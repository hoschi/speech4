import os
import numpy as np
import itertools
from datasets import load_dataset
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode.decoder import build_ctcdecoder
import jiwer
import datetime
import subprocess
import shutil
import librosa
import logging
import argparse
import multiprocessing
from functools import partial

# ==============================================================================
# 1. HELFER-FUNKTIONEN UND WORKER-FUNKTIONEN (sicher für den Import)
# ==============================================================================

def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"

def print_info(msg):
    logging.info(msg)

def print_error(msg):
    logging.error(msg)

def get_report_dir(debug):
    if debug:
        report_dir = os.path.join("server", "reports", "tune-decoder", "debug")
        if os.path.exists(report_dir):
            for f in os.listdir(report_dir):
                fp = os.path.join(report_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
        else:
            os.makedirs(report_dir, exist_ok=True)
        return report_dir
    else:
        report_dir = os.path.join("server", "reports", "tune-decoder")
        os.makedirs(report_dir, exist_ok=True)
        return report_dir

# Die Transformation wird global definiert, damit sie im Worker verfügbar ist.
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()
])

def calculate_wer(prediction, ground_truth):
    return jiwer.wer(
        ground_truth, prediction,
        reference_transform=transform,
        hypothesis_transform=transform
    )

def get_logits(audio, sampling_rate, processor, model):
    """Berechnet die Logits für eine Audiodatei. Benötigt Prozessor und Modell explizit."""
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values).logits[0]
    return logits.cpu().numpy()

def evaluate_params(task_args, labels, lm_path, logits_cache, ground_truths):
    """
    Diese Funktion wird in einem separaten Worker-Prozess ausgeführt.
    Sie erhält die gleichbleibenden Daten (labels, lm_path etc.) über 'partial'.
    """
    alpha, beta = task_args
    try:
        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta,
        )
        
        predictions = [decoder.decode(logits) for logits in logits_cache]
        avg_wer = calculate_wer(predictions, ground_truths)
        
        # Diese Ausgabe kommt jetzt nur noch aus dem Worker
        print(f"[Worker] Alpha: {alpha:.2f}, Beta: {beta:.2f}, Avg. WER: {avg_wer:.4f}")
        return (alpha, beta, avg_wer)

    except Exception as e:
        print_error(f"[Worker ERROR] bei Alpha: {alpha:.2f}, Beta: {beta:.2f} - {e}")
        return (alpha, beta, float('inf'))

def tune_decoder_params(validation_data, labels, lm_path, report_dir, debug, processor, model):
    """
    Hauptfunktion für die Grid-Search.
    Bereitet Daten vor und startet den Multiprocessing-Pool.
    """
    NUM_WORKERS = 4  # Anzahl der parallelen Prozesse

    if debug:
        alpha_range = [0.5]
        beta_range = [1.5]
    else:
        alpha_range = np.arange(0.5, 2.5, 0.2)
        beta_range = np.arange(-1.5, 1.0, 0.25)
    
    print_info(f"[INFO] Teste {len(alpha_range) * len(beta_range)} (alpha, beta)-Kombinationen.")

    # Logits und Ground Truths einmalig im Hauptprozess vorbereiten
    print_info("[INFO] Berechne und cache Logits für alle Validierungsdaten (einmalig im Hauptprozess)...")
    logits_cache = [get_logits(audio, sr, processor, model).astype(np.float32) for audio, _, sr in validation_data]
    ground_truths = [text for _, text, _ in validation_data]

    # Aufgabenliste enthält nur die sich ändernden Parameter
    tasks = list(itertools.product(alpha_range, beta_range))

    best_wer = float('inf')
    best_params = {}
    
    # partial wird verwendet, um die Worker-Funktion mit den statischen Argumenten "vorzuladen".
    # Das ist effizienter als die Daten mit jeder Aufgabe neu zu senden.
    worker_func = partial(
        evaluate_params,
        labels=labels,
        lm_path=lm_path,
        logits_cache=logits_cache,
        ground_truths=ground_truths
    )
    
    ctx = multiprocessing.get_context('spawn')
    print_info(f"[INFO] Starte Grid Search mit einem Pool von {NUM_WORKERS} Prozessen...")

    with ctx.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(worker_func, tasks)

    print_info("[INFO] Grid Search abgeschlossen. Werte Ergebnisse aus...")
    all_results = []
    for alpha, beta, avg_wer in results:
        all_results.append([f"{alpha:.2f}", f"{beta:.2f}", f"{avg_wer:.4f}"])
        if avg_wer < best_wer:
            best_wer = avg_wer
            best_params = {"alpha": alpha, "beta": beta}

    # --- Report-Erstellung (findet nur im Hauptprozess statt) ---
    # ... (Der Code zur Erstellung der CSV-Dateien kann hier wie in meiner vorherigen Antwort folgen) ...

    return best_params, best_wer

# ==============================================================================
# 2. HAUPT-AUSFÜHRUNGSBLOCK
# ==============================================================================

if __name__ == "__main__":
    # Argumente parsen
    parser = argparse.ArgumentParser(description="Tune KenLM Decoder mit Common Voice DE und Wav2Vec2")
    parser.add_argument("--debug", action="store_true", help="Debug-Modus: Nur ein (alpha, beta)-Test, ausführliche Reports im debug-Ordner")
    args = parser.parse_args()
    DEBUG = args.debug

    # Logging und Report-Verzeichnis einrichten
    REPORT_DIR = get_report_dir(DEBUG)
    LOG_PATH = os.path.join(REPORT_DIR, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    # --- Konfiguration ---
    MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german"
    LM_PATH = "server/lm/4gram_de.klm"
    N_VALIDATION = 10
    SEED = 42

    # Überprüfe, ob das LM-Modell existiert
    if not os.path.isfile(LM_PATH):
        print_error(f"[ERROR] KenLM-Modell nicht gefunden: {LM_PATH}\nBitte trainiere oder kopiere das Modell gemäß README.")
        exit(1)

    # --- Laden der Modelle und Daten (jetzt sicher im Hauptprozess) ---
    print_info("[INFO] Lade Wav2Vec2-Modell und Processor ...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()

    # Labels extrahieren (robust gegen Typenfehler)
    try:
        # Prüfe, ob processor ein Wav2Vec2Processor ist und tokenizer besitzt
        from transformers import Wav2Vec2Processor
        if isinstance(processor, Wav2Vec2Processor) and hasattr(processor, "tokenizer"):
            labels = list(processor.tokenizer.get_vocab().keys())
        else:
            raise RuntimeError("Processor ist kein Wav2Vec2Processor oder hat kein tokenizer-Attribut!")
    except Exception as e:
        raise RuntimeError(f"Fehler beim Extrahieren der Labels: {e}")

    print_info("[INFO] Lade Common Voice (DE) Testdaten ...")
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "de", split="test", trust_remote_code=True)
    from datasets import Dataset
    if isinstance(dataset, Dataset):
        dataset = dataset.select(range(N_VALIDATION))
        sample_dict = dataset.to_dict()
        num_samples = len(sample_dict["audio"])
        validation_data = []
        for i in range(num_samples):
            audio_obj = sample_dict["audio"][i]
            text = sample_dict["sentence"][i].strip() if sample_dict["sentence"][i] else ""
            if isinstance(audio_obj, dict):
                audio = audio_obj.get("array", audio_obj)
                sampling_rate = audio_obj.get("sampling_rate", None)
            else:
                audio = audio_obj
                sampling_rate = None
            if sampling_rate is not None and text:
                validation_data.append((audio, text, sampling_rate))
    else:
        # Fallback für IterableDataset, Iterator oder Liste
        dataset_list = list(dataset)[:N_VALIDATION]
        validation_data = []
        for entry in dataset_list:
            audio_obj = entry["audio"]
            text = entry["sentence"].strip() if entry["sentence"] else ""
            if isinstance(audio_obj, dict):
                audio = audio_obj.get("array", audio_obj)
                sampling_rate = audio_obj.get("sampling_rate", None)
            else:
                audio = audio_obj
                sampling_rate = None
            if sampling_rate is not None and text:
                validation_data.append((audio, text, sampling_rate))

    # --- Start der eigentlichen Logik ---
    print_info(f"Speichere Reports unter: {REPORT_DIR}")
    best_params, best_wer = tune_decoder_params(validation_data, labels, LM_PATH, REPORT_DIR, DEBUG, processor, model)
    
    print_info(f"\n[RESULT] Optimale Parameter: {best_params} Beste WER: {best_wer}")