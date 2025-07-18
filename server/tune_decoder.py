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
import csv

# ==============================================================================
# 1. HELFER-FUNKTIONEN UND WORKER-FUNKTION (sicher für den Import)
# ==============================================================================

def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"

def get_report_dir():
    """Erstellt das Report-Verzeichnis, falls nicht vorhanden."""
    report_dir = os.path.join("server", "reports", "tune-decoder")
    os.makedirs(report_dir, exist_ok=True)
    return report_dir

def setup_worker_logging(log_level):
    """Initialisiert das Logging für einen Worker-Prozess."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [Worker-PID:%(process)d] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def print_info(msg):
    logging.info(msg)

def print_error(msg):
    logging.error(msg)

# Die Transformation wird global definiert, damit sie im Worker verfügbar ist.
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])

def calculate_wer(prediction, ground_truth):
    if isinstance(prediction, str): prediction = [prediction]
    if isinstance(ground_truth, str): ground_truth = [ground_truth]
    return jiwer.wer(
        ground_truth, prediction,
        reference_transform=transform,
        hypothesis_transform=transform
    )

def get_logits(audio, sampling_rate, processor, model):
    """Berechnet die Logits für eine Audiodatei."""
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values).logits[0]
    return logits.cpu().numpy()

def evaluate_params(task_args, labels, lm_path, logits_cache, ground_truths):
    """Wird im Worker-Prozess für jede (alpha, beta)-Kombination ausgeführt."""
    alpha, beta = task_args
    try:
        print_info(f"Job für Beta: {beta:.2f} STARTING")
        decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path, alpha=alpha, beta=beta)
        predictions = [decoder.decode(logits) for logits in logits_cache]
        avg_wer = calculate_wer(predictions, ground_truths)
        print_info(f"Job für Beta: {beta:.2f} FINISHED")
        return (alpha, beta, avg_wer, predictions)
    except Exception as e:
        print_error(f"bei Beta: {beta:.2f} - {e}")
        return (alpha, beta, float('inf'), [])

def tune_for_single_alpha(validation_data, labels, lm_path, report_dir, target_alpha, processor, model, best_wer_so_far=None):
    """Führt die Grid Search für einen einzelnen Alpha-Wert und alle Beta-Werte durch."""
    NUM_WORKERS = 1  # Wichtig für deinen PC

    beta_range = np.arange(-1.0, 2.0, 0.25)  # müssen unter 16 sein!!!!

    tasks = [(target_alpha, beta) for beta in beta_range]
    total_tasks = len(tasks)
    print_info(f"Teste {total_tasks} Beta-Werte für Alpha = {target_alpha}.")

    print_info("Berechne und cache Logits für alle Validierungsdaten (nur einmal pro Alpha-Lauf)...")
    logits_cache = [get_logits(audio, sr, processor, model).astype(np.float32) for audio, _, sr in validation_data]
    ground_truths = [text for _, text, _ in validation_data]

    worker_func = partial(evaluate_params, labels=labels, lm_path=lm_path, logits_cache=logits_cache, ground_truths=ground_truths)

    ctx = multiprocessing.get_context('spawn')
    print_info(f"Starte Grid Search mit einem Pool von {NUM_WORKERS} Prozess(en)...")

    alpha_run_results = []
    best_wer = None

    commit = get_git_commit_hash()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    with ctx.Pool(processes=NUM_WORKERS, initializer=setup_worker_logging, initargs=(logging.INFO,)) as pool:
        results_iterator = pool.imap_unordered(worker_func, tasks)
        for result in results_iterator:
            alpha, beta, avg_wer, predictions = result
            print_info(f"Ergebnis für Beta: {beta:.2f} -> Avg. WER: {avg_wer:.4f}")
            alpha_run_results.append([f"{alpha:.2f}", f"{beta:.2f}", f"{avg_wer:.4f}"])
            # Prüfe, ob dieser Beta-Lauf der beste ist
            if (best_wer is None) or (avg_wer < best_wer):
                # Schreibe best_run.csv, wenn besser als best_wer_so_far (oder wenn keiner übergeben)
                if (best_wer_so_far is None) or (avg_wer < best_wer_so_far):
                    best_run_path = os.path.join(report_dir, f"{commit}_best_run.csv")
                    with open(best_run_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["index", "wer", "original", "erkannt", "alpha", "beta"])
                        for idx, (orig, pred) in enumerate(zip(ground_truths, predictions)):
                            wer_val = calculate_wer(pred, orig)
                            writer.writerow([idx, f"{wer_val:.4f}", orig, pred, f"{target_alpha:.2f}", f"{beta:.2f}"])
                        # Schreibe den durchschnittlichen WER als letzte Zeile
                        writer.writerow(["avg", f"{avg_wer:.4f}", '', '', f"{target_alpha:.2f}", f"{beta:.2f}"])
                    print_info(f"Best Run gespeichert unter: {best_run_path} (WER: {avg_wer:.4f})")
                best_wer = avg_wer

    # Schreibe das Ergebnis für diesen Alpha-Lauf in eine eigene CSV-Datei
    commit = get_git_commit_hash()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = os.path.join(report_dir, f"{commit}_{now}_alpha_{target_alpha:.2f}.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "beta", "durchschnittliche_wer"])
        writer.writerows(alpha_run_results)

    print_info(f"Ergebnisse für Alpha {target_alpha} gespeichert in: {csv_path}")


# ==============================================================================
# 2. HAUPT-AUSFÜHRUNGSBLOCK
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune KenLM Decoder für einen einzelnen Alpha-Wert")
    parser.add_argument("--alpha", type=float, required=True, help="Der einzelne Alpha-Wert, der getestet werden soll.")
    parser.add_argument("--best_wer", type=float, required=False, help="Bisher bester WER (optional)")
    args = parser.parse_args()
    TARGET_ALPHA = args.alpha
    BEST_WER_SO_FAR = args.best_wer if hasattr(args, 'best_wer') and args.best_wer is not None else None

    # Logging und Report-Verzeichnis einrichten
    REPORT_DIR = get_report_dir()
    # Log-Datei wird jetzt pro Alpha-Lauf benannt, um Konflikte zu vermeiden
    LOG_PATH = os.path.join(REPORT_DIR, f"log_alpha_{TARGET_ALPHA:.2f}.txt")
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
    N_VALIDATION = 4000
    SEED = 42

    if not os.path.isfile(LM_PATH):
        print_error(f"FEHLER: KenLM-Modell nicht gefunden: {LM_PATH}")
        exit(1)

    print_info(f"Lade Wav2Vec2-Modell ({MODEL_NAME}) und Processor ...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    # Fix: If processor is a tuple, unpack it (for compatibility with some huggingface versions)
    if isinstance(processor, tuple):
        processor = processor[0]
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()

    # Get vocab/labels from processor (standard HuggingFace Wav2Vec2Processor)
    try:
        labels = list(processor.tokenizer.get_vocab().keys())  # type: ignore[attr-defined]
    except AttributeError:
        print_error(f"Wav2Vec2Processor attributes: {dir(processor)}")
        raise AttributeError("Wav2Vec2Processor has no 'tokenizer' with 'get_vocab'. Please check your transformers version or processor object.")

    print_info("Lade Common Voice (DE) Testdaten ...")
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "de", split="test", trust_remote_code=True)
    dataset = dataset.select(range(N_VALIDATION))  # type: ignore[attr-defined]
    dataset = dataset.select_columns(["audio", "sentence"])

    print_info(f"Extrahiere {N_VALIDATION} Audiodateien und Transkripte ...")
    validation_data = []
    for i in range(len(dataset)):
        audio_obj = dataset[i]["audio"]
        text = dataset[i]["sentence"].strip()
        if "array" in audio_obj and "sampling_rate" in audio_obj:
            audio = audio_obj["array"]
            sampling_rate = audio_obj["sampling_rate"]
            if sampling_rate is not None and text:
                validation_data.append((audio, text, sampling_rate))

    # Aufruf der Hauptlogik
    tune_for_single_alpha(validation_data, labels, LM_PATH, REPORT_DIR, TARGET_ALPHA, processor, model, best_wer_so_far=BEST_WER_SO_FAR)

    print_info(f"\n[SUCCESS] Lauf für Alpha {TARGET_ALPHA} beendet.")
