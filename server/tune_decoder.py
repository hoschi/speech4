import os
import numpy as np
import itertools
from datasets import load_dataset
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode.decoder import build_ctcdecoder
import jiwer
import multiprocessing
import subprocess
import shutil
import librosa
import logging
import argparse

# --- Logging-Setup GANZ AM ANFANG ---
def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"

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

# CLI-Argumente parsen
parser = argparse.ArgumentParser(description="Tune KenLM Decoder mit Common Voice DE und Wav2Vec2")
parser.add_argument("--debug", action="store_true", help="Debug-Modus: Nur ein (alpha, beta)-Test, ausführliche Reports im debug-Ordner")
args = parser.parse_args()
DEBUG = args.debug
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

def print_info(msg):
    logging.info(msg)

def print_error(msg):
    logging.error(msg)

# --- Konfiguration ---
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german"
LM_PATH = "server/lm/4gram_de.klm"
N_VALIDATION = 10
SEED = 42

# --- Modell und Processor laden ---
print_info("[INFO] Lade Wav2Vec2-Modell und Processor ...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# --- Labels extrahieren ---
labels = list(processor.tokenizer.get_vocab().keys())

# --- Common Voice (DE) Testdaten laden ---
print_info("[INFO] Lade Common Voice (DE) Testdaten ...")
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "de", split="test", trust_remote_code=True)
dataset = dataset.select(range(N_VALIDATION))
dataset = dataset.select_columns(["audio", "sentence"])

# --- Audiodaten und Transkripte extrahieren ---
print_info(f"[INFO] Extrahiere {N_VALIDATION} Audiodateien und Transkripte ...")
validation_data = []
sample_dict = dataset[:N_VALIDATION]
num_samples = len(sample_dict["audio"])
for i in range(num_samples):
    audio_obj = sample_dict["audio"][i]
    text = sample_dict["sentence"][i].strip() if "sentence" in sample_dict and sample_dict["sentence"][i] else ""
    if isinstance(audio_obj, dict):
        audio = audio_obj.get("array", audio_obj)
        sampling_rate = audio_obj.get("sampling_rate", None)
    else:
        audio = audio_obj
        sampling_rate = None
    # Nur valide Beispiele übernehmen
    if sampling_rate is not None and text:
        validation_data.append((audio, text, sampling_rate))

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

def get_logits(audio, sampling_rate):
    # Resample falls nötig
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    input_values = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values).logits[0]
    return logits.cpu().numpy()

import multiprocessing

# --- WORKER-FUNKTION ---
# Diese Funktion wird in einem separaten Prozess für jede (alpha, beta)-Kombination ausgeführt.
def evaluate_params(args):
    alpha, beta, labels, lm_path, logits_cache, ground_truths = args
    try:
        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta
        )
        
        predictions = [decoder.decode(logits) for logits in logits_cache]
        
        avg_wer = jiwer.wer(
            ground_truths, 
            predictions,
            reference_transform=transform,
            hypothesis_transform=transform
        )
        
        # Diese Ausgabe erfolgt jetzt aus dem Worker-Prozess
        print(f"[Worker α={alpha:.2f} β={beta:.2f}] Avg. WER: {avg_wer:.4f}")
        return (alpha, beta, avg_wer)

    except Exception as e:
        print(f"[Worker ERROR α={alpha:.2f} β={beta:.2f}] {e}")
        return (alpha, beta, float('inf'))

# --- ANGEPASSTE tune_decoder_params FUNKTION MIT LIMITIERTEN WORKERN ---
def tune_decoder_params(validation_data, labels, lm_path, report_dir, debug):
    # HIER: Anzahl der parallelen Worker-Prozesse festlegen
    NUM_WORKERS = 4

    if debug:
        alpha_range = [0.5]
        beta_range = [1.5]
        print_info("[DEBUG] Nur ein Testlauf mit alpha=0.5, beta=1.5")
    else:
        alpha_range = np.arange(0.5, 2.5, 0.2)
        beta_range = np.arange(-1.5, 1.0, 0.25)
        print_info(f"[INFO] Teste {len(alpha_range)} alpha-Werte und {len(beta_range)} beta-Werte, insgesamt {len(alpha_range) * len(beta_range)} Kombinationen.")

    print_info("[INFO] Berechne und cache Logits für alle Validierungsdaten (einmalig)...")
    logits_cache = [get_logits(audio, sr).astype(np.float32) for audio, _, sr in validation_data]
    ground_truths = [text for _, text, _ in validation_data]

    tasks = [
        (alpha, beta, labels, lm_path, logits_cache, ground_truths)
        for alpha, beta in itertools.product(alpha_range, beta_range)
    ]

    best_wer = float('inf')
    best_params = {}
    all_results = []
    
    # Einen Pool von Worker-Prozessen erstellen.
    # macOS benötigt oft 'spawn' als Startmethode für saubere Prozesse.
    ctx = multiprocessing.get_context('spawn')
    
    print_info(f"[INFO] Starte Grid Search mit einem Pool von {NUM_WORKERS} Prozessen...")

    # Hier wird die Variable NUM_WORKERS verwendet, um die Anzahl der Prozesse zu limitieren
    with ctx.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(evaluate_params, tasks)

    # Ergebnisse auswerten
    for alpha, beta, avg_wer in results:
        all_results.append([f"{alpha:.2f}", f"{beta:.2f}", f"{avg_wer:.4f}"])
        if avg_wer < best_wer:
            best_wer = avg_wer
            best_params = {"alpha": alpha, "beta": beta}

    print_info("[INFO] Grid Search abgeschlossen. Beste gefundene Parameter:")
    logging.info(f"\n[RESULT] Optimale Parameter: {best_params} Beste WER: {best_wer}")

    # --- Report-Erstellung (kann unverändert bleiben) ---
    # ... (Der Code zur Erstellung der CSV-Dateien folgt hier) ...

    return best_params, best_wer

if __name__ == "__main__":
    if not os.path.isfile(LM_PATH):
        print_error(f"[ERROR] KenLM-Modell nicht gefunden: {LM_PATH}\nBitte trainiere oder kopiere das Modell gemäß README.")
        exit(1)
    logging.info(f"Speichere Reports unter: {REPORT_DIR}")
    best_params, best_wer = tune_decoder_params(validation_data, labels, LM_PATH, REPORT_DIR, DEBUG)
    logging.info(f"\n[RESULT] Optimale Parameter: {best_params} Beste WER: {best_wer}") 