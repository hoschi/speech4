import os
import numpy as np
import itertools
from datasets import load_dataset
import torch
import jiwer
import datetime
import subprocess
import logging
import argparse
import multiprocessing
from functools import partial
import csv

# Importiere die zentrale ASR-Modell-Klasse und Konstanten
from server.asr_model import ASRModel, LM_PATH, MODEL_NAME

# ==============================================================================
# 1. HELFER-FUNKTIONEN UND WORKER-FUNKTION
# ==============================================================================

def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"

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

def evaluate_params(task_args, labels, lm_path, logits_cache, ground_truths):
    """
    Wird im Worker-Prozess für jede (alpha, beta)-Kombination ausgeführt.
    Baut den Decoder on-the-fly, um Serialisierungsprobleme zu vermeiden.
    """
    alpha, beta = task_args
    # Importiere die Decoder-Funktion hier, da sie im Worker-Kontext benötigt wird.
    from pyctcdecode.decoder import build_ctcdecoder
    
    try:
        print_info(f"Job für Beta: {beta:.2f} STARTING")
        decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path, alpha=alpha, beta=beta)
        predictions = [decoder.decode(logits) for logits in logits_cache]
        avg_wer = calculate_wer(predictions, ground_truths)
        print_info(f"Job für Beta: {beta:.2f} FINISHED")
        return (alpha, beta, avg_wer, predictions)
    except Exception as e:
        print_error(f"Fehler bei Beta: {beta:.2f} - {e}")
        return (alpha, beta, float('inf'), [])

def tune_for_single_alpha(validation_data, asr_model, report_dir, target_alpha, best_wer_so_far=None):
    """Führt die Grid Search für einen einzelnen Alpha-Wert und alle Beta-Werte durch."""
    NUM_WORKERS = 1  # Für lokale Ausführung
    beta_range = np.arange(-2.0, 2.1, 0.25)

    tasks = [(target_alpha, beta) for beta in beta_range]
    total_tasks = len(tasks)
    print_info(f"Teste {total_tasks} Beta-Werte für Alpha = {target_alpha}.")

    print_info("Berechne und cache Logits für alle Validierungsdaten...")
    logits_cache = [asr_model.get_logits(audio, sr).astype(np.float32) for audio, _, sr in validation_data]
    ground_truths = [text for _, text, _ in validation_data]

    # Bereite die Argumente für den Worker vor
    worker_func = partial(evaluate_params, 
                          labels=asr_model.labels, 
                          lm_path=LM_PATH, 
                          logits_cache=logits_cache, 
                          ground_truths=ground_truths)

    ctx = multiprocessing.get_context('spawn')
    print_info(f"Starte Grid Search mit einem Pool von {NUM_WORKERS} Prozess(en)...")

    alpha_run_results = []
    best_wer_for_alpha = float('inf')
    
    commit = get_git_commit_hash()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    with ctx.Pool(processes=NUM_WORKERS, initializer=setup_worker_logging, initargs=(logging.INFO,)) as pool:
        results_iterator = pool.imap_unordered(worker_func, tasks)
        for result in results_iterator:
            alpha, beta, avg_wer, predictions = result
            if avg_wer == float('inf'):
                continue

            print_info(f"Ergebnis für Beta: {beta:.2f} -> Avg. WER: {avg_wer:.4f}")
            alpha_run_results.append([f"{alpha:.2f}", f"{beta:.2f}", f"{avg_wer:.4f}"])

            # Prüfe, ob dieser Lauf der bisher beste ist
            is_best_so_far = (best_wer_so_far is None) or (avg_wer < best_wer_so_far)
            if avg_wer < best_wer_for_alpha and is_best_so_far:
                best_run_path = os.path.join(report_dir, f"{commit}_best_run.csv")
                with open(best_run_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["index", "wer", "original", "erkannt", "alpha", "beta"])
                    for idx, (orig, pred) in enumerate(zip(ground_truths, predictions)):
                        wer_val = calculate_wer(pred, orig)
                        writer.writerow([idx, f"{wer_val:.4f}", orig, pred, f"{target_alpha:.2f}", f"{beta:.2f}"])
                    writer.writerow(["avg", f"{avg_wer:.4f}", '', '', f"{target_alpha:.2f}", f"{beta:.2f}"])
                print_info(f"Neuer bester Lauf gespeichert: {best_run_path} (WER: {avg_wer:.4f})")
                best_wer_for_alpha = avg_wer

    # Schreibe die Ergebnisse für diesen Alpha-Lauf
    csv_path = os.path.join(report_dir, f"{commit}_{now}_alpha_{target_alpha:.2f}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "beta", "durchschnittliche_wer"])
        writer.writerows(sorted(alpha_run_results, key=lambda r: float(r[2])))

    print_info(f"Ergebnisse für Alpha {target_alpha} gespeichert in: {csv_path}")


# ==============================================================================
# 2. HAUPT-AUSFÜHRUNGSBLOCK
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune KenLM Decoder für einen einzelnen Alpha-Wert")
    parser.add_argument("--alpha", type=float, required=True, help="Der einzelne Alpha-Wert, der getestet werden soll.")
    parser.add_argument("--best_wer", type=float, required=False, help="Bisher bester WER (optional)")
    parser.add_argument("--report_dir", type=str, required=True, help="Pfad zum Report-Ordner")
    args = parser.parse_args()
    
    TARGET_ALPHA = args.alpha
    BEST_WER_SO_FAR = args.best_wer
    REPORT_DIR = args.report_dir

    os.makedirs(REPORT_DIR, exist_ok=True)
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
    N_VALIDATION = 400 # Reduziert für schnellere Testläufe, kann erhöht werden
    
    if not os.path.isfile(LM_PATH):
        print_error(f"FEHLER: KenLM-Modell nicht gefunden: {LM_PATH}")
        exit(1)

    # Lade das zentrale ASR-Modell
    # Hinweis: Das Modell wird hier anders initialisiert als im Server, um Flexibilität zu wahren.
    asr_model = ASRModel(model_name="aware-ai/wav2vec2-base-german")

    print_info("Lade Common Voice (DE) Testdaten ...")
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "de", split="test", trust_remote_code=True)
    dataset = dataset.select(range(N_VALIDATION))
    dataset = dataset.select_columns(["audio", "sentence"])

    print_info(f"Extrahiere {N_VALIDATION} Audiodateien und Transkripte ...")
    validation_data = []
    for item in dataset:
        audio_obj = item["audio"]
        text = item["sentence"].strip()
        if "array" in audio_obj and "sampling_rate" in audio_obj and text:
            validation_data.append((audio_obj["array"], text, audio_obj["sampling_rate"]))

    # Aufruf der Hauptlogik
    tune_for_single_alpha(validation_data, asr_model, REPORT_DIR, TARGET_ALPHA, best_wer_so_far=BEST_WER_SO_FAR)

    print_info(f"\n[SUCCESS] Lauf für Alpha {TARGET_ALPHA} beendet.")
