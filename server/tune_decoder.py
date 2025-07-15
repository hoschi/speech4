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

# --- Anpassung der tune_decoder_params Funktion ---
def tune_decoder_params(validation_data, labels, lm_path, report_dir, debug):
    if debug:
        alpha_range = [0.5]
        beta_range = [1.5]
        print_info("[DEBUG] Nur ein Testlauf mit alpha=0.5, beta=1.5")
    else:
        # Reduzierte Reichweite für schnellere Tests, bei Bedarf anpassen
        alpha_range = np.arange(0.5, 2.5, 0.2)
        beta_range = np.arange(-1.5, 1.0, 0.25)
        # too large for my computer
        # alpha_range = np.arange(0, 3.0, 0.2)
        # beta_range = np.arange(-3.0, 3.0, 0.25)

        print_info(f"[INFO] Teste {len(alpha_range)} alpha-Werte und {len(beta_range)} beta-Werte, insgesamt {len(alpha_range) * len(beta_range)} Kombinationen.")

    print_info("[INFO] Berechne und cache Logits für alle Validierungsdaten ...")
    logits_cache = [get_logits(audio, sr).astype(np.float32) for audio, _, sr in validation_data]
    ground_truths = [text for _, text, _ in validation_data]

    import gc
    best_wer = float('inf')
    best_params = {}
    all_results = [] # Nur für die Gesamt-CSV-Datei

    print_info("[INFO] Starte Grid Search für alpha und beta ...")
    for alpha, beta in itertools.product(alpha_range, beta_range):
        # Decoder wird weiterhin hier erstellt, da alpha/beta nicht änderbar sind.
        # Aber wir vermeiden die Speicherung aller Zwischenergebnisse.
        try:
            decoder = build_ctcdecoder(
                labels,
                kenlm_model_path=lm_path,
                alpha=alpha,
                beta=beta
            )
            total_wer = 0
            predictions = []
            for logits in logits_cache:
                pred = decoder.decode(logits)
                predictions.append(pred)

            # Berechne WER für den gesamten Batch
            wer_result = jiwer.wer(
                ground_truths, predictions,
                reference_transform=transform,
                hypothesis_transform=transform
            )
            avg_wer = wer_result

            print_info(f"Alpha: {alpha:.2f}, Beta: {beta:.2f}, Avg. WER: {avg_wer:.4f}")

            # Speichere das aggregierte Ergebnis
            all_results.append([f"{alpha:.2f}", f"{beta:.2f}", f"{avg_wer:.4f}"])

            if avg_wer < best_wer:
                best_wer = avg_wer
                best_params = {"alpha": alpha, "beta": beta}

        except Exception as e:
            print_error(f"Fehler bei Alpha: {alpha:.2f}, Beta: {beta:.2f} - {e}")
        finally:
            # Wichtig: explizit Speicher freigeben
            del decoder
            gc.collect()

    # --- Report-Erstellung NACH der Schleife ---

    # 1. Schreibe die CSV mit den aggregierten Ergebnissen
    if not debug:
        import csv
        commit = get_git_commit_hash()
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(report_dir, f"{commit}_{now}_summary.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["alpha", "beta", "durchschnittliche_wer"])
            writer.writerows(all_results)
        print_info(f"Zusammenfassender Report gespeichert unter: {csv_path}")

    # 2. Erstelle detaillierten Report für die BESTEN Parameter
    if best_params:
        print_info(f"[INFO] Erstelle detaillierten Report für beste Parameter: Alpha={best_params['alpha']:.2f}, Beta={best_params['beta']:.2f}")
        best_decoder = build_ctcdecoder(
            labels, kenlm_model_path=lm_path,
            alpha=best_params["alpha"], beta=best_params["beta"]
        )

        best_run_rows = []
        report_folder = report_dir
        if debug:
            # Eigener Ordner für den besten Lauf im Debug-Modus
            report_folder = os.path.join(report_dir, f"best_run_alpha_{best_params['alpha']:.2f}_beta_{best_params['beta']:.2f}")
            os.makedirs(report_folder, exist_ok=True)

        for idx, (logits, ground_truth) in enumerate(zip(logits_cache, ground_truths)):
            pred = best_decoder.decode(logits)
            wer_val = calculate_wer(pred, ground_truth)
            row = [
                f"{best_params['alpha']:.2f}",
                f"{best_params['beta']:.2f}",
                idx,
                ' '.join(transform(ground_truth)[0]),
                ' '.join(transform(pred)[0]),
                f"{wer_val:.4f}"
            ]
            best_run_rows.append(row)

            if debug:
                 # Korrekter Zugriff auf die zum Index passende Audiodatei
                audio, _, sampling_rate = validation_data[idx]
                audio_path = os.path.join(report_folder, f"sample_{idx:02d}.wav")
                sf.write(audio_path, audio, sampling_rate)
                with open(os.path.join(report_folder, f"sample_{idx:02d}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Original: {row[3]}\n")
                    f.write(f"Erkannt:  {row[4]}\n")
                    f.write(f"WER:      {row[5]}\n")

        if not debug:
            best_csv_path = os.path.join(report_dir, f"{commit}_{now}_best_run_details.csv")
            with open(best_csv_path, "w", encoding="utf-8", newline="") as best_file:
                writer = csv.writer(best_file)
                writer.writerow(["alpha", "beta", "index", "original", "erkannt", "wer"])
                writer.writerows(best_run_rows)
            print_info(f"Detaillierter Report für besten Lauf gespeichert unter: {best_csv_path}")

    return best_params, best_wer

if __name__ == "__main__":
    if not os.path.isfile(LM_PATH):
        print_error(f"[ERROR] KenLM-Modell nicht gefunden: {LM_PATH}\nBitte trainiere oder kopiere das Modell gemäß README.")
        exit(1)
    logging.info(f"Speichere Reports unter: {REPORT_DIR}")
    best_params, best_wer = tune_decoder_params(validation_data, labels, LM_PATH, REPORT_DIR, DEBUG)
    logging.info(f"\n[RESULT] Optimale Parameter: {best_params} Beste WER: {best_wer}") 