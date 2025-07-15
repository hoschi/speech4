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

# Logging-Setup direkt nach den Imports
log_path = os.path.join("server", "reports", "tune-decoder", "debug", "log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
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
dataset = dataset.shuffle(seed=SEED).select(range(N_VALIDATION))
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

# --- Report-Ordner vorbereiten ---
def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"

def get_report_dir():
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    if debug:
        report_dir = os.path.join("server", "reports", "tune-decoder", "debug")
        # Ordner leeren, falls vorhanden
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
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        commit = get_git_commit_hash()
        report_dir = os.path.join("server", "reports", "tune-decoder", f"{commit}_{now}")
        os.makedirs(report_dir, exist_ok=True)
        return report_dir

# --- Anpassung der tune_decoder_params Funktion ---
def tune_decoder_params(validation_data, labels, lm_path, report_dir):
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    if debug:
        alpha_range = [0.5]
        beta_range = [1.5]
        print_info("[DEBUG] Nur ein Testlauf mit alpha=0.5, beta=1.5")
    else:
        alpha_range = np.arange(0.5, 2.5, 0.2)
        beta_range = np.arange(-1.5, 1.0, 0.25)
    best_wer = float('inf')
    best_params = {}
    print_info("[INFO] Starte Grid Search für alpha und beta ...")
    # CSV-File für non-debug
    csv_file = None
    if not debug:
        import csv
        commit = get_git_commit_hash()
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join("server", "reports", "tune-decoder", f"{commit}_{now}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_file = open(csv_path, "w", encoding="utf-8", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["alpha", "beta", "index", "original", "erkannt", "wer"])
    for alpha, beta in itertools.product(alpha_range, beta_range):
        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta
        )
        total_wer = 0
        if debug:
            run_dir = os.path.join(report_dir, f"alpha_{alpha:.2f}_beta_{beta:.2f}")
            os.makedirs(run_dir, exist_ok=True)
        for idx, (audio, ground_truth, sampling_rate) in enumerate(validation_data):
            logits = get_logits(audio, sampling_rate)
            pred = decoder.decode(logits)
            wer_val = calculate_wer(pred, ground_truth)
            total_wer += wer_val
            if debug:
                # Speichere Audio
                audio_path = os.path.join(run_dir, f"sample_{idx:02d}.wav")
                sf.write(audio_path, audio, sampling_rate)
                # Speichere Text und WER (transformiert)
                with open(os.path.join(run_dir, f"sample_{idx:02d}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Original: {' '.join(transform(ground_truth)[0])}\n")
                    f.write(f"Erkannt:  {' '.join(transform(pred)[0])}\n")
                    f.write(f"WER:      {wer_val:.4f}\n")
            else:
                # Schreibe Zeile in CSV (transformiert)
                writer.writerow([
                    f"{alpha:.2f}",
                    f"{beta:.2f}",
                    idx,
                    ' '.join(transform(ground_truth)[0]),
                    ' '.join(transform(pred)[0]),
                    f"{wer_val:.4f}"
                ])
        avg_wer = total_wer / len(validation_data)
        print_info(f"Alpha: {alpha:.2f}, Beta: {beta:.2f}, WER: {avg_wer:.3f}")
        if avg_wer < best_wer:
            best_wer = avg_wer
            best_params = {"alpha": alpha, "beta": beta}
    if csv_file:
        csv_file.close()
    return best_params, best_wer

if __name__ == "__main__":
    if not os.path.isfile(LM_PATH):
        print_error(f"[ERROR] KenLM-Modell nicht gefunden: {LM_PATH}\nBitte trainiere oder kopiere das Modell gemäß README.")
        exit(1)
    report_dir = get_report_dir()
    logging.info(f"Speichere Reports unter: {report_dir}")
    best_params, best_wer = tune_decoder_params(validation_data, labels, LM_PATH, report_dir)
    logging.info(f"\n[RESULT] Optimale Parameter: {best_params} Beste WER: {best_wer}") 