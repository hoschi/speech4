"""
Hauptskript für KenLM-Training und Personalisierung
Führt die Pipeline aus: Preprocessing, Korpus-Zusammenführung, Training, Komprimierung
Verwendet Basis-Korpus und alle Korrekturen in server/corrections/
"""
import os
import glob
import sys
import argparse
from kenlm_personalization import PersonalizedKenLMTrainer

BASE_CORPUS = "server/data/german_base_corpus.txt"
BASE_ARPA = "server/lm/base_model.arpa"
CORRECTIONS_DIR = "server/corrections/"
MARKDOWN_RAW_DIR = "server/markdown_input_raw/"
OUTPUT_DIR = "server/lm/"

def get_correction_files():
    return sorted(glob.glob(os.path.join(CORRECTIONS_DIR, "*.txt")))

def get_markdown_files():
    return sorted(glob.glob(os.path.join(MARKDOWN_RAW_DIR, "*.md")))

def main():
    parser = argparse.ArgumentParser(description="KenLM Training mit adaptivem Pruning")
    parser.add_argument("--regenerate-base-arpa", type=str, default="false", help="Basismodell (ARPA) neu generieren: true|false (default: false)")
    args = parser.parse_args()
    regenerate_base_arpa = args.regenerate_base_arpa.lower() == "true"

    if not os.path.isfile(BASE_CORPUS):
        print(f"[ERROR] Basis-Korpus nicht gefunden: {BASE_CORPUS}")
        sys.exit(1)
    user_corrections = get_correction_files()
    markdown_notes = get_markdown_files()
    if not user_corrections:
        print(f"[WARN] Keine Korrekturen gefunden, es wird nur der Basis-Korpus verwendet.")
    if not markdown_notes:
        print(f"[WARN] Keine Markdown-Notizen gefunden, Personalisierung erfolgt nur mit Korrekturen.")

    # Schritt 1: Markdown-Notizen bereinigen und als temporäre Datei speichern
    cleaned_markdown_path = os.path.join(OUTPUT_DIR, "markdown_cleaned.txt")
    if markdown_notes:
        from kenlm_personalization import process_markdown_notes
        process_markdown_notes(markdown_notes, cleaned_markdown_path)
    else:
        cleaned_markdown_path = None

    # Schritt 2: Korrekturen und bereinigte Markdown-Notizen zusammenführen
    all_personalization_files = user_corrections.copy()
    if cleaned_markdown_path:
        all_personalization_files.append(cleaned_markdown_path)

    # Basiskorpus-ARPA Handling: nur prüfen, ob existiert, falls Flag false
    if not regenerate_base_arpa and not os.path.isfile(BASE_ARPA):
        print(f"[ERROR] Basismodell (ARPA) nicht gefunden: {BASE_ARPA}")
        print(f"Bitte --regenerate-base-arpa=true setzen, um das Basismodell neu zu generieren.")
        sys.exit(1)
    if regenerate_base_arpa:
        print(f"[INFO] Basismodell (ARPA) wird neu generiert...")
    else:
        print(f"[INFO] Existierendes Basismodell (ARPA) wird verwendet: {BASE_ARPA}")

    trainer = PersonalizedKenLMTrainer(
        base_corpus=BASE_CORPUS,
        user_correction_files=all_personalization_files,
        output_dir=OUTPUT_DIR
    )
    model_path, hotwords = trainer.train_adaptive_pruning_pipeline(lambda_mix=0.95, regenerate_base_arpa=regenerate_base_arpa)
    print(f"[SUCCESS] KenLM-Modell gespeichert unter: {model_path}")
    # Hotwords ist ein Pfad zu einer Textdatei, Begriffe einlesen
    from pathlib import Path
    if hotwords and Path(hotwords).is_file():
        hotwords_list = Path(hotwords).read_text(encoding="utf-8").splitlines()
        print(f"[INFO] Hotwords extrahiert: {hotwords_list[:10]} ... (insgesamt {len(hotwords_list)})")
    else:
        print(f"[INFO] Hotwords-Datei nicht gefunden: {hotwords}")

if __name__ == "__main__":
    main()
