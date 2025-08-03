"""
Hauptskript für KenLM-Training und Personalisierung
Führt die Pipeline aus: Preprocessing, Korpus-Zusammenführung, Training, Komprimierung
Verwendet Basis-Korpus und alle Korrekturen in server/corrections/
"""
import os
import glob
import sys
from kenlm_personalization import PersonalizedKenLMTrainer

BASE_CORPUS = "server/data/german_base_corpus.txt"
CORRECTIONS_DIR = "server/corrections/"
MARKDOWN_RAW_DIR = "server/markdown_input_raw/"
OUTPUT_DIR = "server/lm/"

def get_correction_files():
    return sorted(glob.glob(os.path.join(CORRECTIONS_DIR, "*.txt")))

def get_markdown_files():
    return sorted(glob.glob(os.path.join(MARKDOWN_RAW_DIR, "*.md")))

def main():
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
    # Die Korrekturen werden direkt übernommen, Markdown-Notizen werden bereinigt
    # Die PersonalizedKenLMTrainer erwartet beide als user_correction_files
    all_personalization_files = user_corrections.copy()
    if cleaned_markdown_path:
        all_personalization_files.append(cleaned_markdown_path)

    trainer = PersonalizedKenLMTrainer(
        base_corpus=BASE_CORPUS,
        user_correction_files=all_personalization_files,
        output_dir=OUTPUT_DIR
    )
    model_path = trainer.train_complete_pipeline()
    print(f"[SUCCESS] KenLM-Modell gespeichert unter: {model_path}")

if __name__ == "__main__":
    main()
