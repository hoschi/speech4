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
OUTPUT_DIR = "server/lm/"

def get_correction_files():
    return sorted(glob.glob(os.path.join(CORRECTIONS_DIR, "*.txt")))

def main():
    if not os.path.isfile(BASE_CORPUS):
        print(f"[ERROR] Basis-Korpus nicht gefunden: {BASE_CORPUS}")
        sys.exit(1)
    user_corrections = get_correction_files()
    if not user_corrections:
        print(f"[WARN] Keine Korrekturen gefunden, es wird nur der Basis-Korpus verwendet.")
    trainer = PersonalizedKenLMTrainer(
        base_corpus=BASE_CORPUS,
        user_correction_files=user_corrections,
        output_dir=OUTPUT_DIR
    )
    model_path = trainer.train_complete_pipeline()
    print(f"[SUCCESS] KenLM-Modell gespeichert unter: {model_path}")

if __name__ == "__main__":
    main()
