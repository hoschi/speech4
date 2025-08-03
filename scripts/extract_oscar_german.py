from datasets import load_dataset
import os
import time

TARGET_SIZE_GB = 4
TARGET_SIZE_BYTES = TARGET_SIZE_GB * 1024 * 1024 * 1024


def extract_german_text_for_kenlm():
    print("Starte Stream...")
    out_path = "german_base_corpus.txt"
    # Lösche Datei, falls vorhanden
    if os.path.exists(out_path):
        os.remove(out_path)
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="de",
        split="train",
        streaming=True
    )
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset:
            # OSCAR liefert dict mit 'text' oder 'content'
            text = ""
            if isinstance(example, dict):
                text = example.get('text') or example.get('content', '')
            elif isinstance(example, str):
                text = example
            text = text.strip()
            text = text.replace('\n', ' ').replace('\r', ' ')
            # Filter: mindestens 10 Wörter
            if len(text.split()) >= 10:
                f.write(text + '\n')
                written += 1
                if written % 1000 == 0:
                    size_gb = os.path.getsize(out_path) / (1024 * 1024 * 1024)
                    print(f"{written} Zeilen geschrieben... ({size_gb:.2f} GB)")
                if os.path.getsize(out_path) >= TARGET_SIZE_BYTES:
                    print(f"Zielgröße erreicht: {size_gb:.2f} GB, {written} Zeilen geschrieben.")
                    break
    print(f"German corpus extraction completed! {written} lines written, {os.path.getsize(out_path)/(1024*1024*1024):.2f} GB.")

if __name__ == "__main__":
    extract_german_text_for_kenlm()
