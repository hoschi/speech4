from datasets import load_dataset

def extract_german_text_for_kenlm(max_lines=20000):
    print("Starte Stream...")
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="de",
        split="train",
        streaming=True
    )
    written = 0
    with open("german_base_corpus.txt", "w", encoding="utf-8") as f:
        for example in dataset:
            # OSCAR liefert dict mit 'text' oder 'content'
            text = ""
            if isinstance(example, dict):
                text = example.get('text') or example.get('content', '')
            elif isinstance(example, str):
                text = example
            text = text.strip()
            if len(text) > 10:
                f.write(text + '\n')
                written += 1
                if written % 1000 == 0:
                    print(f"{written} Zeilen geschrieben...")
                if written >= max_lines:
                    break
    print(f"German corpus extraction completed! {written} lines written.")

if __name__ == "__main__":
    extract_german_text_for_kenlm()
