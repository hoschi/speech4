from datasets import load_dataset

def extract_german_text_for_kenlm():
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="de",
        streaming=True
    )

    stream = dataset["train"]
    print("Starte Stream...")

    count = 0
    for example in stream:
        # Robust: handle dicts und strings
        if isinstance(example, dict):
            text = example.get("text", "").strip()
        elif isinstance(example, str):
            text = example.strip()
        else:
            continue  # unbekannter Typ – überspringen

        if text:
            print(f"\n--- Beispiel {count + 1} ---")
            print(text[:500])
            count += 1

        if count >= 10:
            break

if __name__ == "__main__":
    extract_german_text_for_kenlm()
