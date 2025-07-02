from datasets import load_dataset

def extract_german_text_for_kenlm(max_lines=20000):
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="de",
        streaming=True,
        use_auth_token=True
    )
    with open("german_base_corpus.txt", "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            # Robust: Dict, Tupel oder String
            if isinstance(example, dict):
                text = example.get('text') or example.get('content', '')
            elif isinstance(example, tuple):
                value = example[1]
                if isinstance(value, dict):
                    text = value.get('text') or value.get('content', '')
                elif isinstance(value, str):
                    text = value
                else:
                    continue
            elif isinstance(example, str):
                text = example
            else:
                continue
            text = text.lower().strip()
            if i < 10:
                print(f"Example {i}: {repr(text)}")
            if len(text) > 10:
                f.write(text + '\n')
            if i+1 >= max_lines:
                break
    print(f"German corpus extraction completed! {max_lines} lines written.")

if __name__ == "__main__":
    extract_german_text_for_kenlm() 