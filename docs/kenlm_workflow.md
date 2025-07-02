# KenLM-Workflow: Training, Integration & Automatisierung

## 1. KenLM-Modell trainieren

- Erzeuge ein n-Gramm-Modell (empfohlen: 3- oder 4-Gramm, Kneser-Ney Glättung):

```bash
lmplz -o 4 --text corpus.txt --arpa corpus.arpa
```

- Komprimiere das Modell für schnellere Nutzung:

```bash
build_binary corpus.arpa corpus.klm
```

Jetzt hast du dein eigenes KenLM-Modell `corpus.klm`.

## 2. Integration in pyctcdecode

- Lade dein Modell im Python-Backend:

```python
from pyctcdecode import build_ctcdecoder
import kenlm

# Lade dein Vokabular (aus dem Tokenizer des ASR-Modells)
unique_vocab = [...]  # Liste aller Token/Wörter

kenlm_model = kenlm.Model('corpus.klm')
decoder = build_ctcdecoder(unique_vocab, kenlm_model, alpha=0.5, beta=1.0)
```

- Nutze den Decoder nach der ASR-Inferenz:

```python
logits = ...  # Output deines Wav2Vec2-CTC-Modells
transcript = decoder.decode(logits)
```

## 3. Kontinuierliche Aktualisierung

- Nach jedem Personalisierungs-Loop:
    - **nur neue, noch nicht verarbeitetete** Korrekturpaare in `corpus.txt` anhängen
    - KenLM-Modell erneut bauen (`lmplz`, `build_binary`)
    - Neues Modell in pyctcdecode laden (ggf. Server neu starten oder Hot-Reload implementieren)

## 4. Automatisierung

- Wenn Training getriggert wird, mache folgendes:
    - Die Textdaten sammelt/vereinigt
    - Das KenLM-Modell neu trainiert
    - Das Modell für die Inferenz bereitstellt

## Quellen und Step-by-Step-Referenz

- [GitHub: KenLM-Training Schritt-für-Schritt](https://github.com/kmario23/KenLM-training)
- [pyctcdecode + KenLM Integration](https://huggingface.co/mesolitica/wav2vec2-xls-r-300m-mixed/discussions/1)
- [Weitere Quellen siehe User-Query] 