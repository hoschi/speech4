<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Erstelle eine Schritt für Schritt Anleitung für meine Coding KI um das Hyperparameter-Tuning mit den Mozilla Common Voice (DE) Daten umzusetzen.

https://www.perplexity.ai/search/bei-meinem-aktuellen-projekt-h-p3YQ8JYSQ0eC2ztoRb1s7Q#2

Hier ist eine **Schritt-für-Schritt-Anleitung**, wie du mit Python und Hugging Face Datasets das Hyperparameter-Tuning für deinen Decoder mit Mozilla Common Voice (DE) umsetzt:

## 1. Voraussetzungen

- Python ≥ 3.8
- Installiere folgende Pakete (z.B. via pip):

```bash
pip install datasets torchaudio soundfile pyctcdecode
```


## 2. Common Voice (DE) Datensatz laden und vorbereiten

```python
from datasets import load_dataset

# Lade 200 zufällige Beispiele aus dem 'test'-Split von Common Voice 12 (Deutsch)
cv = load_dataset("mozilla-foundation/common_voice_12_0", "de", split="test")
cv = cv.shuffle(seed=42).select(range(200))  # 200 zufällige Beispiele

# Optional: nur die benötigten Spalten behalten
cv = cv.select_columns(["audio", "sentence"])
```


## 3. Audio und Transkripte extrahieren

```python
# Extrahiere WAV-Audiodaten und Ground-Truth-Transkripte
validation_data = []
for sample in cv:
    audio = sample["audio"]["array"]  # NumPy-Array (16 kHz)
    text = sample["sentence"].strip()
    validation_data.append((audio, text))
```


## 4. Decoder-Tuning vorbereiten

Stelle sicher, dass du folgende Komponenten hast:

- Dein ASR-Modell (z.B. wav2vec2)
- Die Funktion `build_ctcdecoder` (pyctcdecode)
- Eine Funktion zum Berechnen des WER (z.B. aus jiwer oder selbst implementiert)


## 5. Grid Search für α und β

```python
import itertools
import numpy as np

def tune_decoder_params(validation_data, base_decoder_config, wav2vec_model):
    alpha_range = np.arange(0, 3.0, 0.2)
    beta_range = np.arange(-3.0, 3.0, 0.25)
    best_wer = float('inf')
    best_params = {}

    for alpha, beta in itertools.product(alpha_range, beta_range):
        decoder = build_ctcdecoder(
            base_decoder_config['labels'],
            kenlm_model_path=base_decoder_config['lm_path'],
            alpha=alpha,
            beta=beta
        )

        total_wer = 0
        for audio, ground_truth in validation_data:
            # Modell-Inferenz: logits berechnen
            logits = wav2vec_model(audio)  # ggf. mit torch.no_grad()
            pred = decoder.decode(logits.numpy())
            wer = calculate_wer(pred, ground_truth)
            total_wer += wer
        avg_wer = total_wer / len(validation_data)

        if avg_wer < best_wer:
            best_wer = avg_wer
            best_params = {"alpha": alpha, "beta": beta}

        print(f"Alpha: {alpha}, Beta: {beta}, WER: {avg_wer:.3f}")

    return best_params, best_wer
```


## 6. WER-Berechnung (z.B. mit jiwer)

```python
from jiwer import wer

def calculate_wer(prediction, ground_truth):
    return wer(ground_truth, prediction)
```


## 7. Tuning starten

```python
# Beispiel-Konfiguration
base_decoder_config = {
    "labels": [...],           # Dein Token-Vokabular
    "lm_path": "model.arpa"    # Pfad zu deinem KenLM-Modell
}

# wav2vec_model muss ein Funktion/Objekt sein, das aus Audio-Array die Logits berechnet
best_params, best_wer = tune_decoder_params(validation_data, base_decoder_config, wav2vec_model)
print("Optimale Parameter:", best_params, "Beste WER:", best_wer)
```

**Hinweis:**

- Die Common Voice Daten sind unter CC-BY 4.0 frei verwendbar[^1][^2].
- Die oben gezeigte Auswahl von 200 Beispielen ist für Hyperparameter-Tuning ausreichend und entspricht der gängigen Praxis[^2][^3].

**Damit hast du ein valides, reproduzierbares Setup für das Hyperparameter-Tuning deines Decoders auf deutscher Sprache mit echten Ground-Truth-Daten.**

<div style="text-align: center">⁂</div>

[^1]: https://arxiv.org/pdf/1912.06670.pdf

[^2]: https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0

[^3]: https://huggingface.co/learn/audio-course/chapter5/fine-tuning

[^4]: https://aclanthology.org/2023.emnlp-main.444.pdf

[^5]: http://arxiv.org/pdf/2407.03734.pdf

[^6]: https://arxiv.org/pdf/2309.09443.pdf

[^7]: https://arxiv.org/pdf/2202.13084.pdf

[^8]: https://arxiv.org/pdf/2105.09742.pdf

[^9]: https://www.aclweb.org/anthology/W17-2620.pdf

[^10]: https://discourse.mozilla.org/t/fine-tuning-deepspeech-model-commonvoice-data/41872

[^11]: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

[^12]: https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0

[^13]: https://discourse.mozilla.org/t/language-model-hyperparameter-optimisation/38043

[^14]: https://arxiv.org/pdf/2310.15970.pdf

[^15]: https://arxiv.org/pdf/2007.09127.pdf

[^16]: https://arxiv.org/pdf/2301.07851.pdf

[^17]: https://github.com/AASHISHAG/deepspeech-german

[^18]: https://commonvoice.mozilla.org/datasets

[^19]: https://arxiv.org/pdf/2302.06008.pdf

[^20]: https://blueprints.mozilla.ai/all-blueprints/finetune-an-asr-model-using-common-voice-data

[^21]: https://dagshub.com/kingabzpro/Urdu-ASR-SOTA/pulls/11/files?page=0\&path=README.md

