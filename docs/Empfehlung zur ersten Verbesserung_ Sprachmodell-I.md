<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Empfehlung zur ersten Verbesserung: Sprachmodell-Integration optimieren

Die mit Abstand **größten WER-Reduktionen** bei CTC-basierten Wav2Vec2-Systemen erzielt man in der Regel durch eine robuste **Shallow-Fusion** mit einem domänenspezifischen n-Gramm-Sprachmodell. Bevor Sie an komplexeren Backend-Optimierungen (z. B. Adapter-Feintuning, Punctuation Models, EWC-Feintuning) arbeiten, lohnt es sich daher, zuerst die **CTC-Decoder-Hyperparameter** sowie das **n-Gramm-Modell** zu überarbeiten:

1. Aufbau eines domänenspezifischen n-Gramm-LM
    - Sammeln Sie Transkriptkorrekturen aus realen Nutzersessions und trainieren Sie ein 3- bis 5-Gramm-Modell mit KenLM oder „lmplz“.
    - Verwenden Sie dazu aggressive Vorverarbeitung (Tokenisierung, Entfernung ungewöhnlicher Sonderzeichen) und aktualisieren Sie das LM kontinuierlich mit neuen Korrekturdaten.
2. Feinjustierung der Shallow-Fusion-Gewichte
    - Passen Sie die Gewichtung zwischen Akustik- und Sprachmodell (alpha und beta) systematisch auf einem Validierungs-Set an. Standardwerte, die in der Literatur gute Ergebnisse liefern, sind
– alpha = 0.5 (LM-Gewicht)
– beta = 1.5 (Längen-Normalisierung)
– beam_size = 100
[^1_1]
    - Suchen Sie per Gitter-Suche oder Bayesian-Optimierung das optimale Tripel (alpha, beta, beam_size) für Ihre Domäne.
3. Validierung
    - Messen Sie die WER-Verbesserung ausschließlich durch die LM-Integration (ohne Änderung am Akustikmodell).
    - Typischerweise lassen sich so **10–30% relative WER-Reduktion** erzielen, bevor man weitere, aufwändigere Anpassungen vornimmt.

Erst wenn die WER mit einem gut abgestimmten n-Gramm-LM und optimierten Shallow-Fusion-Parametern noch nicht an die gebrauchsfertigen Systeme von Google/Apple herankommt, sollten Sie gezielt in:

- Adapter-Feintuning (LoRA, EWC) für Domänen- bzw. Nutzerspezifika
- Online‐Punctuation und Truecasing
- CTC-Zweitpass-Rescoring mit LLM
- Encoder-State-Revision

investieren. So stellen Sie sicher, dass Sie die **kostengünstigste Hebelwirkung** zuerst heben und überprüfen können, ob Ihr Gesamt-Pipeline-Ansatz grundsätzlich funktioniert.

––––––––––––––––––––––––––––––––––––––––––––––––––––––––
[^1_1] Zuluaga-Gomez et al.: „We set α = 0.5 and β = 1.5, which corresponds to the LM and length normalization weights. We set the beam size to 100.” [^1_1]

<div style="text-align: center">⁂</div>

[^1_1]: tasks.md

[^1_2]: plan.md

[^1_3]: https://www.mdpi.com/2072-4292/16/13/2442

[^1_4]: https://github.com/parlance/ctcdecode

[^1_5]: https://arxiv.org/pdf/2301.03819.pdf

[^1_6]: https://huggingface.co/docs/transformers/model_doc/wav2vec2

[^1_7]: https://docs.pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html

[^1_8]: https://aircconline.com/ijnlc/V11N6/11622ijnlc01.pdf

[^1_9]: https://publications.idiap.ch/attachments/papers/2022/Juan_SLT2023-2_2023.pdf

[^1_10]: https://docs.pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html

[^1_11]: https://arxiv.org/html/2501.00425v1

[^1_12]: https://github.com/huggingface/transformers/issues/15196

[^1_13]: https://github.com/huggingface/transformers/issues/10459

[^1_14]: https://www.isca-archive.org/interspeech_2021/zhu21f_interspeech.pdf

[^1_15]: https://www.isca-archive.org/interspeech_2022/sriram22_interspeech.pdf

[^1_16]: https://www.isca-archive.org/interspeech_2023/shahin23_interspeech.pdf

[^1_17]: https://publications.idiap.ch/attachments/papers/2022/Vyas_INTERSPEECH_2022.pdf

[^1_18]: https://discuss.huggingface.co/t/improving-performance-of-wav2vec2-fine-tuning-with-word-piece-vocabulary/6292

[^1_19]: https://github.com/kensho-technologies/pyctcdecode

[^1_20]: https://jmlr.csail.mit.edu/manudb/autoreg/reviewer/eFOdi7rU0d4NIc9kqErL/getfile2/8636/manuscript/JMLR-17-098-1.pdf

[^1_21]: https://malaya-speech.readthedocs.io/en/stable/load-stt-ctc-model-pyctcdecode.html

[^1_22]: https://github.com/githubharald/CTCDecoder

[^1_23]: https://malaya-speech.readthedocs.io/en/stable/load-stt-ctc-model-pyctcdecode-mlm.html

[^1_24]: https://www.reddit.com/r/MachineLearning/comments/142t43v/d_hyperparameter_optimization_best_practices/

[^1_25]: https://github.com/kensho-technologies/pyctcdecode/blob/main/pyctcdecode/decoder.py

[^1_26]: https://machinelearningmastery.com/tips-for-tuning-hyperparameters-in-machine-learning-models/

[^1_27]: https://arxiv.org/pdf/2306.05617.pdf

[^1_28]: https://www.isca-archive.org/interspeech_2022/baskar22b_interspeech.pdf

[^1_29]: https://arxiv.org/abs/2306.05617

[^1_30]: https://arxiv.org/html/2405.13018v1

[^1_31]: https://discuss.huggingface.co/t/how-to-train-wav2vec2-in-lora/57074

[^1_32]: https://ceur-ws.org/Vol-3900/Paper14.pdf

[^1_33]: https://arxiv.org/pdf/2305.00115.pdf

[^1_34]: https://ceur-ws.org/Vol-3597/paper17.pdf

[^1_35]: https://www.isca-archive.org/interspeech_2022/futami22_interspeech.pdf

[^1_36]: https://arxiv.org/abs/2305.03837

[^1_37]: https://assets.amazon.science/43/13/104c968c45ea9ed02cffaa1448e0/personalization-of-ctc-speech-recognition-models.pdf

[^1_38]: https://paperswithcode.com/paper/low-rank-adaptation-method-for-wav2vec2-based

[^1_39]: https://arxiv.org/html/2404.01716v1

[^1_40]: https://www.isca-archive.org/interspeech_2024/udupa24_interspeech.pdf

[^1_41]: https://arxiv.org/pdf/2204.00770.pdf

[^1_42]: https://www.ai4europe.eu/research/research-bundles/adaptation-asr-impaired-speech-minimum-resources-adais

[^1_43]: https://publidiap.idiap.ch/attachments/papers/2023/Vanderreydt_ASRU2023_2023.pdf

[^1_44]: https://paperswithcode.com/paper/speaker-adaptation-for-wav2vec2-based

[^1_45]: https://ar5iv.labs.arxiv.org/html/2204.00770

[^1_46]: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12823

[^1_47]: https://arxiv.org/abs/2206.13654

[^1_48]: https://www.alphaxiv.org/overview/2206.13654v1

[^1_49]: https://arxiv.org/pdf/2206.13654.pdf

[^1_50]: https://deepai.org/publication/wav2vec-aug-improved-self-supervised-training-with-limited-data

[^1_51]: https://datascience.columbia.edu/wp-content/uploads/2023/06/38_capstone_poster.pptx-1.pdf

[^1_52]: https://arxiv.org/pdf/1904.05862.pdf

[^1_53]: https://aclanthology.org/2024.findings-acl.681.pdf

[^1_54]: https://ai.meta.com/research/publications/wav2vec-2-0-a-framework-for-self-supervisedlearning-of-speech-representations/

[^1_55]: https://aiola.com/blog/importance-of-punctuation-in-speech-recognition/

[^1_56]: https://kavyamanohar.com/post/evaluating-wav2vec2-bert-asr/

[^1_57]: https://gist.github.com/gkucsko/6d91de8d24d5b8cf65d0240c7edc9755

[^1_58]: https://blog.spheron.network/best-practices-for-llm-hyperparameter-tuning

[^1_59]: https://www.sciencedirect.com/science/article/abs/pii/S0925231224012645

[^1_60]: https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf


---

# Wie mache ich das, gib mir eine detaillierte Schritt für Schritt Anleitung

Hier ist eine **detaillierte Schritt-für-Schritt-Anleitung**, wie du gezielt die Transkriptionsqualität durch Integration eines domänenspezifischen KenLM-Sprachmodells mit Shallow Fusion verbesserst. Damit kannst du sehr schnell beurteilen, ob dein Projekt mit Sprachmodell-Unterstützung das Potenzial hat, mit Google/Apple mitzuhalten.

## 1. **Daten für das Sprachmodell sammeln und vorbereiten**

- Sammle **Korrekturtexte** aus echten Nutzersitzungen oder bestehenden Transkripten.
- **Vorverarbeitung**: Entferne Sonderzeichen, führe Tokenisierung durch, schreibe jeden Satz in eine eigene Zeile, alles in Kleinbuchstaben.
- Speichere das Ergebnis als `corpus.txt` (z. B. im Ordner `server/lm/`).

```python
# Beispiel für einfache Vorverarbeitung in Python
with open("raw_transcripts.txt") as fin, open("corpus.txt", "w") as fout:
    for line in fin:
        line = line.strip().lower()
        line = "".join([c for c in line if c.isalnum() or c.isspace()])
        fout.write(line + "\n")
```


## 2. **KenLM installieren und Sprachmodell trainieren**

- Installiere KenLM und die Python-Bindings (im venv):

```bash
pip install pyctcdecode pypi-kenlm
# Für KenLM-Tools (lmplz, build_binary):
sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
git clone --recursive https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build && cmake .. && make -j4
```

- **Trainiere das n-Gramm-Modell (z. B. 4-Gramm, Kneser-Ney):**

```bash
./kenlm/build/bin/lmplz -o 4 < corpus.txt > corpus.arpa
./kenlm/build/bin/build_binary corpus.arpa corpus.klm
```

*(Optional: Du kannst auch direkt das `.arpa`-Format für pyctcdecode nutzen, dann entfällt der build_binary-Schritt.)*[^2_1][^2_2]

## 3. **Sprachmodell in die ASR-Pipeline integrieren**

- Installiere pyctcdecode:

```bash
pip install pyctcdecode
```

- **Lade das KenLM-Modell und baue den Decoder im Backend:**

```python
from pyctcdecode import build_ctcdecoder

# Dein Vokabular, z. B. aus dem Pretrained-Tokenizer:
vocab_list = [...]  # z.B. tokenizer.get_vocab().keys()

decoder = build_ctcdecoder(
    vocab_list,
    kenlm_model_path="server/lm/corpus.klm",  # oder corpus.arpa
    alpha=0.5,  # Sprachmodellgewicht
    beta=1.5    # Längennormalisierung
)
```

- **Nutze den Decoder für das CTC-Decoding:**

```python
# logits: numpy-Array der Modell-Outputs (Batch x Zeit x Vokabulargröße)
transcript = decoder.decode(logits)
```

*(Siehe auch Beispiel in [^2_3] und [^2_2])*

## 4. **Parameter optimieren**

- Teste verschiedene Werte für `alpha`, `beta` und ggf. `beam_width` auf einem Validierungsset.
- Typische Startwerte: `alpha=0.5`, `beta=1.5`, `beam_width=100`.
- Miss die WER nach jeder Änderung.


## 5. **Automatisierung und Aktualisierung**

- Implementiere ein Skript, das nach jeder Nutzersitzung neue Korrekturen an `corpus.txt` anhängt und das KenLM-Modell neu trainiert.
- Lade das neue Modell automatisch oder nach Server-Neustart.


## 6. **Dokumentation und Monitoring**

- Dokumentiere alle Schritte und Parameter im README.
- Logge, wann das Sprachmodell zuletzt aktualisiert wurde und wie viele Korrekturen eingegangen sind[^2_2].

**Tipp:**
Im Beispielprojekt unter `./other-repos/ovos-stt-plugin-vosk` findest du eine komplette Streaming-Pipeline und KenLM-Integration als Referenz[^2_2].

**Zusammengefasst:**
Mit diesen Schritten kannst du gezielt und schnell die Wirkung eines domänenspezifischen Sprachmodells auf die Transkriptionsqualität testen und optimieren. Erst danach lohnt sich die Implementierung weiterer Features wie Punctuation, Truecasing oder Nutzer-Feintuning[^2_2][^2_4][^2_1][^2_3].

<div style="text-align: center">⁂</div>

[^2_1]: https://github.com/kmario23/KenLM-training

[^2_2]: tasks.md

[^2_3]: https://malaya-speech.readthedocs.io/en/stable/load-stt-ctc-model-pyctcdecode.html

[^2_4]: plan.md

[^2_5]: https://discuss.huggingface.co/t/how-to-create-wav2vec2-with-language-model/12703

[^2_6]: https://www.tensorflow.org/hub/tutorials/wav2vec2_saved_model_finetuning

[^2_7]: https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/asr/asr_language_modeling_and_customization.html

[^2_8]: https://hackernoon.com/lang/de/Arbeiten-mit-wav2vec2-Teil-2-Ausführen-von-Inferenzen-auf-fein-abgestimmten-ASR-Modellen

[^2_9]: https://huggingface.co/docs/transformers/model_doc/wav2vec2?highlight=wav2vec2

[^2_10]: https://huggingface.co/blog/fine-tune-wav2vec2-english

[^2_11]: https://www.cs.ru.nl/bachelors-theses/2022/Thomas_Kolb___1027332___Fine-tuning_Wav2vec2.0_on_caption_data.pdf

[^2_12]: https://aclanthology.org/2024.findings-acl.681.pdf

