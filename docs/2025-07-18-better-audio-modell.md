# Recherche für ein verbessertes ASR-Modell

**Datum:** 2025-07-18

## 1. Problembeschreibung

Das bisherige Akustikmodell (`facebook/wav2vec2-large-xlsr-53-german` und dessen Nachfolger `aware-ai/wav2vec2-xls-r-1b-german`) zeigte in der Analyse der Fehlerdatei (`server/reports/tune-decoder/27bd081_best_run.csv`) deutliche Schwächen. Hauptprobleme waren:

- **Schlechte Erkennung von Eigennamen und Fremdwörtern** (z.B. "Vogue" -> "WOLG").
- **Fehler bei deutschen Komposita** (Wortzusammensetzungen).
- **Allgemeine phonetische Ungenauigkeiten** und Auslassungen.
- Teilweise **komplett fehlerhafte Transkriptionen**, die keinen Bezug zum Original hatten.

Das Ziel war, ein leistungsfähigeres, für Deutsch optimiertes Modell zu finden, das sich nahtlos in die bestehende Architektur mit `pyctcdecode` und KenLM integrieren lässt.

## 2. Recherche und Modell-Alternativen

Die Recherche auf Hugging Face konzentrierte sich auf für Deutsch fein-trainierte Modelle, die auf der `wav2vec2-xls-r-1b` Architektur basieren, da diese eine hohe Grundqualität verspricht.

### Kandidat 1: `jonatasgrosman/wav2vec2-xls-r-1b-german` (Empfehlung)

- **Architektur:** Wav2Vec2 (CTC) - Direkter Ersatz für das alte Modell.
- **Performance:** Exzellente Word Error Rate (WER) von **10.95%** auf dem Common Voice Datensatz.
- **Training:** Wurde auf einem sehr vielfältigen deutschen Datensatz trainiert (Common Voice, TEDx, LibriSpeech, Voxpopuli), was eine hohe Robustheit gegenüber verschiedenen Sprechern, Akzenten und Fachbegriffen verspricht.
- **Vorteil:** Perfekte Kompatibilität mit der bestehenden KenLM-Pipeline.

### Kandidat 2: OpenAI Whisper (`openai/whisper-large-v3`)

- **Architektur:** Encoder-Decoder - **Nicht** direkt kompatibel mit `pyctcdecode`.
- **Performance:** Gilt als State-of-the-Art, besonders bei Satzzeichen und Code-Switching.
- **Nachteil:** Ein Wechsel würde eine **umfangreiche Änderung der Server-Architektur** erfordern, da die gesamte Sprachmodell-Integration neu implementiert werden müsste.

## 3. Entscheidung und Umsetzung

Aufgrund der massiven Qualitätsverbesserung bei gleichzeitig minimalem Implementierungsaufwand wurde die Entscheidung getroffen, auf **`jonatasgrosman/wav2vec2-xls-r-1b-german`** zu wechseln.

Die Änderung wurde am 18. Juli 2025 in der Datei `server/main.py` umgesetzt, indem der `MODEL_NAME` entsprechend aktualisiert wurde.

```python
# Alt
MODEL_NAME = "aware-ai/wav2vec2-xls-r-1b-german"

# Neu
MODEL_NAME = "jonatasgrosman/wav2vec2-xls-r-1b-german"
```

## 4. Nächste Schritte

Obwohl die KenLM-Pipeline nicht neu trainiert werden muss, wird empfohlen, das Skript `tune_decoder.py` erneut auszuführen. Dadurch können die optimalen `alpha`- und `beta`-Gewichte für die Fusion des neuen, besseren Akustikmodells mit dem bestehenden Sprachmodell ermittelt werden, um die bestmögliche Transkriptionsqualität zu erreichen.
