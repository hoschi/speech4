# Projektübersicht

## Ordnerstruktur & Inhalte

- **client/**: React-Frontend für Audioaufnahme, Transkriptanzeige und Korrektureingabe.
- **server/**: Python-Backend (FastAPI) für Streaming-ASR, Korrektur-Upload, KenLM-Training und Modellbereitstellung.
  - **server/data/**: Enthält Trainingsdaten für das Sprachmodell (z.B. `corpus.txt`, Logdateien, temporäre Dateien).
  - **server/corrections/**: Gespeicherte Korrekturtexte und (optional) zugehörige Audiodateien, die von Nutzern hochgeladen wurden.
  - **server/lm/**: Fertig trainierte und komprimierte KenLM-Modelle (z.B. `4gram_de.klm`) für die Inferenz.
  - **server/venv/**: (optional) Python-virtuelle Umgebung für das Backend.
- **docs/**: Dokumentation, Workflows, Codebeispiele und Referenzen.
- **kenlm/**: KenLM-Quellcode, Python-Bindings und Tools (`lmplz`, `build_binary` etc.) für das Training und die Nutzung von n-Gramm-Sprachmodellen.

# speech3

## Setup (Entwicklungsumgebung)

### Voraussetzungen
- asdf
- C++ Build-Tools `brew install cmake`
- Git (für Submodule)

### 1. Repository klonen & Submodule initialisieren
```bash
git clone <repo-url> speech3
cd speech3
git submodule update --init --recursive
```

### 2. Python venv anlegen und aktivieren
```bash
asdf install
python -m venv server/venv
source server/venv/bin/activate
pip install --upgrade pip
pip install -r server/requirements.txt
```

### 3. KenLM-Tools kompilieren
```bash
cd kenlm
mkdir -p build && cd build
cmake ..
make -j4
cd ../..
```

### 4. KenLM-Python-Bindings installieren (im venv)
```bash
source server/venv/bin/activate
pip install ./kenlm
```


### 5. KenLM-Modell trainieren (Basis + Personalisierung)
```bash
source server/venv/bin/activate
python -m server.train_lm
```

### 6. Server starten
```bash
source server/venv/bin/activate
python -m server.main
```

**Hinweis:**

Der Server startet nur, wenn ein fertiges KenLM-Modell (`server/lm/4gram_de.klm`) existiert. Das Training erfolgt über das Hauptskript `server/train_lm.py` und nutzt den Basis-Korpus sowie alle Korrekturen in `server/corrections/`.

## KenLM-Binaries für Training verfügbar machen


Nach dem Kompilieren von KenLM (siehe oben) müssen die Binaries `lmplz` und `build_binary` für das Training im Backend verfügbar sein. Das geht am einfachsten per Symlink in das venv-Bin-Verzeichnis:

```bash
ln -sf $(pwd)/kenlm/build/bin/lmplz server/venv/bin/lmplz
ln -sf $(pwd)/kenlm/build/bin/build_binary server/venv/bin/build_binary
```

Dadurch findet das Backend die Tools automatisch, wenn das Training per Skript ausgelöst wird. Diese Symlinks (und das gesamte venv-Verzeichnis) werden **nicht** versioniert. Jeder Entwickler muss diesen Schritt lokal nach dem Build einmalig ausführen.

## KenLM-Optimierung für 32GB RAM

Das System implementiert automatische KenLM-Optimierung für große Sprachmodelle:

- **Aggressives Pruning** (60-80% Größenreduktion)
- **8-bit Quantisierung** (50-70% zusätzliche Reduktion)
- **Memory Mapping** für lazy loading
- **Automatische Größenüberwachung**

## Hinweise
- Das Verzeichnis `server/venv/` und große Modelle werden nicht versioniert (siehe .gitignore)
- Für die ASR-Qualität ist ein passendes KenLM-Modell erforderlich
- Weitere Features siehe `tasks.md`

## Troubleshooting
- Bei Problemen mit KenLM: Stelle sicher, dass Python 3.10 verwendet wird und alle Build-Tools installiert sind
- Für ARM/Mac: Homebrew clang/cmake empfohlen

## Basis-Korpus für initiales Training (Fallback)

huggingface-cli login

Falls noch keine Korrekturdaten vorliegen, nutze einen großen deutschen Korpus als Startpunkt (z.B. OSCAR-2301). Die Datei `server/data/german_base_corpus.txt` wird als Basis verwendet. Für produktive Nutzung: Korpus regelmäßig mit echten Nutzerdaten/Korrekturen ergänzen!

## Hyperparameter-Tuning für Decoder (Common Voice DE)

Mit dem Skript `server/manager.py` kannst du die optimalen KenLM-Decoder-Parameter (alpha, beta) für das deutsche wav2vec2-Modell auf Basis von Mozilla Common Voice (DE) bestimmen.

Hierfür muss `datasets==3.6.0` eingestellt werden, da die v4 die Daten von Common Voice nicht unterstützt.

### Voraussetzungen
- KenLM-Modell liegt vor (z.B. `server/lm/4gram_de.klm`)

### Ausführung

```bash
source server/venv/bin/activate
python server/manager.py
```

Das Skript lädt automatisch Testbeispiele aus Common Voice (DE), führt eine Grid Search über die Parameter alpha und beta durch und gibt die optimalen Werte sowie die beste WER (Word Error Rate) aus.

**Hinweis:**
- Die Berechnung dauert eine Nacht lang (13h)
- Die Ergebnisse werden im Terminal ausgegeben und als CSV gespeichert in `server/reports/tune-decoder`
- Es macht Sinn einen breiteren run (zB commit `e074f1b`) zu machen und dann die KI die "final csv" Datei anschauen zu lassen um einen zweiten Run mit genau so vielen Kombinationen zu machen, aber mit eingeschränktem Wertebereich um die optimale Kombination zu finden.

Weitere Details siehe `docs/Hyperparameter-Tuning-Anleitung.md`.

### Runs

* `acf601f`
    * "facebook/wav2vec2-large-xlsr-53-german"
    * alpha: 0.2
    * beta: -1
    * examples: 4000
    * wer: 0.2044
* `e3beb96` 2025-07-18 13:16
    * "jonatasgrosman/wav2vec2-xls-r-1b-german"
    * alpha: 0.2
    * beta: -1
    * examples: 4000
    * wer: 0.1218
* `e074f1b` 2025-07-19 11:28
    * "jonatasgrosman/wav2vec2-xls-r-1b-german"
    * alpha: 0.4 (0.0 bis 0.8, 10)
    * beta: -1.2 (-2 bis 2, 10)
    * examples: 336
    * wer: 0.133
* `e8faa02` 2025-07-19 18:01
    * "jonatasgrosman/wav2vec2-xls-r-1b-german"
    * alpha: 0.45 (0.3 bis 0.6, 7)
    * beta: -1.1 (-1.5 bis -0.9, 7)
    * examples: 800
    * wer: 0.1317
* `d5f6ca0` 2025-07-20 09:50
    * "jonatasgrosman/wav2vec2-xls-r-1b-german"
    * examples: 4000
    * alpha: 0.2
    * beta: -1.0
    * wer: 0.1218
    * dauer: 13059s
* `b0e024`
    * "jonatasgrosman/wav2vec2-xls-r-1b-german"
    * examples: 4000
    * alpha: 0.45
    * beta: -1.1
    * wer: 0.1175
    * dauer: 3266s
