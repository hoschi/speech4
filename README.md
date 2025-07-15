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

### 5. Server starten
```bash
source server/venv/bin/activate
python server/main.py
```

## KenLM-Binaries für Training verfügbar machen

Nach dem Kompilieren von KenLM (siehe oben) müssen die Binaries `lmplz` und `build_binary` für das Training im Backend verfügbar sein. Das geht am einfachsten per Symlink in das venv-Bin-Verzeichnis:

```bash
ln -sf $(pwd)/kenlm/build/bin/lmplz server/venv/bin/lmplz
ln -sf $(pwd)/kenlm/build/bin/build_binary server/venv/bin/build_binary
```

- Dadurch findet das Backend die Tools automatisch, wenn das Training per API oder UI ausgelöst wird.
- Diese Symlinks (und das gesamte venv-Verzeichnis) werden **nicht** versioniert. Jeder Entwickler muss diesen Schritt lokal nach dem Build einmalig ausführen.

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

Falls noch nie ein Training stattgefunden hat und keine Korrekturdaten vorliegen, kannst du einen großen deutschen Korpus als Startpunkt nutzen. Empfohlen wird OSCAR-2301:

### Schritt 1: Hugging Face Account & Token
- Erstelle einen Account auf https://huggingface.co
- Gehe zu Settings → Access Tokens und erstelle einen Token (mind. Read-Berechtigung)

### Schritt 2: Zugang zum Dataset beantragen
- Besuche https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
- Akzeptiere die Nutzungsbedingungen (meist sofort freigeschaltet)

### Schritt 3: Authentifizierung mit huggingface-cli
```bash
pip install huggingface_hub[cli]
huggingface-cli login
# Gib deinen Token ein
```

### Schritt 4: Download/Streaming mit der datasets library

- Führe das Skript nach erfolgreichem Login in deiner venv aus:
```bash
source server/venv/bin/activate
pip install datasets
pip install zstandard
python scripts/extract_oscar_german.py
```

- Die Datei `german_base_corpus.txt` kann als Ausgangspunkt für das initiale KenLM-Training verwendet werden, bis genügend echte Korrekturen gesammelt wurden.
- Für produktive Nutzung: Korpus regelmäßig mit echten Nutzerdaten/Korrekturen ergänzen!

## Hyperparameter-Tuning für Decoder (Common Voice DE)

Mit dem Skript `server/tune_decoder.py` kannst du die optimalen KenLM-Decoder-Parameter (alpha, beta) für das deutsche wav2vec2-Modell auf Basis von Mozilla Common Voice (DE) bestimmen.

Hierfür muss `datasets==3.6.0` eingestellt werden, da die v4 die Daten von Common Voice nicht unterstützt.

### Voraussetzungen
- Python venv ist aktiviert (`source server/venv/bin/activate`)
- Alle Abhängigkeiten sind installiert (`pip install -r server/requirements.txt`)
- KenLM-Modell liegt vor (z.B. `server/lm/4gram_de.klm`)

### Ausführung

```bash
source server/venv/bin/activate
python server/tune_decoder.py
```

Das Skript lädt automatisch 200 zufällige Testbeispiele aus Common Voice (DE), führt eine Grid Search über die Parameter alpha und beta durch und gibt die optimalen Werte sowie die beste WER (Word Error Rate) aus.

**Hinweis:**
- Das Skript lädt das Modell `facebook/wav2vec2-large-xlsr-53-german` und verwendet das KenLM-Modell aus `server/lm/4gram_de.klm`.
- Die Berechnung kann je nach Hardware einige Minuten dauern.
- Die Ergebnisse werden im Terminal ausgegeben.

Weitere Details siehe `docs/Hyperparameter-Tuning-Anleitung.md`.

---

Für weitere Fragen siehe `tasks.md` oder melde dich beim Maintainer. 