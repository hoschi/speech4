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

### 5. Sprachmodell (n-Gramm, z.B. 3gram_de.bin) bauen oder herunterladen
- Siehe KenLM-Doku: https://github.com/kpu/kenlm
- Beispiel für 3-Gramm-Modell:
  ```bash
  kenlm/build/bin/lmplz -o 3 < text_corpus.txt > 3gram_de.arpa
  kenlm/build/bin/build_binary 3gram_de.arpa server/lm/3gram_de.bin
  ```
- Alternativ: Fertiges Modell herunterladen und nach `server/lm/` legen

### 6. Server starten
```bash
cd server
source venv/bin/activate
python main.py
```

## Hinweise
- Das Verzeichnis `server/venv/` und große Modelle werden nicht versioniert (siehe .gitignore)
- Für die ASR-Qualität ist ein passendes KenLM-Modell erforderlich
- Weitere Features siehe `tasks.md`

## Troubleshooting
- Bei Problemen mit KenLM: Stelle sicher, dass Python 3.10 verwendet wird und alle Build-Tools installiert sind
- Für ARM/Mac: Homebrew clang/cmake empfohlen

---

Für weitere Fragen siehe `tasks.md` oder melde dich beim Maintainer. 