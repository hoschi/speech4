# Projektübersicht

## Ordnerstruktur & Inhalte

- **client/**: React-Frontend für Audioaufnahme, Transkriptanzeige und Korrektureingabe.
- **server/**: Python-Backend (FastAPI) für Streaming-ASR, Korrektur-Upload, KenLM-Training und Modellbereitstellung.
  - **server/data/**: Enthält Trainingsdaten für das Sprachmodell (z.B. `corpus.txt`, Logdateien, temporäre Dateien).
  - **server/corrections/**: Gespeicherte Korrekturtexte (reine ASR-Ergebnisse, keine Formatierung, kein Cleaning nötig) und (optional) zugehörige Audiodateien, die von Nutzern hochgeladen wurden. Hotwords werden hier nur erkannt wenn sie mit Großbuchstaben anfangen.
  - **server/markdown_input_raw/**: Markdown-Notizen des Nutzers für die Personalisierung (werden vor dem Training automatisch bereinigt/gecleaned).
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



### 2. Conda-Umgebung mit environment.yml anlegen und aktivieren
```bash
asdf install
conda env create -f environment.yml
conda activate speech3
```
Die wichtigsten Tools (inkl. ngram, cmake, clang) werden automatisch installiert.


### 3. KenLM-Tools kompilieren
```bash
cd kenlm
mkdir -p build && cd build
cmake ..
make -j4
cd ../..
```

#### SRILM lokal bauen und Binaries verfügbar machen

Die SRILM-Binaries werden nach dem Build im Ordner `srilm-1.7.3/bin/<MACHINE_TYPE>/` abgelegt und müssen nicht global installiert werden.

**Build-Anleitung (Ordnername: `srilm`):**
1. Lade SRILM von der offiziellen Seite herunter: [SRILM Download](http://www.speech.sri.com/projects/srilm/download.html)
2. Entpacke das SRILM-Archiv und benenne den Ordner in `srilm` um:
   ```bash
   mv srilm-1.7.3 srilm
   cd srilm
   ```
3. Starte den Build. Auf Mac muss gcc mit `brew install gcc` installiert sein, sonst kommt es zu Fehlern. Die gcc version findet man mit `ls /opt/homebrew/bin/gcc-*`:
   ```bash
   make SRILM="$(pwd)" World
   # mac: make SRILM="$(pwd)" CC=gcc-15 World
   # oder falls nötig: make SRILM="$(pwd)" MACHINE_TYPE=i686-m64 World
   ```
   Die Binaries werden in `bin/<MACHINE_TYPE>/` erstellt (z.B. `bin/macosx/ngram`).
4. Teste die Installation:
   ```bash
   ./bin/macosx/ngram -help
   ```

**Integration ins Projekt (Symlink-Empfehlung):**
Lege nach dem Build einen Symlink ins Conda-Bin-Verzeichnis an, damit `ngram` überall verfügbar ist:
```bash
ln -sf $(pwd)/bin/macosx/ngram $(conda info --base)/envs/speech3/bin/ngram
```
Damit kann dein Code einfach `ngram` als Befehl verwenden und muss keinen relativen Pfad kennen.

**Hinweis:**
Die SRILM-Binaries und Symlinks werden nicht versioniert. Jeder Entwickler muss diesen Schritt lokal nach dem Build einmalig ausführen.

### 4. KenLM-Python-Bindings installieren (in Conda-Umgebung)
```bash
conda activate speech3
pip install ./kenlm
```

### 5. KenLM-Modell trainieren (Basis + Personalisierung)

Modell runterladen für NER: `python -m spacy download xx_ent_wiki_sm`

```bash
conda activate speech3
# Standard: existierendes Basismodell (ARPA) wird verwendet
python server/train_lm.py

# Basismodell (ARPA) neu generieren:
python server/train_lm.py --regenerate-base-arpa=true
```
Das Training nutzt die Conda-Umgebung und alle enthaltenen Tools (inkl. ngram).

**Parameter:**
- `--regenerate-base-arpa=true|false` (default: false)
    - true: Basiskorpus wird neu als ARPA-Modell gebaut
    - false: existierendes ARPA wird verwendet
    - Falls kein ARPA existiert und der Parameter false ist, bricht das Skript mit Fehler ab und fordert zur Generierung auf

**Beispiel-Fehlermeldung:**
```
[ERROR] Basismodell (ARPA) nicht gefunden: server/lm/base_model.arpa
Bitte --regenerate-base-arpa=true setzen, um das Basismodell neu zu generieren.
```

**Personalisierungsdaten:**
- Korrekturen: Lege reine Textdateien (ASR-Ergebnisse, keine Formatierung) in `server/corrections/` ab.
- Markdown-Notizen: Lege Markdown-Dateien (`*.md`) in `server/markdown_input_raw/` ab. Diese werden beim Training automatisch bereinigt und verstärkt in den Korpus integriert.


### 6. Server starten
```bash
conda activate speech3
python server/main.py
```

**Hinweis:**

Der Server startet nur, wenn ein fertiges KenLM-Modell (`server/lm/4gram_de.klm`) existiert. Das Training erfolgt über das Hauptskript `server/train_lm.py` und nutzt den Basis-Korpus, alle Korrekturen in `server/corrections/` sowie alle Markdown-Notizen in `server/markdown_input_raw/`.


## KenLM-Binaries für Training verfügbar machen

Nach dem Kompilieren von KenLM (siehe oben) müssen die Binaries `lmplz` und `build_binary` für das Training im Backend verfügbar sein. Das geht am einfachsten per Symlink in das Conda-Bin-Verzeichnis:

```bash
ln -sf $(pwd)/kenlm/build/bin/lmplz $(conda info --base)/envs/speech3/bin/lmplz
ln -sf $(pwd)/kenlm/build/bin/build_binary $(conda info --base)/envs/speech3/bin/build_binary
```

Dadurch findet das Backend die Tools automatisch, wenn das Training per Skript ausgelöst wird. Diese Symlinks (und das gesamte Conda-Umgebungsverzeichnis) werden **nicht** versioniert. Jeder Entwickler muss diesen Schritt lokal nach dem Build einmalig ausführen.

## KenLM-Optimierung für 32GB RAM

Das System implementiert automatische KenLM-Optimierung für große Sprachmodelle:

- **Aggressives Pruning** (60-80% Größenreduktion)
- **8-bit Quantisierung** (50-70% zusätzliche Reduktion)
- **Memory Mapping** für lazy loading
- **Automatische Größenüberwachung**

## Hinweise
- Das Verzeichnis der Conda-Umgebung und große Modelle werden nicht versioniert (siehe .gitignore)
- Für die ASR-Qualität ist ein passendes KenLM-Modell erforderlich
- Weitere Features siehe `tasks.md`

## Troubleshooting
- Bei Problemen mit KenLM: Stelle sicher, dass Python 3.10 verwendet wird und alle Build-Tools installiert sind
- Für ARM/Mac: Homebrew clang/cmake empfohlen

## Basis-Korpus für initiales Training (Fallback)

huggingface-cli login

Falls noch keine Korrekturdaten vorliegen, nutze einen großen deutschen Korpus als Startpunkt (z.B. OSCAR-2301). Die Datei `server/data/german_base_corpus.txt` wird als Basis verwendet.

### Skripte im `scripts`-Ordner

- **extract_oscar_german.py**
  Erstellt einen deutschen Basiskorpus aus dem OSCAR-Datensatz (Huggingface). Die Zielgröße kann im Skript angepasst werden (Standard: 40GB). Das Skript schreibt alle deutschen Textbeispiele mit mindestens 10 Wörtern in die Datei `german_base_corpus.txt`.
  
  **Ausführung:**
  ```bash
  python scripts/extract_oscar_german.py
  ```
  Die erzeugte Datei kann dann nach `server/data/german_base_corpus.txt` verschoben werden. Bei Fehler, siehe hier wie man in pip module löscht und neu installiert. https://www.perplexity.ai/search/traceback-most-recent-call-las-AHUyIVFcRsmr2CuKNJgqww

- **calc_max_examples.py**
  Berechnet die maximal sinnvolle Anzahl an Beispielen (`N_VALIDATION`) für das Hyperparameter-Tuning, sodass ein Grid-Search-Lauf in ca. 13 Stunden abgeschlossen werden kann. Das Skript benötigt die durchschnittliche Zeit pro Kombination (z.B. aus einem Testlauf mit DEBUG=true).
  
  **Ausführung:**
  ```bash
  python scripts/calc_max_examples.py -t <Sekunden pro Kombination>
  ```
  Die Ausgabe zeigt die optimale Beispielanzahl für den nächsten Tuning-Run.

## Hyperparameter-Tuning für Decoder (Common Voice DE)

Mit dem Skript `server/manager.py` kannst du die optimalen KenLM-Decoder-Parameter (alpha, beta) für das deutsche wav2vec2-Modell auf Basis von Mozilla Common Voice (DE) bestimmen.

Hierfür muss `datasets==3.6.0` eingestellt werden, da die v4 die Daten von Common Voice nicht unterstützt.

### Voraussetzungen
- KenLM-Modell liegt vor (z.B. `server/lm/4gram_de.klm`)
- Die optimale Beispielanzahl kann mit `scripts/calc_max_examples.py` berechnet werden (siehe oben).

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
