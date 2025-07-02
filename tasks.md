# Tasks

Dieses Dokument ist nach **Feature-Gruppen** gegliedert. Zu jedem Feature finden sich unter Überschriften je eine Liste mit Aufgaben für **Server** und **Client**.

Unter `./other-repos/ovos-stt-plugin-vosk` findest du ein Beispielprojekt das folgende Features enthält die dir helfen könnten bei der Implementierung:

* **Streaming-ASR** via Vosk API: niedrige Latenz (kleiner 100 ms), CTC/WFST, CPU-optimiert.
* **Code-Switching**: Deutsche Modelle mit englischen Termini.
* **Adapter-Feintuning**: Nutzt Kaldi-Adapter, lässt sich in Personalisierungs-Pipeline einbinden.

## Feature: Basis-Streaming-Inferenz

*Um eine funktionierende Echtzeit-Transkription zu ermöglichen, implementiere im Server die Streaming-Pipeline und im Client die Audio-Erfassung und Anzeige.*

### Server

- [x] Python-Projekt initialisieren mit venv und FastAPI
- [x] Abhängigkeiten installieren (PyTorch, Transformers, PEFT, uvicorn, websockets)
- [x] WebSocket-Endpunkt `/ws/stream` einrichten
- [x] Eingehende 20 ms PCM-Chunks empfangen und in Tensoren umwandeln
- [x] Modell `facebook/wav2vec2-large-xlsr-53-german` auf CPU/Neural Engine laden
- [x] Inferenz-Pipeline (Forward-Pass → Logits → CTC-Decoding) implementieren
- [x] Transkripte als Chunks über WebSocket zurücksenden


### Client

Unter `./other-repos/leon` findest du ein Beispielprojekt das dir helfen kann bei der Audio verarbeitung und anzeige des transkripts. Wichtig ist aber das ich das in einem eingabe feld haben möchte damit ich es später editieren kann

- [x] React-Projekt initialisieren mit Vite und TypeScript
- [x] Mikrofonzugriff via Web-Audio API anfordern
- [x] 20 ms Audio-Chunks erfassen und als PCM-Buffer serialisieren
- [x] WebSocket-Verbindung zu `/ws/stream` aufbauen
- [x] Gesammelte Audio-Chunks in Echtzeit senden
- [x] Eingehende Transkriptions-Chunks anzeigen (Streaming-Update)
- [x] Fehler-Handling bei Verbindungsabbrüchen implementieren

## Feature: Verbesserte ASR-Qualität durch gezielte Backend-Maßnahmen

Alle Details zur Verbesserung der Ist-Situation nach dem letzten Feature: https://www.perplexity.ai/search/bitte-recherchiere-und-fasse-z-Yeh3BqyJQhagazWxD1bKoQ

Die ASR-Qualität im Live-Streaming-Backend wird durch gezielte Maßnahmen deutlich verbessert

### 1. Wortverschmelzungen und falsche Trennungen
- [ ] **Shallow Fusion mit KenLM (pyctcdecode):**
    - [x] KenLM-Tools kompilieren (lmplz, build_binary etc.)
    - [x] Python-Bindings von KenLM im venv installieren
    - [x] README für Setup und Nutzung ergänzen
    - [ ] **Datensammlung & Vorverarbeitung:**
        - Korrekturtexte aus Nutzereingaben sammeln
        - Vorverarbeitung (Sonderzeichen entfernen, Tokenisierung, ein Satz pro Zeile)
        - Alles in `corpus.txt` speichern
    - [ ] **KenLM-Modell trainieren:**
        - n-Gramm-Modell (3- oder 4-Gramm, Kneser-Ney) mit lmplz bauen
        - Komprimieren mit build_binary
        - Modell nach `server/lm/` legen
    - [ ] **Integration in pyctcdecode:**
        - Modell im Backend laden und für Decoding nutzen
    - [ ] **Kontinuierliche Aktualisierung:**
        - Nach jedem Personalisierungs-Loop neue Korrekturen anhängen
        - KenLM-Modell neu bauen und bereitstellen
        - Hot-Reload oder Server-Neustart für neues Modell
    - [ ] **Automatisierung:**
        - Trainings-Trigger automatisiert: Textdaten sammeln, Modell trainieren, bereitstellen
        - Status- und Fehler-Logging
    - [ ] **Dokumentation & Referenzen:**
        - Quellen und Step-by-Step-Referenz in README.md und Code-Kommentaren
- [ ] **Real-Time Encoder State Revision:**
    - Speicherung und Überarbeitung früher Hypothesen mit neuen Frames zur Korrektur von Zusammenziehungen
- [x] .gitignore für Sprachmodelle und Binärdateien angepasst
- [x] Fehler- und Fallback-Handling für KenLM-Integration implementiert

### 2. Interpunktion und Großschreibung
- [ ] **Online-Punctuation-Module:**
    - Leichtgewichtiges ELECTRA-basiertes Modell (z. B. angepasstes `dslim/bert-base-NER`) für inkrementelle Satzzeichen nach CTC
- [ ] **Truecasing-Adapter:**
    - Truecasing-Stufe mit POS-Tagging (spaCy-Deutsch) für Großschreibung von Satzanfängen und Substantiven

### 3. Genauere Wortgrenzen und Alignment
- [ ] **Forced Alignment auf CTC-Logits:**
    - Dynamische Programmierung über CTC-Logit-Lattice für exakte Wort-Zeitstempel (z. B. mit `ctc-forced-aligner`)

### 4. Kontinuierliche Personalisierung
- [ ] **Adapter-Feintuning per LoRA + EWC:**
    - Nutzer-Korrekturen werden für LoRA-Feintuning (r=16, α=32, EWC) genutzt und als Adapter deployed
    - Automatisierter Trainings-Endpoint nach jeder Session
- [ ] **Hotword-Boosting:**
    - Boost-Words/Fachbegriffe mit erhöhtem Score via pyctcdecode

### 5. LLM-gestütztes Rescoring
- [ ] **Zweite Pass-Rescoring mit Transformer-LM:**
    - Nach erster CTC-Hypothese: N-Best-Liste, Bewertung durch LLM (z. B. GPT-4) mit Cross-Attention für komplexe Begriffe

## Feature: Personalisierungs-Loop

*Um kontinuierliches Lernen zu ermöglichen, setze Server-Endpunkte für Corrections und Training und Client-UI für Korrekturen um.*

Erweiterte Dokumentation zu diesem Task ist in `docs/ecw.md`

### Server

- [ ] POST-Endpoint `/upload/corrections` für `(audio_chunk, korrigierter_text)` implementieren
- [ ] Speicherstruktur anlegen: `/corrections` das Dateipaare enthält wie `2025-06-18 15:46.txt` und `2025-06-18 15:46.wav` für Audio und korrigiertem Text
- [ ] Trainings-Trigger realisieren via HTTP-Endpoint `/train/ewc`
- [ ] **Fisher-Information berechnen**  
  - Funktion `get_fisher_diag(model, dataloader)`  
- [ ] **EWCTrainer-Klasse erweitern**  
  - Überschreiben von `compute_loss` mit EWC-Term  
- [ ] **Feintuning-Task**  
  - Skript `run_ewc_training()` für:  
    - Laden des Basismodells  
    - Erzeugen der Datasets A und B  
    - Berechnung von Fisher & alten Parametern  
    - Training mit konfigurierbarem `ewc_lambda`  
    - Speichern und Versionieren des Modells unter `/models/`, logging welche Datensätze in das neue Modell geflossen sind
    - neues Modell laden, hot swapping nicht nötig, downtime ist kein Problem
- [ ] Logging für Performance-Messungen (Latenz, Trainingszeit)
- [ ] mit streamlit unter `/monitoring` ein Monitoring etablieren
    - [ ] welche Traningsdaten noch nicht verarbeitet wurden
    - [ ] Übersicht über die letzten 5 Modelle und wie viele Traningsdaten in das Modell geflossen sind bei dessen Training
    - [ ] letzten 200 Zeilen aus Performance Logging


### Client

- [ ] UI-Komponente zur Bearbeitung transkribierter Zeilen hinzufügen
    - [ ] für jede Aufnahme wird der transkribierte Text in einem einfachen input feld dargestellt das bearbeitet werden kann, sobald die Aufnahme vom Benutzer beendet wird
    - [ ] einen "upload" Button um die daten an `/corrections` zu übertragen
    - [ ] nach einem upload kann dieser nicht nochmal getriggert werden, außer es ist ein Fehler aufgetreten beim upload. Auch der Text kann nicht mehr editiert werden
    - [ ] loading spinner für Upload bis er fertig ist mit Fehlerbehandlung
- [ ] Automatisches Neuladen des neuen Modells nach Training


## Feature: Code-Switching \& Vokabular-Biasing

*Um Fachbegriffe korrekt zu behandeln, implementiere Biasing im Server-Decoder und entsprechende Einstellungen im Client.*

Unter `./other-repos/ASR-Adaptation` findest du ein Beispielprojekt das dir hier helfen kann

### Server

- [ ] Mechanismus zur Prompt-Injection für Vokabular-Biasing umsetzen durch statisches `vocab_bias.json` die vom Client aus geändert werden kann
- [ ] Decoder anpassen, um Bias-Wahrscheinlichkeiten bei der CTC-Dekodierung zu priorisieren
- [ ] Optional: Rescoring-Endpoint `/rescore` zur LLM-gestützten Priorisierung (z. B. GPT-API)


### Client

- [ ] Settings-Tab
    - [ ] Eingabefeld zum Hinzufügen eigener Fachbegriffe zu `vocab_bias.json`
    - [ ] Button um Neustart des Servers der die neuen Begriffe einbetten muss, loading spinner bis Server fertig ist
- [ ] Anzeige der aktiven Bias-Begriffe und Möglichkeit zum Entfernen
- [ ] Option zum temporären Deaktivieren des Biasing


## Feature: Erweiterungen \& Optimierungen

*Um Systemstabilität und -performance zu steigern, integriere Augmentation, CI/CD, Monitoring und UI-Optimierungen.*

### Server

- [ ] Synthetic Data Augmentation via VALL-E X integrieren (API und Lizenz prüfen)
- [ ] Alternative Streaming-Server evaluieren (VOSK, ESPnet-Conformer, Kaldi-Serve)
- [ ] CI/CD-Pipeline für Builds, Tests und Deployments einrichten (Docker, GitHub Actions)


**Regeln für die Coding-KI:**

- Jede Aufgabe wird als **erledigt** markiert, sobald alle zugehörigen Tests und Code-Reviews bestanden sind.
- Entscheidungen zwischen vorgestellten Optionen treffen oder bei Bedarf explizit nachfragen.


## Feature: Improvements

- [ ] Audioaufnahme im Client von ScriptProcessorNode auf AudioWorkletNode umstellen (Web Audio API Best Practice)


