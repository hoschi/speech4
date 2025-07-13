# Tasks
# Tasks

## Aktive Arbeiten

- Fokus liegt aktuell auf **Meilenstein 3: Die echte Personalisierung**. Nachdem die Basis-Pipeline steht und Korrekturen gesammelt werden können, implementieren wir nun das Training und die Nutzung eines vollwertigen KenLM-Sprachmodells.

---

## Meilensteine & Detaillierte Aufgaben

### ✅ Meilenstein 1: Das Fundament (Abgeschlossen)

*Ziel war es, eine stabile ASR-Basis zu schaffen, die live transkribiert. Dies ist nun funktional.*

-   **Server-Aufgaben:**
    -   [x] Python-Projekt mit FastAPI initialisiert.
    -   [x] VOSK-Bibliothek und deutsches Modell installiert.
    -   [x] WebSocket-Endpunkt (`/ws/stream`) implementiert.
    -   [x] VOSK-Recognizer-Instanz pro Verbindung erstellt.
    -   [x] Audio-Chunks an den Recognizer weitergeleitet.
    -   [x] Partielle und finale Ergebnisse an den Client gesendet.
    -   [x] **Code-Bereinigung:** Die irreführende Funktion `init_kenlm_decoder` wurde zu `load_custom_vocabulary` umbenannt, da sie nur das VOSK-Vokabular lädt und nicht mit KenLM arbeitet.
-   **Client-Aufgaben (React):**
    -   [x] React-Projekt aufgesetzt.
    -   [x] UI mit "Start/Stop"-Button und Textfeld erstellt.
    -   [x] Mikrofonzugriff und Audio-Streaming via `MediaRecorder` und WebSocket implementiert.
    -   [x] Empfangene Transkriptionen in der UI angezeigt.

### ✅ Meilenstein 2: Der Basis-Korrektur-Loop (Abgeschlossen)

*Ziel war es, Korrekturen zu sammeln und das Vokabular der ASR-Engine zu erweitern. Das System speichert nun Daten und erstellt eine Wortliste.*

-   **Server-Aufgaben:**
    -   [x] REST-Endpunkt `POST /upload/correction` erstellt, der Audio und Text speichert.
    -   [x] `preprocess_corrections.py` erstellt, um ein `corpus.txt` aus den Korrekturen zu generieren.
    -   [x] REST-Endpunkt `POST /train/lm` implementiert, der das Preprocessing anstößt.
    -   [x] VOSK-Recognizer wird beim Start mit einem benutzerdefinierten Vokabular (Wortliste) aus dem `corpus.txt` initialisiert.
-   **Client-Aufgaben (React):**
    -   [x] UI um ein editierbares Textfeld und einen Upload-Button erweitert.
    -   [x] Logik implementiert, um Audio-Blob und korrigierten Text an den Server zu senden.

### 📋 Meilenstein 3: Die echte Personalisierung mit KenLM (Nächste Iteration)

*Ziel: Von einer einfachen Wortliste zu einem mächtigen N-Gramm-Sprachmodell wechseln, um den Kontext und die Satzstruktur zu lernen. Dies wird die Genauigkeit signifikant verbessern.*

-   **Server-Aufgaben:**
    -   [x] **KenLM-Setup:**
        -   [x] KenLM von GitHub klonen und gemäß der Anleitung für macOS kompilieren.
        -   [x] Einen großen deutschen Basiskorpus (z.B. OSCAR-2301) herunterladen und als `german_base_corpus.txt` bereitstellen.
    -   [ ] **Trainings-Pipeline erweitern:**
        -   [ ] Das `train/lm`-Skript so anpassen, dass es die Nutzer-Korrekturen mit dem großen Basiskorpus zusammenführt.
        -   [ ] Das Skript soll die KenLM-Kommandozeilen-Tools (`lmplz` und `build_binary`) aufrufen, um ein `.arpa`- und dann ein `.klm`-Modell zu trainieren.
    -   [ ] **Migration von VOSK zu pyctcdecode + KenLM:**
        -   [ ] VOSK ist nicht ideal für die tiefe Integration von benutzerdefinierten `.klm`-Modellen. Wir ersetzen den VOSK-Recognizer durch `pyctcdecode`, das speziell für die Fusion mit KenLM-Modellen entwickelt wurde.
        -   [ ] Dazu muss das akustische Modell ausgetauscht werden. Wir verwenden `facebook/wav2vec2-large-xlsr-53-german` als neues Basismodell, da es mit `pyctcdecode` kompatibel ist.
        -   [ ] Die WebSocket-Logik anpassen, um die Chunks an das Wav2Vec2-Modell zu senden und die `logits` dann mit dem `pyctcdecode`-Decoder und unserem KenLM-Modell zu verarbeiten.

### 📋 Meilenstein 4: Optimierung (Zukünftige Iteration)

*Ziel: Die Qualität, Lesbarkeit und Automatisierung weiter verbessern.*

-   [ ] **LLM-Post-Processing:** Einen optionalen Schritt nach der Transkription einfügen, der die Ausgabe an ein lokales LLM zur Korrektur von Grammatik und Interpunktion sendet.
-   [ ] **Automatisierung:** Das KenLM-Training automatisch (z.B. nächtlich per `cron`) ausführen lassen.
-   [ ] **Hyperparameter-Tuning:** Ein Skript (`tune_decoder.py`) erstellen, um die `alpha`- und `beta`-Werte für `pyctcdecode` optimal auf einem Validierungsset (z.B. Common Voice) abzustimmen.

---

**Aufforderung an die KI:**
Aktualisieren Sie task.md, um [Aufgabenname] als erledigt zu markieren und [Aufgabenname] als neue Aufgabe hinzuzufügen.

**Globale Regeln für die KI:**
-   Markiere eine Aufgabe automatisch als erledigt, wenn die Implementierung und die zugehörigen Tests erfolgreich abgeschlossen wurden.
-   Erstelle automatisch neue Unteraufgaben, wenn sich Blocker oder notwendige Zwischenschritte ergeben.

