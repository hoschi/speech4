# Tasks

Dieses Dokument ist nach **Feature-Gruppen** gegliedert. Zu jedem Feature finden sich unter Überschriften je eine Liste mit Aufgaben für **Server** und **Client**. Die Coding-KI markiert jede Checkbox als erledigt, sobald Tests und Implementierungen erfolgreich abgeschlossen sind.

## Feature: Basis-Streaming-Inferenz

*Um eine funktionierende Echtzeit-Transkription zu ermöglichen, implementiere im Server die Streaming-Pipeline und im Client die Audio-Erfassung und Anzeige.*

### Server

- [ ] Python-Projekt initialisieren mit venv und FastAPI
- [ ] Abhängigkeiten installieren (PyTorch, Transformers, PEFT, uvicorn, websockets)
- [ ] WebSocket-Endpunkt `/ws/stream` einrichten
- [ ] Eingehende 20 ms PCM-Chunks empfangen und in Tensoren umwandeln
- [ ] Modell `facebook/wav2vec2-large-xlsr-53-german` auf CPU/Neural Engine laden
- [ ] Inferenz-Pipeline (Forward-Pass → Logits → CTC-Decoding) implementieren
- [ ] Transkripte als Chunks über WebSocket zurücksenden


### Client

- [ ] React-Projekt initialisieren mit Vite und TypeScript
- [ ] Mikrofonzugriff via Web-Audio API anfordern
- [ ] 20 ms Audio-Chunks erfassen und als PCM-Buffer serialisieren
- [ ] WebSocket-Verbindung zu `/ws/stream` aufbauen
- [ ] Gesammelte Audio-Chunks in Echtzeit senden
- [ ] Eingehende Transkriptions-Chunks anzeigen (Streaming-Update)
- [ ] Fehler-Handling bei Verbindungsabbrüchen implementieren


## Feature: Personalisierungs-Loop

*Um kontinuierliches Lernen zu ermöglichen, setze Server-Endpunkte für Corrections und Training und Client-UI für Korrekturen um.*

### Server

- [ ] POST-Endpoint `/upload/corrections` für `(audio_chunk, korrigierter_text)` implementieren
- [ ] Speicherstruktur anlegen:
    - `/data/audio/`
    - `/data/text/`
- [ ] Trainings-Trigger realisieren (wähle eine Option oder ermögliche beides):
    - [ ] manuell via HTTP-Endpoint `/train`
    - [ ] zeitbasiert (Cron-Job)
- [ ] LoRA-Feintuning-Pipeline mit EWC-Integration implementieren (r=16, α=32, dropout=0.1, konfigurierbares λ)
- [ ] Versionierung der Modell-Checkpoints unter `/models/`
- [ ] GET-Endpoint `/model/download` bereitstellen
- [ ] Integrationstests: Upload → Training → Download → Inferenz
- [ ] Logging für Performance-Messungen (Latenz, Trainingszeit)


### Client

- [ ] UI-Komponente zur Bearbeitung transkribierter Zeilen hinzufügen
- [ ] „Speichern“-Button neben jeder Zeile für korrigierten Text
- [ ] Batch-Upload aller Korrekturen (z. B. in Gruppen zu 50 Einträgen)
- [ ] Fortschrittsanzeige für Upload und Training (Polling oder WebSocket)
- [ ] Automatisches Neuladen des neuen Modells nach Training
- [ ] Integrationstest: Korrektur → Upload → neue Inferenz verwenden


## Feature: Code-Switching \& Vokabular-Biasing

*Um Fachbegriffe korrekt zu behandeln, implementiere Biasing im Server-Decoder und entsprechende Einstellungen im Client.*

### Server

- [ ] Mechanismus zur Prompt-Injection für Vokabular-Biasing umsetzen:
    - [ ] Option A: statisches `vocab_bias.json`
    - [ ] Option B: dynamisch aus Nutzer-Feedback
- [ ] Decoder anpassen, um Bias-Wahrscheinlichkeiten bei der CTC-Dekodierung zu priorisieren
- [ ] Optional: Rescoring-Endpoint `/rescore` zur LLM-gestützten Priorisierung (z. B. GPT-API)
- [ ] Tests: korrekte Priorisierung englischer Fachbegriffe und WER-Vergleich


### Client

- [ ] Settings-Tab: Eingabefeld zum Hinzufügen eigener Fachbegriffe
- [ ] Anzeige der aktiven Bias-Begriffe und Möglichkeit zum Entfernen
- [ ] Option zum temporären Deaktivieren des Biasing
- [ ] Tests: Code-Switching-Szenarien simulieren und Verifizierung der Anzeige


## Feature: Erweiterungen \& Optimierungen

*Um Systemstabilität und -performance zu steigern, integriere Augmentation, CI/CD, Monitoring und UI-Optimierungen.*

### Server

- [ ] Synthetic Data Augmentation via VALL-E X integrieren (API \& Lizenz prüfen)
- [ ] Alternative Streaming-Server evaluieren (VOSK, ESPnet-Conformer, Kaldi-Serve)
- [ ] CI/CD-Pipeline für Builds, Tests und Deployments einrichten (Docker, GitHub Actions)
- [ ] Monitoring-Dashboard aufsetzen (Prometheus + Grafana) für Latenz und Fehlerraten


### Client

- [ ] Responsive UI und Dark Mode umsetzen
- [ ] End-to-End-Performance-Test-Suite automatisieren
- [ ] Barrierefreiheit sicherstellen (ARIA-Labels, Screenreader-Kompatibilität)
- [ ] Usability-Tests durchführen und Feedback umsetzen

**Regeln für die Coding-KI:**

- Jede Aufgabe wird als **erledigt** markiert, sobald alle zugehörigen Tests und Code-Reviews bestanden sind.
- Entscheidungen zwischen vorgestellten Optionen treffen oder bei Bedarf explizit nachfragen.
- Keine freien Textantworten: Folge strikt den Checklisten.

