# Empfehlung: Nächste Schritte zur WER-Verbesserung (Stand 03.08.2025)

## Aktueller Stand
- Das System erreicht einen WER von 11,75% (siehe tune_decoder-Run in README.md).
- KenLM ist optimiert und integriert, Grid-Search für Alpha/Beta durchgeführt.
- Fehleranalyse zeigt: Hauptprobleme sind Namen, Fremdwörter, Fachbegriffe und seltene Wörter.
- Großschreibung und Interpunktion sind aktuell nicht relevant für WER.

## Konkrete Empfehlungen

### 1. Vokabular-Biasing / Hotword-Boosting
- Viele Fehler entstehen bei Namen, Fremdwörtern und Fachbegriffen.
- **Umsetzung:**
  - Mechanismus für Vokabular-Biasing im Decoder implementieren (pyctcdecode).
  - `vocab_bias.json` mit typischen Fehlerwörtern, Namen, Fachbegriffen anlegen.
  - Decoder so anpassen, dass diese Wörter gezielt geboostet werden.
  - Testen, ob gezieltes Biasing die Erkennung verbessert.

### 2. Personalisierungs-Loop / Adapter-Feintuning
- Gesammelte Nutzerkorrekturen für LoRA-Feintuning nutzen.
- **Umsetzung:**
  - Trainings-Endpoint `/train/ewc` implementieren.
  - Fisher-Information berechnen, EWCTrainer-Klasse erweitern.
  - Adapter gezielt auf Fehlerwörter und Korrekturen trainieren.
  - Testen, ob Feintuning die Fehler bei schwierigen Wörtern reduziert.

### 3. Monitoring & Logging (Zukunft)
- Monitoring einrichten, um Fehlerwörter und deren Entwicklung gezielt zu tracken.
- Übersicht über Trainingsdaten und Modellversionen bereitstellen.

### 4. Forced Alignment (Zukunft)
- Für exaktere Wortgrenzen und Zeitstempel, aber kein direkter Einfluss auf WER.

## Fazit
**Priorität:**
1. Vokabular-Biasing/Hotword-Boosting umsetzen und testen
2. Personalisierungs-Loop (Feintuning mit Korrekturen) starten
3. Monitoring für gezielte Fehleranalyse

Damit werden die typischen Fehlerquellen (Namen, Fachbegriffe, seltene Wörter) gezielt adressiert und der WER kann weiter gesenkt werden.
