import re
import csv
from datetime import datetime
import numpy as np

LOG_FILE = 'server/reports/tune-decoder/log.txt'
CSV_FILE = 'server/reports/tune-decoder/durations.csv'

# Regex, um Zeitstempel, Alpha und Beta aus den relevanten Zeilen zu extrahieren
# Format: 2025-07-16 11:47:15,013 [INFO] (1/100) Ergebnis erhalten für Alpha: 0.50, Beta: -1.50 -> Avg. WER: 0.4405
log_pattern = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"Ergebnis erhalten für Alpha: ([-\d.]+).*?"
    r"Beta: ([-\d.]+)"
)

parsed_entries = []

print(f"Lese und parse Log-Datei: {LOG_FILE}")

# 1. Log-Datei einlesen und parsen
with open(LOG_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        match = log_pattern.search(line)
        if match:
            timestamp_str, alpha_str, beta_str = match.groups()
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            alpha = float(alpha_str)
            beta = float(beta_str)
            parsed_entries.append({'time': timestamp, 'alpha': alpha, 'beta': beta})

# Nach Zeitstempel sortieren, um die korrekte Reihenfolge sicherzustellen
parsed_entries.sort(key=lambda x: x['time'])

durations_data = []
all_durations_sec = []

print(f"Berechne die Dauer für {len(parsed_entries)} abgeschlossene Jobs...")

# 2. Dauer berechnen und CSV-Daten vorbereiten
# Wir starten bei 1, da die Dauer immer die Differenz zum vorherigen Job ist
for i in range(1, len(parsed_entries)):
    previous_entry = parsed_entries[i-1]
    current_entry = parsed_entries[i]
    
    # Die Dauer eines Jobs ist die Zeitdifferenz zwischen seinem Ende und dem Ende des vorherigen Jobs
    duration = current_entry['time'] - previous_entry['time']
    duration_sec = duration.total_seconds()
    
    # Die Dauer gehört zum *aktuellen* Job
    alpha = current_entry['alpha']
    beta = current_entry['beta']
    
    durations_data.append([f"{alpha:.2f}", f"{beta:.2f}", f"{duration_sec:.2f}"])
    all_durations_sec.append(duration_sec)

# 3. CSV-Datei schreiben
with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['alpha', 'beta', 'dauer_in_sekunden'])
    writer.writerows(durations_data)

print(f"CSV-Datei erfolgreich erstellt: {CSV_FILE}")

# 4. Median und andere Statistiken berechnen
if all_durations_sec:
    median_duration = np.median(all_durations_sec)
    mean_duration = np.mean(all_durations_sec)
    max_duration = np.max(all_durations_sec)
    min_duration = np.min(all_durations_sec)
    
    print("\n--- Analyse der Berechnungszeiten ---")
    print(f"Median der Dauer: {median_duration:.2f} Sekunden")
    print(f"Durchschnittliche Dauer: {mean_duration:.2f} Sekunden")
    print(f"Maximale Dauer: {max_duration:.2f} Sekunden")
    print(f"Minimale Dauer: {min_duration:.2f} Sekunden")
else:
    print("Keine Dauer-Daten gefunden.")