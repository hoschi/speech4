import subprocess
import numpy as np
import os
import glob
import csv

# ==============================================================================
# KONFIGURATION
# ==============================================================================
ALPHA_RANGE = np.arange(0.0, 1, 0.1)
PROGRESS_FILE = 'progress.txt'
# Das Verzeichnis, in dem tune_decoder.py seine CSVs speichert
REPORT_DIR = os.path.join("server", "reports", "tune-decoder")

# ==============================================================================
# HELFER-FUNKTIONEN
# ==============================================================================

def get_completed_alphas():
    """Liest die bereits abgeschlossenen Alpha-Werte aus der Fortschrittsdatei."""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return {float(line.strip()) for line in f}
    except ValueError:
        print("Warnung: Konnte progress.txt nicht als Float parsen. Starte neu.")
        return set()

def mark_alpha_as_completed(alpha):
    """Markiert einen Alpha-Wert als abgeschlossen."""
    with open(PROGRESS_FILE, 'a') as f:
        f.write(f"{alpha}\n")

def summarize_results():
    """
    Sucht alle einzelnen CSV-Dateien, führt sie zusammen und ermittelt die beste Kombination.
    """
    print("\n=============================================")
    print("ZUSAMMENFASSUNG ALLER ERGEBNISSE")
    print("=============================================")
    
    # 1. Alle einzelnen Ergebnis-CSVs finden
    pattern = os.path.join(REPORT_DIR, '*_alpha_*.csv')
    result_files = glob.glob(pattern)
    
    if not result_files:
        print("Keine Ergebnis-Dateien gefunden. Es gibt nichts zusammenzufassen.")
        return

    print(f"{len(result_files)} einzelne CSV-Dateien gefunden, die zusammengeführt werden...")

    all_data = []
    header = []

    # 2. Alle Daten einlesen
    for filename in result_files:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Lese Header nur aus der ersten Datei
            if not header:
                header = next(reader)
            else:
                next(reader) # Überspringe Header in anderen Dateien
            
            for row in reader:
                all_data.append(row)

    if not all_data:
        print("Keine Daten in den Ergebnis-Dateien gefunden.")
        return

    # 3. Finde die beste Zeile (den niedrigsten WER)
    # WER ist in der 3. Spalte (Index 2)
    best_row = min(all_data, key=lambda row: float(row[2]))
    best_alpha = best_row[0]
    best_beta = best_row[1]
    best_wer = float(best_row[2])

    print("\n--- BESTES GLOBALES ERGEBNIS ---")
    print(f"Beste Alpha: {best_alpha}")
    print(f"Beste Beta:  {best_beta}")
    print(f"Beste avg. WER: {best_wer:.4f}")
    
    # 4. Schreibe die finale, zusammengefasste CSV-Datei
    summary_path = os.path.join(REPORT_DIR, 'final_summary.csv')
    try:
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            # Sortiere Daten für eine schönere Übersicht
            all_data.sort(key=lambda row: (float(row[0]), float(row[1])))
            writer.writerows(all_data)
        print(f"\nZusammenfassender Report gespeichert unter: {summary_path}")
    except Exception as e:
        print(f"Fehler beim Schreiben des zusammenfassenden Reports: {e}")

# ==============================================================================
# HAUPT-LOGIK
# ==============================================================================

if __name__ == "__main__":
    completed_alphas = get_completed_alphas()
    all_alphas_rounded = {round(a, 2) for a in ALPHA_RANGE}
    
    # Prüfen, ob noch Arbeit zu tun ist
    is_work_done = all_alphas_rounded.issubset(completed_alphas)
    
    if not is_work_done:
        for alpha in ALPHA_RANGE:
            alpha_rounded = round(alpha, 2)
            if alpha_rounded in completed_alphas:
                print(f"Überspringe bereits abgeschlossenes Alpha: {alpha_rounded}")
                continue

            print(f"\n=============================================")
            print(f"STARTE LAUF FÜR ALPHA = {alpha_rounded}")
            print(f"=============================================")
            
            try:
                command = ["python", "server/tune_decoder.py", "--alpha", str(alpha_rounded)]
                subprocess.run(command, check=True, text=True) # text=True für bessere Ausgabe
                mark_alpha_as_completed(alpha_rounded)
                print(f"--> Lauf für Alpha {alpha_rounded} erfolgreich beendet.")

            except subprocess.CalledProcessError as e:
                print(f"FEHLER: Lauf für Alpha {alpha_rounded} ist fehlgeschlagen. Stoppe den Manager.")
                break
            except KeyboardInterrupt:
                print("\nManager durch Benutzer unterbrochen.")
                break
    else:
        print("Alle Alpha-Läufe sind bereits abgeschlossen.")

    # Führe die Zusammenfassung am Ende aus, egal ob neue Läufe stattfanden oder nicht
    summarize_results()