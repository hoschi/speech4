# manager.py
import subprocess
import numpy as np
import os

ALPHA_RANGE = np.arange(0.2, 1, 0.1) # müssen unter 16 sein!!!!
PROGRESS_FILE = 'progress.txt'

def get_completed_alphas():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, 'r') as f:
        return {float(line.strip()) for line in f}

def mark_alpha_as_completed(alpha):
    with open(PROGRESS_FILE, 'a') as f:
        f.write(f"{alpha}\n")

if __name__ == "__main__":
    completed = get_completed_alphas()
    
    for alpha in ALPHA_RANGE:
        alpha_rounded = round(alpha, 2)
        if alpha_rounded in completed:
            print(f"Überspringe bereits abgeschlossenes Alpha: {alpha_rounded}")
            continue

        print(f"\n=============================================")
        print(f"STARTE LAUF FÜR ALPHA = {alpha_rounded}")
        print(f"=============================================")
        
        try:
            # Baue den Kommandozeilenbefehl
            command = [
                "python",
                "server/tune_decoder.py",
                "--alpha",
                str(alpha_rounded)
            ]
            
            # Führe tune_decoder.py als externen Prozess aus und warte, bis er fertig ist
            subprocess.run(command, check=True)
            
            # Wenn erfolgreich, markiere als erledigt
            mark_alpha_as_completed(alpha_rounded)
            print(f"--> Lauf für Alpha {alpha_rounded} erfolgreich beendet.")

        except subprocess.CalledProcessError as e:
            print(f"FEHLER: Lauf für Alpha {alpha_rounded} ist fehlgeschlagen. Stoppe den Manager.")
            print(f"Fehlermeldung: {e}")
            break
        except KeyboardInterrupt:
            print("\nManager durch Benutzer unterbrochen.")
            break