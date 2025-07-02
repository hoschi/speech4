import subprocess
import sys
import os
import shutil

LOG = "server/data/update_lm.log"
CORPUS = "server/data/corpus.txt"
ARPA = "server/data/corpus.arpa"
KENLM_BIN = "server/data/corpus.klm"
LM_TARGET = "server/lm/4gram_de.klm"


def run(cmd):
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n[RUN] {' '.join(cmd)}\n")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            log.write(result.stdout)
            log.write(result.stderr)
            return result.stdout
        except subprocess.CalledProcessError as e:
            log.write(e.stdout or '')
            log.write(e.stderr or '')
            print(f"[ERROR] {' '.join(cmd)}: {e.stderr}")
            sys.exit(1)

if __name__ == "__main__":
    print("[INFO] Starte Vorverarbeitung...")
    run(["python3", "server/preprocess_corrections.py"])
    if not os.path.isfile(CORPUS):
        print(f"[ERROR] {CORPUS} nicht gefunden!")
        sys.exit(1)
    print("[INFO] Starte KenLM-Training...")
    run(["lmplz", "-o", "4", "--text", CORPUS, "--arpa", ARPA])
    print("[INFO] Komprimiere Modell...")
    run(["build_binary", ARPA, KENLM_BIN])
    print(f"[INFO] Verschiebe Modell nach {LM_TARGET} ...")
    shutil.move(KENLM_BIN, LM_TARGET)
    print(f"[SUCCESS] KenLM-Modell bereit: {LM_TARGET}") 