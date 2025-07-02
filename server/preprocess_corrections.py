import os
import re
from pathlib import Path

CORRECTIONS_DIR = 'server/corrections'
DATA_DIR = 'server/data'
CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.txt')
LOG_PATH = os.path.join(DATA_DIR, 'processed_files.log')

os.makedirs(DATA_DIR, exist_ok=True)

# Lade bereits verarbeitete Dateien
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        processed = set(line.strip() for line in f)
else:
    processed = set()

# Finde alle neuen Korrekturdateien
all_txts = sorted([f for f in os.listdir(CORRECTIONS_DIR) if f.endswith('.txt')])
new_txts = [f for f in all_txts if f not in processed]

if not new_txts:
    print('Keine neuen Korrekturdateien gefunden.')
    exit(0)

def preprocess(text):
    # Entferne Sonderzeichen, Tokenisierung: ein Satz pro Zeile
    text = re.sub(r'[^\wäöüÄÖÜß .,!?\n]', '', text)
    # Splitte in Sätze (sehr einfach)
    sents = re.split(r'[.!?]\s*', text)
    sents = [s.strip() for s in sents if s.strip()]
    return '\n'.join(sents)

with open(CORPUS_PATH, 'a', encoding='utf-8') as corpus, open(LOG_PATH, 'a', encoding='utf-8') as log:
    for fname in new_txts:
        path = os.path.join(CORRECTIONS_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        clean = preprocess(raw)
        corpus.write(clean + '\n')
        log.write(fname + '\n')
        print(f'Korrigiert und hinzugefügt: {fname}') 