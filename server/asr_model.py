"""
Zentrales Modul für das Laden und Verwalten des ASR-Modells (Wav2Vec2),
des Prozessors und des KenLM-Decoders.
"""
import os
import subprocess
import sys
import shutil
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode.decoder import build_ctcdecoder

# --- Konstanten ---
MODEL_NAME = "jonatasgrosman/wav2vec2-xls-r-1b-german" # 350 Beispiele 300s
# MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german" # 350 Beispiele 160s
LM_PATH = "server/lm/4gram_de.klm"
BASE_CORPUS = "german_base_corpus.txt"
CORPUS_DATA_PATH = "server/data/corpus.txt"

# --- KenLM Training & Initialisierung ---

def train_kenlm_pipeline():
    """
    Führt die gesamte Pipeline aus: Vorverarbeitung, KenLM-Training, Modellbereitstellung.
    Nutzt als Fallback den Basis-Korpus, falls noch keine Korrekturen vorliegen.
    """
    LOG = "server/data/update_lm.log"
    ARPA = "server/data/corpus.arpa"
    KENLM_BIN = "server/data/corpus.klm"
    
    # Sicherstellen, dass die Verzeichnisse existieren
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    os.makedirs(os.path.dirname(LM_PATH), exist_ok=True)

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
                raise RuntimeError(f"[ERROR] {' '.join(cmd)}: {e.stderr}")

    # 1. Vorverarbeitung
    run([sys.executable, "server/preprocess_corrections.py"])
    
    use_base = False
    if not os.path.isfile(CORPUS_DATA_PATH) or os.path.getsize(CORPUS_DATA_PATH) < 10:
        if os.path.isfile(BASE_CORPUS):
            run(["cp", BASE_CORPUS, CORPUS_DATA_PATH])
            use_base = True
        else:
            raise RuntimeError(f"[ERROR] Kein Korrektur-Korpus und keine Basisdatei {BASE_CORPUS} gefunden!")

    # 2. KenLM-Training
    lmplz_path = shutil.which("lmplz")
    build_binary_path = shutil.which("build_binary")
    if not lmplz_path or not build_binary_path:
        raise RuntimeError("[ERROR] lmplz oder build_binary nicht im PATH gefunden! Ist KenLM korrekt installiert?")
    
    lmplz_cmd = [lmplz_path, "-o", "4", "--prune", "0", "1", "1", "1", "-S", "80%", "-T", "/tmp", "--skip_symbols", "--text", CORPUS_DATA_PATH, "--arpa", ARPA]
    try:
        run(lmplz_cmd)
    except RuntimeError as e:
        if 'BadDiscountException' in str(e) or 'discount' in str(e):
            print("[INFO] Fallback: Verwende discount_fallback...")
            fallback_cmd = [lmplz_path, "-o", "4", "--discount_fallback", "--prune", "0", "1", "1", "1", "-S", "80%", "-T", "/tmp", "--skip_symbols", "--text", CORPUS_DATA_PATH, "--arpa", ARPA]
            run(fallback_cmd)
        else:
            raise
            
    # 3. Komprimierung
    build_cmd = [build_binary_path, "-a", "22", "-q", "8", "-b", "8", "trie", ARPA, KENLM_BIN]
    run(build_cmd)
    
    # 4. Modell verschieben
    shutil.move(KENLM_BIN, LM_PATH)
    
    if use_base:
        return f"[SUCCESS] KenLM-Basismodell aus {BASE_CORPUS} generiert: {LM_PATH}"
    return f"[SUCCESS] KenLM-Modell aktualisiert und bereit: {LM_PATH}"

def ensure_kenlm_model():
    """
    Stellt sicher, dass das KenLM-Modell existiert. Wenn nicht, wird es aus dem
    Basis-Korpus trainiert.
    """
    if not os.path.isfile(LM_PATH):
        print("[INFO] Kein KenLM-Modell gefunden. Erstelle optimiertes Basismodell...")
        if not os.path.isfile(BASE_CORPUS):
            raise RuntimeError(f"{BASE_CORPUS} nicht gefunden! Bitte gemäß README herunterladen und extrahieren.")
        
        # Stelle sicher, dass der Ziel-Korpus existiert, auch wenn er leer ist
        os.makedirs(os.path.dirname(CORPUS_DATA_PATH), exist_ok=True)
        if not os.path.exists(CORPUS_DATA_PATH):
            with open(CORPUS_DATA_PATH, 'w') as f:
                pass # Leere Datei erstellen
        
        train_kenlm_pipeline()
        print(f"[INFO] Optimiertes Modell erstellt: {LM_PATH}")
    else:
        print(f"[INFO] Vorhandenes KenLM-Modell gefunden: {LM_PATH}")


# --- ASR-Modell-Klasse ---

class ASRModel:
    """
    Kapselt das Wav2Vec2-Modell, den Prozessor und die Decoder-Erstellung.
    """
    def __init__(self):
        print(f"[INFO] Lade ASR-Modell: {MODEL_NAME}")
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        if isinstance(self.processor, tuple):
            self.processor = self.processor[0]
        
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        self.model.eval()
        
        self.labels = self._extract_labels()
        print("[INFO] ASR-Modell und Prozessor erfolgreich geladen.")

    def _extract_labels(self):
        """Extrahiert die Vokabular-Labels aus dem Prozessor."""
        if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'get_vocab'):
            vocab_dict = self.processor.tokenizer.get_vocab()
        elif hasattr(self.processor, 'feature_extractor') and hasattr(self.processor.feature_extractor, 'vocab'):
            vocab_dict = self.processor.feature_extractor.vocab
        else:
            raise AttributeError('Wav2Vec2Processor hat weder tokenizer.get_vocab() noch feature_extractor.vocab!')
        
        return [k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])]

    def get_logits(self, audio_array, sampling_rate):
        """Berechnet die Logits für ein gegebenes Audio-Array."""
        if sampling_rate != 16000:
            # Audio-Array muss beschreibbar sein für librosa
            audio_array = np.ascontiguousarray(audio_array)
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        
        # Normalisierung auf [-1, 1] für Wav2Vec2
        audio_tensor = torch.from_numpy(audio_array).float()
        if torch.max(torch.abs(audio_tensor)) > 1.0:
             audio_tensor /= 32768.0

        input_values = self.processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True).input_values
        with torch.no_grad():
            logits = self.model(input_values).logits[0]
            
        return logits.cpu().numpy()

    def build_decoder(self, alpha=0.2, beta=-1.0, lm_path=LM_PATH, hotwords=None):
        """Erstellt und gibt einen CTC-Decoder mit den angegebenen Parametern und Hotwords zurück."""
        if not os.path.isfile(lm_path):
            print(f"[WARN] KenLM-Modell unter {lm_path} nicht gefunden. Decoder wird ohne Sprachmodell erstellt.")
            return None
        return build_ctcdecoder(
            labels=self.labels,
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta,
            hotwords=hotwords
        )

    def decode_logits(self, logits, decoder=None, hotwords=None, hotword_weight=10.0):
        """
        Dekodiert Logits zu Text. Verwendet den übergebenen Decoder und Hotwords oder fällt auf simples ArgMax-Decoding zurück.
        """
        if decoder:
            if hotwords:
                return decoder.decode(logits, hotwords=hotwords, hotword_weight=hotword_weight)
            else:
                return decoder.decode(logits)
        else:
            predicted_ids = torch.argmax(torch.from_numpy(logits), dim=-1)
            return self.processor.batch_decode([predicted_ids.tolist()])[0]
