from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
from pyctcdecode.decoder import BeamSearchDecoderCTC
import os
import shutil
import datetime
import subprocess
import sys
import threading
import kenlm
from pyctcdecode.alphabet import Alphabet
from pyctcdecode.decoder import build_ctcdecoder
import json

app = FastAPI()

# Modell und Processor beim Start laden
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# NEU: KenLM Sprachmodell laden (falls vorhanden) - Optimiert für 32GB RAM
LM_PATH = "server/lm/4gram_de.klm"

def init_kenlm_decoder(app):
    """
    Stellt sicher, dass das KenLM-Modell existiert und initialisiert den Decoder.
    Hängt den Decoder an app.state.decoder.
    Verwendet optimierte Parameter für 32GB RAM-Kompatibilität.
    """
    import shutil as _shutil
    CORPUS = "server/data/corpus.txt"
    BASE_CORPUS = "german_base_corpus.txt"
    
    if not os.path.isfile(LM_PATH):
        print("[INFO] Kein KenLM-Modell gefunden. Erstelle optimiertes Basismodell...")
        if not os.path.isfile(BASE_CORPUS):
            raise RuntimeError(f"{BASE_CORPUS} nicht gefunden! Bitte gemäß README herunterladen und extrahieren.")
        _shutil.copy(BASE_CORPUS, CORPUS)
        train_kenlm_pipeline()
        print(f"[INFO] Optimiertes Modell erstellt: {LM_PATH}")
    else:
        print(f"[INFO] KenLM-Modell gefunden: {LM_PATH}")
    
    # Extrahiere Labels aus dem Processor
    labels = list(processor.tokenizer.get_vocab().keys())
    app.state.decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=LM_PATH,
        alpha=0.2,  # optional, anpassbar
        beta=-1.0    # optional, anpassbar
    )
    print(f"[INFO] KenLM Decoder geladen: {LM_PATH}")

def remove_word_repeats(text):
    words = text.split()
    result = []
    for w in words:
        if not result or w != result[-1]:
            result.append(w)
    return ' '.join(result)

WINDOW_SIZE = 48000  # 3 Sekunden @ 16kHz
STRIDE = 12000      # 0.75 Sekunden vorne und hinten (insgesamt 1.5s Overlap)
OVERLAP = STRIDE * 2

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    buffer = np.zeros(0, dtype=np.int16)
    full_audio = []
    hypotheses = []  # Buffer für Hypothesen: Liste aus Dicts mit 'start', 'end', 'text'
    audio_offset = 0  # Gesamtanzahl Samples, die bereits verarbeitet wurden
    try:
        while True:
            try:
                message = await websocket.receive()
                print(f"[WS] Empfangen: type={message.get('type')} keys={list(message.keys())} size={len(message.get('bytes', b'')) if 'bytes' in message else '-'} text={message.get('text', '')[:100]}")
            except RuntimeError as e:
                # Verbindung wurde vom Client geschlossen
                break
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Empfange neuen Audio-Chunk vom Client und füge ihn zum Buffer hinzu
                    chunk = np.frombuffer(message["bytes"], dtype=np.int16)
                    buffer = np.concatenate([buffer, chunk])
                    full_audio.append(chunk)
                    # Solange genug Samples für ein Fenster vorhanden sind, führe Inferenz durch
                    while len(buffer) >= WINDOW_SIZE:
                        # Extrahiere ein Fenster (z.B. 3 Sekunden) für die Inferenz
                        audio_chunk = buffer[:WINDOW_SIZE].copy()
                        # Wandle PCM in Float-Tensor um und normalisiere auf [-1, 1]
                        audio_tensor = torch.from_numpy(audio_chunk).float() / 32768.0
                        # Feature-Extraktion und Padding via Processor
                        input_values = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_values
                        with torch.no_grad():
                            # Modell-Inferenz: Erzeuge Logits (Wahrscheinlichkeiten für jedes Token pro Frame)
                            logits = model(input_values).logits
                        n_frames = logits.shape[1]
                        # Berechne wie viele Frames vorne und hinten als Overlap ignoriert werden (für Kontext)
                        stride_frames = int(n_frames * STRIDE / WINDOW_SIZE)
                        # Nutze nur die mittleren Frames für das aktuelle Fenster (Overlapping-Chunking)
                        middle_logits = logits[0][stride_frames:n_frames-stride_frames] if n_frames > 2*stride_frames else logits[0]
                        # Berechne Start- und Endposition des aktuellen Chunks im Gesamtaudio
                        chunk_start = audio_offset + stride_frames * int(WINDOW_SIZE / n_frames) if n_frames > 0 else audio_offset
                        chunk_end = audio_offset + (n_frames - stride_frames) * int(WINDOW_SIZE / n_frames) if n_frames > 0 else audio_offset + WINDOW_SIZE
                        if middle_logits.shape[0] > 0:
                            # Dekodiere die Logits zu Text (mit KenLM, falls vorhanden)
                            if app.state.decoder:
                                transcription = app.state.decoder.decode(middle_logits.cpu().numpy())
                            else:
                                predicted_ids = torch.argmax(middle_logits, dim=-1)
                                tokens = predicted_ids.tolist()
                                transcription = processor.batch_decode([tokens])[0]
                            # Verhindere Überschneidungen mit vorherigem Chunk
                            if hypotheses and hypotheses[-1]['end'] > chunk_start:
                                hypotheses[-1]['end'] = chunk_start
                            # Speichere Hypothese und sende sie an den Client
                            hypotheses.append({'start': chunk_start, 'end': chunk_end, 'text': transcription})
                            await websocket.send_json({
                                'type': 'hypothesis',
                                'start': chunk_start,
                                'end': chunk_end,
                                'text': transcription
                            })
                        # Entferne das verarbeitete Fenster, lasse Overlap für Kontext stehen
                        buffer = buffer[WINDOW_SIZE - OVERLAP:]
                        audio_offset += WINDOW_SIZE - OVERLAP
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if isinstance(data, dict) and data.get("text") == "final":
                            # Finale Nachricht vom Client: Gesamtes Audio verarbeiten
                            if full_audio:
                                all_audio = np.concatenate(full_audio)
                                audio_tensor = torch.from_numpy(all_audio).float() / 32768.0
                                input_values = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_values
                                with torch.no_grad():
                                    logits = model(input_values).logits
                                if logits.shape[1] > 0:
                                    # Dekodiere das gesamte Audio zu einem finalen Transkript
                                    if app.state.decoder:
                                        transcription = app.state.decoder.decode(logits[0].cpu().numpy())
                                    else:
                                        predicted_ids = torch.argmax(logits, dim=-1)
                                        tokens = predicted_ids[0].tolist()
                                        transcription = processor.batch_decode([tokens])[0]
                                    # Sende das finale Transkript an den Client
                                    await websocket.send_json({
                                        'type': 'final',
                                        'text': transcription
                                    })
                    except Exception:
                        pass  # ignore non-JSON text messages
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })

@app.post("/upload/correction")
async def upload_correction(text: str = Form(...), audio: UploadFile = File(None)):
    """
    Speichert Korrekturtext (und optional zugehörige Audiodatei) in server/corrections/.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = f"server/corrections/{now}"
    text_path = base_path + ".txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    audio_path = None
    if audio is not None:
        audio_path = base_path + ".wav"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    return {"status": "ok", "text_file": text_path, "audio_file": audio_path}

def train_kenlm_pipeline():
    """
    Führt die gesamte Pipeline aus: Vorverarbeitung, KenLM-Training, Modellbereitstellung.
    Nutzt als Fallback den Basis-Korpus (german_base_corpus.txt), falls noch keine Korrekturen vorliegen.
    """
    LOG = "server/data/update_lm.log"
    CORPUS = "server/data/corpus.txt"
    BASE_CORPUS = "german_base_corpus.txt"
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
                raise RuntimeError(f"[ERROR] {' '.join(cmd)}: {e.stderr}")

    # 1. Vorverarbeitung (immer venv-Python)
    run([sys.executable, "server/preprocess_corrections.py"])
    # Prüfe, ob Korpus leer ist oder keine Korrekturen vorliegen
    use_base = False
    if not os.path.isfile(CORPUS) or os.path.getsize(CORPUS) < 10:
        if os.path.isfile(BASE_CORPUS):
            run(["cp", BASE_CORPUS, CORPUS])
            use_base = True
        else:
            raise RuntimeError(f"[ERROR] Kein Korrektur-Korpus und keine Basisdatei {BASE_CORPUS} gefunden!")
    # 2. KenLM-Training
    lmplz_path = shutil.which("lmplz")
    build_binary_path = shutil.which("build_binary")
    if not lmplz_path or not build_binary_path:
        raise RuntimeError("[ERROR] lmplz oder build_binary nicht im PATH gefunden! Ist KenLM korrekt installiert?")
    # Optimierte Parameter für 32GB RAM: Aggressives Pruning + Quantisierung
    lmplz_cmd = [lmplz_path, "-o", "4", "--prune", "0", "1", "1", "1", "-S", "80%", "-T", "/tmp", "--skip_symbols", "--text", CORPUS, "--arpa", ARPA]

    try:
        run(lmplz_cmd)
    except RuntimeError as e:
        # Fallback: --discount_fallback bei kleinen Daten
        if 'BadDiscountException' in str(e) or 'discount' in str(e):
            print("[INFO] Fallback: Verwende discount_fallback...")
            fallback_cmd = [lmplz_path, "-o", "4", "--discount_fallback", "--prune", "0", "1", "1", "1", "-S", "80%", "-T", "/tmp", "--skip_symbols", "--text", CORPUS, "--arpa", ARPA]
            run(fallback_cmd)
        else:
            raise
    # 3. Optimierte Komprimierung mit 8-bit Quantisierung
    build_cmd = [build_binary_path, "-a", "22", "-q", "8", "-b", "8", "trie", ARPA, KENLM_BIN]
    run(build_cmd)
    # 4. Modell verschieben
    shutil.move(KENLM_BIN, LM_TARGET)
    if use_base:
        return f"[SUCCESS] KenLM-Basismodell aus {BASE_CORPUS} generiert: {LM_TARGET}"
    return f"[SUCCESS] KenLM-Modell bereit: {LM_TARGET}"

@app.post("/train/lm")
def train_lm():
    """
    Führt die KenLM-Trainingspipeline direkt im Python-Prozess aus.
    """
    try:
        output = train_kenlm_pipeline()
        return JSONResponse(content={"status": "success", "output": output})
    except Exception as e:
        return JSONResponse(content={"status": "error", "output": str(e)}, status_code=500)



# Initialisierung beim Serverstart (blockierend, garantiert Decoder)
init_kenlm_decoder(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
