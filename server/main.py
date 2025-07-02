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

app = FastAPI()

# Modell und Processor beim Start laden
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# NEU: KenLM Sprachmodell laden (falls vorhanden)
LM_PATH = "server/lm/4gram_de.klm"

def init_kenlm_decoder(app):
    """
    Stellt sicher, dass das KenLM-Modell existiert und initialisiert den Decoder.
    Hängt den Decoder an app.state.decoder.
    """
    import shutil as _shutil
    CORPUS = "server/data/corpus.txt"
    BASE_CORPUS = "german_base_corpus.txt"
    if not os.path.isfile(LM_PATH):
        print("[INFO] Kein KenLM-Modell gefunden. Erstelle Basismodell aus german_base_corpus.txt ...")
        if not os.path.isfile(BASE_CORPUS):
            raise RuntimeError(f"{BASE_CORPUS} nicht gefunden! Bitte gemäß README herunterladen und extrahieren.")
        _shutil.copy(BASE_CORPUS, CORPUS)
        train_kenlm_pipeline()
        print(f"[INFO] Basismodell erfolgreich generiert: {LM_PATH}")
    else:
        print(f"[INFO] KenLM-Modell gefunden: {LM_PATH}")
    labels = list(processor.tokenizer.get_vocab().keys())
    app.state.decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=LM_PATH,
        alpha=0.5,  # optional, anpassbar
        beta=1.0    # optional, anpassbar
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
            except RuntimeError as e:
                # Verbindung wurde vom Client geschlossen
                break
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    chunk = np.frombuffer(message["bytes"], dtype=np.int16)
                    buffer = np.concatenate([buffer, chunk])
                    full_audio.append(chunk)
                    while len(buffer) >= WINDOW_SIZE:
                        audio_chunk = buffer[:WINDOW_SIZE].copy()
                        audio_tensor = torch.from_numpy(audio_chunk).float() / 32768.0
                        input_values = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_values
                        with torch.no_grad():
                            logits = model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        tokens = predicted_ids[0].tolist()
                        n_tokens = len(tokens)
                        stride_tokens = int(n_tokens * STRIDE / WINDOW_SIZE)
                        middle_tokens = tokens[stride_tokens:n_tokens-stride_tokens] if n_tokens > 2*stride_tokens else []
                        chunk_start = audio_offset + stride_tokens * int(WINDOW_SIZE / n_tokens) if n_tokens > 0 else audio_offset
                        chunk_end = audio_offset + (n_tokens - stride_tokens) * int(WINDOW_SIZE / n_tokens) if n_tokens > 0 else audio_offset + WINDOW_SIZE
                        if middle_tokens:
                            if app.state.decoder:
                                transcription = app.state.decoder.decode(np.array(middle_tokens))
                            else:
                                transcription = processor.batch_decode([middle_tokens])[0]
                            if hypotheses and hypotheses[-1]['end'] > chunk_start:
                                hypotheses[-1]['end'] = chunk_start
                            hypotheses.append({'start': chunk_start, 'end': chunk_end, 'text': transcription})
                            await websocket.send_json({
                                'type': 'hypothesis',
                                'start': chunk_start,
                                'end': chunk_end,
                                'text': transcription
                            })
                        buffer = buffer[WINDOW_SIZE - OVERLAP:]
                        audio_offset += WINDOW_SIZE - OVERLAP
                elif "text" in message and message["text"] == "final":
                    if full_audio:
                        all_audio = np.concatenate(full_audio)
                        audio_tensor = torch.from_numpy(all_audio).float() / 32768.0
                        input_values = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_values
                        with torch.no_grad():
                            logits = model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        tokens = predicted_ids[0].tolist()
                        if tokens:
                            if app.state.decoder:
                                transcription = app.state.decoder.decode(np.array(tokens))
                            else:
                                transcription = processor.batch_decode([tokens])[0]
                            await websocket.send_json({
                                'type': 'final',
                                'text': transcription
                            })
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
    lmplz_cmd = [lmplz_path, "-o", "4","-T","8", "--skip_symbols", "--text", CORPUS, "--arpa", ARPA]

    try:
        run(lmplz_cmd)
    except RuntimeError as e:
        # Fallback: --discount_fallback bei kleinen Daten
        if 'BadDiscountException' in str(e) or 'discount' in str(e):
            lmplz_cmd.insert(3, "--discount_fallback")
            run(lmplz_cmd)
        else:
            raise
    # 3. Komprimieren
    run([build_binary_path, ARPA, KENLM_BIN])
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