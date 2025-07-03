from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
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
from vosk import Model, KaldiRecognizer

app = FastAPI()

# CORS für React-Dev-Server aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Port deines Vite-Servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modell und Processor beim Start laden
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# NEU: KenLM Sprachmodell laden (falls vorhanden)
LM_PATH = "server/lm/4gram_de.klm"

VOSK_MODEL_PATH = "server/models/vosk-model-de-0.21"
if not os.path.exists(VOSK_MODEL_PATH):
    raise RuntimeError(f"VOSK-Modell nicht gefunden unter {VOSK_MODEL_PATH}. Bitte gemäß README herunterladen.")

vosk_model = Model(VOSK_MODEL_PATH)
app.state.custom_vocabulary = []

def init_kenlm_decoder(app):
    """
    Lädt die Vokabelliste aus den bisherigen Korrekturen.
    """
    if os.path.exists("server/data/corpus.txt"):
        print(f"[INFO] Lade benutzerdefiniertes Vokabular aus server/data/corpus.txt")
        with open("server/data/corpus.txt", "r", encoding="utf-8") as f:
            words = set(word for line in f for word in line.lower().split())
            app.state.custom_vocabulary = list(words)
        print(f"[INFO] {len(app.state.custom_vocabulary)} einzigartige Wörter geladen.")

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

    # Erstelle einen VOSK-Recognizer mit dem optionalen Vokabular
    vocabulary = app.state.custom_vocabulary
    if vocabulary:
        recognizer = KaldiRecognizer(vosk_model, 16000, json.dumps(vocabulary, ensure_ascii=False))
        print("[INFO] VOSK Recognizer mit benutzerdefiniertem Vokabular initialisiert.")
    else:
        recognizer = KaldiRecognizer(vosk_model, 16000)
        print("[INFO] VOSK Recognizer ohne benutzerdefiniertes Vokabular initialisiert.")

    try:
        while True:
            try:
                data = await websocket.receive()
                print(f"[WS] Empfangen: type={data.get('type')}")
            except RuntimeError as e:
                break
            if data.get('bytes'):
                # Audio-Daten verarbeiten
                if recognizer.AcceptWaveform(data['bytes']):
                    result = json.loads(recognizer.Result())
                    await websocket.send_text(json.dumps(result))
                else:
                    result = json.loads(recognizer.PartialResult())
                    await websocket.send_text(json.dumps(result))
            if data.get('text'):
                if json.loads(data['text']).get('eof'):
                    final_result = json.loads(recognizer.FinalResult())
                    await websocket.send_text(json.dumps(final_result))
                    break
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"[ERROR] WebSocket: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

@app.post("/upload/correction")
async def upload_correction(text: str = Form(...), audio: UploadFile = File(...)):
    """
    Speichert Korrekturtext (und optional zugehörige Audiodatei) in server/corrections/.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = f"server/corrections/{now}"
    text_path = base_path + ".txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    audio_path = base_path + ".wav"
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    return {"status": "ok", "text_file": text_path, "audio_file": audio_path}

@app.post("/train/lm")
def train_lm():
    """
    Löst die Vorverarbeitung der Korrekturen aus, um das Vokabular zu aktualisieren.
    """
    try:
        result = subprocess.run([sys.executable, "server/preprocess_corrections.py"], capture_output=True, text=True, check=True)
        output = result.stdout
        init_kenlm_decoder(app)
        return JSONResponse(content={"status": "success", "output": output})
    except Exception as e:
        return JSONResponse(content={"status": "error", "output": str(e)}, status_code=500)

@app.on_event("startup")
def startup_event():
    init_kenlm_decoder(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["server"])
