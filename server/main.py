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
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel

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

def load_custom_vocabulary(app):
    """
    Lädt die Vokabelliste aus den bisherigen Korrekturen für VOSK.
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
    recognizer = KaldiRecognizer(vosk_model, 16000, json.dumps(vocabulary, ensure_ascii=False)) if vocabulary else KaldiRecognizer(vosk_model, 16000)
    
    # Aktiviere mehrere Alternativen (N-Best-Liste)
    recognizer.SetMaxAlternatives(3)  # Erstelle bis zu 3 Alternativen
    
    if vocabulary:
        print("[INFO] VOSK Recognizer mit benutzerdefiniertem Vokabular und N-Best-Alternativen initialisiert.")
    else:
        print("[INFO] VOSK Recognizer ohne benutzerdefiniertes Vokabular und N-Best-Alternativen initialisiert.")

    try:
        while True:
            # Empfange die nächste Nachricht vom Client
            message = await websocket.receive()
            
            # Prüfe, ob es sich um Binärdaten (Audio) handelt
            if "bytes" in message and message["bytes"]:
                audio_data = message["bytes"]
                # Konvertiere bytes zu Int16Array für VOSK
                import array
                audio_array = array.array('h', audio_data)
                
                # Verarbeite das Audio, aber sende nur partielle Ergebnisse
                # VOSK entscheidet selbst, wann es AcceptWaveform() True zurückgibt
                # aber wir ignorieren das und senden nur PartialResult()
                recognizer.AcceptWaveform(audio_array.tobytes())
                
                # Sende immer nur partielle Ergebnisse, nie finale
                partial_result_json = recognizer.PartialResult()
                await websocket.send_text(partial_result_json)
            
            # Prüfe, ob es sich um eine Textnachricht handelt (für Steuersignale)
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    # EOF-Signal vom Client, um die Transkription abzuschließen
                    if data.get('eof') == 1:
                        final_result_json = recognizer.FinalResult()
                        # Sende ein spezielles finales Ergebnis
                        final_data = json.loads(final_result_json)
                        final_data['type'] = 'final'  # Markiere als finales Ergebnis
                        
                        # VOSK gibt bereits Alternativen zurück, wenn SetMaxAlternatives gesetzt ist
                        # Die Alternativen sind in der JSON-Struktur enthalten
                        if 'alternatives' in final_data:
                            print(f"[INFO] VOSK lieferte {len(final_data['alternatives'])} echte Alternativen")
                        
                        await websocket.send_text(json.dumps(final_data))
                        # Beende die Schleife, nachdem das Endergebnis gesendet wurde
                        break
                except json.JSONDecodeError:
                    # Ignoriere Textnachrichten, die kein valides JSON sind
                    print(f"[WARN] Ungültige Textnachricht vom Client empfangen: {message['text']}")
                    pass

    except WebSocketDisconnect:
        print("[INFO] WebSocket disconnected.")
    except Exception as e:
        print(f"[ERROR] Ein unerwarteter Fehler ist aufgetreten: {e}")
        # Optional: Sende eine Fehlermeldung an den Client
        await websocket.send_text(json.dumps({'type': 'error', 'message': str(e)}))
    finally:
        print("[INFO] Schließe WebSocket-Verbindung.")

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
        load_custom_vocabulary(app)
        return JSONResponse(content={"status": "success", "output": output})
    except Exception as e:
        return JSONResponse(content={"status": "error", "output": str(e)}, status_code=500)

# Request-Model für Ollama
class OllamaRequest(BaseModel):
    text: str

@app.post("/ollama/stream")
async def ollama_stream(req: OllamaRequest):
    """
    Streamt die Antwort des Ollama-Modells 'asr-fixer' auf den gegebenen Text.
    """
    async def stream_gen():
        url = "http://localhost:11434/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "asr-fixer",
            "messages": [
                {"role": "user", "content": req.text}
            ],
            "stream": True
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                async for chunk in response.aiter_text():
                    if chunk:
                        # Logge den Roh-Chunk
                        print("[OLLAMA-RAW]", chunk.strip())
                        # Versuche, das JSON zu parsen und nur den Text zu extrahieren
                        try:
                            for line in chunk.strip().splitlines():
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                        except Exception as e:
                            print(f"[OLLAMA-STREAM-ERROR] {e}")
                            continue
    return StreamingResponse(stream_gen(), media_type="text/plain")

@app.on_event("startup")
def startup_event():
    load_custom_vocabulary(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["server"])
