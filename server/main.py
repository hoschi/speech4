from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import torch
import numpy as np
import shutil
import datetime
import json

# Importiere die zentrale ASR-Modell-Klasse und Initialisierungsfunktionen
from asr_model import ASRModel, LM_PATH
import pathlib
import os

app = FastAPI()

# --- Globale Objekte ---
asr_model = ASRModel()
# Hotwords laden, falls vorhanden
hotwords_path = pathlib.Path("personalized_lm_output/hotwords.txt")
hotwords = None
if hotwords_path.exists():
    with open(hotwords_path, "r", encoding="utf-8") as f:
        hotwords = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Hotwords für Decoder geladen: {hotwords[:10]} ... (insgesamt {len(hotwords)})")
# Prüfe, ob KenLM-Modell existiert, ansonsten Fehler und Abbruch
if not os.path.isfile(LM_PATH):
    raise RuntimeError(f"[ERROR] KenLM-Modell nicht gefunden unter {LM_PATH}. Bitte zuerst mit 'python -m server.train_lm' trainieren.")
app.state.decoder = asr_model.build_decoder(alpha=0.45, beta=-1.1, hotwords=hotwords)
if app.state.decoder:
    print(f"[INFO] Standard-Decoder erfolgreich geladen.")
else:
    print("[WARN] Konnte Standard-Decoder nicht laden. Fällt auf Decoding ohne Sprachmodell zurück.")


# --- Streaming-Konstanten ---
WINDOW_SIZE = 48000  # 3 Sekunden @ 16kHz
STRIDE = 12000      # 0.75 Sekunden vorne und hinten (insgesamt 1.5s Overlap)
OVERLAP = STRIDE * 2

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Nimmt einen kontinuierlichen Audiostrom per WebSocket entgegen, führt ASR in Chunks
    durch und sendet Hypothesen sowie ein finales Transkript zurück.
    """
    await websocket.accept()
    buffer = np.zeros(0, dtype=np.int16)
    full_audio = []
    hypotheses = []
    audio_offset = 0

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
                    chunk = np.frombuffer(message["bytes"], dtype=np.int16)
                    buffer = np.concatenate([buffer, chunk])
                    full_audio.append(chunk)

                    while len(buffer) >= WINDOW_SIZE:
                        audio_chunk = buffer[:WINDOW_SIZE].copy()

                        # Logits mit der zentralen Methode berechnen
                        logits = asr_model.get_logits(audio_chunk, sampling_rate=16000)

                        n_frames = logits.shape[0]
                        stride_frames = int(n_frames * STRIDE / WINDOW_SIZE)

                        middle_logits = logits[stride_frames : n_frames - stride_frames] if n_frames > 2 * stride_frames else logits

                        chunk_start = audio_offset + stride_frames * int(WINDOW_SIZE / n_frames) if n_frames > 0 else audio_offset
                        chunk_end = audio_offset + (n_frames - stride_frames) * int(WINDOW_SIZE / n_frames) if n_frames > 0 else audio_offset + WINDOW_SIZE

                        if middle_logits.shape[0] > 0:
                            # Dekodierung über die zentrale Methode
                            transcription = asr_model.decode_logits(middle_logits, app.state.decoder)

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

                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if isinstance(data, dict) and data.get("text") == "final":
                            if full_audio:
                                all_audio = np.concatenate(full_audio)
                                # Finale Logits und Transkription
                                final_logits = asr_model.get_logits(all_audio, sampling_rate=16000)
                                final_transcription = asr_model.decode_logits(final_logits, app.state.decoder)

                                await websocket.send_json({
                                    'type': 'final',
                                    'text': final_transcription
                                })
                    except (json.JSONDecodeError, KeyError):
                        pass # Ignoriere Nachrichten, die nicht dem erwarteten Format entsprechen

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        await websocket.send_json({'type': 'error', 'message': str(e)})


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
    if audio and audio.filename:
        audio_path = base_path + ".wav"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

    return {"status": "ok", "text_file": text_path, "audio_file": audio_path}





if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
