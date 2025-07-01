from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

app = FastAPI()

# Modell und Processor beim Start laden
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german"
processor_obj = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
if isinstance(processor_obj, tuple):
    processor = processor_obj[0]
else:
    processor = processor_obj
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

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
                        if middle_tokens:
                            transcription = processor.batch_decode([middle_tokens])[0]
                            await websocket.send_text(transcription)
                        buffer = buffer[WINDOW_SIZE - OVERLAP:]
                elif "text" in message and message["text"] == "final":
                    # Finales Transkript für das gesamte Audio
                    if full_audio:
                        all_audio = np.concatenate(full_audio)
                        audio_tensor = torch.from_numpy(all_audio).float() / 32768.0
                        input_values = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_values
                        with torch.no_grad():
                            logits = model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        tokens = predicted_ids[0].tolist()
                        if tokens:
                            transcription = processor.batch_decode([tokens])[0]
                            await websocket.send_text("[FINAL] " + transcription)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 