from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = FastAPI()

# Modell und Processor beim Start laden
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)  # type: ignore
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

WINDOW_SIZE = 5120  # 320ms @ 16kHz
OVERLAP = 2560      # 50% Overlap

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    buffer = np.zeros(0, dtype=np.int16)
    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16)
            buffer = np.concatenate([buffer, chunk])
            while len(buffer) >= WINDOW_SIZE:
                audio_tensor = torch.from_numpy(buffer[:WINDOW_SIZE].copy()).float() / 32768.0
                input_values = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_values  # type: ignore
                with torch.no_grad():
                    logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]  # type: ignore
                await websocket.send_text(transcription)
                # Sliding Window: entferne nur OVERLAP Samples, nicht das ganze Fenster
                buffer = buffer[OVERLAP:]
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 