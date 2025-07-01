### Plan

#### Zweck
Dieses Dokument beschreibt die übergeordnete Vision, die Systemarchitektur, technische Beschränkungen, den Stack und die eingesetzten Werkzeuge für das selbstlernende, personalisierte Transkriptionssystem.

#### Vision auf hoher Ebene
Ein leichtgewichtiger, personalisierter Speech-to-Text-Service, der mit weniger als 5 Minuten Nutzerdaten startklar ist, kontinuierlich aus Korrekturen lernt und Echtzeit-Transkription mit Code-Switching zwischen Deutsch und englischen Fachbegriffen ermöglicht.

#### Architektur
- **Server:** MacBook Pro M1 (32 GB RAM, Apple Neural Engine)  
- **Client:** React-Web-Frontend (Web-Audio API für Audio-Capture, Anzeige der Transkripte)  
- **Kommunikation:** Bidirektionale WebSocket-Verbindung zum Streamen von 20 ms PCM-Chunks und Transkriptions-Chunks  
- **Inferenz:** `facebook/wav2vec2-large-xlsr-53-german` über PyTorch auf CPU/Neural Engine (< 100 ms/Chunk)  
- **Batch-Training:** Python-Endpoint mit HuggingFace Transformers + PEFT (LoRA-Adapter + EWC)

#### Beschränkungen
- Keine NVIDIA-GPU verfügbar (nur CPU/Neural Engine)  
- Echtzeit-Anforderung: kleiner 100 ms Latenz pro 20 ms Audio (Realtime-Factor kleiner 5×)  
- Nur kostenlose, quelloffene Frameworks (HuggingFace, PEFT, PyTorch)  

#### Technischer Stack
| Ebene               | Technologie                                    |
|---------------------|------------------------------------------------|
| Modellbasis         | facebook/wav2vec2-large-xlsr-53-german         |
| Adapter-Feintuning  | LoRA (r=16, α=32, dropout=0.1)                 |
| Forgetting-Schutz   | Elastic Weight Consolidation (EWC)             |
| Streaming-Server    | Python + WebSocket (uvicorn, websockets)       |
| Client-Frontend     | React, Web-Audio API                           |
| Batch-Training      | HuggingFace Transformers + PEFT + PyTorch      |

#### Werkzeuge
- Python 3.10, venv benutzen
- PyTorch  
- HuggingFace Transformers & PEFT  
- FastAPI für Endpoints  
- React, TypeScript, Vite
- Web-Audio API  
