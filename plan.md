### Plan

#### Zweck
Dieses Dokument beschreibt die übergeordnete Vision, die Systemarchitektur, technische Beschränkungen, den Stack und die eingesetzten Werkzeuge für das selbstlernende, personalisierte Transkriptionssystem.

#### Vision auf hoher Ebene
Ein leichtgewichtiger, personalisierter Speech-to-Text-Service, der mit weniger als 5 Minuten Nutzerdaten startklar ist, kontinuierlich aus Korrekturen lernt und Echtzeit-Transkription mit Code-Switching zwischen Deutsch und englischen Fachbegriffen ermöglicht.

#### Architektur
- **Server:** MacBook Pro M1 (32 GB RAM, Apple Neural Engine)  
- **Client:** React-Web-Frontend (Web-Audio API für Audio-Capture, Anzeige der Transkripte)  
- **Kommunikation:** Bidirektionale WebSocket-Verbindung zum Streamen von 20 ms PCM-Chunks und Transkriptions-Chunks  
- **Inferenz (Batch \& Streaming):**
    - Basis-Akustikmodell: `wav2vec2-xls-r-1B-german` über PyTorch auf CPU/Neural Engine (kleiner 100 ms/Chunk)
    - Streaming-optimiertes Modell: `wav2vec-S` für niedrige Latenz und konsistente Echtzeit-Qualität
- **Batch-Training:** Python-Endpoint mit HuggingFace Transformers + PEFT (LoRA-Adapter + EWC)
- **Sprachmodell-Optimierung:** KenLM mit aggressivem Pruning und Quantisierung für 32GB RAM-Kompatibilität

#### Beschränkungen
- Keine NVIDIA-GPU verfügbar (nur CPU/Neural Engine)  
- Echtzeit-Anforderung: kleiner 100 ms Latenz pro 20 ms Audio (Realtime-Factor kleiner 5×)  
- Nur kostenlose, quelloffene Frameworks (HuggingFace, PEFT, PyTorch)  
- **KenLM-Modellgröße:** Optimiert für 32GB RAM durch Pruning und Quantisierung

#### Technischer Stack
| Ebene               | Technologie                                    |
|---------------------|------------------------------------------------|
| Modellbasis         | wav2vec2-xls-r-1B-german                       |
| Adapter-Feintuning  | LoRA (r=16, α=32, dropout=0.1)                 |
| Forgetting-Schutz   | Elastic Weight Consolidation (EWC)             |
| Streaming-Server    | Python + WebSocket (uvicorn, websockets)       |
| Client-Frontend     | React, Web-Audio API                           |
| Batch-Training      | HuggingFace Transformers + PEFT + PyTorch      |
| **Sprachmodell**    | **KenLM mit aggressivem Pruning + 8-bit Quantisierung** |

#### KenLM-Optimierungsstrategie
- **Aggressives Pruning:** 60-80% Größenreduktion durch Entfernung seltener n-Gramme
- **Binärformat + Quantisierung:** Zusätzliche 50-70% Reduktion durch 8-bit Quantisierung
- **Memory Mapping:** Lazy Loading für Modelle größer als verfügbarer RAM
- **Optimierte Parameter:** `-o 4 --prune 0 1 1 1 -S 80%` für maximale Effizienz

#### Werkzeuge
- Python 3.10, venv benutzen
- PyTorch  
- HuggingFace Transformers & PEFT  
- FastAPI für Endpoints  
- React, TypeScript, Vite
- Web-Audio API  
- **KenLM-Tools:** lmplz, build_binary mit optimierten Parametern
