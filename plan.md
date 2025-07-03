# PLANUNG.md

## 1. Zweck & Vision auf hoher Ebene

Entwicklung eines personalisierten, selbstlernenden Speech-to-Text-Systems, das in Echtzeit die deutsche Sprache (inklusive englischer Fachbegriffe) transkribiert. Das System soll aus Nutzerkorrekturen kontinuierlich lernen und eine höhere Genauigkeit als standardmäßige ASR-Dienste erreichen. Die gesamte Verarbeitung findet auf lokaler Hardware statt, um Datenschutz und Unabhängigkeit zu gewährleisten.

## 2. Architektur

-   **Server:** Ein Python-Backend auf einem MacBook Pro (M1, 32 GB RAM), das ohne dedizierte NVIDIA-GPU auskommt. Es nutzt CPU-optimierte Open-Source-Software.
-   **Client:** Ein schlankes React-Web-Frontend, dessen einzige Aufgaben die Audio-Erfassung, das Streaming an den Server und die Anzeige der zurückgelieferten Transkriptionen sind.
-   **Kommunikation:** Eine bidirektionale WebSocket-Verbindung für das Echtzeit-Streaming von Audio-Chunks (Client → Server) und Transkriptions-Ergebnissen (Server → Client).

## 3. Phasen der Implementierung

-   **Phase 1 (Abgeschlossen):** Aufbau einer stabilen Echtzeit-Streaming-Pipeline mit VOSK als ASR-Engine.
-   **Phase 2 (Abgeschlossen):** Implementierung des Korrektur-Loops. Das System kann Korrekturpaare (Audio + Text) speichern und daraus ein benutzerdefiniertes Vokabular für VOSK erstellen.
-   **Phase 3 (Nächste Schritte):** Verbesserung der Personalisierungslogik durch ein vollwertiges N-Gramm-Sprachmodell (KenLM) und optionales LLM-Post-Processing.

## 4. Technischer Stack

| Ebene | Technologie | Begründung |
| :--- | :--- | :--- |
| **Backend** | Python 3.10+, FastAPI, Uvicorn | Modern, performant, ideal für WebSockets und REST-APIs. |
| **ASR-Engine** | VOSK | Beste Wahl für kostenlose, CPU-basierte Echtzeit-Streaming-ASR auf macOS mit Vokabular-Unterstützung. |
| **Sprachmodell** | *Initial:* Benutzerdefiniertes Vokabular. *Final:* KenLM N-Gramm-Modell. | KenLM ist der logische nächste Schritt für eine tiefere Sprachkontext-Integration. |
| **Frontend** | React, TypeScript, Vite | Moderner, schneller und typsicherer Stack für Web-UIs. |
| **Kommunikation** | WebSocket API (browser-nativ) | Standard für latenzarme, bidirektionale Kommunikation. |
| **Audio-Erfassung**| Web Audio API (`MediaRecorder`) | Standard im Browser für den Zugriff auf das Mikrofon. |

## 5. Beschränkungen

-   **Hardware:** Kein Einsatz von NVIDIA-GPUs. Die Lösung muss auf einem Apple M1 Chip CPU-optimiert laufen.
-   **Software:** Es darf ausschließlich kostenlose Open-Source-Software ohne laufende Kosten verwendet werden.
-   **Performance:** Das System muss eine Echtzeit-Verarbeitung mit einer Latenz von < 300ms pro Audio-Chunk ermöglichen (realistisch für VOSK).

## 6. Werkzeuge

-   **Versionskontrolle:** Git
-   **Paketmanager:** `pip` für Python, `npm` für Node.js

---

**Aufforderung an die KI:**

Verwenden Sie die Struktur und die Entscheidungen, die in plan.md skizziert sind. Beziehen Sie sich zu Beginn jeder neuen Konversation auf dieses Dokument, um den Kontext und die Projektziele sicherzustellen."
