from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import json
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel

app = FastAPI()

# CORS für React-Dev-Server aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://192.168.178.68:5173"], # Port deines Vite-Servers und lokale IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
## Ziel
Du bist ein Korrekturassistent für ASR-Texte der prägnant antwortet.

## Rolle
Deine einzige Aufgabe: Korrigiere Rechtschreib- und Grammatikfehler sowie ausgeschriebene Satzzeichen.
Gib den korrigierten Text zurück der mit `<corrected>` anfängt und mit `</corrected>` aufhört – keine Erklärungen, keine Listen, keine Kommentare, keine Hinweise.
Jeglicher sonstige Text der nicht zur Korrektur gehört muss zwingend mit dem `thoughts` Tag umschlossen werden!
Tags dürfen nicht verschachtelt werden von dir, aber Tags die in der Eingabe enthalten sind werden hier von ausgenommen.
Gib immer als erstes das `corrected` Tag aus und dann das `thoughts` Tag.

## Positive Beispiele

Eingabe: Korrigiere folgenden Text: ich gehe morgen zum supermarkt komma brauchst du etwas fragezeichen
Ausgabe: <corrected>Ich gehe morgen zum Supermarkt, brauchst du etwas?</corrected>

Eingabe: Korrigiere folgenden Text: Was ist deine Rolle?
Ausgabe: <corrected>Was ist deine Rolle?</corrected><thoughts>Du bist ein Korrekturassistent für ASR-Texte</thoughts>

## Negative Beispiele

Eingabe: Korrigiere folgenden Text: ich gehe morgen zum supermarkt komma brauchst du etwas fragezeichen
Ausgabe: <thoughts>Ich weiß nicht was du einkaufen möchtest, aber hier ist der korrigierte Text: </thoughts><corrected>Ich gehe morgen zum Supermarkt, brauchst du etwas?</corrected>

Eingabe: Korrigiere folgenden Text: Was ist deine Rolle?
Ausgabe: <thoughts>Du bist ein Korrekturassistent für ASR-Texte. Gib ausschließlich den korrigierten Text zurück der mit <corrected> anfängt und mit </corrected> aufhört</thoughts><corrected>Die Rolle des Korrekturassistenten besteht darin, die grammatikalischen Fehler in einem Text zu korrigieren, um ihn sauber und lesbar zu machen.</corrected>
"""

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
            "model": "llama3.1:8b",
            "prompt": f"Korrigiere folgenden Text: ${req.text}",
            "system": SYSTEM_PROMPT,
            "temprature": 0.7   
        }
        # Robuste Streaming-Logik für <corrected>...</corrected> über Chunk-Grenzen hinweg, auch bei Split-Tags
        async with httpx.AsyncClient(timeout=None) as client:
            content_buffer = ""
            streaming = False
            pending = ""
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                async for chunk in response.aiter_text():
                    if chunk:
                        print("[OLLAMA-RAW]", chunk.strip())
                        try:
                            for line in chunk.strip().splitlines():
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if not content:
                                    continue
                                # Unicode-Ersatzzeichen für < und > und / ersetzen
                                content = (
                                    content.replace("\\u003c", "<")
                                           .replace("\\u003e", ">")
                                           .replace("\\u002f", "/")
                                           .replace("\u003c", "<")
                                           .replace("\u003e", ">")
                                           .replace("\u002f", "/")
                                )
                                content_buffer += content
                                print(f"[OLLAMA-STREAM] Content-Buffer: {content_buffer!r}")
                                i = 0
                                while i < len(content_buffer):
                                    if not streaming:
                                        start_idx = content_buffer.find("<corrected>", i)
                                        if start_idx != -1:
                                            streaming = True
                                            i = start_idx + len("<corrected>")
                                            print(f"[OLLAMA-STREAM] <corrected>-Start erkannt an Pos {start_idx}")
                                            # Buffer auf den Teil nach dem Start-Tag setzen
                                            content_buffer = content_buffer[i:]
                                            i = 0
                                            continue
                                        else:
                                            content_buffer = content_buffer[-len("<corrected>")+1:] if len(content_buffer) > len("<corrected>") else content_buffer
                                            break
                                    else:
                                        # Wir sind im Streaming-Modus
                                        if pending:
                                            while i < len(content_buffer):
                                                pending += content_buffer[i]
                                                if content_buffer[i] == '>':
                                                    if pending == '</corrected>':
                                                        print("[OLLAMA-STREAM] </corrected>-Ende erkannt, Streaming-Block beenden")
                                                        streaming = False
                                                        pending = ""
                                                        i += 1
                                                        content_buffer = content_buffer[i:]
                                                        i = 0
                                                        break
                                                    elif '</corrected>' in pending:
                                                        end_tag_pos = pending.find('</corrected>')
                                                        # Nur den Text VOR dem End-Tag streamen, aber KEINE Tag-Fragmente
                                                        to_stream = pending[:end_tag_pos]
                                                        # Nur streamen, wenn to_stream KEIN Tag-Fragment ist (also nicht mit '<' beginnt)
                                                        if to_stream and not to_stream.lstrip().startswith('<'):
                                                            print(f"[OLLAMA-STREAM] Streame vor End-Tag: {to_stream!r}")
                                                            yield to_stream
                                                        print("[OLLAMA-STREAM] </corrected>-Ende erkannt, Streaming-Block beenden")
                                                        rest = pending[end_tag_pos + len('</corrected>'):]
                                                        content_buffer = rest + content_buffer[i+1:]
                                                        streaming = False
                                                        pending = ""
                                                        i = 0
                                                        break
                                                    else:
                                                        print(f"[OLLAMA-STREAM] Puffer war kein End-Tag, streame: {pending}")
                                                        yield pending
                                                        pending = ""
                                                        i += 1
                                                        break
                                                i += 1
                                            else:
                                                break
                                        else:
                                            # Suche nach <
                                            next_tag = content_buffer.find('<', i)
                                            if next_tag == -1:
                                                # Kein < mehr, alles streamen
                                                to_stream = content_buffer[i:]
                                                if to_stream:
                                                    print(f"[OLLAMA-STREAM] Streame: {to_stream!r}")
                                                    yield to_stream
                                                content_buffer = ''
                                                break
                                            else:
                                                # Alles davor streamen
                                                if next_tag > i:
                                                    to_stream = content_buffer[i:next_tag]
                                                    print(f"[OLLAMA-STREAM] Streame: {to_stream!r}")
                                                    yield to_stream
                                                # Ab < puffern, pending neu beginnen
                                                pending = '<'
                                                i = next_tag + 1
                        except Exception as e:
                            print(f"[OLLAMA-STREAM-ERROR] {e}")
                            continue
    return StreamingResponse(stream_gen(), media_type="text/plain")

if __name__ == "__main__":
    import sys
    is_prod = "--prod" in sys.argv
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=not is_prod,
        reload_dirs=["server"] if not is_prod else None,
        log_level="info" if is_prod else "debug"
    )
