import './index.css'
import AudioRecorder from './components/AudioRecorder'
import type { TranscriptMessage } from './components/AudioRecorder'
import TranscriptEditor from './components/TranscriptEditor'
//import TrainButton from './components/TrainButton'
import { useState, useRef } from 'react'
// Für Streaming-API
type OllamaStreamState = {
  loading: boolean;
  error: string | null;
  output: string;
};

type Alternative = {
  text: string;
  confidence: number;
};

function App() {
  const [transcript, setTranscript] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isRecording, setIsRecording] = useState(false)
  const [alternatives, setAlternatives] = useState<Alternative[]>([])
  const [ollama, setOllama] = useState<OllamaStreamState>({ loading: false, error: null, output: '' });
  const audioRecorderRef = useRef<{ cleanup: () => void } | null>(null)

  // Diese Funktion wird an AudioRecorder übergeben, um Recording-Status zu setzen
  const handleRecordingChange = (rec: boolean) => {
    setIsRecording(rec)
    if (rec) {
      setTranscript('') // Editor leeren, wenn neue Aufnahme startet
      setAudioBlob(null);
      setAlternatives([]); // Alternativen zurücksetzen
    }
  }

  const handleTranscriptChunk = (msg: TranscriptMessage) => {
    console.log(msg)
    if (msg.type === 'hypothesis') {
      if (msg.text.trim() !== '') {
        setTranscript(msg.text)
      }
    } else if (msg.type === 'final') {
      // Nur setzen, wenn das finale Ergebnis nicht leer ist
      if (msg.text && msg.text.trim() !== '') {
        setTranscript(msg.text)
      }
      // Alternativen setzen, falls vorhanden
      if (msg.alternatives) {
        setAlternatives(msg.alternatives)
      }
      setIsRecording(false);
      // KEIN Cleanup mehr hier!
    } else if (msg.type === 'error') {
      setError(msg.message)
    }
  }

  const handleRecordingComplete = (blob: Blob) => {
    console.log(`Audio-Blob empfangen, Größe: ${blob.size} bytes`);
    setAudioBlob(blob);
  };

  // Ollama-Streaming-Handler
  const handleOllama = async () => {
    setOllama({ loading: true, error: null, output: '' });
    try {
      const response = await fetch('/ollama/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: transcript })
      });
      if (!response.body) throw new Error('Keine Streaming-Antwort vom Server');
      const reader = response.body.getReader();
      let result = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = new TextDecoder().decode(value);
        result += chunk;
        setOllama((prev) => ({ ...prev, output: result }));
      }
      setOllama((prev) => ({ ...prev, loading: false }));
    } catch (e: unknown) {
      let msg = 'Fehler beim Streamen';
      if (e && typeof e === 'object' && 'message' in e && typeof (e as { message?: string }).message === 'string') {
        msg = (e as { message?: string }).message as string;
      }
      setOllama({ loading: false, error: msg, output: '' });
    }
  };

  return (
    <div className="app-card">
      {error && <div style={{ color: 'red', margin: '1rem 0' }}>Fehler: {error}</div>}
      {ollama.error && <div style={{ color: 'red', marginBottom: 8 }}>{ollama.error}</div>}
      <TranscriptEditor
        transcript={transcript}
        onTranscriptChange={isRecording ? () => {} : setTranscript}
        disabled={isRecording}
        audioBlob={audioBlob}
        alternatives={alternatives}
      />
      <div>
      <AudioRecorder
        ref={audioRecorderRef as React.RefObject<{ cleanup: () => void } | null>}
        onTranscriptChunk={handleTranscriptChunk}
        onRecordingChange={handleRecordingChange}
        onRecordingComplete={handleRecordingComplete}
      />
      <button
        onClick={handleOllama}
        disabled={isRecording || !transcript.trim() || ollama.loading}
        style={{ minWidth: 180, marginBottom: 8 }}
      >
        {ollama.loading ? 'Ollama denkt...' : 'Ollama-Korrektur (asr-fixer)'}
      </button>
      </div>
      <textarea
        value={ollama.output}
        rows={4}
        style={{ width: '100%', minHeight: '4em'}}
        placeholder="Ollama-Output erscheint hier..."
      />
    </div>
  )
}

export default App
