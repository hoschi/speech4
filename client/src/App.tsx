import './index.css'
import AudioRecorder from './components/AudioRecorder'
import type { TranscriptMessage } from './components/AudioRecorder'
import TranscriptEditor from './components/TranscriptEditor'
import TrainButton from './components/TrainButton'
import { useState, useRef } from 'react'

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

  return (
    <div className="app-card">
      <h1>Speech-to-Text Streaming Demo</h1>
      <AudioRecorder
        ref={audioRecorderRef as React.RefObject<{ cleanup: () => void } | null>}
        onTranscriptChunk={handleTranscriptChunk}
        onRecordingChange={handleRecordingChange}
        onRecordingComplete={handleRecordingComplete}
      />
      {error && <div style={{ color: 'red', margin: '1rem 0' }}>Fehler: {error}</div>}
      <TranscriptEditor
        transcript={transcript}
        onTranscriptChange={isRecording ? () => {} : setTranscript}
        disabled={isRecording}
        audioBlob={audioBlob}
        alternatives={alternatives}
      />
      <TrainButton />
    </div>
  )
}

export default App
