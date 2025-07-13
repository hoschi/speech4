import './index.css'
import AudioRecorder from './components/AudioRecorder'
import type { TranscriptMessage } from './components/AudioRecorder'
import TranscriptEditor from './components/TranscriptEditor'
import TrainButton from './components/TrainButton'
import { useState, useRef } from 'react'

function App() {
  const [hypotheses, setHypotheses] = useState<TranscriptMessage[]>([])
  const [userTranscript, setUserTranscript] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isRecording, setIsRecording] = useState(false)
  const audioRecorderRef = useRef<{ cleanup: () => void } | null>(null)

  // Diese Funktion wird an AudioRecorder übergeben, um Recording-Status zu setzen
  const handleRecordingChange = (rec: boolean) => {
    setIsRecording(rec)
    if (rec) {
      setUserTranscript('') // Editor leeren, wenn neue Aufnahme startet
      setAudioBlob(null);
      setHypotheses([])
    }
  }

  const handleTranscriptChunk = (msg: TranscriptMessage) => {
    console.log(msg)
    if (msg.type === 'hypothesis') {
      if (msg.text.trim() !== '') {
        // Ersetze alle Hypothesen mit der neuesten
        setHypotheses([msg])
      }
    } else if (msg.type === 'final') {
      setHypotheses([])
      // Nur setzen, wenn das finale Ergebnis nicht leer ist
      if (msg.text && msg.text.trim() !== '') {
        setUserTranscript(msg.text)
      }
      // Ressourcen abbauen und Aufnahme-Status zurücksetzen
      audioRecorderRef.current?.cleanup();
      setIsRecording(false);
    } else if (msg.type === 'error') {
      setError(msg.message)
    }
  }

  const handleRecordingComplete = (blob: Blob) => {
    console.log(`Audio-Blob empfangen, Größe: ${blob.size} bytes`);
    setAudioBlob(blob);
  };

  // Laufendes Transkript aus Hypothesen zusammensetzen
  const liveTranscript = hypotheses
    .filter((h): h is { type: 'hypothesis', start: number, end: number, text: string } => h.type === 'hypothesis')
    .sort((a, b) => a.start - b.start)
    .map(h => h.text)
    .join(' ')
    .replace(/ +/g, ' ')
    .trim()
    //console.log(liveTranscript, hypotheses)

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
        transcript={isRecording ? liveTranscript : userTranscript}
        onTranscriptChange={isRecording ? () => {} : setUserTranscript}
        disabled={isRecording}
        audioBlob={audioBlob}
      />
      <TrainButton />
    </div>
  )
}

export default App
