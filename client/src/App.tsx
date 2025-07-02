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
  const [isRecording, setIsRecording] = useState(false)
  const audioRecorderRef = useRef<{ cleanup: () => void } | null>(null)

  // Diese Funktion wird an AudioRecorder übergeben, um Recording-Status zu setzen
  const handleRecordingChange = (rec: boolean) => {
    setIsRecording(rec)
    if (rec) {
      setUserTranscript('') // Editor leeren, wenn neue Aufnahme startet
      setHypotheses([])
    }
  }

  const handleTranscriptChunk = (msg: TranscriptMessage) => {
    console.log(msg)
    if (msg.type === 'hypothesis') {
      if (msg.text.trim() !== '') {
        setHypotheses(prev => [...prev, msg])
      }
    } else if (msg.type === 'final') {
      setHypotheses([])
      if (msg.text.trim() !== '') {
        setUserTranscript(msg.text)
      }
      // Ressourcen abbauen und Aufnahme-Status zurücksetzen
      audioRecorderRef.current?.cleanup();
      setIsRecording(false);
    } else if (msg.type === 'error') {
      setError(msg.message)
    }
  }

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
      <AudioRecorder ref={audioRecorderRef} onTranscriptChunk={handleTranscriptChunk} onRecordingChange={handleRecordingChange} />
      {error && <div style={{ color: 'red', margin: '1rem 0' }}>Fehler: {error}</div>}
      <TranscriptEditor
        transcript={isRecording ? liveTranscript : userTranscript}
        setTranscript={isRecording ? () => {} : setUserTranscript}
        disabled={isRecording}
      />
      <TrainButton />
    </div>
  )
}

export default App
