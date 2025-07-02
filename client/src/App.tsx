import './index.css'
import AudioRecorder from './components/AudioRecorder'
import type { TranscriptMessage } from './components/AudioRecorder'
import TranscriptEditor from './components/TranscriptEditor'
import TrainButton from './components/TrainButton'
import { useState } from 'react'

function App() {
  const [hypotheses, setHypotheses] = useState<TranscriptMessage[]>([])
  const [final, setFinal] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleTranscriptChunk = (msg: TranscriptMessage) => {
    if (msg.type === 'hypothesis') {
      setHypotheses(prev => {
        const filtered = prev.filter(h => h.type !== 'hypothesis' || h.end <= msg.start)
        return [...filtered, msg]
      })
    } else if (msg.type === 'final') {
      setFinal(msg.text)
      setHypotheses([])
    } else if (msg.type === 'error') {
      setError(msg.message)
    }
  }

  const transcript = final !== null
    ? final
    : hypotheses
      .filter((h): h is { type: 'hypothesis', start: number, end: number, text: string } => h.type === 'hypothesis')
      .sort((a, b) => a.start - b.start)
      .map(h => h.text)
      .join(' ')
      .replace(/ +/g, ' ')
      .trim()

  return (
    <div className="app-card">
      <h1>Speech-to-Text Streaming Demo</h1>
      <AudioRecorder onTranscriptChunk={handleTranscriptChunk} />
      {error && <div style={{ color: 'red', margin: '1rem 0' }}>Fehler: {error}</div>}
      <TranscriptEditor transcript={transcript} setTranscript={() => {}} />
      <TrainButton />
    </div>
  )
}

export default App
