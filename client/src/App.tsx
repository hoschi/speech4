import './index.css'
import AudioRecorder from './components/AudioRecorder'
import TranscriptEditor from './components/TranscriptEditor'
import TrainButton from './components/TrainButton'
import { useState } from 'react'

function App() {
  const [transcript, setTranscript] = useState('')

  return (
    <div className="app-card">
      <h1>Speech-to-Text Streaming Demo</h1>
      <AudioRecorder onTranscriptChunk={chunk => setTranscript(prev => prev + chunk)} />
      <TranscriptEditor transcript={transcript} setTranscript={setTranscript} />
      <TrainButton />
    </div>
  )
}

export default App
