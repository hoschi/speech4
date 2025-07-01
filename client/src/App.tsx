import './index.css'
import AudioRecorder from './components/AudioRecorder'
import TranscriptEditor from './components/TranscriptEditor'
import { useState } from 'react'

function App() {
  const [transcript, setTranscript] = useState('')

  return (
    <div className="app-card">
      <h1>Speech-to-Text Streaming Demo</h1>
      <AudioRecorder onTranscriptChunk={chunk => setTranscript(prev => prev + chunk)} />
      <TranscriptEditor transcript={transcript} setTranscript={setTranscript} />
    </div>
  )
}

export default App
