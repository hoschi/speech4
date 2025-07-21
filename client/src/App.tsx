import './index.css'
import { useRef, useState } from 'react'

function App() {
  const [transcript, setTranscript] = useState('')
  const [correction, setCorrection] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const transcriptRef = useRef<HTMLTextAreaElement>(null)
  const correctionRef = useRef<HTMLTextAreaElement>(null)

  // Aufnahme-Logik (Dummy, bitte ggf. mit echter Logik ersetzen)
  const handleRecord = () => {
    setIsRecording(r => !r)
    // Hier ggf. echte Aufnahme-Logik einbauen
    if (!isRecording) {
      setTranscript('')
      setCorrection('')
    }
  }

  // Ollama-Korrektur
  const handleCorrection = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/ollama-correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: transcript })
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setCorrection(data.correction || '')
    } catch (e: any) {
      setError(e.message || 'Fehler bei der Korrektur')
    } finally {
      setLoading(false)
    }
  }

  // In Zwischenablage kopieren
  const handleCopy = async () => {
    if (correctionRef.current) {
      await navigator.clipboard.writeText(correctionRef.current.value)
    }
  }

  return (
    <div style={{height:'100vh',width:'100vw',display:'flex',flexDirection:'column',justifyContent:'center',alignItems:'stretch',padding:0,margin:0,background:'#fff',boxSizing:'border-box'}}>
      <textarea
        ref={transcriptRef}
        value={transcript}
        onChange={e => setTranscript(e.target.value)}
        placeholder="Transkript hier eingeben oder aufnehmen..."
        style={{flex:'0 0 30%',fontSize:'1.2em',padding:'1em',border:'1px solid #ccc',borderRadius:'0.7em',margin:'1em',resize:'none',minHeight:80,maxHeight:200}}
        disabled={isRecording}
      />
      <div style={{display:'flex',flexDirection:'row',gap:'1em',justifyContent:'center',alignItems:'center',margin:'0 1em'}}>
        <button
          onClick={handleRecord}
          className={isRecording ? 'button-main stop' : 'button-main'}
          style={{flex:1,minWidth:0}}
        >
          {isRecording ? 'Stop' : 'Aufnahme'}
        </button>
        <button
          onClick={handleCorrection}
          className='button-main'
          style={{flex:1,minWidth:0}}
          disabled={loading || !transcript.trim()}
        >
          {loading ? 'Korrigiere...' : 'Korrektur'}
        </button>
        <button
          onClick={handleCopy}
          className='button-main'
          style={{flex:1,minWidth:0}}
          disabled={!correction.trim()}
        >
          Kopieren
        </button>
      </div>
      <textarea
        ref={correctionRef}
        value={correction}
        onChange={e => setCorrection(e.target.value)}
        placeholder="Korrigierter Text..."
        style={{flex:'1 1 40%',fontSize:'1.2em',padding:'1em',border:'1px solid #ccc',borderRadius:'0.7em',margin:'1em',resize:'vertical',minHeight:100}}
      />
      {error && <div style={{color:'red',textAlign:'center',marginBottom:'1em'}}>{error}</div>}
    </div>
  )
}

export default App
