import React, { useRef, useState } from 'react';

type TranscriptEditorProps = {
  transcript: string;
  setTranscript: (t: string) => void;
};

const TranscriptEditor: React.FC<TranscriptEditorProps> = ({ transcript, setTranscript }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState(false);

  const handleUpload = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      const formData = new FormData();
      formData.append('text', transcript);
      const res = await fetch('/upload/correction', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(await res.text());
      setSuccess(true);
      setUploaded(true);
      setTranscript('');
    } catch (e: unknown) {
      if (e instanceof Error) {
        setError(e.message || 'Fehler beim Upload');
      } else {
        setError('Fehler beim Upload');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ margin: '2rem 0', display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
      <label htmlFor="transcript" className="transcript-label">Transkript:</label>
      <input
        id="transcript"
        type="text"
        value={transcript}
        onChange={e => setTranscript(e.target.value)}
        className="transcript-input"
        ref={inputRef}
        disabled={uploaded}
      />
      <button
        onClick={handleUpload}
        disabled={loading || !transcript.trim() || uploaded}
        style={{ marginTop: '1rem', minWidth: 120 }}
      >
        {loading ? 'Hochladen...' : uploaded ? 'Hochgeladen' : 'Korrektur hochladen'}
      </button>
      {success && <div style={{ color: 'green', marginTop: 8 }}>Erfolgreich hochgeladen!</div>}
      {error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
    </div>
  );
};

export default TranscriptEditor; 