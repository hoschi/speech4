import React, { useState } from 'react';

type TranscriptEditorProps = {
  transcript: string;
  onTranscriptChange: (t: string) => void;
  audioBlob: Blob | null;
  disabled?: boolean;
};

const TranscriptEditor: React.FC<TranscriptEditorProps> = ({ transcript, onTranscriptChange, audioBlob, disabled = false }) => {
  const [isDirty, setIsDirty] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState(false);

  const handleUpload = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    if (!audioBlob) {
        setError("Keine Audiodatei für den Upload vorhanden.");
        setLoading(false);
        return;
    }

    try {
      const formData = new FormData();
      formData.append('text', transcript);
      formData.append('audio', audioBlob, 'correction.wav');

      const res = await fetch('http://localhost:8000/upload/correction', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(await res.text());

      setSuccess(true);
      setUploaded(true);
      onTranscriptChange('');

    } catch (e: unknown) {
      if (e instanceof Error) {
        const msg = e.message || 'Fehler beim Upload';
        console.error(msg, e);
        setError(msg);
      } else {
        setError('Fehler beim Upload');
      }
    }
    setLoading(false);
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onTranscriptChange(e.target.value);
    if (!isDirty) setIsDirty(true);
  }

  return (
    <div style={{ margin: '2rem 0', display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
      <label htmlFor="transcript" className="transcript-label">Transkript:</label>
      <textarea
        id="transcript"
        value={transcript}
        rows={4}
        onChange={handleChange}
        className="transcript-input"
        disabled={uploaded || disabled}
        style={{ resize: 'vertical', minHeight: '4em', maxWidth: 420 }}
      />
      <button
        onClick={handleUpload}
        disabled={loading || !transcript.trim() || !isDirty || uploaded || disabled || !audioBlob}
        style={{ marginTop: '1rem', minWidth: 120 }}
      >
        {loading ? 'Hochladen...' : uploaded ? '✅ Hochgeladen' : 'Korrektur hochladen'}
      </button>
      {success && <div style={{ color: 'green', marginTop: 8 }}>Erfolgreich hochgeladen!</div>}
      {error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
    </div>
  );
};

export default TranscriptEditor; 