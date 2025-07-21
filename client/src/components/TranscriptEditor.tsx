import React, { useState } from 'react';

type Alternative = {
  text: string;
  confidence: number;
};

type TranscriptEditorProps = {
  transcript: string;
  onTranscriptChange: (t: string) => void;
  audioBlob: Blob | null;
  disabled?: boolean;
  alternatives?: Alternative[];
};

const TranscriptEditor: React.FC<TranscriptEditorProps> = ({ 
  transcript, 
  onTranscriptChange, 
  audioBlob, 
  disabled = false,
  alternatives = []
}) => {
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

  const handleAlternativeClick = (alternativeText: string) => {
    onTranscriptChange(alternativeText);
    setIsDirty(true);
  };

  return (
    <div>
      <textarea
        id="transcript"
        value={transcript}
        rows={4}
        onChange={handleChange}
        className="transcript-input"
        disabled={uploaded || disabled}
        style={{ resize: 'vertical', minHeight: '4em' }}
      />
      
      {/* Alternativen anzeigen */}
      {alternatives.length > 0 && (
        <div>
          <label style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0.5rem', display: 'block' }}>
            VOSK-Alternativen:
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {alternatives.map((alt, index) => (
              <button
                key={index}
                onClick={() => handleAlternativeClick(alt.text)}
                disabled={uploaded || disabled}
                style={{
                  padding: '0.5rem',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  background: 'white',
                  cursor: uploaded || disabled ? 'default' : 'pointer',
                  textAlign: 'left',
                  fontSize: '0.9rem',
                  opacity: uploaded || disabled ? 0.6 : 1,
                }}
                onMouseEnter={(e) => {
                  if (!uploaded && !disabled) {
                    e.currentTarget.style.background = '#f5f5f5';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!uploaded && !disabled) {
                    e.currentTarget.style.background = 'white';
                  }
                }}
              >
                <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>
                  {alt.text}
                </div>
                <div style={{ fontSize: '0.8rem', color: '#666' }}>
                  Konfidenz: {(alt.confidence * 100).toFixed(1)}%
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
      
      <button
        onClick={handleUpload}
        disabled={loading || !transcript.trim() || !isDirty || uploaded || disabled || !audioBlob}
        style={{ marginTop: '1rem', minWidth: 120, display:'none' }}
      >
        {loading ? 'Hochladen...' : uploaded ? '✅ Hochgeladen' : 'Korrektur hochladen'}
      </button>
      {success && <div style={{ color: 'green', marginTop: 8 }}>Erfolgreich hochgeladen!</div>}
      {error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
    </div>
  );
};

export default TranscriptEditor; 