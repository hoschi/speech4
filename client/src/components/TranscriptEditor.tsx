import React, { useRef } from 'react';

type TranscriptEditorProps = {
  transcript: string;
  setTranscript: (t: string) => void;
};

const TranscriptEditor: React.FC<TranscriptEditorProps> = ({ transcript, setTranscript }) => {
  const inputRef = useRef<HTMLInputElement>(null);

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
      />
    </div>
  );
};

export default TranscriptEditor; 