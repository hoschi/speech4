import React from 'react';

type TranscriptEditorProps = {
  transcript: string;
  setTranscript: (t: string) => void;
};

const TranscriptEditor: React.FC<TranscriptEditorProps> = ({ transcript, setTranscript }) => {
  return (
    <div>
      <label htmlFor="transcript">Transkript:</label>
      <input
        id="transcript"
        type="text"
        value={transcript}
        onChange={e => setTranscript(e.target.value)}
        style={{ width: '100%', fontSize: '1.2em', marginTop: 8 }}
      />
    </div>
  );
};

export default TranscriptEditor; 