import React, { useState } from 'react';

const TrainButton: React.FC = () => {
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainResult, setTrainResult] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);

  const handleTrain = async () => {
    setTrainLoading(true);
    setTrainResult(null);
    setTrainError(null);
    try {
      const res = await fetch('http://localhost:8000/train/lm', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        setTrainResult(data.output);
      } else {
        setTrainError(data.output);
      }
    } catch {
      setTrainError('Fehler beim Training oder keine Verbindung zum Server.');
    } finally {
      setTrainLoading(false);
    }
  };

  return (
    <div style={{ margin: '2rem 0', display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
      <button
        onClick={handleTrain}
        disabled={trainLoading}
        style={{ minWidth: 180 }}
      >
        {trainLoading ? 'Training läuft...' : 'KenLM-Training starten'}
      </button>
      {trainResult && (
        <div style={{ color: 'green', marginTop: 12, whiteSpace: 'pre-wrap', maxWidth: 600 }}>
          <b>Training erfolgreich:</b>
          <br />{trainResult}
        </div>
      )}
      {trainError && (
        <div style={{ color: 'red', marginTop: 12, whiteSpace: 'pre-wrap', maxWidth: 600 }}>
          <b>Fehler beim Training:</b>
          <br />{trainError}
        </div>
      )}
    </div>
  );
};

export default TrainButton; 