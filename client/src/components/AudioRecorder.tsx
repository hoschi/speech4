import { useRef, useState, useImperativeHandle, forwardRef } from 'react';
import { z } from 'zod';

// Zod-Schemas für VOSK-Formate
const VoskPartial = z.object({ partial: z.string() });
const VoskFinal = z.object({
  text: z.string().optional(), // jetzt optional!
  alternatives: z.array(z.object({
    text: z.string(),
    confidence: z.number(),
  })).optional(),
  type: z.string().optional(),
});
const ErrorChunk = z.object({ type: z.literal('error'), message: z.string() });

type HypothesisChunk = { type: 'hypothesis', start: number, end: number, text: string };
type FinalChunk = { type: 'final', text: string, alternatives?: { text: string, confidence: number }[] };
type ErrorChunkType = { type: 'error', message: string };
export type TranscriptMessage = HypothesisChunk | FinalChunk | ErrorChunkType;

type AudioRecorderProps = {
  onTranscriptChunk: (chunk: TranscriptMessage) => void;
  onRecordingChange?: (rec: boolean) => void;
  onRecordingComplete?: (blob: Blob) => void;
  onFinal?: () => void;
};

const AudioRecorder = forwardRef((props: AudioRecorderProps, ref) => {
  const { onTranscriptChunk, onRecordingChange, onRecordingComplete } = props;
  const wsRef = useRef<WebSocket | null>(null);
  const [recording, setRecording] = useState(false);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Cleanup-Methode für App
  useImperativeHandle(ref, () => ({
    cleanup: () => {
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(console.error);
        audioContextRef.current = null;
      }
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
      if (wsRef.current) {
        if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
          console.log('[WebSocket] Cleanup: Schließe WebSocket (cleanup)');
          wsRef.current.close();
        } else {
          console.log('[WebSocket] Cleanup: WebSocket bereits geschlossen');
        }
        wsRef.current = null;
      }
      audioChunksRef.current = [];
    }
  }));

  const startRecording = async () => {
    if (recording) return;
    audioChunksRef.current = []; // Reset chunks
    setWsStatus('connecting');

    // WebSocket verbinden
    wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
    wsRef.current.binaryType = 'arraybuffer';
    wsRef.current.onopen = () => {
      setWsStatus('connected');
      console.log('[WebSocket] Geöffnet');
    };
    wsRef.current.onclose = () => {
      setWsStatus('disconnected');
      console.log('[WebSocket] Geschlossen');
      // Jetzt erst cleanup!
      wsRef.current = null;
      // ggf. weitere Aufräumarbeiten
    };
    wsRef.current.onerror = (e) => {
      setWsStatus('error');
      console.log('[WebSocket] Fehler', e);
    };
    wsRef.current.onmessage = (event) => {
      try {
        const raw = JSON.parse(event.data);
        let parsed: TranscriptMessage;
        if (VoskPartial.safeParse(raw).success) {
          parsed = {
            type: 'hypothesis',
            start: 0,
            end: 0,
            text: raw.partial,
          };
        } else if (VoskFinal.safeParse(raw).success) {
          // Falls text fehlt, nimm den besten aus alternatives (typisch: erster ist der beste)
          let text = raw.text;
          if ((!text || text.trim() === '') && Array.isArray(raw.alternatives) && raw.alternatives.length > 0) {
            text = raw.alternatives[0].text;
          }
          parsed = {
            type: 'final',
            text: text || '',
            alternatives: raw.alternatives,
          };
        } else if (ErrorChunk.safeParse(raw).success) {
          parsed = raw;
        } else {
          console.warn('Unbekanntes Nachrichtenformat:', raw);
          return;
        }
        onTranscriptChunk(parsed);
        if (parsed.type === 'final') {
          stopRecording(true);
        }
      } catch (e) {
        console.error('Fehler beim Parsen der WebSocket-Nachricht:', e);
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });
    mediaStreamRef.current = stream;

    // AudioContext für PCM-Konvertierung erstellen
    audioContextRef.current = new AudioContext({ sampleRate: 16000 });
    const source = audioContextRef.current.createMediaStreamSource(stream);
    
    // ScriptProcessor für PCM-Konvertierung
    processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);
    
    processorRef.current.onaudioprocess = (event) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const inputData = event.inputBuffer.getChannelData(0);
        // Konvertiere Float32 zu Int16 PCM
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
        }
        wsRef.current.send(pcmData.buffer);
      }
    };

    source.connect(processorRef.current);
    processorRef.current.connect(audioContextRef.current.destination);

    // MediaRecorder für Blob-Speicherung (nicht für WebSocket)
    mediaRecorderRef.current = new MediaRecorder(stream);
    mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
    };

    mediaRecorderRef.current.start(250);

    setRecording(true);
    if (typeof onRecordingChange === 'function') onRecordingChange(true);
  };

  const stopRecording = (fromServer: boolean = false) => {
    console.log("stopRecording")
    if (!recording && !fromServer) return;

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.stop();
    }

    // AudioContext-Ressourcen bereinigen
    if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Sage dem Server, dass die Aufnahme beendet ist, damit er das Endergebnis schickt
        console.log('[WebSocket] Sende EOF an Server, readyState:', wsRef.current.readyState);
        wsRef.current.send(JSON.stringify({ 'eof': 1 }));
    } else if (wsRef.current) {
        console.log('[WebSocket] Konnte EOF nicht senden, readyState:', wsRef.current.readyState);
    }

    // Erstelle den finalen Audio-Blob
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
    if (onRecordingComplete) {
        onRecordingComplete(audioBlob);
    }

    // Ressourcen freigeben
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());

    setRecording(false);
    if (typeof onRecordingChange === 'function' && !fromServer) {
        onRecordingChange(false);
    }

    // WebSocket wird nach finaler Nachricht vom Server geschlossen
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
      <button
        onClick={recording ? () => stopRecording(false) : startRecording}
        className={`button-main${recording ? ' stop' : ''}`}
      >
        {recording ? 'Stop Recording' : 'Start Recording'}
      </button>
      {wsStatus === 'error' && (
        <span className="status-error">Verbindungsfehler zum Server</span>
      )}
    </div>
  );
});

export default AudioRecorder; 