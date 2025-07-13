import React, { useRef, useState, useImperativeHandle, forwardRef } from 'react';
import { z } from 'zod';

const HypothesisChunk = z.object({
  type: z.literal('hypothesis'),
  start: z.number(),
  end: z.number(),
  text: z.string(),
});
const FinalChunk = z.object({
  type: z.literal('final'),
  text: z.string(),
});
const ErrorChunk = z.object({
  type: z.literal('error'),
  message: z.string(),
});

type HypothesisChunk = z.infer<typeof HypothesisChunk>;
type FinalChunk = z.infer<typeof FinalChunk>;
type ErrorChunk = z.infer<typeof ErrorChunk>;
export type TranscriptMessage = HypothesisChunk | FinalChunk | ErrorChunk;

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
      wsRef.current?.close();
      wsRef.current = null;
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
    wsRef.current.onopen = () => setWsStatus('connected');
    wsRef.current.onclose = () => setWsStatus('disconnected');
    wsRef.current.onerror = () => setWsStatus('error');
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // VOSK gibt 'partial' oder 'text' zurück
        const isFinal = 'text' in data;
        const text = isFinal ? data.text : data.partial;

        if (text && text.trim()) {
            if (isFinal) {
                onTranscriptChunk({ type: 'final', text });
                // Stoppt die Aufnahme serverseitig ausgelöst
                stopRecording(true);
            } else {
                onTranscriptChunk({
                    type: 'hypothesis',
                    start: 0, // VOSK liefert keine verlässlichen Timestamps per default
                    end: 0,
                    text: text,
                });
            }
        }

      } catch (e) {
        console.error("Fehler beim Parsen der WebSocket-Nachricht:", e);
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
        wsRef.current.send(JSON.stringify({ 'eof': 1 }));
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
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem', margin: '1.5rem 0' }}>
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