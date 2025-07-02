import React, { useRef, useState } from 'react';
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

const MessageSchema = z.discriminatedUnion('type', [HypothesisChunk, FinalChunk, ErrorChunk]);

type HypothesisChunk = z.infer<typeof HypothesisChunk>;
type FinalChunk = z.infer<typeof FinalChunk>;
type ErrorChunk = z.infer<typeof ErrorChunk>;
export type TranscriptMessage = HypothesisChunk | FinalChunk | ErrorChunk;

type AudioRecorderProps = {
  onTranscriptChunk: (chunk: TranscriptMessage) => void;
};

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onTranscriptChunk }) => {
  const wsRef = useRef<WebSocket | null>(null);
  const [recording, setRecording] = useState(false);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const BUFFER_SIZE = 16000; // 1 Sekunde @ 16kHz
  const sampleBufferRef = useRef<Int16Array>(new Int16Array(0));

  const startRecording = async () => {
    if (recording) return;
    setWsStatus('connecting');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;
    audioContextRef.current = new window.AudioContext({ sampleRate: 16000 });
    const source = audioContextRef.current.createMediaStreamSource(stream);
    const processor = audioContextRef.current.createScriptProcessor(1024, 1, 1); // 64ms @ 16kHz = 1024 samples
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      // PCM 16bit Little Endian
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        pcm[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
      }
      // Buffering: sammle PCM bis mindestens 5120 Samples erreicht sind
      const prev = sampleBufferRef.current;
      const combined = new Int16Array(prev.length + pcm.length);
      combined.set(prev, 0);
      combined.set(pcm, prev.length);
      let offset = 0;
      while (combined.length - offset >= BUFFER_SIZE) {
        const chunk = combined.slice(offset, offset + BUFFER_SIZE);
        wsRef.current?.send(chunk.buffer);
        offset += BUFFER_SIZE;
      }
      // Rest im Buffer behalten
      sampleBufferRef.current = combined.slice(offset);
    };
    source.connect(processor);
    processor.connect(audioContextRef.current.destination);
    processorRef.current = processor;
    // WebSocket verbinden
    wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
    wsRef.current.onopen = () => setWsStatus('connected');
    wsRef.current.onclose = () => {
      setWsStatus('disconnected');
      stopRecording();
    };
    wsRef.current.onerror = () => {
      setWsStatus('error');
      stopRecording();
    };
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const parsed = MessageSchema.safeParse(data);
        if (parsed.success) {
          onTranscriptChunk(parsed.data);
        }
      } catch {
        // Fallback: ignoriere untypisierte Nachrichten
      }
    };
    setRecording(true);
  };

  const stopRecording = () => {
    setRecording(false);
    processorRef.current?.disconnect();
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    wsRef.current?.close();
    sampleBufferRef.current = new Int16Array(0); // Buffer leeren
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem', margin: '1.5rem 0' }}>
      <button
        onClick={recording ? stopRecording : startRecording}
        className={`button-main${recording ? ' stop' : ''}`}
      >
        {recording ? 'Stop Recording' : 'Start Recording'}
      </button>
      {wsStatus === 'error' && (
        <span className="status-error">Verbindungsfehler zum Server</span>
      )}
    </div>
  );
};

export default AudioRecorder; 