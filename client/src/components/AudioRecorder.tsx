import React, { useRef, useState } from 'react';

type AudioRecorderProps = {
  onTranscriptChunk: (chunk: string) => void;
};

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onTranscriptChunk }) => {
  const wsRef = useRef<WebSocket | null>(null);
  const [recording, setRecording] = useState(false);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const startRecording = async () => {
    if (recording) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;
    audioContextRef.current = new window.AudioContext({ sampleRate: 16000 });
    const source = audioContextRef.current.createMediaStreamSource(stream);
    const processor = audioContextRef.current.createScriptProcessor(256, 1, 1); // 16ms @ 16kHz = 256 samples
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      // PCM 16bit Little Endian
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        pcm[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
      }
      wsRef.current?.send(pcm.buffer);
    };
    source.connect(processor);
    processor.connect(audioContextRef.current.destination);
    processorRef.current = processor;
    // WebSocket verbinden
    wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
    wsRef.current.onmessage = (event) => {
      onTranscriptChunk(event.data);
    };
    setRecording(true);
  };

  const stopRecording = () => {
    setRecording(false);
    processorRef.current?.disconnect();
    audioContextRef.current?.close();
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    wsRef.current?.close();
  };

  return (
    <div>
      <button onClick={recording ? stopRecording : startRecording}>
        {recording ? 'Stop Recording' : 'Start Recording'}
      </button>
    </div>
  );
};

export default AudioRecorder; 