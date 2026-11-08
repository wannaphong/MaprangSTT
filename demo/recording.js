// Microphone recording & audio file decoding for MaprangSTT web demo.
// Adapted from afshinm/cactus-needle demo/recording.ts (Apache-2.0).

export const SAMPLE_RATE = 16_000;
export const MAX_DURATION_SECONDS = 30;

/**
 * Decode any browser-supported audio buffer (WAV, MP3, WebM, Ogg, M4A)
 * into mono 16 kHz Float32Array samples in [-1, 1], capped at 30 seconds.
 */
export async function decodeAudioBuffer(
  encodedBuffer,
  signal,
) {
  signal?.throwIfAborted();
  // Create a temporary AudioContext to decode the compressed audio at its native rate first,
  // then resample via OfflineAudioContext at 16 kHz mono for maximum browser compatibility.
  const tempCtx = new AudioContext();
  let nativeDecoded;
  try {
    nativeDecoded = await tempCtx.decodeAudioData(encodedBuffer.slice(0));
  } finally {
    void tempCtx.close().catch(() => {});
  }
  signal?.throwIfAborted();
  const targetFrames = Math.min(
    Math.max(1, Math.ceil(nativeDecoded.duration * SAMPLE_RATE)),
    MAX_DURATION_SECONDS * SAMPLE_RATE,
  );
  const offline = new OfflineAudioContext(1, targetFrames, SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = nativeDecoded;
  source.connect(offline.destination);
  source.start(0);
  const rendered = await offline.startRendering();
  signal?.throwIfAborted();
  const data = rendered.getChannelData(0);
  const samples = new Float32Array(Math.min(data.length, MAX_DURATION_SECONDS * SAMPLE_RATE));
  for (let i = 0; i < samples.length; i++) {
    const v = data[i] ?? 0;
    samples[i] = Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0;
  }
  return {
    samples,
    durationInSeconds: samples.length / SAMPLE_RATE,
  };
}

/**
 * Encode mono 16 kHz Float32Array samples into a 16-bit PCM WAV Blob
 * so the user can play back or download their recorded/decoded audio.
 */
export function encodeWavBlob(samples, sampleRate = SAMPLE_RATE) {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true);  // PCM format = 1
  view.setUint16(22, 1, true);  // Mono = 1 channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);  // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i] ?? 0));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Start capturing microphone audio with live level metering and optional auto-stop on silence.
 */
export async function startRecording(signal, options = {}) {
  const autoStopOnSilence = options.autoStopOnSilence ?? false;
  signal.throwIfAborted();
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
    throw new Error('Voice input needs a browser with microphone recording support.');
  }
  const context = new AudioContext();
  void context.resume().catch(() => {});
  const requested = navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const release = (stream) => {
    for (const track of stream.getTracks()) track.stop();
  };
  void requested.then(
    (stream) => {
      if (signal.aborted) release(stream);
    },
    () => {},
  );
  let abortPermission;
  const cancelled = new Promise((_resolve, reject) => {
    abortPermission = () => reject(signal.reason);
    signal.addEventListener('abort', abortPermission, { once: true });
  });
  let stream;
  try {
    stream = await Promise.race([requested, cancelled]);
  } catch (error) {
    void context.close();
    throw error;
  } finally {
    signal.removeEventListener('abort', abortPermission);
  }
  if (signal.aborted) {
    release(stream);
    void context.close();
    signal.throwIfAborted();
  }
  let recorder;
  let source;
  let analyser;
  try {
    recorder = new MediaRecorder(stream);
    source = context.createMediaStreamSource(stream);
    analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
  } catch (error) {
    release(stream);
    void context.close();
    throw error;
  }
  const timeDomain = new Float32Array(analyser.fftSize);
  const chunks = [];
  let timer;
  let activity;
  let level = 0;
  let released = false;
  let failed = false;
  const cleanup = () => {
    clearTimeout(timer);
    clearInterval(activity);
    if (released) return;
    released = true;
    release(stream);
    source.disconnect();
    void context.close();
  };
  const stop = () => {
    if (recorder.state !== 'inactive') recorder.stop();
    cleanup();
  };
  const startedAt = performance.now();
  let noiseFloor = 0.003;
  let speechStarted;
  let heardSpeech = false;
  let lastSpeech = startedAt;
  const detectActivity = () => {
    if (context.state !== 'running') return;
    analyser.getFloatTimeDomainData(timeDomain);
    let sum = 0;
    for (let i = 0; i < timeDomain.length; i++) {
      sum += timeDomain[i] * timeDomain[i];
    }
    level = Math.sqrt(sum / timeDomain.length);
    const now = performance.now();
    if (level > Math.max(0.008, noiseFloor * 3)) {
      speechStarted ??= now;
      if (now - speechStarted >= 150) heardSpeech = true;
      lastSpeech = now;
    } else {
      speechStarted = undefined;
      noiseFloor = noiseFloor * 0.95 + level * 0.05;
      if (autoStopOnSilence && heardSpeech && now - lastSpeech >= 2000) stop();
    }
  };
  const result = new Promise((resolve, reject) => {
    const abort = () => {
      stop();
      reject(signal.reason);
    };
    signal.addEventListener('abort', abort, { once: true });
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    recorder.onerror = () => {
      failed = true;
      stop();
      reject(new Error('Microphone recording failed. Try again.'));
    };
    recorder.onstop = async () => {
      cleanup();
      signal.removeEventListener('abort', abort);
      if (failed) return;
      try {
        signal.throwIfAborted();
        const encoded = await new Blob(chunks, { type: recorder.mimeType }).arrayBuffer();
        const { samples } = await decodeAudioBuffer(encoded, signal);
        resolve(samples);
      } catch (error) {
        reject(error);
      }
    };
    try {
      recorder.start();
      timer = setTimeout(stop, MAX_DURATION_SECONDS * 1000);
      activity = setInterval(detectActivity, 50);
    } catch (error) {
      signal.removeEventListener('abort', abort);
      cleanup();
      reject(error);
    }
  });
  return {
    stop,
    result,
    startedAt,
    level() {
      return released ? 0 : level;
    },
  };
}
