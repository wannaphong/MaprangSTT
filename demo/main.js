// MaprangSTT Web Demo powered by cactus-needle (Whistle WASM)
// Model: https://huggingface.co/pythainlp/MaprangSTT/resolve/main/maprang.cact

import {
  createWhistle,
  NeedleError,
} from './cactus-needle/dist/browser/index.js';
import {
  decodeAudioBuffer,
  encodeWavBlob,
  startRecording,
} from './recording.js';

export const MODEL_URL =
  'https://huggingface.co/pythainlp/MaprangSTT/resolve/main/maprang.cact';
export const MODEL_SHA256 =
  '6fd4b1a6a12a1b74cdef035ece814b89eb566a05a1789c10ad8fd714379057c0';

export const SAMPLES = [
  {
    id: 'cv_1',
    title: 'Common Voice #1 (Standard Thai)',
    url: './samples/cv_1.wav',
    duration: '8.4s',
    reference: 'ฉัน เช่นกัน ดังนั้น ฉัน พยายาม ที่จะ ได้รับ มากมาย ทุกที่ ที่ ฉัน ไป',
  },
  {
    id: 'cv_2',
    title: 'Common Voice #2 (Standard Thai)',
    url: './samples/cv_2.wav',
    duration: '6.4s',
    reference: 'สถาปัตยกรรม เหล่านั้น ก็ ไม่ ได้ เป็น สิ่งปลูกสร้าง ที่ มี อยู่ จริง',
  },
  {
    id: 'cv_3',
    title: 'Common Voice #3 (Standard Thai)',
    url: './samples/cv_3.wav',
    duration: '5.4s',
    reference: 'สถาน จัดกิจกรรม ที่ ดี ที่สุด ที่อยู่ ใกล้เคียง',
  },
  {
    id: 'fleurs_1',
    title: 'Google FLEURS (Spoken Numbers 1967)',
    url: './samples/fleurs_1.wav',
    duration: '11.0s',
    reference:
      'ตาม ข่าว ที่ รั่วไหล ออกมา เอกสาร ฉบับ นี้ จะ อ้างอิง ถึง ข้อพิพาท เรื่อง พรมแดน ซึ่ง ปาเลสไตน์ ต้องการ โดย อ้างอิง จาก พรมแดน ก่อน สงคราม ตะวันออกกลาง ใน ปี หนึ่ง พัน เก้า ร้อย หกสิบ เจ็ด',
  },
];

// DOM elements
const el = (selector) => {
  const node = document.querySelector(selector);
  if (!node) throw new Error(`Missing element: ${selector}`);
  return node;
};

const statusBadge = el('#model-status-badge');
const statusText = el('#status-text');
const progressContainer = el('#progress-container');
const progressBar = el('#progress-bar');
const progressPercent = el('#progress-percent');
const loadModelBtn = el('#load-model-btn');
const recordBtn = el('#record-btn');
const recordLabel = el('#record-label');
const autoStopCheckbox = el('#auto-stop-checkbox');
const fileInput = el('#audio-file-input');
const dropZone = el('#drop-zone');
const languageSelect = el('#language-select');
const timestampsCheckbox = el('#timestamps-checkbox');
const removeSpacesCheckbox = el('#remove-spaces-checkbox');
const recordingMeter = el('#recording-meter');
const recordingCanvas = el('#recording-canvas');
const recordingTime = el('#recording-time');
const audioPreviewBox = el('#audio-preview-box');
const audioPlayer = el('#audio-player');
const audioSourceLabel = el('#audio-source-label');
const reTranscribeBtn = el('#retranscribe-btn');
const transcriptOutput = el('#transcript-output');
const copyBtn = el('#copy-btn');
const referenceBox = el('#reference-box');
const referenceText = el('#reference-text');
const statsBox = el('#stats-box');
const statDuration = el('#stat-duration');
const statInference = el('#stat-inference');
const statTtft = el('#stat-ttft');
const statTps = el('#stat-tps');
const wordsBox = el('#words-box');
const wordsList = el('#words-list');
const sampleButtonsContainer = el('#sample-buttons');

let whistle;
let activeController;
let activeRecording;
let stopMeter;
let lastAudioSamples;
let lastAudioUrl;
let lastRawText = '';
let lastReference = '';

function setStatus(msg, isError = false) {
  statusText.textContent = msg;
  statusText.dataset.error = String(isError);
}

function updateModelBadge(state, detail = '') {
  statusBadge.dataset.state = state;
  if (state === 'ready') {
    statusBadge.textContent = 'Model Ready (Cached in Browser)';
    loadModelBtn.hidden = true;
    progressContainer.hidden = true;
  } else if (state === 'loading') {
    statusBadge.textContent = detail || 'Loading MaprangSTT (32.5 MB)…';
    loadModelBtn.disabled = true;
  } else if (state === 'error') {
    statusBadge.textContent = 'Model Load Failed';
    loadModelBtn.hidden = false;
    loadModelBtn.disabled = false;
    loadModelBtn.textContent = 'Retry Download';
  } else {
    statusBadge.textContent = 'Not Loaded (32.5 MB download on first use)';
    loadModelBtn.hidden = false;
    loadModelBtn.disabled = false;
    loadModelBtn.textContent = 'Preload Model (32.5 MB)';
  }
}

function formatErrorMessage(error) {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError')
      return 'Allow microphone access in your browser, then try again.';
    if (error.name === 'NotFoundError')
      return 'No microphone found. Connect one and try again.';
    if (error.name === 'NotReadableError')
      return 'The microphone is busy. Close other recording apps and try again.';
    if (error.name === 'EncodingError')
      return 'Could not decode that audio file. Please try WAV, MP3, M4A, or Ogg.';
  }
  if (error instanceof NeedleError) {
    if (error.code === 'DOWNLOAD_FAILED')
      return 'Could not download MaprangSTT from Hugging Face. Check your connection and try again.';
    if (error.code === 'INTEGRITY_ERROR')
      return 'Downloaded model failed SHA-256 integrity verification. Please try again.';
    if (error.code === 'UNSUPPORTED_ENVIRONMENT')
      return 'This browser cannot run WebAssembly workers. Use a current Chrome, Edge, Firefox, or Safari over HTTPS/localhost.';
  }
  return error instanceof Error ? error.message : String(error);
}

function formatBytes(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function ensureWhistle(signal) {
  if (whistle) return whistle;
  updateModelBadge('loading');
  progressContainer.hidden = false;
  progressBar.style.width = '0%';
  progressPercent.textContent = '0%';
  setStatus('Downloading MaprangSTT (maprang.cact) from Hugging Face…');

  try {
    whistle = await createWhistle({
      model: MODEL_URL,
      modelSha256: MODEL_SHA256,
      abortSignal: signal,
      onDownloadProgress({ loaded, total, percentage }) {
        const pct = Math.round(percentage ?? (total ? (loaded / total) * 100 : 0));
        progressBar.style.width = `${pct}%`;
        progressPercent.textContent = total
          ? `${pct}% (${formatBytes(loaded)} / ${formatBytes(total)})`
          : `${formatBytes(loaded)}`;
        if (pct >= 100) {
          updateModelBadge('loading', 'Initializing WASM Speech Engine…');
          setStatus('Starting Whistle WebAssembly worker…');
        } else {
          updateModelBadge('loading', `Downloading Model… ${pct}%`);
          setStatus(`Downloading maprang.cact from Hugging Face… ${pct}%`);
        }
      },
    });
    updateModelBadge('ready');
    setStatus('MaprangSTT loaded and ready for offline browser transcription.');
    return whistle;
  } catch (error) {
    whistle = undefined;
    if (!signal?.aborted) {
      updateModelBadge('error');
    } else {
      updateModelBadge('idle');
    }
    throw error;
  }
}

function renderTranscriptText() {
  if (!lastRawText) return;
  const text = removeSpacesCheckbox.checked
    ? lastRawText.replace(/\s+/g, '')
    : lastRawText;
  transcriptOutput.value = text;
  copyBtn.disabled = !text;
}

function renderWordTimestamps(words) {
  wordsList.replaceChildren();
  if (!words || !words.length || !timestampsCheckbox.checked) {
    wordsBox.hidden = true;
    return;
  }
  wordsBox.hidden = false;
  for (const item of words) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'word-chip';
    const prob = Math.round((item.probability ?? 0) * 100);
    const timeRange = `${item.start.toFixed(2)}s–${item.end.toFixed(2)}s`;
    chip.title = `Jump to ${item.start.toFixed(2)}s (${prob}% confidence)`;
    chip.dataset.confidence = prob >= 80 ? 'high' : prob >= 50 ? 'med' : 'low';

    const wordSpan = document.createElement('span');
    wordSpan.className = 'word-chip-text';
    wordSpan.textContent = item.word;

    const timeSpan = document.createElement('span');
    timeSpan.className = 'word-chip-time';
    timeSpan.textContent = timeRange;

    chip.append(wordSpan, timeSpan);
    chip.addEventListener('click', () => {
      if (audioPlayer.src) {
        audioPlayer.currentTime = item.start;
        void audioPlayer.play().catch(() => {});
      }
    });
    wordsList.append(chip);
  }
}

function setControlsBusy(busy, isRecording = false) {
  for (const btn of sampleButtonsContainer.querySelectorAll('button')) {
    btn.disabled = busy;
  }
  fileInput.disabled = busy;
  languageSelect.disabled = busy;
  reTranscribeBtn.disabled = busy || !lastAudioSamples;
  if (isRecording) {
    recordBtn.disabled = false;
    recordBtn.dataset.recording = 'true';
    recordBtn.setAttribute('aria-pressed', 'true');
    recordLabel.textContent = 'Stop Recording';
  } else {
    recordBtn.disabled = busy;
    recordBtn.dataset.recording = 'false';
    recordBtn.setAttribute('aria-pressed', 'false');
    recordLabel.textContent = 'Record Microphone';
  }
}

function setAudioPreview(samples, labelText, reference = '') {
  lastAudioSamples = samples;
  lastReference = reference;
  if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
  const wavBlob = encodeWavBlob(samples);
  lastAudioUrl = URL.createObjectURL(wavBlob);
  audioPlayer.src = lastAudioUrl;
  audioSourceLabel.textContent = labelText;
  audioPreviewBox.hidden = false;
  reTranscribeBtn.disabled = false;

  if (reference) {
    referenceText.textContent = reference;
    referenceBox.hidden = false;
  } else {
    referenceText.textContent = '';
    referenceBox.hidden = true;
  }
}

async function runTranscription(samples, sourceLabel, reference = '') {
  if (activeController) {
    activeController.abort();
  }
  const controller = new AbortController();
  activeController = controller;
  setControlsBusy(true, false);

  try {
    setAudioPreview(samples, sourceLabel, reference);
    const session = await ensureWhistle(controller.signal);
    controller.signal.throwIfAborted();

    setStatus('Transcribing audio in WebAssembly worker…');
    const lang = languageSelect.value;
    const wordTimestamps = timestampsCheckbox.checked;

    const t0 = performance.now();
    const res = await session.transcribe(samples, {
      ...(lang ? { language: lang } : {}),
      wordTimestamps,
      abortSignal: controller.signal,
    });
    const elapsedMs = performance.now() - t0;

    lastRawText = (res.text || '').trim();
    if (!lastRawText) {
      transcriptOutput.value = '';
      copyBtn.disabled = true;
      setStatus('No speech detected in the audio clip.');
      statsBox.hidden = true;
      wordsBox.hidden = true;
      return;
    }

    renderTranscriptText();
    renderWordTimestamps(res.words);

    statDuration.textContent = `${res.durationInSeconds.toFixed(2)} s`;
    statInference.textContent = `${(elapsedMs / 1000).toFixed(2)} s (${(
      res.durationInSeconds /
      (elapsedMs / 1000)
    ).toFixed(1)}× real-time)`;
    statTtft.textContent = `${Math.round(res.timeToFirstTokenMs)} ms`;
    statTps.textContent = `${res.tokensPerSecond.toFixed(1)} tok/s`;
    statsBox.hidden = false;

    setStatus(
      `Transcribed ${res.durationInSeconds.toFixed(1)}s audio in ${(
        elapsedMs / 1000
      ).toFixed(2)}s (${res.tokensPerSecond.toFixed(1)} tokens/s).`,
    );
  } catch (error) {
    if (
      controller.signal.aborted ||
      (error instanceof NeedleError && ['CLOSED', 'WORKER_ERROR'].includes(error.code))
    ) {
      if (whistle && error instanceof NeedleError) {
        await whistle.close().catch(() => {});
        whistle = undefined;
        updateModelBadge('idle');
      }
    }
    setStatus(
      controller.signal.aborted ? 'Cancelled.' : formatErrorMessage(error),
      !controller.signal.aborted,
    );
  } finally {
    if (activeController === controller) {
      activeController = undefined;
    }
    setControlsBusy(false, false);
  }
}

function showRecordingMeter(capture) {
  const ctx = recordingCanvas.getContext('2d');
  const levels = [];
  let prev = 0;
  let frame = 0;
  recordingMeter.hidden = false;

  const draw = (now) => {
    frame = requestAnimationFrame(draw);
    if (now - prev < 50) return;
    prev = now;
    const secs = Math.min(30, Math.floor((now - capture.startedAt) / 1000));
    recordingTime.textContent = `0:${String(secs).padStart(2, '0')} / 0:30`;
    if (!ctx) return;
    const width = recordingCanvas.clientWidth || 300;
    const height = recordingCanvas.clientHeight || 36;
    const dpr = window.devicePixelRatio || 1;
    if (
      recordingCanvas.width !== Math.round(width * dpr) ||
      recordingCanvas.height !== Math.round(height * dpr)
    ) {
      recordingCanvas.width = Math.round(width * dpr);
      recordingCanvas.height = Math.round(height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    levels.push(Math.min(1, capture.level() * 6));
    const count = Math.floor(width / 5);
    levels.splice(0, Math.max(0, levels.length - count));
    for (let i = 0; i < count; i++) {
      const lvl = levels[i - (count - levels.length)] ?? 0;
      const bar = Math.max(3, lvl * (height - 4));
      ctx.fillStyle = lvl > 0.025 ? '#dc2626' : '#d1d5db';
      ctx.beginPath();
      ctx.roundRect(i * 5, (height - bar) / 2, 2.5, bar, 1);
      ctx.fill();
    }
  };
  draw(performance.now());
  return () => {
    cancelAnimationFrame(frame);
    recordingMeter.hidden = true;
  };
}

async function handleMicrophoneClick() {
  if (activeRecording) {
    activeRecording.stop();
    recordBtn.disabled = true;
    return;
  }
  if (activeController) return;

  const controller = new AbortController();
  activeController = controller;
  setControlsBusy(true, false);

  try {
    setStatus('Requesting microphone access…');
    activeRecording = await startRecording(controller.signal, {
      autoStopOnSilence: autoStopCheckbox.checked,
    });
    stopMeter = showRecordingMeter(activeRecording);
    setControlsBusy(true, true);
    setStatus(
      autoStopCheckbox.checked
        ? 'Listening… Speak in Thai (stops automatically after a 2s pause or press Stop).'
        : 'Recording… Speak in Thai and press Stop Recording when finished (up to 30s).',
    );
    const samples = await activeRecording.result;
    stopMeter?.();
    stopMeter = undefined;
    activeRecording = undefined;
    activeController = undefined;
    await runTranscription(samples, 'Microphone Recording');
  } catch (error) {
    stopMeter?.();
    stopMeter = undefined;
    activeRecording?.stop();
    activeRecording = undefined;
    activeController = undefined;
    setControlsBusy(false, false);
    setStatus(
      controller.signal.aborted ? 'Recording cancelled.' : formatErrorMessage(error),
      !controller.signal.aborted,
    );
  }
}

async function handleAudioFile(file) {
  if (!file || activeController) return;
  const controller = new AbortController();
  activeController = controller;
  setControlsBusy(true, false);

  try {
    setStatus(`Decoding "${file.name}" to 16 kHz mono PCM…`);
    const buffer = await file.arrayBuffer();
    const { samples, durationInSeconds } = await decodeAudioBuffer(
      buffer,
      controller.signal,
    );
    activeController = undefined;
    const note =
      durationInSeconds >= 30 ? ' (trimmed to first 30s)' : '';
    await runTranscription(samples, `Uploaded: ${file.name}${note}`);
  } catch (error) {
    activeController = undefined;
    setControlsBusy(false, false);
    setStatus(formatErrorMessage(error), true);
  } finally {
    fileInput.value = '';
  }
}

async function handleSampleSelect(sample) {
  if (activeController) return;
  const controller = new AbortController();
  activeController = controller;
  setControlsBusy(true, false);

  try {
    setStatus(`Loading sample "${sample.title}"…`);
    const res = await fetch(sample.url, { signal: controller.signal });
    if (!res.ok) throw new Error(`Failed to fetch sample (${res.status})`);
    const buffer = await res.arrayBuffer();
    const { samples } = await decodeAudioBuffer(buffer, controller.signal);
    activeController = undefined;
    await runTranscription(samples, sample.title, sample.reference);
  } catch (error) {
    activeController = undefined;
    setControlsBusy(false, false);
    setStatus(formatErrorMessage(error), true);
  }
}

// Wire up UI events
loadModelBtn.addEventListener('click', async () => {
  if (activeController || whistle) return;
  const controller = new AbortController();
  activeController = controller;
  setControlsBusy(true, false);
  try {
    await ensureWhistle(controller.signal);
  } catch (error) {
    setStatus(formatErrorMessage(error), true);
  } finally {
    activeController = undefined;
    setControlsBusy(false, false);
  }
});

recordBtn.addEventListener('click', () => {
  void handleMicrophoneClick();
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) void handleAudioFile(file);
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.dataset.dragover = 'true';
});
dropZone.addEventListener('dragleave', () => {
  dropZone.dataset.dragover = 'false';
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.dataset.dragover = 'false';
  const file = e.dataTransfer?.files?.[0];
  if (file) void handleAudioFile(file);
});

reTranscribeBtn.addEventListener('click', () => {
  if (lastAudioSamples && !activeController) {
    void runTranscription(
      lastAudioSamples,
      audioSourceLabel.textContent || 'Audio',
      lastReference,
    );
  }
});

removeSpacesCheckbox.addEventListener('change', () => {
  renderTranscriptText();
});

timestampsCheckbox.addEventListener('change', () => {
  if (lastAudioSamples && whistle && !activeController) {
    void runTranscription(
      lastAudioSamples,
      audioSourceLabel.textContent || 'Audio',
      lastReference,
    );
  }
});

copyBtn.addEventListener('click', async () => {
  const text = transcriptOutput.value;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    const orig = copyBtn.textContent;
    copyBtn.textContent = 'Copied!';
    setTimeout(() => {
      copyBtn.textContent = orig;
    }, 1500);
  } catch {
    transcriptOutput.select();
    document.execCommand('copy');
  }
});

// Populate sample buttons
for (const sample of SAMPLES) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'sample-btn';
  btn.innerHTML = `<span class="sample-btn-title">${sample.title}</span><span class="sample-btn-meta">${sample.duration}</span>`;
  btn.title = sample.reference;
  btn.addEventListener('click', () => {
    void handleSampleSelect(sample);
  });
  sampleButtonsContainer.append(btn);
}

window.addEventListener('pagehide', () => {
  activeController?.abort();
  activeRecording?.stop();
  void whistle?.close();
  if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
});
