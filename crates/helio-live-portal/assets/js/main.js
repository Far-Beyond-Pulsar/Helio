// main.js - entry point for portal frontend, handles websocket and updates
import { renderNodeGraph } from './nodeGraph.js';

const statusEl = document.getElementById('status');
const frameEl = document.getElementById('frame');
const frameMsEl = document.getElementById('frameMs');
const gpuTotalEl = document.getElementById('gpuTotal');
const cpuTotalEl = document.getElementById('cpuTotal');
const objCountEl = document.getElementById('objCount');
const lightCountEl = document.getElementById('lightCount');
const bbCountEl = document.getElementById('bbCount');
const drawCallsEl = document.getElementById('drawCalls');

let lastSnapshot = null;
let cyInstance = null;

// ── Recording state ───────────────────────────────────────────────────────────
const CHUNK_SIZE = 100; // Flush to IndexedDB every 100 frames
const recordingState = {
  isRecording: false,
  buffer: [],           // Temporary buffer (max CHUNK_SIZE frames)
  chunks: [],           // Array of chunk metadata
  startTime: null,
  frameCount: 0,
  db: null,             // IndexedDB instance
  lastBadgeUpdate: 0    // Throttle badge updates
};

// ── Replay state ──────────────────────────────────────────────────────────────
const replayState = {
  isActive: false,
  snapshots: [],
  currentIndex: 0,
  playing: false,
  playbackSpeed: 1.0,
  playbackInterval: null
};

// ── copy helpers ─────────────────────────────────────────────────────────────

const toastEl = document.getElementById('copyToast');
let toastTimer = null;

function showToast(msg) {
  if (!toastEl) return;
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 1800);
}

async function copyJson(data, btn) {
  const json = JSON.stringify(data, null, 2);
  try {
    await navigator.clipboard.writeText(json);
    showToast('Copied frame data to clipboard ✓');
    if (btn) {
      btn.classList.add('copied');
      setTimeout(() => btn.classList.remove('copied'), 1200);
    }
  } catch {
    // Fallback for browsers without clipboard API permission
    const ta = document.createElement('textarea');
    ta.value = json;
    ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    showToast('Copied to clipboard ✓');
  }
}

const btnCopyFrame = document.getElementById('btnCopyFrame');

if (btnCopyFrame) {
  btnCopyFrame.addEventListener('click', () => {
    if (!lastSnapshot) return;
    const { scene_delta: _omit, ...rest } = lastSnapshot;
    copyJson(rest, btnCopyFrame);
  });
}

// ── render ────────────────────────────────────────────────────────────────────

function render(snapshot) {
  lastSnapshot = snapshot;

  // Update stat cards
  frameEl.textContent = snapshot.frame;
  frameMsEl.innerHTML = `${snapshot.frame_time_ms.toFixed(2)}<span class="stat-unit">ms</span>`;
  gpuTotalEl.innerHTML = `${snapshot.total_gpu_ms.toFixed(2)}<span class="stat-unit">ms</span>`;
  cpuTotalEl.innerHTML = `${snapshot.total_cpu_ms.toFixed(2)}<span class="stat-unit">ms</span>`;

  if (objCountEl) objCountEl.textContent = snapshot.object_count;
  if (lightCountEl) lightCountEl.textContent = snapshot.light_count;
  if (bbCountEl) bbCountEl.textContent = snapshot.billboard_count;
  if (drawCallsEl) drawCallsEl.textContent = snapshot.draw_calls.total;

  // Update timings table (in modal)
  const totalGpu = snapshot.total_gpu_ms || 0;
  const totalCpu = snapshot.total_cpu_ms || 0;

  // Prepare data for table (expose to global scope for sorting)
  window.timingsData = (snapshot.pass_timings || []).map(t => ({
    name: t.name,
    gpu: t.gpu_ms,
    cpu: t.cpu_ms,
    gpuPct: totalGpu > 0 ? parseFloat(((t.gpu_ms / totalGpu) * 100).toFixed(1)) : 0,
    cpuPct: totalCpu > 0 ? parseFloat(((t.cpu_ms / totalCpu) * 100).toFixed(1)) : 0,
  }));

  // Render table (respects current sort)
  if (window.renderTimingsTable) {
    const sorted = window.sortColumn
      ? [...window.timingsData].sort((a, b) => {
          const aVal = a[window.sortColumn];
          const bVal = b[window.sortColumn];
          const dir = window.sortDirection === 'asc' ? 1 : -1;
          if (typeof aVal === 'string') {
            return dir * aVal.toLowerCase().localeCompare(bVal.toLowerCase());
          }
          return dir * (aVal - bVal);
        })
      : window.timingsData;
    window.renderTimingsTable(sorted);
  }

  // Render pipeline graph (full screen)
  cyInstance = renderNodeGraph(snapshot, '');

  // Render timing tree if available
  if (window.renderTimingTreeGraph) {
    window.renderTimingTreeGraph(snapshot);
  }

  window.perfWindows?.update(snapshot);

  const dt = new Date(snapshot.timestamp_ms).toLocaleTimeString();
  statusEl.textContent = `Connected · ${dt}`;
}

// ── WebSocket connection ──────────────────────────────────────────────────────

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws`);

ws.onopen = () => {
  statusEl.textContent = 'Connected · waiting for first frame...';
};

ws.onclose = () => {
  statusEl.textContent = 'Disconnected';
};

ws.onerror = () => {
  statusEl.textContent = 'Socket error';
};

ws.onmessage = (evt) => {
  // Ignore WebSocket messages when in replay mode
  if (replayState.isActive) return;

  try {
    const data = JSON.parse(evt.data);
    if (Array.isArray(data)) {
      // Batch of snapshots sent at once; process them in order
      for (const snap of data) {
        // Capture to recording buffer if recording is active
        if (recordingState.isRecording) {
          captureFrame(snap);
        }
        render(snap);
      }
    } else {
      if (recordingState.isRecording) {
        captureFrame(data);
      }
      render(data);
    }
  } catch (e) {
    console.error('parse error', e);
  }
};

// ── Recording functions (chunked with IndexedDB) ─────────────────────────────

// Initialize IndexedDB for recording storage
async function initRecordingDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('HelioRecordings', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('chunks')) {
        db.createObjectStore('chunks', { keyPath: 'id' });
      }
    };
  });
}

// Capture a single frame (buffered, flushes to IndexedDB when chunk is full)
async function captureFrame(snapshot) {
  recordingState.buffer.push(snapshot);
  recordingState.frameCount++;

  // Throttle badge updates (every 30 frames)
  if (recordingState.frameCount - recordingState.lastBadgeUpdate >= 30) {
    updateRecordingUI();
    recordingState.lastBadgeUpdate = recordingState.frameCount;
  }

  // Flush chunk to IndexedDB when buffer is full
  if (recordingState.buffer.length >= CHUNK_SIZE) {
    await flushChunk();
  }
}

// Flush current buffer to IndexedDB
async function flushChunk() {
  if (recordingState.buffer.length === 0 || !recordingState.db) return;

  const chunkId = recordingState.chunks.length;
  const chunk = {
    id: chunkId,
    snapshots: recordingState.buffer.slice()
  };

  return new Promise((resolve, reject) => {
    const transaction = recordingState.db.transaction(['chunks'], 'readwrite');
    const store = transaction.objectStore('chunks');
    const request = store.add(chunk);

    request.onsuccess = () => {
      recordingState.chunks.push(chunkId);
      recordingState.buffer = [];
      resolve();
    };
    request.onerror = () => reject(request.error);
  });
}

// Update recording UI (throttled)
function updateRecordingUI() {
  const badge = document.getElementById('recordingBadge');
  if (badge && recordingState.isRecording) {
    // Show frame count in K for large numbers
    const count = recordingState.frameCount;
    badge.textContent = count >= 10000 ? `${(count / 1000).toFixed(1)}K` : count;
  }
}

async function startRecording() {
  try {
    // Initialize IndexedDB
    recordingState.db = await initRecordingDB();

    // Clear previous recording data
    const transaction = recordingState.db.transaction(['chunks'], 'readwrite');
    const store = transaction.objectStore('chunks');
    await new Promise((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    recordingState.isRecording = true;
    recordingState.buffer = [];
    recordingState.chunks = [];
    recordingState.startTime = Date.now();
    recordingState.frameCount = 0;
    recordingState.lastBadgeUpdate = 0;

    const btn = document.getElementById('btnRecord');
    const badge = document.getElementById('recordingBadge');
    if (btn) btn.classList.add('recording');
    if (badge) badge.textContent = '●';

    console.log('Recording started');
  } catch (err) {
    showToast(`Failed to start recording: ${err.message}`);
    console.error('Recording initialization failed:', err);
  }
}

async function stopRecording() {
  if (!recordingState.isRecording) return;

  recordingState.isRecording = false;

  const btn = document.getElementById('btnRecord');
  const badge = document.getElementById('recordingBadge');
  if (btn) btn.classList.remove('recording');
  if (badge) badge.textContent = '';

  console.log(`Recording stopped: ${recordingState.frameCount} frames`);

  // Flush remaining buffer
  if (recordingState.buffer.length > 0) {
    await flushChunk();
  }

  // Auto-download the recording
  if (recordingState.frameCount > 0) {
    await downloadRecording();
  } else {
    showToast('No frames recorded');
  }
}

async function downloadRecording() {
  try {
    showToast('Preparing download...');

    // Collect all chunks from IndexedDB
    const allSnapshots = [];

    for (const chunkId of recordingState.chunks) {
      const chunk = await new Promise((resolve, reject) => {
        const transaction = recordingState.db.transaction(['chunks'], 'readonly');
        const store = transaction.objectStore('chunks');
        const request = store.get(chunkId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });

      if (chunk && chunk.snapshots) {
        allSnapshots.push(...chunk.snapshots);
      }
    }

    if (allSnapshots.length === 0) {
      showToast('No frames to download');
      return;
    }

    const duration = allSnapshots[allSnapshots.length - 1].timestamp_ms - allSnapshots[0].timestamp_ms;

    const recording = {
      version: 1,
      meta: {
        recorded_at: new Date().toISOString(),
        total_frames: allSnapshots.length,
        duration_ms: duration
      },
      snapshots: allSnapshots
    };

    // Stream download using chunked JSON serialization
    const jsonStart = '{"version":1,"meta":' + JSON.stringify(recording.meta) + ',"snapshots":[';
    const jsonEnd = ']}';

    const parts = [jsonStart];

    // Add snapshots in chunks to avoid blocking
    for (let i = 0; i < allSnapshots.length; i++) {
      if (i > 0) parts.push(',');
      parts.push(JSON.stringify(allSnapshots[i]));

      // Yield to UI every 100 frames
      if (i % 100 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
        showToast(`Serializing... ${Math.round((i / allSnapshots.length) * 100)}%`);
      }
    }

    parts.push(jsonEnd);

    const blob = new Blob(parts, { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `helio-recording-${Date.now()}.helio-recording`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    // Cleanup after a delay
    setTimeout(() => URL.revokeObjectURL(url), 1000);

    showToast(`Downloaded ${allSnapshots.length.toLocaleString()} frames`);
  } catch (err) {
    showToast(`Download failed: ${err.message}`);
    console.error('Download failed:', err);
  }
}

// ── Replay functions ──────────────────────────────────────────────────────────

function loadRecording(recording) {
  try {
    // Validate recording format
    if (!recording.version || !recording.snapshots || !Array.isArray(recording.snapshots)) {
      throw new Error('Invalid recording format');
    }

    if (recording.snapshots.length === 0) {
      throw new Error('Recording is empty');
    }

    // Enter replay mode
    replayState.isActive = true;
    replayState.snapshots = recording.snapshots;
    replayState.currentIndex = 0;
    replayState.playing = false;

    // Close WebSocket to prevent live updates
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }

    // Show replay controls
    const replayBar = document.getElementById('replayControls');
    if (replayBar) replayBar.classList.add('active');

    // Update slider max
    const slider = document.getElementById('replaySlider');
    if (slider) {
      slider.max = replayState.snapshots.length - 1;
      slider.value = 0;
    }

    // Render first frame
    renderReplayFrame(0);

    showToast(`Loaded recording: ${recording.snapshots.length} frames`);
    console.log('Replay mode activated', recording.meta);
  } catch (e) {
    showToast(`Error loading recording: ${e.message}`);
    console.error('Failed to load recording:', e);
  }
}

function renderReplayFrame(index) {
  if (index < 0 || index >= replayState.snapshots.length) return;

  replayState.currentIndex = index;
  const snapshot = replayState.snapshots[index];

  // Render the frame
  render(snapshot);

  // Update UI
  updateReplayUI();
}

function updateReplayUI() {
  const slider = document.getElementById('replaySlider');
  const timeEl = document.getElementById('replayTime');
  const durationEl = document.getElementById('replayDuration');

  if (slider) slider.value = replayState.currentIndex;

  if (timeEl) {
    timeEl.textContent = `Frame ${replayState.currentIndex + 1} / ${replayState.snapshots.length}`;
  }

  if (durationEl && replayState.snapshots.length > 0) {
    const current = replayState.snapshots[replayState.currentIndex];
    const first = replayState.snapshots[0];
    const last = replayState.snapshots[replayState.snapshots.length - 1];
    const currentTime = (current.timestamp_ms - first.timestamp_ms) / 1000;
    const totalTime = (last.timestamp_ms - first.timestamp_ms) / 1000;
    durationEl.textContent = `${currentTime.toFixed(2)}s / ${totalTime.toFixed(2)}s`;
  }
}

function toggleReplayPlayback() {
  replayState.playing = !replayState.playing;

  const playIcon = document.getElementById('replayPlayIcon');
  const pauseIcon = document.getElementById('replayPauseIcon');

  if (replayState.playing) {
    if (playIcon) playIcon.style.display = 'none';
    if (pauseIcon) pauseIcon.style.display = 'block';
    startReplayPlayback();
  } else {
    if (playIcon) playIcon.style.display = 'block';
    if (pauseIcon) pauseIcon.style.display = 'none';
    stopReplayPlayback();
  }
}

function startReplayPlayback() {
  if (replayState.playbackInterval) return;

  // Calculate average frame time for playback
  let avgFrameTime = 16.67; // Default to 60fps
  if (replayState.snapshots.length > 1) {
    const first = replayState.snapshots[0];
    const last = replayState.snapshots[replayState.snapshots.length - 1];
    avgFrameTime = (last.timestamp_ms - first.timestamp_ms) / replayState.snapshots.length;
  }

  const playbackInterval = avgFrameTime / replayState.playbackSpeed;

  replayState.playbackInterval = setInterval(() => {
    if (replayState.currentIndex >= replayState.snapshots.length - 1) {
      // Reached the end
      stopReplayPlayback();
      replayState.playing = false;
      const playIcon = document.getElementById('replayPlayIcon');
      const pauseIcon = document.getElementById('replayPauseIcon');
      if (playIcon) playIcon.style.display = 'block';
      if (pauseIcon) pauseIcon.style.display = 'none';
    } else {
      renderReplayFrame(replayState.currentIndex + 1);
    }
  }, playbackInterval);
}

function stopReplayPlayback() {
  if (replayState.playbackInterval) {
    clearInterval(replayState.playbackInterval);
    replayState.playbackInterval = null;
  }
}

function stepReplayFrame(delta) {
  stopReplayPlayback();
  replayState.playing = false;
  const playIcon = document.getElementById('replayPlayIcon');
  const pauseIcon = document.getElementById('replayPauseIcon');
  if (playIcon) playIcon.style.display = 'block';
  if (pauseIcon) pauseIcon.style.display = 'none';

  const newIndex = Math.max(0, Math.min(replayState.snapshots.length - 1, replayState.currentIndex + delta));
  renderReplayFrame(newIndex);
}

function cyclePlaybackSpeed() {
  const speeds = [0.25, 0.5, 1.0, 2.0, 4.0];
  const currentIdx = speeds.indexOf(replayState.playbackSpeed);
  const nextIdx = (currentIdx + 1) % speeds.length;
  replayState.playbackSpeed = speeds[nextIdx];

  const btn = document.getElementById('replaySpeed');
  if (btn) btn.textContent = `${replayState.playbackSpeed}×`;

  // Restart playback if currently playing
  if (replayState.playing) {
    stopReplayPlayback();
    startReplayPlayback();
  }
}

function exitReplayMode() {
  // Stop playback
  stopReplayPlayback();
  replayState.playing = false;

  // Clear replay state
  replayState.isActive = false;
  replayState.snapshots = [];
  replayState.currentIndex = 0;

  // Hide replay controls
  const replayBar = document.getElementById('replayControls');
  if (replayBar) replayBar.classList.remove('active');

  // Reconnect WebSocket
  location.reload(); // Simple approach: reload to reconnect
}

// ── File upload handling ──────────────────────────────────────────────────────

function handleFileUpload(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const recording = JSON.parse(e.target.result);
      loadRecording(recording);
    } catch (err) {
      showToast(`Failed to parse recording: ${err.message}`);
      console.error('File parse error:', err);
    }
  };
  reader.readAsText(file);
}

// ── Button event handlers ─────────────────────────────────────────────────────

// Record button
const btnRecord = document.getElementById('btnRecord');
if (btnRecord) {
  btnRecord.addEventListener('click', () => {
    if (recordingState.isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });
}

// Upload button
const btnUpload = document.getElementById('btnUpload');
const fileInput = document.getElementById('fileUploadInput');
if (btnUpload && fileInput) {
  btnUpload.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileUpload(file);
    // Reset input so the same file can be selected again
    e.target.value = '';
  });
}

// Drag and drop upload
const dropZone = document.getElementById('uploadDropZone');
if (dropZone) {
  // Show drop zone on drag over page
  let dragCounter = 0;
  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (dragCounter === 1 && !replayState.isActive) {
      dropZone.classList.add('active');
    }
  });

  document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter === 0) {
      dropZone.classList.remove('active');
    }
  });

  document.addEventListener('dragover', (e) => {
    e.preventDefault();
  });

  document.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    dropZone.classList.remove('active');

    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  });

  // Click on drop zone to trigger file picker
  dropZone.addEventListener('click', () => {
    if (fileInput) fileInput.click();
  });
}

// Replay controls
const replaySlider = document.getElementById('replaySlider');
if (replaySlider) {
  replaySlider.addEventListener('input', (e) => {
    const index = parseInt(e.target.value, 10);
    renderReplayFrame(index);
  });
}

const replayPlay = document.getElementById('replayPlay');
if (replayPlay) {
  replayPlay.addEventListener('click', toggleReplayPlayback);
}

const replayStepBack = document.getElementById('replayStepBack');
if (replayStepBack) {
  replayStepBack.addEventListener('click', () => stepReplayFrame(-1));
}

const replayStepForward = document.getElementById('replayStepForward');
if (replayStepForward) {
  replayStepForward.addEventListener('click', () => stepReplayFrame(1));
}

const replaySpeed = document.getElementById('replaySpeed');
if (replaySpeed) {
  replaySpeed.addEventListener('click', cyclePlaybackSpeed);
}

const replayExit = document.getElementById('replayExit');
if (replayExit) {
  replayExit.addEventListener('click', exitReplayMode);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (!replayState.isActive) return;

  // Prevent shortcuts if user is typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  switch (e.key) {
    case ' ':
      e.preventDefault();
      toggleReplayPlayback();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      stepReplayFrame(-1);
      break;
    case 'ArrowRight':
      e.preventDefault();
      stepReplayFrame(1);
      break;
    case 'Escape':
      e.preventDefault();
      exitReplayMode();
      break;
  }
});

// ── Export for debugging ──────────────────────────────────────────────────────
window.helioPortal = {
  getLastSnapshot: () => lastSnapshot,
  getGraphInstance: () => cyInstance,
  recording: recordingState,
  replay: replayState,
};
