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

// ── Central Data Store (Single Source of Truth) ──────────────────────────────
// All UI elements MUST read from this central store instead of maintaining their own state
const centralStore = {
  mode: 'live', // 'live' or 'replay'

  // Rolling history for charts (last 256 frames) - SINGLE SOURCE OF TRUTH
  history: {
    maxSize: 256,
    snapshots: []
  },

  // Add snapshot to rolling history (only in live mode)
  addSnapshot(snapshot) {
    this.history.snapshots.push(snapshot);
    if (this.history.snapshots.length > this.history.maxSize) {
      this.history.snapshots.shift();
    }
  },

  // Set history from replay (replaces entire history)
  setHistory(snapshots, currentIndex) {
    // Take a window of snapshots centered around currentIndex
    const start = Math.max(0, currentIndex - this.history.maxSize + 1);
    const end = currentIndex + 1;
    this.history.snapshots = snapshots.slice(start, end);
  },

  // Clear all state
  clear() {
    this.history.snapshots = [];
  },

  // Get rolling history (for all UI components to read)
  getHistory() {
    return this.history.snapshots;
  },

  // Get current snapshot (last in history)
  getCurrent() {
    return this.history.snapshots[this.history.snapshots.length - 1] || null;
  }
};

let currentDataSource = 'live'; // 'live' or 'replay' (deprecated, use centralStore.mode)

// ── Recording state ───────────────────────────────────────────────────────────
const BATCH_SIZE = 10; // Send frames in batches of 10
const recordingState = {
  isRecording: false,
  worker: null,
  frameCount: 0,
  batchBuffer: [],       // Accumulate frames before sending to worker
  uiUpdateScheduled: false
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

let ws = null;

if (window.HELIO_STATIC_MODE) {
  // Static profiler mode — no live server, load recordings from file or URL
  const _staticProfileParam = new URLSearchParams(location.search).get('profile');
  if (_staticProfileParam) {
    // ?profile= present — load it immediately, skip the upload screen
    statusEl.textContent = 'Loading profile…';
    // loadRemoteProfile is defined later; defer one microtask so all top-level
    // code (including the function definition) has run before we call it.
    Promise.resolve().then(() => loadRemoteProfile(_staticProfileParam));
  } else {
    statusEl.textContent = 'Offline viewer · drop a .helio-recording file or click Upload';
    // Automatically surface the drop zone so users know what to do
    const dropZoneEl = document.getElementById('uploadDropZone');
    if (dropZoneEl) dropZoneEl.classList.add('active');
  }
} else {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

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
    if (currentDataSource === 'replay') return;

    try {
      const data = JSON.parse(evt.data);
      if (Array.isArray(data)) {
        // Batch of snapshots sent at once; process them in order
        for (const snap of data) {
          processSnapshot(snap);
        }
      } else {
        processSnapshot(data);
      }
    } catch (e) {
      console.error('parse error', e);
    }
  };
} // end !HELIO_STATIC_MODE

// Process a snapshot from either live or replay source
function processSnapshot(snapshot) {
  // Add to central store (single source of truth)
  if (centralStore.mode === 'live') {
    centralStore.addSnapshot(snapshot);
  }

  // Record if active (batch and send to worker)
  if (recordingState.isRecording && recordingState.worker) {
    recordingState.batchBuffer.push(snapshot);
    recordingState.frameCount++;

    // Flush batch when full
    if (recordingState.batchBuffer.length >= BATCH_SIZE) {
      recordingState.worker.postMessage({
        type: 'capture-batch',
        payload: recordingState.batchBuffer.slice()
      });
      recordingState.batchBuffer = [];
    }

    // Schedule UI update using requestIdleCallback (non-blocking)
    if (!recordingState.uiUpdateScheduled) {
      recordingState.uiUpdateScheduled = true;
      (requestIdleCallback || requestAnimationFrame)(() => {
        updateRecordingUI();
        recordingState.uiUpdateScheduled = false;
      });
    }
  }

  // Render to UI (reads from central store)
  render(snapshot);
}

// ── UI State Management ──────────────────────────────────────────────────────

// Clear all accumulated UI state (for replay mode switch)
function clearUIState() {
  // Clear central store (single source of truth)
  centralStore.clear();

  // Clear perf windows state if available (will be deprecated - should read from centralStore)
  if (window.perfWindows && window.perfWindows.clear) {
    window.perfWindows.clear();
  }

  // Clear global timing data
  window.timingsData = [];

  console.log('UI state cleared (central store + legacy)');
}

// ── Recording functions (Web Worker based) ───────────────────────────────────

function updateRecordingUI() {
  const badge = document.getElementById('recordingBadge');
  if (badge && recordingState.isRecording) {
    const count = recordingState.frameCount;
    badge.textContent = count >= 10000 ? `${(count / 1000).toFixed(1)}K` : count;
  }
}

function startRecording() {
  if (window.HELIO_STATIC_MODE) {
    showToast('Recording is not available in offline viewer mode');
    return;
  }
  try {
    // Create recording worker
    recordingState.worker = new Worker('./js/recordingWorker.js');
    recordingState.frameCount = 0;
    recordingState.batchBuffer = [];
    recordingState.uiUpdateScheduled = false;

    // Handle worker messages
    recordingState.worker.onmessage = (e) => {
      const { type, progress, parts, totalFrames, error } = e.data;

      switch (type) {
        case 'init-complete':
          console.log('Recording worker initialized');
          break;

        case 'progress':
          // Worker confirms batch received (optional logging)
          break;

        case 'serialize-progress':
          showToast(`Serializing... ${progress}%`);
          break;

        case 'stop-complete':
          if (parts && parts.length > 0) {
            // Download the recording
            const blob = new Blob(parts, { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `helio-recording-${Date.now()}.helio-recording`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            setTimeout(() => URL.revokeObjectURL(url), 1000);
            showToast(`Downloaded ${totalFrames.toLocaleString()} frames`);
          } else {
            showToast('No frames recorded');
          }

          // Cleanup worker
          if (recordingState.worker) {
            recordingState.worker.terminate();
            recordingState.worker = null;
          }
          break;

        case 'error':
          showToast(`Recording error: ${error}`);
          console.error('Recording error:', error);
          break;
      }
    };

    recordingState.worker.onerror = (err) => {
      showToast(`Worker error: ${err.message}`);
      console.error('Worker error:', err);
    };

    // Initialize worker
    recordingState.worker.postMessage({ type: 'init' });

    recordingState.isRecording = true;

    const btn = document.getElementById('btnRecord');
    const badge = document.getElementById('recordingBadge');
    if (btn) btn.classList.add('recording');
    if (badge) badge.textContent = '●';

    console.log('Recording started (batched in-memory mode)');
  } catch (err) {
    showToast(`Failed to start recording: ${err.message}`);
    console.error('Recording initialization failed:', err);
  }
}

function stopRecording() {
  if (!recordingState.isRecording || !recordingState.worker) return;

  recordingState.isRecording = false;

  const btn = document.getElementById('btnRecord');
  const badge = document.getElementById('recordingBadge');
  if (btn) btn.classList.remove('recording');
  if (badge) badge.textContent = '';

  // Flush any remaining batched frames
  if (recordingState.batchBuffer.length > 0) {
    recordingState.worker.postMessage({
      type: 'capture-batch',
      payload: recordingState.batchBuffer.slice()
    });
    recordingState.batchBuffer = [];
  }

  console.log(`Recording stopped: ${recordingState.frameCount} frames`);

  // Tell worker to stop and prepare download
  showToast('Preparing download...');
  recordingState.worker.postMessage({ type: 'stop' });
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

    // CRITICAL: Clear all accumulated UI state
    clearUIState();

    // Switch to replay mode in central store
    centralStore.mode = 'replay';
    currentDataSource = 'replay';
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

    // Hide the upload drop zone now that a recording is loaded
    const dropZoneEl = document.getElementById('uploadDropZone');
    if (dropZoneEl) dropZoneEl.classList.remove('active');

    // Update slider max
    const slider = document.getElementById('replaySlider');
    if (slider) {
      slider.max = replayState.snapshots.length - 1;
      slider.value = 0;
    }

    // Render first frame (this will populate central store with replay history)
    renderReplayFrame(0);

    showToast(`Loaded recording: ${recording.snapshots.length.toLocaleString()} frames`);
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

  // Update central store with replay history (window of snapshots up to current index)
  centralStore.setHistory(replayState.snapshots, index);

  // Render the frame (UI reads from central store)
  render(snapshot);

  // Update replay UI
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

  // Clear UI state before switching back to live
  clearUIState();

  // Switch back to live data source in central store
  centralStore.mode = 'live';
  currentDataSource = 'live';

  if (window.HELIO_STATIC_MODE) {
    // In static mode there is no server to reconnect to — just show the upload prompt
    statusEl.textContent = 'Offline viewer · drop a .helio-recording file or click Upload';
    const dropZoneEl = document.getElementById('uploadDropZone');
    if (dropZoneEl) dropZoneEl.classList.add('active');
  } else {
    // Reconnect WebSocket
    location.reload();
  }
}

// ── Load progress helpers ─────────────────────────────────────────────────────

function formatBytes(n) {
  if (n < 1024)    return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

function showLoadProgress(label) {
  const overlay = document.getElementById('loadOverlay');
  if (!overlay) return;
  document.getElementById('loadOverlayLabel').textContent = label;
  document.getElementById('loadOverlayFill').style.width = '0%';
  document.getElementById('loadOverlayPct').textContent = '0%';
  overlay.classList.add('active');
}

function updateLoadProgress(fraction, label) {
  const pct = Math.round(Math.min(Math.max(fraction, 0), 1) * 100);
  const fill = document.getElementById('loadOverlayFill');
  const pctEl = document.getElementById('loadOverlayPct');
  if (fill) fill.style.width = pct + '%';
  if (pctEl) pctEl.textContent = pct + '%';
  if (label != null) {
    const lbl = document.getElementById('loadOverlayLabel');
    if (lbl) lbl.textContent = label;
  }
}

function hideLoadProgress() {
  const overlay = document.getElementById('loadOverlay');
  if (overlay) overlay.classList.remove('active');
}

// ── File upload handling ──────────────────────────────────────────────────────

function handleFileUpload(file) {
  if (!file) return;

  // Hide the drop zone immediately
  const dropZoneEl = document.getElementById('uploadDropZone');
  if (dropZoneEl) dropZoneEl.classList.remove('active');

  showLoadProgress(`Reading ${file.name}…`);

  const reader = new FileReader();

  reader.onprogress = (e) => {
    if (e.lengthComputable) {
      // Reserve 0–80% for reading, 80–100% for parsing
      updateLoadProgress((e.loaded / e.total) * 0.8,
        `Reading ${file.name}… (${formatBytes(e.loaded)} / ${formatBytes(e.total)})`);
    }
  };

  reader.onload = (e) => {
    updateLoadProgress(0.85, 'Parsing…');
    // Yield to the browser so the progress UI updates before JSON.parse blocks
    setTimeout(() => {
      try {
        const recording = JSON.parse(e.target.result);
        updateLoadProgress(1, 'Done');
        setTimeout(() => { hideLoadProgress(); loadRecording(recording); }, 200);
      } catch (err) {
        hideLoadProgress();
        showToast(`Failed to parse recording: ${err.message}`);
        console.error('File parse error:', err);
      }
    }, 16);
  };

  reader.onerror = () => {
    hideLoadProgress();
    showToast('Failed to read file');
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

// ── Upload menu ──────────────────────────────────────────────────────────────
const fileInput      = document.getElementById('fileUploadInput');
const btnUpload      = document.getElementById('btnUpload');
const uploadMenu     = document.getElementById('uploadMenu');
const uploadMenuWrap = document.getElementById('uploadMenuWrap');
const menuDevice     = document.getElementById('uploadMenuDevice');
const menuGithub     = document.getElementById('uploadMenuGithub');
const menuUrlRow     = document.getElementById('uploadMenuUrlRow');
const menuUrlInput   = document.getElementById('uploadMenuUrlInput');
const menuUrlGo      = document.getElementById('uploadMenuUrlGo');

function openUploadMenu() {
  if (!uploadMenu) return;
  uploadMenu.classList.add('open');
  btnUpload.setAttribute('aria-expanded', 'true');
  // Reset URL row each open unless GitHub was already selected
  menuUrlRow.classList.remove('visible');
  menuUrlInput.value = '';
}

function closeUploadMenu() {
  if (!uploadMenu) return;
  uploadMenu.classList.remove('open');
  btnUpload.setAttribute('aria-expanded', 'false');
  menuUrlRow.classList.remove('visible');
}

if (btnUpload) {
  btnUpload.addEventListener('click', (e) => {
    e.stopPropagation();
    uploadMenu.classList.contains('open') ? closeUploadMenu() : openUploadMenu();
  });
}

// "From device" option
if (menuDevice && fileInput) {
  menuDevice.addEventListener('click', () => {
    closeUploadMenu();
    fileInput.click();
  });
}

// "From GitHub URL" option — reveal the URL row
if (menuGithub) {
  menuGithub.addEventListener('click', () => {
    menuUrlRow.classList.add('visible');
    menuUrlInput.focus();
  });
}

// Load URL and update the page URL
async function submitGithubUrl() {
  const raw = menuUrlInput.value.trim();
  if (!raw) return;
  closeUploadMenu();
  await loadRemoteProfile(raw);
  // Update browser URL so the page is now shareable
  if (remoteProfileUrl) {
    const newUrl = `${location.pathname}?profile=${encodeURIComponent(remoteProfileUrl)}`;
    history.replaceState(null, '', newUrl);
  }
}

if (menuUrlGo) {
  menuUrlGo.addEventListener('click', submitGithubUrl);
}
if (menuUrlInput) {
  menuUrlInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); submitGithubUrl(); }
    if (e.key === 'Escape') closeUploadMenu();
  });
}

// File-input change handler
if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileUpload(file);
    e.target.value = '';
  });
}

// Close menu when clicking outside
document.addEventListener('click', (e) => {
  if (uploadMenuWrap && !uploadMenuWrap.contains(e.target)) closeUploadMenu();
});

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
  // (but not when clicking the URL row — it has its own pointer-events)
  dropZone.addEventListener('click', (e) => {
    const urlRow = document.getElementById('dropZoneUrlRow');
    if (urlRow && urlRow.contains(e.target)) return; // let URL row handle itself
    if (fileInput) fileInput.click();
  });

  // Drop zone GitHub URL submit
  const dropZoneUrlInput = document.getElementById('dropZoneUrlInput');
  const dropZoneUrlGo    = document.getElementById('dropZoneUrlGo');

  async function submitDropZoneUrl() {
    const raw = dropZoneUrlInput?.value.trim();
    if (!raw) return;
    dropZone.classList.remove('active');
    await loadRemoteProfile(raw);
    if (remoteProfileUrl) {
      const newUrl = `${location.pathname}?profile=${encodeURIComponent(remoteProfileUrl)}`;
      history.replaceState(null, '', newUrl);
    }
  }

  if (dropZoneUrlGo) {
    dropZoneUrlGo.addEventListener('click', (e) => { e.stopPropagation(); submitDropZoneUrl(); });
  }
  if (dropZoneUrlInput) {
    dropZoneUrlInput.addEventListener('click', (e) => e.stopPropagation());
    dropZoneUrlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.stopPropagation(); submitDropZoneUrl(); }
    });
  }
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

// ── Remote profile loading via ?profile= URL param ─────────────────────────
// Allows sharing profiles stored on GitHub (or any CORS-accessible URL) by
// encoding the raw file URL as the `profile` query parameter.
// Example:
//   ?profile=https%3A%2F%2Fraw.githubusercontent.com%2Fuser%2Frepo%2Fmain%2Frun.helio-recording
//
// GitHub blob URLs are automatically converted to raw URLs:
//   https://github.com/user/repo/blob/main/file → https://raw.githubusercontent.com/user/repo/main/file

// Track which URL the current profile was loaded from (for share-link copying)
let remoteProfileUrl = null;

/**
 * Convert a github.com blob URL to its raw.githubusercontent.com equivalent.
 * All other URLs are returned unchanged.
 */
function toRawUrl(url) {
  // https://github.com/{owner}/{repo}/blob/{ref}/{path}
  const m = url.match(/^https:\/\/github\.com\/([^/]+\/[^/]+)\/blob\/(.+)$/);
  if (m) return `https://raw.githubusercontent.com/${m[1]}/${m[2]}`;
  return url;
}

/**
 * Validate that a URL is safe to fetch: must be a valid absolute HTTPS URL.
 */
function validateProfileUrl(url) {
  let parsed;
  try { parsed = new URL(url); } catch { throw new Error('Invalid URL'); }
  if (parsed.protocol !== 'https:') throw new Error('Only HTTPS URLs are supported');
  return parsed.href;
}

async function loadRemoteProfile(rawInput) {
  let resolvedUrl;
  try {
    resolvedUrl = validateProfileUrl(toRawUrl(rawInput));
  } catch (e) {
    showToast(`Invalid profile URL: ${e.message}`);
    return;
  }

  statusEl.textContent = 'Fetching remote profile…';
  const dotEl = document.getElementById('statusDot');
  if (dotEl) dotEl.className = 'status-dot';

  showLoadProgress('Connecting…');

  try {
    const res = await fetch(resolvedUrl, { credentials: 'omit' });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

    // Stream the response body so we can track download progress
    const contentLength = +res.headers.get('Content-Length') || 0;
    const bodyReader = res.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await bodyReader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (contentLength) {
        updateLoadProgress(
          (received / contentLength) * 0.8,
          `Downloading… ${formatBytes(received)} / ${formatBytes(contentLength)}`
        );
      } else {
        // Content-Length unavailable — show bytes received, indeterminate bar
        updateLoadProgress(
          0.4,
          `Downloading… ${formatBytes(received)}`
        );
      }
    }

    // Assemble chunks into a single buffer
    updateLoadProgress(0.85, 'Parsing…');
    await new Promise(r => setTimeout(r, 16)); // let UI update before blocking parse

    let recording;
    try {
      const total = chunks.reduce((s, c) => s + c.length, 0);
      const buf = new Uint8Array(total);
      let pos = 0;
      for (const chunk of chunks) { buf.set(chunk, pos); pos += chunk.length; }
      recording = JSON.parse(new TextDecoder().decode(buf));
    } catch { throw new Error('Response is not valid JSON'); }

    updateLoadProgress(1, 'Done');
    await new Promise(r => setTimeout(r, 200));
    hideLoadProgress();

    remoteProfileUrl = resolvedUrl;
    loadRecording(recording);

    // Update the browser URL so the loaded profile is directly shareable
    const newUrl = `${location.pathname}?profile=${encodeURIComponent(resolvedUrl)}`;
    history.replaceState(null, '', newUrl);

    // Show the share-link button in the replay bar
    const shareBtn = document.getElementById('replayCopyLink');
    if (shareBtn) shareBtn.style.display = '';
  } catch (e) {
    hideLoadProgress();
    statusEl.textContent = 'Failed to load remote profile';
    showToast(`Remote profile error: ${e.message}`);
    console.error('loadRemoteProfile failed:', e);
  }
}

function copyShareLink() {
  if (!remoteProfileUrl) return;
  const shareUrl =
    `${location.origin}${location.pathname}?profile=${encodeURIComponent(remoteProfileUrl)}`;
  if (navigator.clipboard) {
    navigator.clipboard.writeText(shareUrl).then(() => showToast('Share link copied \u2713'));
  } else {
    const ta = document.createElement('textarea');
    ta.value = shareUrl;
    ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    showToast('Share link copied \u2713');
  }
}

const replayCopyLinkBtn = document.getElementById('replayCopyLink');
if (replayCopyLinkBtn) {
  replayCopyLinkBtn.addEventListener('click', copyShareLink);
}

// Hide the share button and clear remoteProfileUrl when exiting replay
document.getElementById('replayExit')?.addEventListener('click', () => {
  remoteProfileUrl = null;
  const shareBtn = document.getElementById('replayCopyLink');
  if (shareBtn) shareBtn.style.display = 'none';
  // Strip the ?profile= param from the URL
  history.replaceState(null, '', location.pathname);
}, true /* capture: runs before the existing exit listener */);

// Check for ?profile= on load (live/server mode only — static mode handles this above)
if (!window.HELIO_STATIC_MODE) {
  const profileParam = new URLSearchParams(location.search).get('profile');
  if (profileParam) {
    requestAnimationFrame(() => loadRemoteProfile(profileParam));
  }
}

// ── Export for debugging and cross-module access ─────────────────────────────
window.centralStore = centralStore; // Export central store globally for perf.js and other modules
window.helioPortal = {
  getLastSnapshot: () => lastSnapshot,
  getGraphInstance: () => cyInstance,
  recording: recordingState,
  replay: replayState,
  store: centralStore, // Access central store
};
