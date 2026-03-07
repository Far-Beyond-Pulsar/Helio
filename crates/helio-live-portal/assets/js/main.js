// main.js - entry point for portal frontend, handles websocket and updates
import { renderNodeGraph } from './nodeGraph.js';

const statusEl = document.getElementById('status');
const frameEl = document.getElementById('frame');
const frameMsEl = document.getElementById('frameMs');
const ftfMsEl = document.getElementById('ftfMs');
const gpuTotalEl = document.getElementById('gpuTotal');
const cpuTotalEl = document.getElementById('cpuTotal');
const objCountEl = document.getElementById('objCount');
const lightCountEl = document.getElementById('lightCount');
const bbCountEl = document.getElementById('bbCount');
const rowsEl = document.getElementById('rows');
const viewTop = document.getElementById('viewTop');
const viewFront = document.getElementById('viewFront');
const viewSide = document.getElementById('viewSide');

let lastSnapshot = null;

// Ring buffer of the last 10 complete frame snapshots (scene_delta excluded
// to keep the payload readable — full scene data can be large).
const MAX_HISTORY = 10;
const frameHistory = [];
const historyCountEl = document.getElementById('historyCount');

function pushHistory(snapshot) {
  // Store a lightweight copy: omit scene_delta (can be huge) but keep everything else.
  const { scene_delta: _omit, ...rest } = snapshot;
  frameHistory.push(rest);
  if (frameHistory.length > MAX_HISTORY) frameHistory.shift();
  if (historyCountEl) historyCountEl.textContent = frameHistory.length;
}

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
    showToast('Copied stats + graph to clipboard ✓');
    if (btn) {
      const orig = btn.querySelector('.btn-label') ? btn.querySelector('.btn-label').textContent : btn.textContent;
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
const btnCopyHistory = document.getElementById('btnCopyHistory');

if (btnCopyFrame) {
  btnCopyFrame.addEventListener('click', () => {
    if (!lastSnapshot) return;
    const { scene_delta: _omit, ...rest } = lastSnapshot;
    copyJson(rest, btnCopyFrame);
  });
}

if (btnCopyHistory) {
  btnCopyHistory.addEventListener('click', () => {
    if (!frameHistory.length) return;
    copyJson(frameHistory, btnCopyHistory);
  });
}

// ── render ────────────────────────────────────────────────────────────────────

let cyInstance = null;
const searchBox = document.getElementById('searchBox');

function render(snapshot) {
  lastSnapshot = snapshot;
  pushHistory(snapshot);

  frameEl.textContent = snapshot.frame;
  frameMsEl.textContent = snapshot.frame_time_ms.toFixed(2);
  ftfMsEl.textContent = snapshot.frame_to_frame_ms.toFixed(2);
  gpuTotalEl.textContent = snapshot.total_gpu_ms.toFixed(2);
  cpuTotalEl.textContent = snapshot.total_cpu_ms.toFixed(2);

  // draw call metrics (added 3/2026)
  const drawCallsEl = document.getElementById('drawCalls');
  const drawDetailEl = document.getElementById('drawDetail');
  if (drawCallsEl) drawCallsEl.textContent = snapshot.draw_calls.total;
  if (drawDetailEl) drawDetailEl.textContent = `opaque ${snapshot.draw_calls.opaque}, transparent ${snapshot.draw_calls.transparent}`;

  // scene counts
  if (objCountEl) objCountEl.textContent = snapshot.object_count;
  if (lightCountEl) lightCountEl.textContent = snapshot.light_count;
  if (bbCountEl) bbCountEl.textContent = snapshot.billboard_count;

  // ignore scene data for now

  const totalGpu = snapshot.total_gpu_ms || 0;
  const totalCpu = snapshot.total_cpu_ms || 0;

  rowsEl.innerHTML = '';
  for (const t of snapshot.pass_timings || []) {
    const gpuPct = totalGpu > 0 ? ((t.gpu_ms / totalGpu) * 100).toFixed(1) : '0';
    const cpuPct = totalCpu > 0 ? ((t.cpu_ms / totalCpu) * 100).toFixed(1) : '0';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${t.name}</td><td class="gpu">${t.gpu_ms.toFixed(3)}</td><td class="cpu">${cpuPct}%</td><td class="gpu">${gpuPct}%</td>`;
    rowsEl.appendChild(tr);
  }

  const filter = searchBox ? searchBox.value.toLowerCase() : '';
  cyInstance = renderNodeGraph(snapshot, filter);
  if (window.renderTimingTreeGraph) window.renderTimingTreeGraph(snapshot);
  // skip scene projections

  const dt = new Date(snapshot.timestamp_ms).toLocaleTimeString();
  statusEl.textContent = `Connected · last update ${dt}`;
}

if (searchBox) {
  searchBox.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase();
    if (lastSnapshot) {
      cyInstance = renderNodeGraph(lastSnapshot, q);
    }
  });
}

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws`);
ws.onopen = () => { statusEl.textContent = 'Connected · waiting for first frame...'; };
ws.onclose = () => { statusEl.textContent = 'Disconnected'; };
ws.onerror = () => { statusEl.textContent = 'Socket error'; };
ws.onmessage = (evt) => {
  try {
    const data = JSON.parse(evt.data);
    if (Array.isArray(data)) {
      // batch of snapshots sent at once; process them in order
      for (const snap of data) {
        render(snap);
      }
    } else {
      render(data);
    }
  } catch (e) {
    console.error('parse error', e);
  }
};
