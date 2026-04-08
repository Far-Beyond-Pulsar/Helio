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
  try {
    const data = JSON.parse(evt.data);
    if (Array.isArray(data)) {
      // Batch of snapshots sent at once; process them in order
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

// ── Search functionality (optional - can be added later) ─────────────────────
// Search is currently removed from the new design to focus on the graph
// Can be added back as a modal or overlay feature if needed

// ── Export for debugging ──────────────────────────────────────────────────────
window.helioPortal = {
  getLastSnapshot: () => lastSnapshot,
  getGraphInstance: () => cyInstance,
};
