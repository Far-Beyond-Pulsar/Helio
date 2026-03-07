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

let cyInstance = null;
const searchBox = document.getElementById('searchBox');

function render(snapshot) {
  lastSnapshot = snapshot;
  frameEl.textContent = snapshot.frame;
  frameMsEl.textContent = snapshot.frame_time_ms.toFixed(2);
  ftfMsEl.textContent = snapshot.frame_to_frame_ms.toFixed(2);
  gpuTotalEl.textContent = snapshot.total_gpu_ms.toFixed(2);
  cpuTotalEl.textContent = snapshot.total_cpu_ms.toFixed(2);

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
