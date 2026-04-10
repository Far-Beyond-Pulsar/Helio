// perf.js — Performance insight tool windows
// Provides four live tool windows driven by PortalFrameSnapshot data:
//   1. Frame Time History  — rolling sparkline of frame_time_ms / frame_to_frame_ms
//   2. GPU Pass Flamegraph — horizontal bars sorted by GPU cost, colour-coded by severity
//   3. Performance Alerts  — real-time spike / budget-overrun detection log
//   4. Draw Call Inspector — three-channel sparkline (total / opaque / transparent)
//
// Exposes: window.perfWindows = { update(snapshot), clearAlerts() }

(function () {
  'use strict';

  // ── Ring buffer ─────────────────────────────────────────────────────────────
  function Ring(max) {
    this.buf = [];
    this.max = max;
  }
  Ring.prototype.push = function (v) {
    this.buf.push(v);
    if (this.buf.length > this.max) this.buf.shift();
  };
  Ring.prototype.vals  = function () { return this.buf; };
  Ring.prototype.last  = function () { return this.buf[this.buf.length - 1] ?? 0; };
  Ring.prototype.count = function () { return this.buf.length; };

  // ── Shared state ────────────────────────────────────────────────────────────
  // NOTE: Ring buffers are DEPRECATED - now reading from window.centralStore.getHistory()
  // These are kept for backwards compatibility but should not accumulate data
  const N        = 256;          // history depth
  const frameMs  = new Ring(N);
  const f2fMs    = new Ring(N);
  const dcTotal  = new Ring(N);
  const dcOpaque = new Ring(N);
  const dcTrans  = new Ring(N);

  const passAvg = {};            // name → EMA of gpu_ms
  const ALPHA   = 0.05;          // EMA smoothing factor

  const alerts     = [];
  const MAX_ALERTS = 100;        // Limit to last 100 alerts
  let   allReplayAlerts = [];    // Pre-computed per-frame alerts for bi-directional replay seek

  let lastPassTimings = [];

  // ── Frame history crosshair state ───────────────────────────────────────────
  const fhMouse = { x: null, y: null };   // CSS-pixel coords within canvas
  let   fhListenerAttached = false;

  function wireFhMouse() {
    if (fhListenerAttached) return;
    const canvas = document.getElementById('frameHistoryCanvas');
    if (!canvas) return;
    fhListenerAttached = true;
    canvas.addEventListener('mousemove', e => {
      fhMouse.x = e.offsetX;
      fhMouse.y = e.offsetY;
      drawFrameHistory();
    });
    canvas.addEventListener('mouseleave', () => {
      fhMouse.x = null;
      fhMouse.y = null;
      drawFrameHistory();
    });
  }

  // ── CSS token cache ─────────────────────────────────────────────────────────
  let _css = null;
  function css() {
    if (_css) return _css;
    const s = getComputedStyle(document.documentElement);
    const g = n => s.getPropertyValue(n).trim();
    _css = {
      fg:        g('--fg'),
      fgMuted:   g('--fg-muted'),
      fgSubtle:  g('--fg-subtle'),
      border:    g('--border'),
      accent:    g('--accent'),
      success:   g('--success'),
      attention: g('--attention'),
      danger:    g('--danger'),
      gpu:       g('--gpu-color'),
      cpu:       g('--cpu-color'),
      canvas:    g('--canvas'),
      canvasInset: g('--canvas-inset'),
    };
    return _css;
  }

  // ── Visibility guard ────────────────────────────────────────────────────────
  function isActive(el) {
    return el && el.closest('.modal-container.active');
  }

  // ── Canvas helpers ──────────────────────────────────────────────────────────
  // For position:absolute fill canvases — sizes to parent container
  function fitCanvas(canvas) {
    const dpr  = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    if (W < 1 || H < 1) return null;
    // Only reallocate buffer on actual resize (expensive)
    const pw = Math.ceil(W * dpr), ph = Math.ceil(H * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width  = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    return { ctx, W, H };
  }

  // For scrollable canvases with dynamic height — sizes to parent width
  function fitScrollCanvas(canvas, desiredH) {
    const dpr  = window.devicePixelRatio || 1;
    const par  = canvas.parentElement;
    const W    = par ? par.clientWidth : 0;
    if (W < 1) return null;
    canvas.style.height = desiredH + 'px';
    const pw = Math.ceil(W * dpr), ph = Math.ceil(desiredH * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width  = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, desiredH);
    return { ctx, W, H: desiredH };
  }

  // Rounded-rect path
  function rRect(ctx, x, y, w, h, r) {
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(x, y, w, h, r);
    } else {
      r = Math.min(r, w / 2, h / 2);
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }
  }

  // Polyline helper — vals spans the full width W regardless of buf length
  function polyline(ctx, vals, W, H, maxV, yTop, segH) {
    const n = vals.length;
    if (n < 2) return;
    vals.forEach((v, i) => {
      const x = (i / (n - 1)) * W;
      const y = yTop + segH - Math.max(0, (v / maxV)) * segH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  1.  Frame Time History
  // ════════════════════════════════════════════════════════════════════════════
  const BUDGET    = 16.667;
  const BUDGET2X  = 33.333;

  function drawFrameHistory() {
    const canvas = document.getElementById('frameHistoryCanvas');
    if (!isActive(canvas)) return;
    wireFhMouse();
    const r = fitCanvas(canvas);
    if (!r) return;
    const { ctx, W, H } = r;
    const c  = css();

    const fmv  = frameMs.vals();
    const f2fv = f2fMs.vals();
    if (fmv.length < 2) {
      ctx.fillStyle = c.fgSubtle;
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data…', W / 2, H / 2);
      return;
    }

    const maxV = Math.max(BUDGET2X * 1.5, ...fmv, ...f2fv);

    // ── budget reference lines ──
    [[BUDGET, 'rgba(63,185,80,0.3)', '16ms'], [BUDGET2X, 'rgba(248,81,73,0.3)', '33ms']].forEach(([ms, col, lbl]) => {
      const y = H - (ms / maxV) * H;
      ctx.strokeStyle = col;
      ctx.lineWidth   = 1;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(W, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.font      = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(lbl, 4, y - 3);
    });

    // ── spike column fills ──
    ctx.fillStyle = 'rgba(248,81,73,0.09)';
    fmv.forEach((v, i) => {
      if (v > BUDGET2X) {
        const n = fmv.length;
        const x = (i / (n - 1)) * W;
        ctx.fillRect(x - 1, 0, 3, H);
      }
    });

    // ── gradient fill under frame_time line ──
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(240,136,62,0.22)');
    grad.addColorStop(1, 'rgba(240,136,62,0.0)');
    ctx.fillStyle = grad;
    const n = fmv.length;
    ctx.beginPath();
    ctx.moveTo(0, H);
    fmv.forEach((v, i) => {
      const x = (i / (n - 1)) * W;
      const y = H - (v / maxV) * H;
      ctx.lineTo(x, y);
    });
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fill();

    // ── frame-to-frame line (subtle) ──
    const nf = f2fv.length;
    ctx.beginPath();
    ctx.strokeStyle = c.fgSubtle;
    ctx.lineWidth   = 1;
    ctx.lineJoin    = 'round';
    f2fv.forEach((v, i) => {
      const x = (i / (nf - 1)) * W;
      const y = H - (v / maxV) * H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // ── frame_time line ──
    ctx.beginPath();
    ctx.strokeStyle = c.gpu;
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = 'round';
    fmv.forEach((v, i) => {
      const x = (i / (n - 1)) * W;
      const y = H - (v / maxV) * H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // ── current-value label ──
    const last  = frameMs.last();
    const fps   = last > 0 ? Math.round(1000 / last) : 0;
    const label = `${last.toFixed(2)} ms  |  ${fps} fps`;
    const col   = last > BUDGET2X ? c.danger : last > BUDGET ? c.attention : c.success;
    ctx.fillStyle = col;
    ctx.font      = 'bold 12px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(label, W - 8, 20);
    ctx.textAlign = 'left';

    // ── mouse crosshair ──
    if (fhMouse.x !== null) {
      const mx = Math.max(0, Math.min(W, fhMouse.x));

      // Snap to nearest data sample
      const idx     = Math.round((mx / W) * (n - 1));
      const clampedIdx = Math.max(0, Math.min(n - 1, idx));
      const snapVal = fmv[clampedIdx] ?? 0;
      const snapX   = (clampedIdx / (n - 1)) * W;
      const snapY   = H - (snapVal / maxV) * H;

      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.lineWidth   = 1;
      ctx.strokeStyle = 'rgba(139,148,158,0.55)';  // --fg-muted tinted

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(snapX, 0);
      ctx.lineTo(snapX, H);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(0, snapY);
      ctx.lineTo(W, snapY);
      ctx.stroke();

      ctx.setLineDash([]);

      // Intersection dot
      const dotCol = snapVal > BUDGET2X ? c.danger : snapVal > BUDGET ? c.attention : c.success;
      ctx.beginPath();
      ctx.arc(snapX, snapY, 4, 0, Math.PI * 2);
      ctx.fillStyle   = dotCol;
      ctx.fill();
      ctx.strokeStyle = 'rgba(13,17,23,0.8)';
      ctx.lineWidth   = 1.5;
      ctx.stroke();

      // Value tooltip
      const tipMs  = snapVal.toFixed(2);
      const tipFps = snapVal > 0 ? Math.round(1000 / snapVal) : 0;
      const tipTxt = `${tipMs} ms  ·  ${tipFps} fps`;
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      const tw  = ctx.measureText(tipTxt).width;
      const PAD = 6;
      const th  = 18;
      let tx = snapX + 10;
      let ty = snapY - th - 6;
      if (tx + tw + PAD * 2 > W) tx = snapX - tw - PAD * 2 - 10;
      if (ty < 0) ty = snapY + 6;

      ctx.fillStyle = 'rgba(22,27,34,0.92)';
      ctx.beginPath();
      ctx.roundRect(tx - PAD, ty - 1, tw + PAD * 2, th, 4);
      ctx.fill();
      ctx.strokeStyle = dotCol;
      ctx.lineWidth   = 1;
      ctx.stroke();

      ctx.fillStyle   = dotCol;
      ctx.textAlign   = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(tipTxt, tx, ty + th / 2);
      ctx.textBaseline = 'alphabetic';

      ctx.restore();
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  2.  GPU Pass Flamegraph
  // ════════════════════════════════════════════════════════════════════════════
  const FG_ROW_H = 24;
  const FG_GAP   = 3;
  const FG_PAD   = 6;

  function passColor(gpuMs) {
    if (gpuMs < 0.5)  return '#3fb950';
    if (gpuMs < 2.0)  return '#56d364';
    if (gpuMs < 5.0)  return '#d29922';
    if (gpuMs < 10.0) return '#f0883e';
    return '#f85149';
  }

  function drawFlamegraph() {
    const canvas = document.getElementById('flamegraphCanvas');
    if (!isActive(canvas)) return;

    const passes = lastPassTimings.slice().sort((a, b) => b.gpu_ms - a.gpu_ms);
    if (!passes.length) return;

    const contentH = FG_PAD + passes.length * (FG_ROW_H + FG_GAP) + FG_PAD;
    const r = fitScrollCanvas(canvas, contentH);
    if (!r) return;
    const { ctx, W, H } = r;
    const c = css();

    const totalGpu = passes.reduce((s, p) => s + p.gpu_ms, 0) || 1;
    const LABEL_W  = 168;
    const BAR_X    = LABEL_W + 10;
    const BAR_W    = Math.max(W - BAR_X - 10, 10);

    ctx.textBaseline = 'middle';
    ctx.font         = '11px JetBrains Mono, monospace';

    // Header
    ctx.fillStyle = c.fgSubtle;
    ctx.font      = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`Σ GPU: ${totalGpu.toFixed(2)} ms`, W - FG_PAD, FG_PAD + FG_ROW_H / 4);
    ctx.textAlign = 'left';
    ctx.font      = '11px JetBrains Mono, monospace';

    passes.forEach((p, i) => {
      const y   = FG_PAD + i * (FG_ROW_H + FG_GAP);
      const pct = p.gpu_ms / totalGpu;
      const bw  = Math.max(pct * BAR_W, 2);
      const col = passColor(p.gpu_ms);

      // Bar track
      ctx.fillStyle   = 'rgba(255,255,255,0.03)';
      ctx.beginPath();
      rRect(ctx, BAR_X, y, BAR_W, FG_ROW_H, 3);
      ctx.fill();

      // Bar fill
      ctx.globalAlpha = 0.88;
      ctx.fillStyle   = col;
      ctx.beginPath();
      rRect(ctx, BAR_X, y, bw, FG_ROW_H, 3);
      ctx.fill();
      ctx.globalAlpha = 1;

      // Pass name
      const truncN = p.name.length > 22 ? p.name.slice(0, 20) + '…' : p.name;
      ctx.fillStyle    = c.fg;
      ctx.textAlign    = 'left';
      ctx.fillText(truncN, FG_PAD, y + FG_ROW_H / 2);

      // Stats right of bar
      const statsX = BAR_X + bw + 7;
      if (statsX + 130 < W) {
        ctx.fillStyle = c.fgMuted;
        ctx.fillText(`${p.gpu_ms.toFixed(2)} ms  ${(pct * 100).toFixed(1)}%`, statsX, y + FG_ROW_H / 2);
      }

      // cpu hint inside bar if there's room  
      if (bw > 60 && p.cpu_ms != null) {
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.font      = '9px JetBrains Mono, monospace';
        ctx.fillText(`cpu ${p.cpu_ms.toFixed(2)}`, BAR_X + 5, y + FG_ROW_H / 2);
        ctx.font      = '11px JetBrains Mono, monospace';
      }
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  3.  Performance Alerts
  // ════════════════════════════════════════════════════════════════════════════
  const ALERT_PASS_THRESHOLD = 2.0;   // ms — flag any pass heavier than this when spiking
  const ALERT_SPIKE_FACTOR   = 2.5;   // × EMA average
  const ALERT_FRAME_BUDGET   = BUDGET2X;

  function severityOf(gpuMs, avg) {
    if (gpuMs > ALERT_FRAME_BUDGET)          return 'critical';
    if (avg > 0 && gpuMs > avg * 3)          return 'critical';
    return 'warning';
  }

  function checkAlerts(snapshot) {
    const now  = new Date().toLocaleTimeString();
    const newA = [];

    // Frame budget overrun
    if (snapshot.frame_time_ms > ALERT_FRAME_BUDGET) {
      newA.push({
        time: now, level: 'critical',
        msg: `Frame budget overrun: ${snapshot.frame_time_ms.toFixed(2)} ms  (${(snapshot.frame_time_ms / BUDGET * 100).toFixed(0)}% of 60fps budget)`,
      });
    }

    // Per-pass spikes vs rolling EMA
    for (const p of (snapshot.pass_timings || [])) {
      const prev = passAvg[p.name];
      // Only alert once the EMA is warm (a few frames of data)
      const warm = prev !== undefined && prev > 0.05;
      const spike = warm && p.gpu_ms > prev * ALERT_SPIKE_FACTOR;
      if (spike) {
        newA.push({
          time: now,
          level: severityOf(p.gpu_ms, prev),
          msg: `${p.name}: ${p.gpu_ms.toFixed(2)} ms GPU   (${(p.gpu_ms / prev).toFixed(1)}× avg of ${prev.toFixed(2)} ms)`,
        });
      }
      // Update EMA
      passAvg[p.name] = prev === undefined ? p.gpu_ms : prev + ALPHA * (p.gpu_ms - prev);
    }

    if (newA.length) {
      alerts.unshift(...newA);
      if (alerts.length > MAX_ALERTS) alerts.length = MAX_ALERTS;
      renderAlerts();
    }

    const badge = document.getElementById('alertBadge');
    if (badge) badge.textContent = alerts.length > 0 ? (alerts.length > 99 ? '99+' : alerts.length) : '';
  }

  function renderAlerts() {
    const list    = document.getElementById('alertList');
    const countEl = document.getElementById('alertCount');
    if (countEl) countEl.textContent = `${alerts.length} alert${alerts.length !== 1 ? 's' : ''}`;
    if (!list) return;
    list.innerHTML = alerts.map(a =>
      `<div class="alert-row alert-${a.level}">` +
        `<span class="alert-time">${a.time}</span>` +
        `<span class="alert-msg">${a.msg}</span>` +
      `</div>`
    ).join('');
  }

  function clearAlerts() {
    alerts.length = 0;
    renderAlerts();
    const badge = document.getElementById('alertBadge');
    if (badge) badge.textContent = '';
  }

  // Pre-compute every alert that would fire across the full recording.
  // Called once when a recording is loaded so that replay seek is O(1) and
  // works correctly in both forward and backward directions.
  function precomputeAlerts(snapshots) {
    allReplayAlerts = [];
    // Use a fresh EMA table — do NOT pollute the live `passAvg` state
    const localAvg = {};

    snapshots.forEach((snapshot, frameIndex) => {
      const ts = snapshot.timestamp_ms
        ? new Date(snapshot.timestamp_ms).toLocaleTimeString()
        : `Frame ${frameIndex + 1}`;

      // Frame budget overrun
      if (snapshot.frame_time_ms > ALERT_FRAME_BUDGET) {
        allReplayAlerts.push({
          frameIndex, time: ts, level: 'critical',
          msg: `Frame budget overrun: ${snapshot.frame_time_ms.toFixed(2)} ms  ` +
               `(${(snapshot.frame_time_ms / BUDGET * 100).toFixed(0)}% of 60fps budget)`,
        });
      }

      // Per-pass spikes vs rolling EMA
      for (const p of (snapshot.pass_timings || [])) {
        const prev  = localAvg[p.name];
        const warm  = prev !== undefined && prev > 0.05;
        const spike = warm && p.gpu_ms > prev * ALERT_SPIKE_FACTOR;
        if (spike) {
          allReplayAlerts.push({
            frameIndex, time: ts,
            level: severityOf(p.gpu_ms, prev),
            msg: `${p.name}: ${p.gpu_ms.toFixed(2)} ms GPU   ` +
                 `(${(p.gpu_ms / prev).toFixed(1)}\u00d7 avg of ${prev.toFixed(2)} ms)`,
          });
        }
        localAvg[p.name] = prev === undefined ? p.gpu_ms : prev + ALPHA * (p.gpu_ms - prev);
      }
    });
  }

  // Show only alerts whose frameIndex <= currentIndex (newest frame first).
  // Called by main.js on every renderReplayFrame so rewind clears future alerts.
  function syncReplayAlerts(currentIndex) {
    // Filter and reverse so the most-recently-triggered alerts appear at the top
    const visible = [];
    for (let i = allReplayAlerts.length - 1; i >= 0; i--) {
      if (allReplayAlerts[i].frameIndex <= currentIndex) visible.push(allReplayAlerts[i]);
      if (visible.length >= MAX_ALERTS) break;
    }

    const list    = document.getElementById('alertList');
    const countEl = document.getElementById('alertCount');
    const badge   = document.getElementById('alertBadge');

    if (countEl) countEl.textContent = `${visible.length} alert${visible.length !== 1 ? 's' : ''}`;
    if (badge)   badge.textContent   = visible.length > 0 ? (visible.length > 99 ? '99+' : visible.length) : '';
    if (!list)   return;

    list.innerHTML = visible.map(a =>
      `<div class="alert-row alert-${a.level}">` +
        `<span class="alert-time">${a.time}</span>` +
        `<span class="alert-msg">${a.msg}</span>` +
      `</div>`
    ).join('');
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  4.  Draw Call Inspector
  // ════════════════════════════════════════════════════════════════════════════
  const DC_CHANNELS = [
    { ring: null, label: 'Total',       hex: '#388bfd', fillA: 0.18 },
    { ring: null, label: 'Opaque',      hex: '#3fb950', fillA: 0.15 },
    { ring: null, label: 'Transparent', hex: '#f0883e', fillA: 0.15 },
  ];

  function drawDrawCalls() {
    const canvas = document.getElementById('drawCallCanvas');
    if (!isActive(canvas)) return;
    const r = fitCanvas(canvas);
    if (!r) return;
    const { ctx, W, H } = r;
    const c = css();

    const tv = dcTotal.vals();
    const ov = dcOpaque.vals();
    const rv = dcTrans.vals();
    if (tv.length < 2) {
      ctx.fillStyle = c.fgSubtle;
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data…', W / 2, H / 2);
      return;
    }

    const SEG  = H / 3;
    const maxV = Math.max(...tv, 1);
    const PAD  = 4;

    const channels = [
      { vals: tv, hex: '#388bfd', label: `Total: ${dcTotal.last()}` },
      { vals: ov, hex: '#3fb950', label: `Opaque: ${dcOpaque.last()}` },
      { vals: rv, hex: '#f0883e', label: `Transparent: ${dcTrans.last()}` },
    ];

    // Dividers
    ctx.strokeStyle = c.border;
    ctx.lineWidth   = 1;
    ctx.setLineDash([2, 4]);
    [SEG, SEG * 2].forEach(y => {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    });
    ctx.setLineDash([]);

    channels.forEach(({ vals, hex, label }, ci) => {
      const yTop = ci * SEG;
      const segH = SEG - 1;
      const segN = vals.length;

      // Parse hex to rgba
      const rr = parseInt(hex.slice(1, 3), 16);
      const gg = parseInt(hex.slice(3, 5), 16);
      const bb = parseInt(hex.slice(5, 7), 16);

      // Gradient fill under the line
      const grad = ctx.createLinearGradient(0, yTop, 0, yTop + segH);
      grad.addColorStop(0,   `rgba(${rr},${gg},${bb},0.20)`);
      grad.addColorStop(1,   `rgba(${rr},${gg},${bb},0.00)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(0, yTop + segH);
      polyline(ctx, vals, W, segH, maxV, yTop, segH - PAD);
      ctx.lineTo(W, yTop + segH);
      ctx.closePath();
      ctx.fill();

      // Line
      ctx.strokeStyle = hex;
      ctx.lineWidth   = 1.5;
      ctx.lineJoin    = 'round';
      ctx.beginPath();
      polyline(ctx, vals, W, segH, maxV, yTop, segH - PAD);
      ctx.stroke();

      // Label
      ctx.fillStyle    = hex;
      ctx.font         = 'bold 11px JetBrains Mono, monospace';
      ctx.textBaseline = 'top';
      ctx.textAlign    = 'left';
      ctx.fillText(label, 8, yTop + 5);
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  Main update — called from main.js render()
  // ════════════════════════════════════════════════════════════════════════════
  function update(snapshot) {
    // CRITICAL: Read from central store instead of accumulating our own data
    // This fixes wonky rewind behavior in replay mode
    const history = window.centralStore ? window.centralStore.getHistory() : [];

    // Rebuild ring buffers from central store (do NOT accumulate!)
    frameMs.buf = history.map(s => s.frame_time_ms || 0);
    f2fMs.buf = history.map(s => s.frame_to_frame_ms || 0);
    dcTotal.buf = history.map(s => (s.draw_calls && s.draw_calls.total) || 0);
    dcOpaque.buf = history.map(s => (s.draw_calls && s.draw_calls.opaque) || 0);
    dcTrans.buf = history.map(s => (s.draw_calls && s.draw_calls.transparent) || 0);

    lastPassTimings = snapshot.pass_timings || [];

    // Alerts first (also updates passAvg EMA) - only in live mode
    if (!window.centralStore || window.centralStore.mode === 'live') {
      checkAlerts(snapshot);
      // Limit alerts to MAX_ALERTS
      if (alerts.length > MAX_ALERTS) {
        alerts.splice(0, alerts.length - MAX_ALERTS);
      }
    }

    // Charts
    drawFrameHistory();
    drawFlamegraph();
    drawDrawCalls();
  }

  // Clear all accumulated state (for replay mode switch)
  function clear() {
    // Clear ring buffers
    frameMs.buf = [];
    f2fMs.buf = [];
    dcTotal.buf = [];
    dcOpaque.buf = [];
    dcTrans.buf = [];

    // Clear pass averages
    Object.keys(passAvg).forEach(key => delete passAvg[key]);

    // Clear alerts
    alerts.length = 0;
    allReplayAlerts = [];
    renderAlerts();

    // Clear last pass timings
    lastPassTimings = [];

    console.log('Perf windows state cleared');
  }

  window.perfWindows = { update, clearAlerts, clear, precomputeAlerts, syncReplayAlerts };
})();
