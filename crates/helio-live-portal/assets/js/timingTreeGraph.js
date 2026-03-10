// timingTreeGraph.js — pure DOM + SVG timing tree.
// Data: snapshot.stage_timings = [{ id, name, ms, children?: [...] }, ...]

(function () {
  'use strict';

  const NW = 188, NH = 52, HG = 40, VG = 80, STEP = NW + HG;

  function msColor(ms) {
    if (ms >= 8)  return '#7f2c21';
    if (ms >= 4)  return '#5f3a27';
    if (ms >= 1)  return '#3f4a2f';
    return '#161b22';
  }

  // slot width = enough to hold this node's children side-by-side
  function slotW(node) {
    const n = (node.children || []).length;
    if (n === 0) return NW;
    return Math.max(NW, n * STEP - HG);
  }

  // draw a straight horizontal bezier from (x1,y) to (x2,y)
  function hLine(svg, x1, y, x2, color, sw) {
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', `M${x1} ${y} L${x2} ${y}`);
    p.setAttribute('fill', 'none');
    p.setAttribute('stroke', color);
    p.setAttribute('stroke-width', String(sw));
    svg.appendChild(p);
  }

  // draw a cubic drop curve from (x1,y1) down to (x2,y2)
  function dropLine(svg, x1, y1, x2, y2) {
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const my = (y1 + y2) / 2;
    p.setAttribute('d', `M${x1} ${y1} C${x1} ${my},${x2} ${my},${x2} ${y2}`);
    p.setAttribute('fill', 'none');
    p.setAttribute('stroke', '#56d364');
    p.setAttribute('stroke-width', '1.5');
    svg.appendChild(p);
  }

  function makeCard(node, x, y, dotLeft, dotRight, dotBottom) {
    const el = document.createElement('div');
    el.style.cssText = [
      'position:absolute',
      `left:${x}px`,
      `top:${y}px`,
      `width:${NW}px`,
      `height:${NH}px`,
      `background:${msColor(node.ms)}`,
      'border:1px solid #30363d',
      'border-radius:6px',
      'box-sizing:border-box',
      'padding:0 12px',
      "font-family:'JetBrains Mono',monospace",
      'color:#e6edf3',
      'display:flex',
      'flex-direction:column',
      'justify-content:center',
      'overflow:hidden',
    ].join(';');

    const title = document.createElement('div');
    title.style.cssText = 'font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
    title.textContent = node.name;

    const ms = document.createElement('div');
    ms.style.cssText = 'font-size:11px;color:#8b949e;margin-top:2px;white-space:nowrap';
    ms.textContent = node.ms.toFixed(node.ms < 1 ? 3 : 2) + ' ms';

    el.appendChild(title);
    el.appendChild(ms);

    function addDot(left, top) {
      const d = document.createElement('div');
      d.style.cssText = `position:absolute;width:8px;height:8px;border-radius:50%;background:#d29922;left:${left}px;top:${top}px`;
      el.appendChild(d);
    }
    if (dotLeft)   addDot(-5, NH/2 - 4);
    if (dotRight)  addDot(NW - 3, NH/2 - 4);
    if (dotBottom) addDot(NW/2 - 4, NH - 3);

    return el;
  }

  let _dbgFrame = 0;

  window.renderTimingTreeGraph = function (snapshot) {
    const container = document.getElementById('timingTreeGraph');
    if (!container) return;

    if (++_dbgFrame % 100 === 0) {
      console.log('[timingTree]', JSON.stringify(snapshot.stage_timings?.map(s => ({ id: s.id, ms: s.ms, kids: (s.children||[]).length }))));
    }

    const roots = snapshot.stage_timings || [];
    if (!roots.length) { container.innerHTML = ''; return; }

    // ── slot geometry ───────────────────────────────────────────────────────
    const slots = roots.map(r => slotW(r));
    const slotLeft = [];
    let cur = 0;
    for (let i = 0; i < slots.length; i++) {
      slotLeft.push(cur);
      cur += slots[i] + HG;
    }
    const totalW = cur - HG;
    const maxKids = Math.max(0, ...roots.map(r => (r.children || []).length));
    const totalH = maxKids > 0 ? NH + VG + NH + 32 : NH + 32;

    // ── reset container ─────────────────────────────────────────────────────
    container.innerHTML = '';
    container.style.position = 'relative';
    container.style.overflowX = 'auto';

    // inner wrap (scrollable content at correct size)
    const wrap = document.createElement('div');
    wrap.style.cssText = `position:relative;width:${totalW}px;height:${totalH}px`;
    container.appendChild(wrap);

    // SVG for edges behind the cards
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', totalW);
    svg.setAttribute('height', totalH);
    svg.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none';
    wrap.appendChild(svg);

    // ── place nodes and draw edges ──────────────────────────────────────────
    const childRowY = NH + VG;

    for (let i = 0; i < roots.length; i++) {
      const r      = roots[i];
      const kids   = r.children || [];
      const w      = slots[i];
      const sl     = slotLeft[i];

      // parent node: centered within its slot
      const px = sl + (w - NW) / 2;
      const py = 0;

      wrap.appendChild(makeCard(r, px, py,
        /*dotLeft*/   i > 0,
        /*dotRight*/  i < roots.length - 1,
        /*dotBottom*/ kids.length > 0
      ));

      // horizontal pipeline edge: right edge of prev card → left edge of this card
      if (i > 0) {
        const prevSl = slotLeft[i - 1];
        const prevW  = slots[i - 1];
        const prevPx = prevSl + (prevW - NW) / 2;
        hLine(svg, prevPx + NW, py + NH/2, px, '#30363d', 2);
      }

      // children
      if (kids.length > 0) {
        const parentCx = px + NW / 2;

        for (let ki = 0; ki < kids.length; ki++) {
          const kx = sl + ki * STEP;
          const ky = childRowY;

          wrap.appendChild(makeCard(kids[ki], kx, ky,
            /*dotLeft*/   ki > 0,
            /*dotRight*/  ki < kids.length - 1,
            /*dotBottom*/ false
          ));

          if (ki === 0) {
            // drop from parent bottom-center to first child top-center
            dropLine(svg, parentCx, py + NH, kx + NW/2, ky);
          } else {
            // sibling chain: right of prev → left of this
            const prevKx = sl + (ki - 1) * STEP;
            hLine(svg, prevKx + NW, ky + NH/2, kx, '#56d364', 1.5);
          }
        }
      }
    }
  };
})();
