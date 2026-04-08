// helioGraph.js — shared ReactFlow graph factory used by both pipeline and timing graphs.
// Must be loaded as a plain <script> after React/ReactDOM/ReactFlow UMD bundles.
// Exposes: window.HelioGraph = { createGraph, NODE_W, NODE_H, H_GAP, V_GAP, BULGE_R }
(function () {
  'use strict';

  const RF             = window.ReactFlow || {};
  const React          = window.React;
  const ReactDOM       = window.ReactDOM;
  const ReactFlowComp  = RF.ReactFlow;
  const Handle         = RF.Handle;
  const Background     = RF.Background;
  const Controls       = RF.Controls;
  const Position             = RF.Position || { Left: 'left', Right: 'right', Top: 'top', Bottom: 'bottom' };
  // Safe hook — falls back to a no-op when running outside ReactFlow context or on older builds
  const useUpdateNodeInternals = RF.useUpdateNodeInternals || (() => () => {});

  // ── layout constants ────────────────────────────────────────────────────────
  const NODE_W      = 188;
  const NODE_H      = 52;
  const H_GAP       = 40;
  const V_GAP       = 80;
  const BULGE_R     = 11;
  const BAR_CHART_H = 76;  // height of the expanded metrics panel
  const MAX_HISTORY = 24;  // max samples retained per node

  // ── warning tooltip constants ───────────────────────────────────────────────
  const TOOLTIP_W  = 162;
  const TOOLTIP_PX = 10;
  const TOOLTIP_PY = 8;
  const TOOLTIP_LH = 15;
  const TOOLTIP_CR = 6;
  const CHEV_BASE  = 14;
  const CHEV_H     = 9;

  // ── comet edge animation (inject once) ─────────────────────────────────────
  const COMET_MS = 2200;
  if (!document.getElementById('helio-edge-anim')) {
    const s = document.createElement('style');
    s.id = 'helio-edge-anim';
    s.textContent = '@keyframes helioCometAnim { from { stroke-dashoffset:1.15 } to { stroke-dashoffset:0 } }';
    document.head.appendChild(s);
  }

  // ── GradientEdge — shared, stateless ───────────────────────────────────────
  function GradientEdge({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {} }) {
    const gsp = RF.getSmoothStepPath;
    if (!gsp) return null;
    const [edgePath] = gsp({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition, borderRadius: 12 });
    const color = style.stroke || '#30363d';
    const sw    = style.strokeWidth || 2;
    const phaseRef = React.useRef(null);
    if (phaseRef.current === null) phaseRef.current = `${-(performance.now() % COMET_MS)}ms`;
    return React.createElement('g', null,
      React.createElement('path', { d: edgePath, fill: 'none', stroke: color, strokeWidth: sw, strokeOpacity: 0.15 }),
      React.createElement('path', {
        d: edgePath, fill: 'none', pathLength: '1',
        stroke: color, strokeWidth: sw + 1, strokeLinecap: 'round', strokeDasharray: '0.15 1',
        style: {
          animationName: 'helioCometAnim', animationDuration: `${COMET_MS}ms`,
          animationTimingFunction: 'linear', animationIterationCount: 'infinite',
          animationFillMode: 'none', animationDelay: phaseRef.current,
          filter: `drop-shadow(0 0 5px ${color}) drop-shadow(0 0 2px ${color})`,
        },
      }),
    );
  }

  const edgeTypes = { gradient: GradientEdge };

  // ── createGraph(containerId) ────────────────────────────────────────────────
  // Returns { render(nodes, edges) }
  // Each call creates isolated state so two graphs can coexist on one page.
  function createGraph(containerId) {
    const _nodeData    = new Map();
    const _nodeSubs    = new Map();
    const _warnSubs    = new Map();
    const _historyData = new Map(); // id → number[]  (ring-buffer of timing values)
    let reactRoot    = null;
    let structureKey = '';
    let reactFlowInstance = null;

    function _pushHistory(id, timeStr) {
      const m = (timeStr || '').match(/[\d.]+/);
      if (!m) return;
      const val = parseFloat(m[0]);
      if (isNaN(val)) return;
      const arr = _historyData.get(id) || [];
      arr.push(val);
      if (arr.length > MAX_HISTORY) arr.shift();
      _historyData.set(id, arr);
    }

    // ── BarPanel — metrics history chart ───────────────────────────────────
    function BarPanel({ id, data }) {
      const ce      = React.createElement;
      const history = _historyData.get(id) || [];
      const accent  = (data.kind === 'pass') ? '#388bfd' : '#d29922';
      const padX    = 8;
      const barsW   = NODE_W - padX * 2;
      const barsH   = 36;

      const inner = history.length === 0
        ? ce('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' } },
            ce('span', { style: { fontSize: '10px', color: '#484f58' } }, 'no data yet')
          )
        : ce(React.Fragment, null,
            ce('div', { style: { fontSize: '10px', color: '#484f58', marginBottom: '3px', letterSpacing: '0.05em', textTransform: 'uppercase' } }, 'History'),
            ce('svg', { width: barsW, height: barsH, style: { display: 'block' } },
              ce('line', { x1: 0, y1: barsH * 0.5, x2: barsW, y2: barsH * 0.5, stroke: '#21262d', strokeWidth: 1 }),
              ...(() => {
                const max  = Math.max(...history, 0.001);
                const n    = history.length;
                const barW = (barsW - (n - 1)) / n;
                return history.map((v, i) =>
                  ce('rect', {
                    key: i, rx: 1.5,
                    x: i * (Math.max(1, barW) + 1),
                    y: barsH - Math.max(2, (v / max) * barsH),
                    width: Math.max(1, barW),
                    height: Math.max(2, (v / max) * barsH),
                    fill: accent,
                    opacity: 0.25 + 0.75 * ((i + 1) / n),
                  })
                );
              })(),
            ),
            ce('div', { style: { display: 'flex', justifyContent: 'space-between', marginTop: '3px', fontSize: '10px', color: '#484f58' } },
              ce('span', null, '0'),
              ce('span', null, `${Math.max(...history, 0.001).toFixed(2)} ms`),
            ),
          );

      return ce('div', {
        style: {
          position: 'absolute', top: `${NODE_H}px`, left: 0,
          width: `${NODE_W}px`, height: `${BAR_CHART_H}px`,
          background: 'transparent',
          padding: `4px ${padX}px`, boxSizing: 'border-box', overflow: 'hidden',
          fontFamily: 'JetBrains Mono, monospace',
        },
      }, inner);
    }

    // ── WarningTooltip ──────────────────────────────────────────────────────
    function WarningTooltip({ id }) {
      const [warning, setWarning] = React.useState(() => (_nodeData.get(id) || {}).warning || null);
      React.useEffect(() => {
        _warnSubs.set(id, setWarning);
        return () => { _warnSubs.delete(id); };
      }, [id]);
      const ce    = React.createElement;
      const stats = warning ? (warning.stats || []) : [];
      const TW    = TOOLTIP_W;
      const TH    = TOOLTIP_PY * 2 + TOOLTIP_LH + stats.length * TOOLTIP_LH + (stats.length ? 3 : 0);
      const R     = TOOLTIP_CR;
      const mx    = TW / 2;
      const d     = [
        `M ${R} 0`, `L ${TW - R} 0`, `Q ${TW} 0 ${TW} ${R}`,
        `L ${TW} ${TH - R}`, `Q ${TW} ${TH} ${TW - R} ${TH}`,
        `L ${mx + CHEV_BASE / 2} ${TH}`, `L ${mx} ${TH + CHEV_H}`,
        `L ${mx - CHEV_BASE / 2} ${TH}`, `L ${R} ${TH}`,
        `Q 0 ${TH} 0 ${TH - R}`, `L 0 ${R}`, `Q 0 0 ${R} 0`, 'Z',
      ].join(' ');
      return ce('div', {
        style: {
          position: 'absolute', top: `${-(TH + CHEV_H + 6)}px`, left: `${(NODE_W - TW) / 2}px`,
          width: `${TW}px`, height: `${TH + CHEV_H}px`,
          overflow: 'visible', pointerEvents: 'none', zIndex: 20,
          display: warning ? 'block' : 'none',
        },
      },
        ce('svg', { style: { position: 'absolute', top: 0, left: 0, overflow: 'visible' }, width: TW, height: TH + CHEV_H },
          ce('path', { d, fill: 'rgba(210,153,34,0.10)', stroke: 'rgba(210,153,34,0.78)', strokeWidth: 1 }),
        ),
        ce('div', {
          style: {
            position: 'absolute', top: `${TOOLTIP_PY}px`, left: `${TOOLTIP_PX}px`,
            width: `${TW - TOOLTIP_PX * 2}px`,
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '11px', lineHeight: `${TOOLTIP_LH}px`, color: '#e3b341', userSelect: 'none',
          },
        },
          ce('div', { style: { fontWeight: 700 } }, warning ? `\u26a0 ${warning.message}` : ''),
          ...stats.map((s, i) => ce('div', { key: i, style: { color: '#d29922', opacity: 0.88 } }, s)),
        ),
      );
    }

    // ── PipelineNode ────────────────────────────────────────────────────────
    function PipelineNode({ id, data: initialData }) {
      const [data, setData]     = React.useState(() => _nodeData.get(id) || initialData);
      const [expanded, setExpanded] = React.useState(false);
      React.useEffect(() => {
        const latest = _nodeData.get(id);
        if (latest) setData(latest);
        _nodeSubs.set(id, setData);
        return () => { _nodeSubs.delete(id); };
      }, [id]);

      const ce       = React.createElement;
      const dotColor = data.kind === 'pass' ? '#388bfd' : '#d29922';
      const { left = false, right = false, bottom = false } = data.connections || {};
      const W = NODE_W, H = NODE_H, R = 6, BR = BULGE_R, OX = BR, half = H / 2;

      const parts = [`M ${OX + R} 0`, `L ${OX + W - R} 0`, `Q ${OX + W} 0 ${OX + W} ${R}`];
      if (right) {
        parts.push(`L ${OX + W} ${half - BR}`, `A ${BR} ${BR} 0 0 1 ${OX + W} ${half + BR}`);
      }
      parts.push(`L ${OX + W} ${H - R}`, `Q ${OX + W} ${H} ${OX + W - R} ${H}`);
      if (bottom) {
        const cx = OX + W / 2;
        parts.push(`L ${cx + BR} ${H}`, `A ${BR} ${BR} 0 0 1 ${cx - BR} ${H}`);
      }
      parts.push(`L ${OX + R} ${H}`, `Q ${OX} ${H} ${OX} ${H - R}`);
      if (left) {
        parts.push(`L ${OX} ${half + BR}`, `A ${BR} ${BR} 0 0 1 ${OX} ${half - BR}`);
      }
      parts.push(`L ${OX} ${R}`, `Q ${OX} 0 ${OX + R} 0`, 'Z');

      // Expansion area: bottom-rounded rect drawn under the header card.
      // Container height stays fixed at NODE_H so ReactFlow's node measurement never
      // changes — handles and edges remain anchored to the header's midpoint.
      const EH = H + BAR_CHART_H;
      const expandPath = `M ${OX} ${H} L ${OX+W} ${H} L ${OX+W} ${EH-R} Q ${OX+W} ${EH} ${OX+W-R} ${EH} L ${OX+R} ${EH} Q ${OX} ${EH} ${OX} ${EH-R} Z`;

      return ce('div', {
        style: { position: 'relative', overflow: 'visible', width: `${W}px`, height: `${H}px`, cursor: 'pointer' },
        onClick: () => setExpanded(ex => !ex),
      },
        ce('svg', {
          style: { position: 'absolute', left: `${-BR}px`, top: 0, overflow: 'visible', pointerEvents: 'none', zIndex: 0 },
          width: W + 2 * BR, height: expanded ? EH : H,
        },
          expanded && ce('path', { d: expandPath, fill: '#0d1117', stroke: '#30363d', strokeWidth: 1 }),
          expanded && ce('line', { x1: OX, y1: H, x2: OX + W, y2: H, stroke: '#21262d', strokeWidth: 1 }),
          ce('path', { d: parts.join(' '), fill: '#161b22', stroke: '#30363d', strokeWidth: 1 }),
          left   && ce('circle', { cx: OX - 1,         cy: half,    r: 4.5, fill: dotColor }),
          right  && ce('circle', { cx: OX + W + 1,     cy: half,    r: 4.5, fill: dotColor }),
          bottom && ce('circle', { cx: OX + W / 2,     cy: H + 1,   r: 4.5, fill: '#388bfd' }),
        ),
        ce('div', {
          style: {
            position: 'absolute', top: 0, left: 0,
            zIndex: 1, width: `${W}px`, height: `${H}px`, display: 'flex',
            flexDirection: 'column', justifyContent: 'center',
            padding: `0 14px 0 ${BR + 8}px`, color: '#e6edf3',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '13px', boxSizing: 'border-box', pointerEvents: 'none',
          },
        },
          ce('div', { style: { fontWeight: 600, marginBottom: data.time ? '3px' : 0, letterSpacing: '0.01em', whiteSpace: 'nowrap' } }, data.title),
          data.time && ce('div', { style: { fontSize: '11px', color: '#8b949e', whiteSpace: 'nowrap' } }, data.time),
        ),
        Handle && ce(Handle, { type: 'target', position: Position.Left,   style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
        Handle && ce(Handle, { type: 'target', id: 'left', position: Position.Left,   style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
        Handle && ce(Handle, { type: 'source', position: Position.Right,  style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
        Handle && ce(Handle, { type: 'source', id: 'right', position: Position.Right,  style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
        Handle && ce(Handle, { type: 'source', id: 'bottom', position: Position.Bottom, style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
        ce(WarningTooltip, { id }),
        expanded && ce(BarPanel, { id, data }),
      );
    }

    const nodeTypes = { pipeline: PipelineNode, pass: PipelineNode };

    // ── Load/Save viewport state ────────────────────────────────────────────
    function loadViewport(containerId) {
      try {
        const saved = localStorage.getItem(`helio-viewport-${containerId}`);
        if (saved) {
          const viewport = JSON.parse(saved);
          return { x: viewport.x || 0, y: viewport.y || 0, zoom: viewport.zoom || 1 };
        }
      } catch (e) {
        console.warn('Failed to load viewport:', e);
      }
      return null;
    }

    function saveViewport(containerId, viewport) {
      try {
        localStorage.setItem(`helio-viewport-${containerId}`, JSON.stringify(viewport));
      } catch (e) {
        console.warn('Failed to save viewport:', e);
      }
    }

    // ── GraphInner — zero state, never re-renders ───────────────────────────
    function GraphInner({ initNodes, initEdges, containerId, onReactFlowInit }) {
      const [isFirstRender, setIsFirstRender] = React.useState(true);
      const savedViewport = React.useMemo(() => loadViewport(containerId), [containerId]);

      // Debounced viewport save to avoid spamming localStorage
      const saveTimer = React.useRef(null);
      const handleMove = React.useCallback((event, viewport) => {
        clearTimeout(saveTimer.current);
        saveTimer.current = setTimeout(() => {
          saveViewport(containerId, viewport);
        }, 300);
      }, [containerId]);

      React.useEffect(() => {
        if (isFirstRender) {
          setIsFirstRender(false);
        }
      }, [isFirstRender]);

      const handleInit = React.useCallback((reactFlowInstance) => {
        if (onReactFlowInit) {
          onReactFlowInit(reactFlowInstance);
        }
      }, [onReactFlowInit]);

      return React.createElement(ReactFlowComp, {
        nodes: initNodes, edges: initEdges,
        onNodesChange: () => {}, onEdgesChange: () => {},
        nodeTypes, edgeTypes,
        fitView: isFirstRender && !savedViewport,
        fitViewOptions: { padding: 0.15 },
        defaultViewport: savedViewport || undefined,
        onMove: handleMove,
        onInit: handleInit,
        nodesDraggable: true, nodesConnectable: false, elementsSelectable: true,
        proOptions: { hideAttribution: true },
        style: { background: '#0d1117' },
      },
        Background && React.createElement(Background, { color: '#21262d', gap: 20, style: { opacity: 0.5 } }),
        Controls   && React.createElement(Controls,   { style: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px' } }),
      );
    }

    // ── public render method ────────────────────────────────────────────────
    function render(nodes, edges) {
      const container = document.getElementById(containerId);
      if (!container) { console.warn(`HelioGraph: no #${containerId} container`); return; }
      if (!reactRoot) reactRoot = ReactDOM.createRoot(container);

      const key = nodes.map(n => n.id).join(',');
      if (key !== structureKey) {
        // Structure changed — full mount; seed history for immediate expand
        for (const n of nodes) { _nodeData.set(n.id, n.data); _pushHistory(n.id, n.data.time); }
        structureKey = key;
        const onReactFlowInit = (instance) => { reactFlowInstance = instance; };
        reactRoot.render(React.createElement(GraphInner, { initNodes: nodes, initEdges: edges, containerId, onReactFlowInit }));
      } else {
        // Data-only update — push directly to each node's state, never re-render the graph
        for (const n of nodes) {
          _nodeData.set(n.id, n.data);
          _pushHistory(n.id, n.data.time);
          const cb  = _nodeSubs.get(n.id); if (cb)  cb(n.data);
          const wcb = _warnSubs.get(n.id); if (wcb) wcb(n.data.warning || null);
        }
      }
    }

    // ── public API for controlling the graph ───────────────────────────────
    function focusNode(nodeId) {
      if (!reactFlowInstance) {
        console.warn('ReactFlow instance not initialized yet');
        return;
      }
      const node = reactFlowInstance.getNode(nodeId);
      if (!node) {
        console.warn(`Node ${nodeId} not found`);
        return;
      }

      console.log('Focusing on node:', nodeId, node);

      // Try ReactFlow's setCenter API first
      if (typeof reactFlowInstance.setCenter === 'function') {
        const x = node.position.x + NODE_W / 2;
        const y = node.position.y + NODE_H / 2;
        console.log('Using setCenter:', x, y);
        reactFlowInstance.setCenter(x, y, { zoom: 1.2, duration: 800 });
      }
      // Fallback: use setViewport directly
      else if (typeof reactFlowInstance.setViewport === 'function') {
        const x = node.position.x + NODE_W / 2;
        const y = node.position.y + NODE_H / 2;
        const viewport = reactFlowInstance.getViewport();
        console.log('Using setViewport fallback');
        reactFlowInstance.setViewport({
          x: -x * 1.2 + window.innerWidth / 2,
          y: -y * 1.2 + window.innerHeight / 2,
          zoom: 1.2,
        }, { duration: 800 });
      }
      // Last resort: instant jump
      else {
        console.warn('No viewport animation API available, checking available methods:', Object.keys(reactFlowInstance));
      }
    }

    return { render, focusNode };
  }

  window.HelioGraph = { createGraph, NODE_W, NODE_H, H_GAP, V_GAP, BULGE_R };
})();
