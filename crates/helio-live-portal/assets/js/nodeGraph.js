// nodeGraph.js – GitHub-Actions-styled pipeline graph with React Flow v11
//
// React, ReactDOM, and ReactFlow (v11) must be loaded as UMD globals before this module.

// ── resolve RF globals ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// The reactflow v11 UMD build sets window.ReactFlow to an object of named exports.
const RF = window.ReactFlow || {};
console.log('ReactFlow UMD keys:', Object.keys(RF));

// ReactFlow v11 UMD: main component is `ReactFlow` named export, NOT `.default`
const ReactFlowComp = RF.ReactFlow;
// ReactFlowProvider is NOT needed — ReactFlow itself is its own provider
const Handle        = RF.Handle;
const Background    = RF.Background;
const Controls      = RF.Controls;
const MarkerType    = RF.MarkerType   || { ArrowClosed: 'arrowclosed' };
const Position      = RF.Position     || { Left: 'left', Right: 'right', Top: 'top', Bottom: 'bottom' };

if (!ReactFlowComp)  console.error('ReactFlow component not found in window.ReactFlow – check CDN URL');

let reactRoot         = null;
let graphStructureKey = '';        // comma-joined node IDs — detects structural changes
const _nodeData       = new Map(); // id → latest node data; populated before first mount
const _nodeSubs       = new Map(); // id → setData callback registered by each PipelineNode
const _warnSubs       = new Map(); // id → setWarning callback registered by each WarningTooltip

// ── layout constants (must precede PipelineNode so the SVG path can use them) ─
const NODE_W  = 188;
const NODE_H  = 52;
const H_GAP   = 40;
const V_GAP   = 80;
const BULGE_R = 11;   // radius of the status bulge that protrudes from the left edge

// ── warning tooltip constants ─────────────────────────────────────────────────
const TOOLTIP_W  = 162;  // tooltip width px
const TOOLTIP_PX = 10;   // horizontal text padding
const TOOLTIP_PY = 8;    // vertical text padding
const TOOLTIP_LH = 15;   // line-height
const TOOLTIP_CR = 6;    // corner radius
const CHEV_BASE  = 14;   // chevron base width
const CHEV_H     = 9;    // chevron height

// Rounded-rect SVG path with a downward-pointing chevron at the bottom centre.
// Always mounted — hides via display:none when no warning so there is never a
// mount/unmount cycle that could cause a visible flicker. Subscribes to its own
// entry in _warnSubs so PipelineNode never re-renders for warning changes.
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

  const d = [
    `M ${R} 0`,
    `L ${TW - R} 0`,
    `Q ${TW} 0 ${TW} ${R}`,
    `L ${TW} ${TH - R}`,
    `Q ${TW} ${TH} ${TW - R} ${TH}`,
    `L ${mx + CHEV_BASE / 2} ${TH}`,
    `L ${mx} ${TH + CHEV_H}`,          // chevron tip — points at the flagged card
    `L ${mx - CHEV_BASE / 2} ${TH}`,
    `L ${R} ${TH}`,
    `Q 0 ${TH} 0 ${TH - R}`,
    `L 0 ${R}`,
    `Q 0 0 ${R} 0`,
    'Z',
  ].join(' ');

  return ce('div', {
    style: {
      position:      'absolute',
      top:           `${-(TH + CHEV_H + 6)}px`,
      left:          `${(NODE_W - TW) / 2}px`,
      width:         `${TW}px`,
      height:        `${TH + CHEV_H}px`,
      overflow:      'visible',
      pointerEvents: 'none',
      zIndex:        20,
      display:       warning ? 'block' : 'none', // hide in-place — never unmount
    },
  },
    ce('svg', {
      style: { position: 'absolute', top: 0, left: 0, overflow: 'visible' },
      width: TW, height: TH + CHEV_H,
    },
      ce('path', {
        d,
        fill:        'rgba(210,153,34,0.10)',
        stroke:      'rgba(210,153,34,0.78)',
        strokeWidth: 1,
      }),
    ),
    ce('div', {
      style: {
        position:   'absolute',
        top:        `${TOOLTIP_PY}px`,
        left:       `${TOOLTIP_PX}px`,
        width:      `${TW - TOOLTIP_PX * 2}px`,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", monospace',
        fontSize:   '11px',
        lineHeight: `${TOOLTIP_LH}px`,
        color:      '#e3b341',
        userSelect: 'none',
      },
    },
      ce('div', { style: { fontWeight: 700 } }, warning ? `\u26a0 ${warning.message}` : ''),
      ...stats.map((s, i) =>
        ce('div', { key: i, style: { color: '#d29922', opacity: 0.88 } }, s)
      ),
    ),
  );
}

// ── anomaly detection ─────────────────────────────────────────────────────────
// Returns a map of node-id → { message, stats[] } for any node whose timing is
// a statistical outlier within its group (mean + 2σ, minimum absolute threshold).
function detectAnomalies(snapshot) {
  const anomalies = {};

  const flagOutliers = (pairs, message, unit, minMs) => {
    const vals = pairs.map(p => p.val).filter(v => v > 0);
    if (vals.length < 2) return;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std  = Math.sqrt(vals.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / vals.length);
    const thr  = Math.max(mean + 2 * std, minMs);
    for (const { id, val } of pairs) {
      if (val > thr) {
        anomalies[id] = {
          message,
          stats: [
            `${val.toFixed(2)} ${unit}`,
            `${((val / mean) * 100).toFixed(0)}% of group avg`,
          ],
        };
      }
    }
  };

  flagOutliers(
    (snapshot.stage_timings || []).map(s => ({ id: s.id, val: s.ms })),
    'High CPU time', 'ms', 2
  );

  const passes = snapshot.pass_timings || [];
  flagOutliers(passes.map(p => ({ id: p.name, val: p.gpu_ms })), 'GPU bottleneck', 'ms GPU', 1);
  // CPU flagging only for passes not already caught by GPU check
  flagOutliers(
    passes.filter(p => !anomalies[p.name]).map(p => ({ id: p.name, val: p.cpu_ms })),
    'CPU bottleneck', 'ms CPU', 1
  );

  return anomalies;
}

// ── custom node component ─────────────────────────────────────────────────────
// Card is a single SVG path: rounded-rect + optional semicircular bulges on
// left / right / bottom — one per connected handle, zero seams.
function PipelineNode({ id, data: initialData }) {
  // Each node owns its own live state, subscribed to the module-level store.
  // This means status/time updates never touch ReactFlowComp or its edges.
  const [data, setData] = React.useState(() => _nodeData.get(id) || initialData);
  React.useEffect(() => {
    const latest = _nodeData.get(id);
    if (latest) setData(latest);
    _nodeSubs.set(id, setData);
    return () => { _nodeSubs.delete(id); };
  }, [id]);

  const ce       = React.createElement;
  const dotColor = data.kind === 'pass' ? '#388bfd' : '#d29922';
  const { left = false, right = false, bottom = false } = data.connections || {};

  const W    = NODE_W;
  const H    = NODE_H;
  const R    = 6;       // card corner radius
  const BR   = BULGE_R;
  const OX   = BR;      // card left edge in SVG space (SVG overflows by BR on each side)
  const half = H / 2;

  // Build the outline path with optional bulges.
  // Going clockwise from top-left corner.
  const parts = [
    `M ${OX + R} 0`,
    `L ${OX + W - R} 0`,
    `Q ${OX + W} 0 ${OX + W} ${R}`,
  ];
  // right bulge: protrudes rightward, arc goes downward with sweep=1
  if (right) {
    parts.push(`L ${OX + W} ${half - BR}`);
    parts.push(`A ${BR} ${BR} 0 0 1 ${OX + W} ${half + BR}`);
  }
  parts.push(
    `L ${OX + W} ${H - R}`,
    `Q ${OX + W} ${H} ${OX + W - R} ${H}`,
  );
  // bottom bulge: protrudes downward, arc goes leftward with sweep=1
  if (bottom) {
    const cx = OX + W / 2;
    parts.push(`L ${cx + BR} ${H}`);
    parts.push(`A ${BR} ${BR} 0 0 1 ${cx - BR} ${H}`);
  }
  parts.push(
    `L ${OX + R} ${H}`,
    `Q ${OX} ${H} ${OX} ${H - R}`,
  );
  // left bulge: protrudes leftward, arc goes upward with sweep=1
  if (left) {
    parts.push(`L ${OX} ${half + BR}`);
    parts.push(`A ${BR} ${BR} 0 0 1 ${OX} ${half - BR}`);
  }
  parts.push(`L ${OX} ${R}`, `Q ${OX} 0 ${OX + R} 0`, 'Z');
  const d = parts.join(' ');

  // Dot cx values: same inset from the flat face of each bulge.
  // Left bulge flat face is at SVG x=OX; dot is 1px inside (rightward).
  // Right bulge flat face is at SVG x=OX+W; dot is 1px inside (leftward).
  const leftDotCx   = OX - 1;             // 1px left of card edge
  const rightDotCx  = OX + W + 1;         // 1px right of card edge
  const bottomDotCy = H + 1;              // 1px below card edge
  const bottomDotCx = OX + W / 2;

  return ce('div', {
    style: { position: 'relative', overflow: 'visible', width: `${W}px`, height: `${H}px` },
  },

    // ── single SVG: fill + border + dots ─────────────────────────────────
    ce('svg', {
      style: { position: 'absolute', left: `${-BR}px`, top: 0,
               overflow: 'visible', pointerEvents: 'none', zIndex: 0 },
      width: W + 2 * BR, height: H,
    },
      ce('path', { d, fill: '#161b22', stroke: '#30363d', strokeWidth: 1 }),
      left   && ce('circle', { cx: leftDotCx,   cy: half,        r: 4.5, fill: dotColor }),
      right  && ce('circle', { cx: rightDotCx,  cy: half,        r: 4.5, fill: dotColor }),
      bottom && ce('circle', { cx: bottomDotCx, cy: bottomDotCy, r: 4.5, fill: '#388bfd' }),
    ),

    // ── text content ──────────────────────────────────────────────────────
    ce('div', {
      style: {
        position: 'relative', zIndex: 1,
        height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center',
        padding:    `0 14px 0 ${BR + 8}px`,
        color:      '#e6edf3',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", monospace',
        fontSize:   '13px', boxSizing: 'border-box',
      }
    },
      ce('div', { style: { fontWeight: 600, marginBottom: data.time ? '3px' : 0, letterSpacing: '0.01em', whiteSpace: 'nowrap' } }, data.title),
      data.time && ce('div', { style: { fontSize: '11px', color: '#8b949e', whiteSpace: 'nowrap' } }, data.time),
    ),

    // ── ReactFlow handles (invisible — routing only, SVG is the visual) ──
    Handle && ce(Handle, { type: 'target', position: Position.Left,
      style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
    Handle && ce(Handle, { type: 'target', id: 'left', position: Position.Left,
      style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
    Handle && ce(Handle, { type: 'source', position: Position.Right,
      style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
    Handle && ce(Handle, { type: 'source', id: 'right', position: Position.Right,
      style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
    Handle && ce(Handle, { type: 'source', id: 'bottom', position: Position.Bottom,
      style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
    ce(WarningTooltip, { id }),  // always mounted — manages own visibility
  );
}

const nodeTypes = { pipeline: PipelineNode, pass: PipelineNode };

// ── animated gradient edge ────────────────────────────────────────────────────
// Uses SVG pathLength="1" to normalize all dash lengths to 0–1 of actual path,
// eliminating any dependency on getTotalLength(). CSS animation runs on the
// compositor thread and is immune to JS re-renders. animationDelay is computed
// from wall-clock time once at mount and held constant in a ref — React's
// reconciler only patches DOM properties that *change*, so animationDelay is
// never touched again, even when ReactFlow updates edge geometry.
//
// Keyframe: dashoffset 1.15 → 0 with dasharray "0.15 1" (period 1.15):
//   D=1.15 → comet at path start (0)
//   D=0.15 → comet at path end (1.0)
//   D∈[0,0.15) → comet past path end (invisible) — seamless loop back to start
if (!document.getElementById('helio-edge-anim')) {
  const s = document.createElement('style');
  s.id = 'helio-edge-anim';
  s.textContent = '@keyframes helioCometAnim { from { stroke-dashoffset: 1.15 } to { stroke-dashoffset: 0 } }';
  document.head.appendChild(s);
}

const COMET_DURATION_MS = 2200; // ms per full source-to-target pass

function GradientEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {} }) {
  const getSmoothStepPath = RF.getSmoothStepPath;
  if (!getSmoothStepPath) return null;
  const [edgePath] = getSmoothStepPath({
    sourceX, sourceY, sourcePosition,
    targetX, targetY, targetPosition,
    borderRadius: 12,
  });
  const color = style.stroke || '#30363d';
  const sw    = style.strokeWidth || 2;

  // Capture wall-clock phase exactly once per component instance.
  // React.useRef persists across re-renders of the same instance, so
  // phaseRef.current is set on first render and never overwritten.
  // Constant animationDelay → React never patches the DOM property → no reset.
  const phaseRef = React.useRef(null);
  if (phaseRef.current === null) {
    phaseRef.current = `${-(performance.now() % COMET_DURATION_MS)}ms`;
  }

  return React.createElement('g', null,
    // dim base track — always visible so the wire reads clearly
    React.createElement('path', {
      d: edgePath, fill: 'none',
      stroke: color, strokeWidth: sw, strokeOpacity: 0.15,
    }),
    // comet — CSS-animated, never touched by React after first paint
    React.createElement('path', {
      d: edgePath, fill: 'none',
      pathLength: '1',            // normalize: all dash values are fractions of path length
      stroke: color,
      strokeWidth: sw + 1,
      strokeLinecap: 'round',
      strokeDasharray: '0.15 1',  // 15% comet, 100% gap (≥ path length)
      style: {
        animationName:           'helioCometAnim',
        animationDuration:       `${COMET_DURATION_MS}ms`,
        animationTimingFunction: 'linear',
        animationIterationCount: 'infinite',
        animationFillMode:       'none',
        animationDelay:          phaseRef.current, // constant → never resets
        filter:                  `drop-shadow(0 0 5px ${color}) drop-shadow(0 0 2px ${color})`,
      },
    }),
  );
}

const edgeTypes = { gradient: GradientEdge };

// ── layout computation ────────────────────────────────────────────────────────

function computeLayout(snapshot, filter) {
  const rawNodes  = [];
  const rawEdges  = [];
  const anomalies = detectAnomalies(snapshot);

  // Top-level stages come entirely from the backend — no field names hardcoded here.
  const topSteps = (snapshot.stage_timings || []).map(s => ({
    id: s.id, label: s.name, val: s.ms,
  }));

  const match = (s) => !filter || s.toLowerCase().includes(filter);

  for (let i = 0; i < topSteps.length; i++) {
    const s = topSteps[i];
    const time = s.val.toFixed(2) + ' ms';
    if (!match(s.label) && !match(time)) continue;
    rawNodes.push({
      id:       s.id,
      type:     'pipeline',
      position: { x: i * (NODE_W + H_GAP), y: 0 },
      data:     { title: s.label, time, kind: 'top', warning: anomalies[s.id] },
    });
    if (i > 0) {
      rawEdges.push({
        id:     `e_${topSteps[i-1].id}_${s.id}`,
        source: topSteps[i-1].id,
        target: s.id,
        type:   'gradient',
        style:  { stroke: '#30363d', strokeWidth: 2 },
      });
    }
  }

  const pipelineStageId = snapshot.pipeline_stage_id || null;
  const passOrder = snapshot.pipeline_order || [];
  if (pipelineStageId && passOrder.length > 0) {
    const pipelineNode   = rawNodes.find(n => n.id === pipelineStageId);
    const pipelineCenter = pipelineNode ? pipelineNode.position.x + NODE_W / 2 : 0;
    const startX         = pipelineCenter;
    const visiblePasses  = [];

    for (let i = 0; i < passOrder.length; i++) {
      const name = passOrder[i];
      const t    = (snapshot.pass_timings || []).find(p => p.name === name) || { gpu_ms: 0, cpu_ms: 0 };
      const time = `GPU ${t.gpu_ms.toFixed(2)} ms  ·  CPU ${t.cpu_ms.toFixed(2)} ms`;
      if (!match(name) && !match(time)) continue;

      const vi = visiblePasses.length;
      visiblePasses.push(name);

      rawNodes.push({
        id:       name,
        type:     'pass',
        position: { x: startX + vi * (NODE_W + H_GAP), y: NODE_H + V_GAP },
        data:     { title: name, time, kind: 'pass', warning: anomalies[name] },
      });

      if (vi === 0) {
        rawEdges.push({
          id: `e_${pipelineStageId}_${name}`, source: pipelineStageId, sourceHandle: 'bottom',
          target: name, targetHandle: 'left', type: 'gradient',
          style: { stroke: '#388bfd', strokeWidth: 2 },
        });
      } else {
        const prev = visiblePasses[vi - 1];
        rawEdges.push({
          id: `e_pass_${prev}_${name}`, source: prev, sourceHandle: 'right',
          target: name, targetHandle: 'left', type: 'gradient',
          style: { stroke: '#388bfd', strokeWidth: 2 },
        });
      }
    }
  }

  // ── derive which handles are actually connected for each node ────────────
  // This tells PipelineNode which bulges and dots to render.
  const conn = {};
  const ensure = id => { if (!conn[id]) conn[id] = { left: false, right: false, bottom: false }; };
  for (const e of rawEdges) {
    ensure(e.source); ensure(e.target);
    if (e.sourceHandle === 'bottom') conn[e.source].bottom = true;
    else conn[e.source].right = true;
    conn[e.target].left = true; // all our target handles are 'left' or default-left
  }

  const nodes = rawNodes.map(n => ({
    ...n,
    data: { ...n.data, connections: conn[n.id] || { left: false, right: false, bottom: false } },
  }));

  return { nodes, edges: rawEdges };
}

// ── inner flow component ─────────────────────────────────────────────────────
// GraphInner holds NO state and never re-renders after mount.
// Each PipelineNode subscribes to _nodeSubs for its own live data, so status
// and timing changes bypass this component and ReactFlowComp entirely.
// Edge DOM elements are never touched → CSS animations run uninterrupted.
function GraphInner({ initNodes, initEdges }) {
  return React.createElement(ReactFlowComp,
    {
      nodes: initNodes, edges: initEdges,
      onNodesChange: () => {},
      onEdgesChange: () => {},
      nodeTypes,
      edgeTypes,
      fitView: true,
      fitViewOptions: { padding: 0.15 },
      nodesDraggable:    true,
      nodesConnectable:  false,
      elementsSelectable: true,
      proOptions: { hideAttribution: true },
      style: { background: '#0d1117' },
    },
    Background  && React.createElement(Background,  { color: '#21262d', gap: 20, style: { opacity: 0.5 } }),
    Controls    && React.createElement(Controls,    { style: { background: '#161b22', border: '1px solid #30363d', borderRadius: '6px' } }),
  );
}

// ── public API ────────────────────────────────────────────────────────────────
export function renderNodeGraph(snapshot, filter = '') {
  const container = document.getElementById('cy');
  if (!container) { console.warn('nodeGraph: no #cy container'); return null; }

  const { nodes, edges } = computeLayout(snapshot, filter);

  if (!reactRoot) {
    reactRoot = ReactDOM.createRoot(container);
  }

  const structureKey = nodes.map(n => n.id).join(',');

  if (structureKey !== graphStructureKey) {
    // Structure changed (or first render) — pre-populate store then full mount.
    // Edges get new DOM elements and animations restart from zero (correct: new graph).
    for (const n of nodes) _nodeData.set(n.id, n.data);
    graphStructureKey = structureKey;
    reactRoot.render(
      React.createElement(GraphInner, { initNodes: nodes, initEdges: edges })
    );
  } else {
    // Data-only update (status/time changed, same node IDs).
    // Push directly to each PipelineNode's own setState — GraphInner and
    // ReactFlowComp are never touched, so edge SVG elements are never recreated.
    for (const n of nodes) {
      _nodeData.set(n.id, n.data);
      const cb = _nodeSubs.get(n.id);
      if (cb) cb(n.data);
      const wcb = _warnSubs.get(n.id);
      if (wcb) wcb(n.data.warning || null);
    }
  }

  return { root: reactRoot };
}
