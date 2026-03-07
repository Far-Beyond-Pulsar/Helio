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

let reactRoot = null;

// ── layout constants (must precede PipelineNode so the SVG path can use them) ─
const NODE_W  = 188;
const NODE_H  = 52;
const H_GAP   = 40;
const V_GAP   = 80;
const BULGE_R = 11;   // radius of the status bulge that protrudes from the left edge

// ── custom node component ─────────────────────────────────────────────────────
// The card shape is drawn as a SINGLE SVG path:
//   rounded-rect body + left-protruding semicircle for the status pip.
// One continuous stroke means zero border seams.
function PipelineNode({ data }) {
  const ce       = React.createElement;
  const dotColor = data.kind === 'pass' ? '#388bfd' : '#d29922';

  const W  = NODE_W;
  const H  = NODE_H;
  const R  = 6;        // card corner radius
  const BR = BULGE_R;
  // OX = x of the card's left edge in SVG coordinate space.
  // The SVG canvas extends BR px to the left of the card to accommodate the bulge.
  const OX   = BR;
  const half = H / 2;

  // Clockwise path: rounded rect with a left-protruding semicircular bump.
  // Sweep-flag 0 on the arc = counterclockwise in screen space = leftward.
  const d = [
    `M ${OX + R} 0`,
    `L ${OX + W - R} 0`,
    `Q ${OX + W} 0 ${OX + W} ${R}`,
    `L ${OX + W} ${H - R}`,
    `Q ${OX + W} ${H} ${OX + W - R} ${H}`,
    `L ${OX + R} ${H}`,
    `Q ${OX} ${H} ${OX} ${H - R}`,
    `L ${OX} ${half + BR}`,
    `A ${BR} ${BR} 0 0 1 ${OX} ${half - BR}`,
    `L ${OX} ${R}`,
    `Q ${OX} 0 ${OX + R} 0`,
    'Z',
  ].join(' ');

  return ce('div', {
    style: { position: 'relative', overflow: 'visible', width: `${W}px`, height: `${H}px` },
  },

    // ── single SVG: fill + border + status dot ────────────────────────────
    ce('svg', {
      style: {
        position:      'absolute',
        left:          `${-BR}px`,
        top:           0,
        overflow:      'visible',
        pointerEvents: 'none',
        zIndex:        0,
      },
      width:  W + BR,
      height: H,
    },
      ce('path', { d, fill: '#161b22', stroke: '#30363d', strokeWidth: 1 }),
      // status dot centred in the protruding half-circle
      ce('circle', {
        cx: OX - BR * 0.55 + 5, cy: half, r: 4.5,
        fill: dotColor,
      }),
    ),

    // ── text content ──────────────────────────────────────────────────────
    ce('div', {
      style: {
        position:       'relative',
        zIndex:         1,
        height:         '100%',
        display:        'flex',
        flexDirection:  'column',
        justifyContent: 'center',
        // left pad clears the inner half of the bulge (BR px inside the card)
        padding:    `0 14px 0 ${BR + 8}px`,
        color:      '#e6edf3',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", monospace',
        fontSize:   '13px',
        boxSizing:  'border-box',
      }
    },
      ce('div', { style: { fontWeight: 600, marginBottom: data.time ? '3px' : 0, letterSpacing: '0.01em', whiteSpace: 'nowrap' } }, data.title),
      data.time && ce('div', { style: { fontSize: '11px', color: '#8b949e', whiteSpace: 'nowrap' } }, data.time),
    ),

    // ── ReactFlow handles (invisible — SVG provides all visuals) ─────────
    data.kind === 'top'  && Handle && ce(Handle, { type: 'target', position: Position.Left,
      style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
    data.kind === 'pass' && Handle && ce(Handle, { type: 'target', id: 'left', position: Position.Left,
      style: { background: 'transparent', border: 'none', width: 0, height: 0 } }),
    data.kind === 'top'  && Handle && ce(Handle, { type: 'source', position: Position.Right,
      style: { background: '#30363d', border: 'none', width: '6px', height: '6px' } }),
    data.kind === 'pass' && Handle && ce(Handle, { type: 'source', id: 'right', position: Position.Right,
      style: { background: '#30363d', border: 'none', width: '6px', height: '6px' } }),
    data.kind === 'top'  && Handle && ce(Handle, { type: 'source', id: 'bottom', position: Position.Bottom,
      style: { background: '#388bfd', border: 'none', width: '6px', height: '6px' } }),
  );
}

const nodeTypes = { pipeline: PipelineNode, pass: PipelineNode };

// ── layout computation ────────────────────────────────────────────────────────

function computeLayout(snapshot, filter) {
  const qNodes = [];
  const qEdges = [];

  const topSteps = [
    { id: 'untracked', label: 'Untracked',      val: snapshot.untracked_ms || 0 },
    { id: 'prep',      label: 'Prep',            val: snapshot.prep_ms      || 0 },
    { id: 'pipeline',  label: 'Render Pipeline', val: snapshot.graph_ms     || 0 },
    { id: 'aa',        label: 'AA',              val: snapshot.aa_ms        || 0 },
    { id: 'resolve',   label: 'Resolve',         val: snapshot.resolve_ms   || 0 },
    { id: 'submit',    label: 'Submit',          val: snapshot.submit_ms    || 0 },
    { id: 'poll',      label: 'Poll',            val: snapshot.poll_ms      || 0 },
  ];

  const match = (s) => !filter || s.toLowerCase().includes(filter);
  const edgeStyle = { stroke: '#30363d', strokeWidth: 2 };
  const marker    = { type: MarkerType.ArrowClosed, color: '#30363d' };

  for (let i = 0; i < topSteps.length; i++) {
    const s = topSteps[i];
    const time = s.val.toFixed(2) + ' ms';
    if (!match(s.label) && !match(time)) continue;
    qNodes.push({
      id:       s.id,
      type:     'pipeline',
      position: { x: i * (NODE_W + H_GAP), y: 0 },
      data:     { title: s.label, time, kind: 'top' },
    });
    if (i > 0) {
      qEdges.push({
        id:        `e_${topSteps[i-1].id}_${s.id}`,
        source:    topSteps[i-1].id,
        target:    s.id,
        type:      'smoothstep',
        style:     edgeStyle,
        markerEnd: marker,
      });
    }
  }

  const passOrder = snapshot.pipeline_order || [];
  if (passOrder.length > 0) {
    const pipelineIdx = 2;
    // First child's left edge sits at the parent's center-x so the elbow edge
    // drops straight down then turns right:  PARENT
    //                                           |__ CHILD1 → CHILD2 …
    const pipelineX      = pipelineIdx * (NODE_W + H_GAP);
    const pipelineCenter = pipelineX + NODE_W / 2;
    const startX         = pipelineCenter; // child[0].x = parent center → elbow aligns

    // track which pass nodes actually passed the filter so we can chain correctly
    const visiblePasses = [];

    for (let i = 0; i < passOrder.length; i++) {
      const name = passOrder[i];
      const t    = (snapshot.pass_timings || []).find(p => p.name === name) || { gpu_ms: 0, cpu_ms: 0 };
      const time = `GPU ${t.gpu_ms.toFixed(2)} ms  ·  CPU ${t.cpu_ms.toFixed(2)} ms`;
      if (!match(name) && !match(time)) continue;

      const vi = visiblePasses.length; // index among visible passes
      visiblePasses.push(name);

      qNodes.push({
        id:       name,
        type:     'pass',
        position: { x: startX + vi * (NODE_W + H_GAP), y: NODE_H + V_GAP },
        data:     { title: name, time, kind: 'pass' },
      });

      if (vi === 0) {
        // parent bottom → first child left  (the │__ elbow)
        qEdges.push({
          id:           `e_pipeline_${name}`,
          source:       'pipeline',
          sourceHandle: 'bottom',
          target:       name,
          targetHandle: 'left',
          type:         'smoothstep',
          style:        { stroke: '#388bfd', strokeWidth: 2 },
          markerEnd:    { type: MarkerType.ArrowClosed, color: '#388bfd' },
        });
      } else {
        // previous child right → this child left  (sequential chain)
        const prev = visiblePasses[vi - 1];
        qEdges.push({
          id:           `e_pass_${prev}_${name}`,
          source:       prev,
          sourceHandle: 'right',
          target:       name,
          targetHandle: 'left',
          type:         'smoothstep',
          style:        { stroke: '#388bfd', strokeWidth: 2 },
          markerEnd:    { type: MarkerType.ArrowClosed, color: '#388bfd' },
        });
      }
    }
  }

  return { nodes: qNodes, edges: qEdges };
}

// ── inner flow component (plain React.useState — UMD build omits RF hooks) ────
function GraphInner({ initNodes, initEdges }) {
  const [nodes, setNodes] = React.useState(initNodes);
  const [edges, setEdges] = React.useState(initEdges);

  React.useEffect(() => { setNodes(initNodes); }, [initNodes]);
  React.useEffect(() => { setEdges(initEdges); }, [initEdges]);

  return React.createElement(ReactFlowComp,
    {
      nodes, edges,
      onNodesChange: () => {},
      onEdgesChange: () => {},
      nodeTypes,
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
  console.log(`nodeGraph: ${nodes.length} nodes, ${edges.length} edges`);

  if (!reactRoot) {
    reactRoot = ReactDOM.createRoot(container);
  }

  // Render GraphInner directly — ReactFlowComp is its own provider
  reactRoot.render(
    React.createElement(GraphInner, { initNodes: nodes, initEdges: edges })
  );

  return { root: reactRoot };
}
