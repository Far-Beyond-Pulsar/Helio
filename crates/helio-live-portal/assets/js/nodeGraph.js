// nodeGraph.js — Render Pipeline graph.
// Owns pipeline-specific layout + anomaly detection only.
// All ReactFlow rendering is delegated to window.HelioGraph (helioGraph.js).

const { NODE_W, NODE_H, H_GAP, V_GAP } = window.HelioGraph;

let _pipelineGraph = null; // HelioGraph instance for #cy

// ────────────────────────────────────────────────────────────────────────────
//  Anomaly detection
// ────────────────────────────────────────────────────────────────────────────
function detectAnomalies(snapshot) {
  const anomalies = {};
  const flag = (pairs, message, unit, minMs) => {
    const vals = pairs.map(p => p.val).filter(v => v > 0);
    if (vals.length < 2) return;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std  = Math.sqrt(vals.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / vals.length);
    const thr  = Math.max(mean + 2 * std, minMs);
    for (const { id, val } of pairs) {
      if (val > thr) {
        anomalies[id] = {
          message,
          stats: [`${val.toFixed(2)} ${unit}`, `${((val / mean) * 100).toFixed(0)}% of group avg`],
        };
      }
    }
  };

  flag((snapshot.stage_timings || []).map(s => ({ id: s.id, val: s.ms })), 'High CPU time', 'ms', 2);

  const passes = snapshot.pass_timings || [];
  flag(passes.map(p => ({ id: p.name, val: p.gpu_ms })), 'GPU bottleneck', 'ms GPU', 1);
  flag(
    passes.filter(p => !anomalies[p.name]).map(p => ({ id: p.name, val: p.cpu_ms })),
    'CPU bottleneck', 'ms CPU', 1
  );

  return anomalies;
}

// ────────────────────────────────────────────────────────────────────────────
//  Layout
//
//  Row 0 (y=0):              [Stage] → [Stage] → [Stage] → [Stage]
//  Row 1 (y=NODE_H+V_GAP):      |_[Child] → [Child]           |_[Child]
//
//  Top-level nodes come from snapshot.stage_timings (the tree roots).
//  Each root's .children array populates row 1 directly below that root.
//  GPU passes (pipeline_order) hang as children under the stage whose id
//  matches snapshot.pipeline_stage_id; if that id is absent from
//  stage_timings a synthetic node is appended to row 0.
// ────────────────────────────────────────────────────────────────────────────
function computeLayout(snapshot, filter) {
  const rawNodes  = [];
  const rawEdges  = [];
  const anomalies = detectAnomalies(snapshot);
  const match     = s => !filter || s.toLowerCase().includes(filter.toLowerCase());

  const stageRoots      = snapshot.stage_timings || [];
  const pipelineStageId = snapshot.pipeline_stage_id || null;
  const passOrder       = snapshot.pipeline_order    || [];
  const hasPasses       = !!(pipelineStageId && passOrder.length > 0);

  // If the pipeline stage id is not already a root in stage_timings, append a
  // synthetic node so passes have a parent in row 0.
  const pipelineInRoots = stageRoots.some(s => s.id === pipelineStageId);
  const row0 = pipelineInRoots || !hasPasses
    ? stageRoots
    : [...stageRoots, { id: pipelineStageId, name: 'Pipeline', ms: snapshot.total_gpu_ms || 0, children: [] }];

  // Build the full child list for each stage root (before filtering) so we can
  // compute how wide each parent's slot needs to be.
  function buildChildren(s) {
    const isPipeline = hasPasses && s.id === pipelineStageId;
    if (isPipeline) {
      return passOrder.map(name => {
        const t = (snapshot.pass_timings || []).find(p => p.name === name) || { gpu_ms: 0, cpu_ms: 0 };
        return {
          id: name, name,
          time: `GPU ${t.gpu_ms.toFixed(2)} ms  ·  CPU ${t.cpu_ms.toFixed(2)} ms`,
          kind: 'pass', nodeType: 'pass',
        };
      });
    }
    return (s.children || []).map(c => ({
      id: c.id, name: c.name,
      time: c.ms.toFixed(2) + ' ms',
      kind: 'sub', nodeType: 'pipeline',
    }));
  }

  // ── Pre-pass: decide which row-0 nodes are visible, count visible children ─
  // A row-0 node is visible if its own text matches OR any child matches.
  const visibleRow0 = [];
  for (const s of row0) {
    const time     = s.ms.toFixed(2) + ' ms';
    const children = buildChildren(s).filter(c => match(c.name) || match(c.time));
    if (!match(s.name) && !match(time) && children.length === 0) continue;
    visibleRow0.push({ stage: s, visibleChildren: children });
  }

  // Collect every ID that lives in row 0 so we can exclude them from row 1
  // (guards against nodes that appear as both a root and a child in the data).
  const row0Ids = new Set(visibleRow0.map(({ stage }) => stage.id));

  // Annotate each row-0 entry with safe (deduplicated) children.
  const layoutRows = visibleRow0.map(({ stage, visibleChildren }) => ({
    stage,
    children: visibleChildren.filter(c => !row0Ids.has(c.id)),
  }));

  // ── Row 0: place each parent at curX; slot width = max(1, numChildren) ────
  // This ensures children never overlap the next parent.
  let curX      = 0;
  let prevTopId = null;
  const stageX  = new Map(); // stage id → pixel x

  for (const { stage: s, children } of layoutRows) {
    const time  = s.ms.toFixed(2) + ' ms';
    const slots = Math.max(1, children.length);
    const x     = curX;
    stageX.set(s.id, x);

    rawNodes.push({
      id: s.id, type: 'pipeline',
      position: { x, y: 0 },
      data: { title: s.name, time, kind: 'top', warning: anomalies[s.id] },
    });
    if (prevTopId !== null) {
      rawEdges.push({
        id: `e_top_${prevTopId}_${s.id}`,
        source: prevTopId, target: s.id,
        type: 'gradient', style: { stroke: '#30363d', strokeWidth: 2 },
      });
    }
    prevTopId = s.id;
    curX += slots * (NODE_W + H_GAP);
  }

  // ── Row 1: children directly below their parent ──────────────────────────
  for (const { stage: s, children } of layoutRows) {
    if (children.length === 0) continue;

    const isPipelineStage = hasPasses && s.id === pipelineStageId;
    const edgeColor       = isPipelineStage ? '#388bfd' : '#56d364';
    const edgeWidth       = isPipelineStage ? 2 : 1.5;
    const parentX         = stageX.get(s.id);

    let prevChildId = null;
    let cx          = parentX;

    for (const child of children) {
      rawNodes.push({
        id: child.id, type: child.nodeType,
        position: { x: cx, y: NODE_H + V_GAP },
        data: { title: child.name, time: child.time, kind: child.kind, warning: anomalies[child.id] },
      });

      rawEdges.push(prevChildId === null
        ? {
            // First child: drop from parent's bottom handle
            id: `e_${s.id}_${child.id}`,
            source: s.id, sourceHandle: 'bottom',
            target: child.id, targetHandle: 'left',
            type: 'gradient', style: { stroke: edgeColor, strokeWidth: edgeWidth },
          }
        : {
            // Subsequent children: chain right
            id: `e_${prevChildId}_${child.id}`,
            source: prevChildId, sourceHandle: 'right',
            target: child.id, targetHandle: 'left',
            type: 'gradient', style: { stroke: edgeColor, strokeWidth: edgeWidth },
          }
      );

      prevChildId = child.id;
      cx += NODE_W + H_GAP;
    }
  }

  // ── Derive which handles are connected (drives bulge/dot rendering) ───────
  const conn   = {};
  const ensure = id => { if (!conn[id]) conn[id] = { left: false, right: false, bottom: false }; };
  for (const e of rawEdges) {
    ensure(e.source); ensure(e.target);
    if (e.sourceHandle === 'bottom') conn[e.source].bottom = true;
    else conn[e.source].right = true;
    conn[e.target].left = true;
  }

  const nodes = rawNodes.map(n => ({
    ...n,
    data: { ...n.data, connections: conn[n.id] || { left: false, right: false, bottom: false } },
  }));

  return { nodes, edges: rawEdges };
}

// ────────────────────────────────────────────────────────────────────────────
//  Public API
// ────────────────────────────────────────────────────────────────────────────
export function renderNodeGraph(snapshot, filter = '') {
  if (!_pipelineGraph) _pipelineGraph = window.HelioGraph.createGraph('cy');
  const { nodes, edges } = computeLayout(snapshot, filter);
  _pipelineGraph.render(nodes, edges);
  return _pipelineGraph;
}
