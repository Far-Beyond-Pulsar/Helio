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
// ────────────────────────────────────────────────────────────────────────────
function computeLayout(snapshot, filter) {
  const rawNodes  = [];
  const rawEdges  = [];
  const anomalies = detectAnomalies(snapshot);
  const topSteps  = (snapshot.stage_timings || []).map(s => ({ id: s.id, label: s.name, val: s.ms }));
  const match     = s => !filter || s.toLowerCase().includes(filter);

  for (let i = 0; i < topSteps.length; i++) {
    const s    = topSteps[i];
    const time = s.val.toFixed(2) + ' ms';
    if (!match(s.label) && !match(time)) continue;
    rawNodes.push({
      id: s.id, type: 'pipeline',
      position: { x: i * (NODE_W + H_GAP), y: 0 },
      data: { title: s.label, time, kind: 'top', warning: anomalies[s.id] },
    });
    if (i > 0) {
      rawEdges.push({
        id: `e_${topSteps[i-1].id}_${s.id}`,
        source: topSteps[i-1].id, target: s.id,
        type: 'gradient', style: { stroke: '#30363d', strokeWidth: 2 },
      });
    }
  }

  const pipelineStageId = snapshot.pipeline_stage_id || null;
  const passOrder       = snapshot.pipeline_order || [];
  if (pipelineStageId && passOrder.length > 0) {
    const pipelineNode   = rawNodes.find(n => n.id === pipelineStageId);
    const pipelineCenter = pipelineNode ? pipelineNode.position.x + NODE_W / 2 : 0;
    const visiblePasses  = [];

    for (let i = 0; i < passOrder.length; i++) {
      const name = passOrder[i];
      const t    = (snapshot.pass_timings || []).find(p => p.name === name) || { gpu_ms: 0, cpu_ms: 0 };
      const time = `GPU ${t.gpu_ms.toFixed(2)} ms  ·  CPU ${t.cpu_ms.toFixed(2)} ms`;
      if (!match(name) && !match(time)) continue;

      const vi = visiblePasses.length;
      visiblePasses.push(name);
      rawNodes.push({
        id: name, type: 'pass',
        position: { x: pipelineCenter + vi * (NODE_W + H_GAP), y: NODE_H + V_GAP },
        data: { title: name, time, kind: 'pass', warning: anomalies[name] },
      });

      if (vi === 0) {
        rawEdges.push({
          id: `e_${pipelineStageId}_${name}`,
          source: pipelineStageId, sourceHandle: 'bottom',
          target: name, targetHandle: 'left',
          type: 'gradient', style: { stroke: '#388bfd', strokeWidth: 2 },
        });
      } else {
        const prev = visiblePasses[vi - 1];
        rawEdges.push({
          id: `e_pass_${prev}_${name}`,
          source: prev, sourceHandle: 'right',
          target: name, targetHandle: 'left',
          type: 'gradient', style: { stroke: '#388bfd', strokeWidth: 2 },
        });
      }
    }
  }

  // Derive which handles are connected so PipelineNode can render bulges/dots correctly
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
