// timingTreeGraph.js — Frame Timing Breakdown graph.
// Owns timing-specific layout only.
// All ReactFlow rendering is delegated to window.HelioGraph (helioGraph.js).

let _timingGraph = null; // HelioGraph instance for #timingTreeGraph

function computeTimingLayout(snapshot) {
  const { NODE_W, NODE_H, H_GAP } = window.HelioGraph;
  const stages = snapshot.stage_timings || [];

  const nodes = stages.map((s, i) => ({
    id: s.id || String(i),
    type: 'pipeline',
    position: { x: i * (NODE_W + H_GAP), y: 0 },
    data: {
      title: s.name,
      time: `${s.ms.toFixed(2)} ms`,
      kind: 'top',
      connections: {
        left:   i > 0,
        right:  i < stages.length - 1,
        bottom: false,
      },
    },
  }));

  const edges = stages.slice(1).map((s, i) => ({
    id: `et_${i}`,
    source: stages[i].id || String(i),
    target: s.id || String(i + 1),
    type: 'gradient',
    style: { stroke: '#30363d', strokeWidth: 2 },
  }));

  return { nodes, edges };
}

window.renderTimingTreeGraph = function (snapshot) {
  if (!window.HelioGraph) { console.warn('timingTreeGraph: window.HelioGraph not ready'); return; }
  if (!_timingGraph) _timingGraph = window.HelioGraph.createGraph('timingTreeGraph');
  const { nodes, edges } = computeTimingLayout(snapshot);
  _timingGraph.render(nodes, edges);
};


