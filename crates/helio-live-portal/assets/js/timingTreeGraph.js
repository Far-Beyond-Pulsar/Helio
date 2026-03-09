// timingTreeGraph.js — Frame Timing Breakdown graph.
// Owns timing-specific layout only.
// All ReactFlow rendering is delegated to window.HelioGraph (helioGraph.js).

let _timingGraph = null; // HelioGraph instance for #timingTreeGraph

function computeTimingLayout(snapshot) {
  const { NODE_W, NODE_H, H_GAP, V_GAP } = window.HelioGraph;
  const stages = snapshot.stage_timings || [];

  // Pipeline-level entries (no '/' in name) go in the top row.
  // Sub-scope entries (e.g. "depth_prepass/compile") go in a second row
  // anchored under their parent entry so the relationship is visible.
  const topStages = stages.filter(s => !s.name.includes('/'));
  const subStages = stages.filter(s => s.name.includes('/'));

  const nodes = [];
  const edges = [];

  // ── Top row ─────────────────────────────────────────────────────────────
  topStages.forEach((s, i) => {
    nodes.push({
      id: s.id || String(i),
      type: 'pipeline',
      position: { x: i * (NODE_W + H_GAP), y: 0 },
      data: {
        title: s.name,
        time: `${s.ms.toFixed(2)} ms`,
        kind: 'top',
        connections: {
          left:   i > 0,
          right:  i < topStages.length - 1,
          bottom: false,       // updated below if it has sub-scope children
        },
      },
    });
    if (i > 0) {
      edges.push({
        id: `et_${i}`,
        source: topStages[i - 1].id || String(i - 1),
        target: s.id || String(i),
        type: 'gradient',
        style: { stroke: '#30363d', strokeWidth: 2 },
      });
    }
  });

  // ── Sub-scope row ────────────────────────────────────────────────────────
  // Group by parent prefix (part before '/').
  const groupedSubs = {};
  for (const s of subStages) {
    const prefix = s.name.split('/')[0]; // "depth_prepass" etc.
    if (!groupedSubs[prefix]) groupedSubs[prefix] = [];
    groupedSubs[prefix].push(s);
  }

  for (const [prefix, subs] of Object.entries(groupedSubs)) {
    // Find the parent node in the top row by matching its id or name.
    const parentIdx = topStages.findIndex(s => s.id === prefix || s.name === prefix);
    if (parentIdx < 0) continue;
    const parentNode = nodes[parentIdx];

    // Give the parent a bottom connection dot.
    parentNode.data = { ...parentNode.data, connections: { ...parentNode.data.connections, bottom: true } };

    subs.forEach((s, si) => {
      const x = parentNode.position.x + si * (NODE_W + H_GAP);
      const y = NODE_H + V_GAP;
      nodes.push({
        id: s.id || `sub_${prefix}_${si}`,
        type: 'pipeline',
        position: { x, y },
        data: {
          title: s.name,
          time: `${s.ms.toFixed(3)} ms`,
          kind: 'top',
          connections: { left: si > 0, right: si < subs.length - 1, bottom: false },
        },
      });
      if (si === 0) {
        edges.push({
          id: `esub_${prefix}_${si}`,
          source: parentNode.id || String(parentIdx),
          sourceHandle: 'bottom',
          target: s.id || `sub_${prefix}_${si}`,
          targetHandle: 'left',
          type: 'gradient',
          style: { stroke: '#56d364', strokeWidth: 1.5 },
        });
      } else {
        edges.push({
          id: `esub_${prefix}_${si}`,
          source: subs[si - 1].id || `sub_${prefix}_${si - 1}`,
          sourceHandle: 'right',
          target: s.id || `sub_${prefix}_${si}`,
          targetHandle: 'left',
          type: 'gradient',
          style: { stroke: '#56d364', strokeWidth: 1.5 },
        });
      }
    });
  }

  return { nodes, edges };
}

window.renderTimingTreeGraph = function (snapshot) {
  if (!window.HelioGraph) { console.warn('timingTreeGraph: window.HelioGraph not ready'); return; }
  if (!_timingGraph) _timingGraph = window.HelioGraph.createGraph('timingTreeGraph');
  const { nodes, edges } = computeTimingLayout(snapshot);
  _timingGraph.render(nodes, edges);
};


