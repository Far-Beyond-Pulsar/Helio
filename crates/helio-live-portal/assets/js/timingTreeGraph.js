// timingTreeGraph.js — Frame Timing Breakdown tree.
// Layout algorithm:
//   1. Split stages into top-level (no '/') and sub-scopes (has '/').
//   2. Group sub-scopes by parent prefix.
//   3. Each top-level node occupies a "slot" wide enough for:
//        max(NODE_W, children_count * (NODE_W + H_GAP) - H_GAP)
//      Slots are laid out left-to-right with H_GAP between them.
//   4. Top-level node is horizontally centered within its slot.
//   5. Children are left-aligned within their parent's slot (they fill it exactly).
//   No overlap is possible because slot widths are guaranteed.

let _timingGraph = null;

function computeTimingLayout(snapshot) {
  const { NODE_W, NODE_H, H_GAP, V_GAP } = window.HelioGraph;

  const stages = snapshot.stage_timings || [];
  const topStages = stages.filter(s => !s.name.includes('/'));
  const subStages  = stages.filter(s =>  s.name.includes('/'));

  // Group sub-scopes by parent prefix (part before first '/').
  /** @type {Record<string, typeof subStages>} */
  const groups = {};
  for (const s of subStages) {
    const key = s.name.split('/')[0];
    (groups[key] ??= []).push(s);
  }

  // ── Slot widths ─────────────────────────────────────────────────────────────
  // A slot must fit both the parent node AND its children side-by-side.
  const STEP = NODE_W + H_GAP;

  const slotW = topStages.map(s => {
    const key  = s.id ?? s.name;
    const kids = groups[key];
    if (!kids || kids.length === 0) return NODE_W;
    const childrenW = kids.length * STEP - H_GAP;
    return Math.max(NODE_W, childrenW);
  });

  // Left edge of each slot.
  const slotX = [];
  let cur = 0;
  for (let i = 0; i < topStages.length; i++) {
    slotX.push(cur);
    cur += slotW[i] + H_GAP;
  }

  const nodes = [];
  const edges = [];

  // ── Top row ──────────────────────────────────────────────────────────────────
  topStages.forEach((s, i) => {
    const id = s.id ?? String(i);
    const x  = slotX[i] + (slotW[i] - NODE_W) / 2; // centered in slot
    const hasChildren = !!(groups[s.id ?? s.name]?.length);

    nodes.push({
      id,
      type: 'pipeline',
      position: { x, y: 0 },
      data: {
        title: s.name,
        time: `${s.ms.toFixed(2)} ms`,
        kind: 'top',
        connections: {
          left:   i > 0,
          right:  i < topStages.length - 1,
          bottom: hasChildren,
        },
      },
    });

    if (i > 0) {
      edges.push({
        id: `et_${i}`,
        source: topStages[i - 1].id ?? String(i - 1),
        target: id,
        type: 'gradient',
        style: { stroke: '#30363d', strokeWidth: 2 },
      });
    }
  });

  // ── Sub-scope row ────────────────────────────────────────────────────────────
  // Children fill their parent's slot from the left edge.
  topStages.forEach((s, i) => {
    const key  = s.id ?? s.name;
    const kids = groups[key];
    if (!kids || kids.length === 0) return;

    const parentId = s.id ?? String(i);

    kids.forEach((k, ki) => {
      const id = k.id ?? `sub_${key}_${ki}`;
      const x  = slotX[i] + ki * STEP;
      const y  = NODE_H + V_GAP;

      nodes.push({
        id,
        type: 'pipeline',
        position: { x, y },
        data: {
          title: k.name,
          time: `${k.ms.toFixed(3)} ms`,
          kind: 'sub',
          connections: {
            left:  ki > 0,
            right: ki < kids.length - 1,
            bottom: false,
          },
        },
      });

      if (ki === 0) {
        // Parent → first child (vertical drop).
        edges.push({
          id: `esub_${key}_0`,
          source: parentId,
          sourceHandle: 'bottom',
          target: id,
          targetHandle: 'left',
          type: 'gradient',
          style: { stroke: '#56d364', strokeWidth: 1.5 },
        });
      } else {
        // Sibling chain.
        const prevId = kids[ki - 1].id ?? `sub_${key}_${ki - 1}`;
        edges.push({
          id: `esub_${key}_${ki}`,
          source: prevId,
          sourceHandle: 'right',
          target: id,
          targetHandle: 'left',
          type: 'gradient',
          style: { stroke: '#56d364', strokeWidth: 1.5 },
        });
      }
    });
  });

  return { nodes, edges };
}

window.renderTimingTreeGraph = function (snapshot) {
  if (!window.HelioGraph) { console.warn('timingTreeGraph: window.HelioGraph not ready'); return; }
  if (!_timingGraph) _timingGraph = window.HelioGraph.createGraph('timingTreeGraph');
  const { nodes, edges } = computeTimingLayout(snapshot);
  _timingGraph.render(nodes, edges);
};
