// timingTreeGraph.js
// ReactFlow-based Frame Timing Breakdown graph using UMD globals

const RF = window.ReactFlow || {};
const React = window.React;
const ReactDOM = window.ReactDOM;
const ReactFlowComp = RF.ReactFlow;
const Background = RF.Background;
const Controls = RF.Controls;

const nodeStyle = {
  borderRadius: 6,
  background: '#161b22',
  border: '1px solid #30363d',
  color: '#e6edf3',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
  boxShadow: '0 2px 8px #010409',
};

function TimingNode({ data }) {
  return React.createElement('div', { style: nodeStyle },
    React.createElement('div', { style: { fontWeight: 600, color: '#e6edf3', fontSize: 13 } }, data.label),
    React.createElement('div', { style: { color: '#8b949e', fontSize: 12 } }, `${data.ms} ms`)
  );
}

function TimingTreeGraph({ nodes, edges }) {
  return React.createElement(
    ReactFlowComp,
    {
      nodes,
      edges,
      nodeTypes: { timing: TimingNode },
      fitView: true,
      style: { width: '100%', height: 560, background: '#0d1117' }
    },
    React.createElement(Background, { color: '#30363d', gap: 32, size: 1 }),
    React.createElement(Controls, null)
  );
}

window.renderTimingTreeGraph = function(snapshot) {
  const treeDiv = document.getElementById('timingTreeGraph');
  if (!treeDiv) return;
  const stages = snapshot.stage_timings || [];
  const nodes = stages.map((s, i) => ({
    id: s.id || String(i),
    type: 'timing',
    position: { x: i * 180, y: 0 },
    data: { label: s.name, ms: s.ms }
  }));
  const edges = stages.slice(1).map((s, i) => ({
    id: `e${i}`,
    source: stages[i].id || String(i),
    target: s.id || String(i+1),
    animated: true,
    style: { stroke: '#388bfd', strokeWidth: 2 }
  }));
  ReactDOM.render(
    React.createElement(TimingTreeGraph, { nodes, edges }),
    treeDiv
  );
};
