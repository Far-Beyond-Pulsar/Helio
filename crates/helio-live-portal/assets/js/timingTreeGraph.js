// timingTreeGraph.js
// ReactFlow-based Frame Timing Breakdown graph
import { ReactFlow, Handle, Background, Controls } from '/vendor/reactflow.min.js';
import { useEffect, useState } from '/vendor/react.development.js';

// Node style matches pipeline graph
const nodeStyle = {
  borderRadius: 6,
  background: '#161b22',
  border: '1px solid #30363d',
  color: '#e6edf3',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
  boxShadow: '0 2px 8px #010409',
};

function TimingNode({ data }) {
  return (
    <div style={nodeStyle}>
      <div style={{ fontWeight: 600, color: '#e6edf3', fontSize: 13 }}>{data.label}</div>
      <div style={{ color: '#8b949e', fontSize: 12 }}>{data.ms} ms</div>
    </div>
  );
}

export function TimingTreeGraph({ nodes, edges }) {
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={{ timing: TimingNode }}
      fitView
      style={{ width: '100%', height: 560, background: '#0d1117' }}
    >
      <Background color="#30363d" gap={32} size={1} />
      <Controls />
    </ReactFlow>
  );
}
