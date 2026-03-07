// timingTree.js - draws the timing breakdown tree using SVG elements

export function drawTimingTree(snapshot) {
  const treeEl = document.getElementById('timingTree');
  while (treeEl.firstChild) treeEl.removeChild(treeEl.firstChild);

  const frameToFrame = snapshot.frame_to_frame_ms || 0;
  const frameTm = snapshot.frame_time_ms || 0;
  const stages  = snapshot.stage_timings || [];
  const untrackedStage = stages.find(s => s.id === 'untracked');
  const untracked = untrackedStage ? untrackedStage.ms : (snapshot.untracked_ms || 0);

  const rootX = 80;
  const rootY = 50;
  const colWidth = 240;
  const rowHeight = 60;

  function boxColor(ms) {
    if (ms >= 2.0) return '#7f2c21';
    if (ms >= 1.0) return '#5f3a27';
    if (ms >= 0.5) return '#3f4a2f';
    return '#27414f';
  }

  function drawBox(x, y, label, value, parent = null) {
    if (parent) {
      const line = document.createElementNS('http://www.w3.org/2000/svg','line');
      line.setAttribute('x1', String(parent.x+70));
      line.setAttribute('y1', String(parent.y+28));
      line.setAttribute('x2', String(x+35));
      line.setAttribute('y2', String(y));
      line.setAttribute('stroke', '#5c7388');
      line.setAttribute('stroke-width','2');
      treeEl.appendChild(line);
    }
    const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('x', String(x));
    rect.setAttribute('y', String(y));
    rect.setAttribute('width','140');
    rect.setAttribute('height','50');
    rect.setAttribute('rx','8');
    rect.setAttribute('fill', boxColor(value));
    rect.setAttribute('stroke','#6c89a5');
    rect.setAttribute('stroke-width','2');
    treeEl.appendChild(rect);
    const title = document.createElementNS('http://www.w3.org/2000/svg','text');
    title.setAttribute('x', String(x+8));
    title.setAttribute('y', String(y+16));
    title.setAttribute('fill','#e8f0f8');
    title.setAttribute('font-size','11');
    title.setAttribute('font-weight','700');
    title.textContent = label;
    treeEl.appendChild(title);
    const valText = document.createElementNS('http://www.w3.org/2000/svg','text');
    valText.setAttribute('x', String(x+8));
    valText.setAttribute('y', String(y+35));
    valText.setAttribute('fill','#ffb08a');
    valText.setAttribute('font-size','12');
    valText.setAttribute('font-weight','700');
    valText.textContent = value.toFixed(2) + 'ms';
    treeEl.appendChild(valText);
    return {x,y,width:140,height:50};
  }

  const root = drawBox(rootX, rootY, 'Frame-to-frame', frameToFrame);
  const untrackBox = drawBox(rootX, rootY+100,'Untracked',untracked, root);
  const renderBox = drawBox(rootX+colWidth, rootY+100,'render()', frameTm, root);
  const y2 = rootY+200;
  const boxes = stages
    .filter(s => s.id !== 'untracked')
    .map(s => ({ label: s.name, val: s.ms }));
  for (let i=0;i<boxes.length;i++){
    const x = rootX + colWidth + (i%3)*200;
    const y = y2 + Math.floor(i/3)*80;
    drawBox(x,y,boxes[i].label, boxes[i].val, renderBox);
  }
  const legX=rootX, legY=rootY+420;
  const legText=document.createElementNS('http://www.w3.org/2000/svg','text');
  legText.setAttribute('x',String(legX));
  legText.setAttribute('y',String(legY));
  legText.setAttribute('fill','#94a8bb');
  legText.setAttribute('font-size','11');
  legText.textContent = 'Tree: Frame-to-frame splits into untracked (app) and render(). render() breaks down into CPU/GPU stages.';
  treeEl.appendChild(legText);
}
