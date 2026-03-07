// scene.js - scene layout projection and delta application logic

let sceneState = {
  objects: new Map(),
  lights: new Map(),
  billboards: new Map(),
  camera: null
};
// ids moved in most recent delta (for highlight)
let movedObjectIds = new Set();

export function applySceneDelta(delta) {
  if (!delta) return;
  movedObjectIds.clear();
  if (delta.moved_object_ids) {
    for (const id of delta.moved_object_ids) {
      movedObjectIds.add(id);
    }
  }
  if (delta.object_changes) {
    for (const obj of delta.object_changes) {
      sceneState.objects.set(obj.id, obj);
    }
  }
  if (delta.removed_object_ids) {
    for (const id of delta.removed_object_ids) {
      sceneState.objects.delete(id);
    }
  }
  if (delta.light_changes) {
    for (const light of delta.light_changes) {
      sceneState.lights.set(light.id, light);
    }
  }
  if (delta.removed_light_ids) {
    for (const id of delta.removed_light_ids) {
      sceneState.lights.delete(id);
    }
  }
  if (delta.billboard_changes) {
    for (const bb of delta.billboard_changes) {
      sceneState.billboards.set(bb.id, bb);
    }
  }
  if (delta.removed_billboard_ids) {
    for (const id of delta.removed_billboard_ids) {
      sceneState.billboards.delete(id);
    }
  }
  if (delta.camera !== undefined) {
    sceneState.camera = delta.camera;
  }
}

export function getSceneLayoutFromState() {
  return {
    objects: Array.from(sceneState.objects.values()),
    lights: Array.from(sceneState.lights.values()),
    billboards: Array.from(sceneState.billboards.values()),
    camera: sceneState.camera
  };
}

// draw coordinate axes on canvas
function drawAxes(ctx, w, h) {
  ctx.strokeStyle = '#1f3142';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
  ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
  ctx.stroke();
}

export function drawSceneProjection(canvas, layout, mode) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0c141b';
  ctx.fillRect(0,0,w,h);
  drawAxes(ctx, w, h);
  if (!layout) return;
  let minA=Infinity, maxA=-Infinity, minB=Infinity, maxB=-Infinity;
  function extract(obj){
    const c=obj.bounds_center;
    if(mode==='top')return[c[0],c[2]];
    if(mode==='front')return[c[0],c[1]];
    return[c[2],c[1]];
  }
  for(const o of layout.objects||[]){
    const [a,b]=extract(o);
    minA=Math.min(minA,a);maxA=Math.max(maxA,a);
    minB=Math.min(minB,b);maxB=Math.max(maxB,b);
  }
  for(const l of layout.lights||[]){
    const c=l.position;
    const p=mode==='top'? [c[0],c[2]]:
            mode==='front'? [c[0],c[1]]:[c[2],c[1]];
    minA=Math.min(minA,p[0]);maxA=Math.max(maxA,p[0]);
    minB=Math.min(minB,p[1]);maxB=Math.max(maxB,p[1]);
  }
  if(!isFinite(minA)){minA=-10;maxA=10;minB=-10;maxB=10;}
  const spanA=Math.max(1,maxA-minA);
  const spanB=Math.max(1,maxB-minB);
  const pad=14;
  const sx=(w-pad*2)/spanA;
  const sy=(h-pad*2)/spanB;
  const s=Math.min(sx,sy);
  function toScreen(a,b){
    const x=pad+(a-minA)*s;
    const y=h-(pad+(b-minB)*s);
    return[x,y];
  }
  for(const o of layout.objects||[]){
    const [a,b]=extract(o);
    const [x,y]=toScreen(a,b);
    const r=Math.max(1.5,o.bounds_radius*s*0.2);
    ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
    if (movedObjectIds.has(o.id)) {
      ctx.strokeStyle = '#ff5f4a';
      ctx.lineWidth = 2.5;
    } else {
      ctx.strokeStyle = '#304a60';
      ctx.lineWidth = 1;
    }
    ctx.stroke();
  }
  ctx.fillStyle='#ffb06e';
  for(const l of layout.lights||[]){
    const c=l.position;
    const p=mode==='top'? [c[0],c[2]]:
            mode==='front'? [c[0],c[1]]:[c[2],c[1]];
    const [x,y]=toScreen(p[0],p[1]);
    ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();
  }
  ctx.fillStyle='#7ff1b2';
  for(const b of layout.billboards||[]){
    const c=b.position;
    const p=mode==='top'? [c[0],c[2]]:
            mode==='front'? [c[0],c[1]]:[c[2],c[1]];
    const [x,y]=toScreen(p[0],p[1]);
    ctx.fillRect(x-1,y-1,2,2);
  }
  if(layout.camera){
    const c=layout.camera.position;
    const p=mode==='top'? [c[0],c[2]]:
            mode==='front'? [c[0],c[1]]:[c[2],c[1]];
    const [x,y]=toScreen(p[0],p[1]);
    ctx.fillStyle='#69d8ff';
    ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();
  }
}
