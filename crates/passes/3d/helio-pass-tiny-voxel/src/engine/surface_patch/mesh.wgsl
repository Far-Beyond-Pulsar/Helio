struct MeshCamera {
    view:mat4x4<f32>,proj:mat4x4<f32>,view_proj:mat4x4<f32>,inv_view_proj:mat4x4<f32>,
    position_near:vec4<f32>,forward_far:vec4<f32>,jitter_frame:vec4<f32>,prev_view_proj:mat4x4<f32>,
}
@group(1) @binding(0) var<storage,read> mesh_cameras:array<MeshCamera>;
@group(0) @binding(34) var<storage,read> mesh_quads:array<Quad>;
@group(0) @binding(35) var<storage,read> mesh_ready:array<u32>;
@group(0) @binding(36) var mesh_visibility:texture_2d<u32>;
struct MeshVertex {
    @builtin(position) clip:vec4<f32>,
    @location(0) @interpolate(flat) identity:vec3<u32>,
}
fn mesh_low(slot:u32)->vec3<i32> {
    return (patch_settings.low+vec3<i32>(i32(slot%8u),i32(slot/8u%8u),i32(slot/64u)))*i32(patch_settings.step*32u);
}
fn mesh_clip(slot:u32,local:vec3<f32>)->vec4<f32> {
    // Subtract integer planetary anchors BEFORE conversion; retain the same
    // fraction and camera rotation/projection used to materialize primary rays.
    let relative=(vec3<f32>(mesh_low(slot)-p.origin.xyz)+local*f32(patch_settings.step)-p.fraction.xyz)*0.1;
    let camera=mesh_cameras[0];
    let view=(camera.view*vec4<f32>(relative,0.0)).xyz;
    let w=-view.z;
    return vec4<f32>(view.x*camera.proj[0][0]+camera.jitter_frame.x*w,
        view.y*camera.proj[1][1]+camera.jitter_frame.y*w,0.000001,w);
}
@vertex fn mesh_vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) index:u32)->MeshVertex {
    let q=mesh_quads[index];let point=quad_vertex(q,vertex);
    if point.w==0.0 {return MeshVertex(vec4<f32>(0.0,0.0,0.0,1.0),vec3<u32>(0u));}
    return MeshVertex(mesh_clip(q.extent>>12u,point.xyz),vec3<u32>(q.origin_face_material,q.extent,1u));
}
@vertex fn missing_vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) slot:u32)->MeshVertex {
    if mesh_ready[slot]!=0u {return MeshVertex(vec4<f32>(0.0,0.0,0.0,1.0),vec3<u32>(0u));}
    let face=vertex/6u;let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    var corner=vertex%6u;
    if (face&1u)!=0u {corner=select(corner-1u,corner+1u,corner%3u==1u);if vertex%3u==0u {corner=vertex%6u;}}
    let uv=array<vec2<f32>,6>(vec2<f32>(0.0),vec2<f32>(32.0,0.0),vec2<f32>(32.0),vec2<f32>(0.0),vec2<f32>(32.0),vec2<f32>(0.0,32.0))[corner];
    var local=vec3<f32>(0.0);local[axis]=select(32.0,0.0,(face&1u)!=0u);local[u]=uv.x;local[v]=uv.y;
    return MeshVertex(mesh_clip(slot,local),vec3<u32>(0u,0u,2u));
}
@fragment fn mesh_identity(vertex:MeshVertex)->@location(0) vec2<u32> {return vertex.identity.xy;}
fn mesh_page_slot(cell:vec3<i32>)->u32 {
    let low=voxel_low(cell,i32(patch_settings.step*32u));
    let q=low/i32(patch_settings.step*32u)-patch_settings.low;
    if any(q<vec3<i32>(0)) || any(q>=vec3<i32>(8)) {return 0xffffffffu;}
    return u32(q.x+8*q.y+64*q.z);
}
// Independent prefix certificate: raster markers are an early rejection, but
// hardware edge quantization must never turn an unavailable page into air.
fn mesh_prefix(plane:i32,axis:u32,rd:vec3<f32>)->bool {
    let direction=select(vec3<i32>(1),vec3<i32>(-1),rd<vec3<f32>(0.0));
    let span=i32(patch_settings.step*32u);
    var low=voxel_low(p.origin.xyz,span);
    low-=select(vec3<i32>(0),vec3<i32>(span),(direction<vec3<i32>(0)) & (low==p.origin.xyz) & (p.fraction.xyz==vec3<f32>(0.0)));
    let end=patch_delta(plane,p.origin[axis],p.fraction[axis],direction[axis]);
    for(var visit=0u;visit<25u;visit++) {
        let slot=mesh_page_slot(low);
        if slot==0xffffffffu || mesh_ready[slot]==0u {return false;}
        let boundary=low+select(vec3<i32>(0),vec3<i32>(span),direction>vec3<i32>(0));
        let times=array<vec2<f32>,3>(patch_delta(boundary.x,p.origin.x,p.fraction.x,direction.x),
            patch_delta(boundary.y,p.origin.y,p.fraction.y,direction.y),patch_delta(boundary.z,p.origin.z,p.fraction.z,direction.z));
        var next=0u;
        for(var a=1u;a<3u;a++) {if patch_compare(times[a],abs(rd[a]),times[next],abs(rd[next]))<0 {next=a;}}
        if patch_compare(end,abs(rd[axis]),times[next],abs(rd[next]))<=0 {return true;}
        for(var a=0u;a<3u;a++) {if a==next || patch_compare(times[a],abs(rd[a]),times[next],abs(rd[next]))==0 {low[a]+=direction[a]*span;}}
    }
    return false;
}
fn mesh_candidate(index:u32,candidate:vec4<u32>) {
    if ((candidate.x>>21u)&3u)==0u {return;}
    let rd=primary_hits[index].normal;
    let step=i32(patch_settings.step);
    var start=voxel_low(p.origin.xyz,step);
    start-=select(vec3<i32>(0),vec3<i32>(step),(rd<vec3<f32>(0.0)) & (start==p.origin.xyz) & (p.fraction.xyz==vec3<f32>(0.0)));
    if stored_cached_material(stored_cached_page(start),start)!=0u {return;}
    // Preserve very near intersections that the internal projection could clip.
    // Exactly-on-plane starts already own the interval in the ray direction.
    let first=start+select(vec3<i32>(0),vec3<i32>(step),rd>vec3<f32>(0.0));
    let first_delta=abs(vec3<f32>(first-p.origin.xyz)-p.fraction.xyz);
    if any(first_delta<abs(rd)*(0.001*f32(step))) {return;}
    let face=(candidate.x>>18u)&7u;let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    if rd[axis]==0.0 || (rd[axis]>0.0)!=((face&1u)!=0u) {return;}
    let local=vec3<i32>(i32(candidate.x&63u),i32((candidate.x>>6u)&63u),i32((candidate.x>>12u)&63u));
    let low=mesh_low(candidate.y>>12u)+local*step;
    let delta=vec3<f32>(low-p.origin.xyz)-p.fraction.xyz;
    let t=delta[axis]/rd[axis];
    if t<=0.0 || t*0.1>p.settings.x {return;}
    let point=(rd*t-delta)/f32(step);
    let width=f32(candidate.y&63u);let height=f32((candidate.y>>6u)&63u);
    // Never clamp an out-of-rectangle pixel ray onto nearby geometry.
    if point[u]<0.0 || point[u]>=width || point[v]<0.0 || point[v]>=height {return;}
    // Conservative rasterization retains every potentially earlier face. A
    // pixel near a transverse cell plane still has ambiguous f32 interval/depth
    // ordering, including internal planes of a merged rectangle. Exact DDA
    // handles this band; it is a fallback guard, not a displaced surface.
    if abs(point[u]-round(point[u]))<0.001 || abs(point[v]-round(point[v]))<0.001 {return;}
    if !mesh_prefix(low[axis],axis,rd) {return;}
    var cell=low;cell[u]+=i32(floor(point[u]))*step;cell[v]+=i32(floor(point[v]))*step;
    cell[axis]-=select(0,step,(face&1u)==0u);
    let material=(candidate.x>>21u)&3u;
    if stored_cached_material(stored_cached_page(cell),cell)!=material {return;}
    // Bit 26 attributes this hit to raster; bit 27 is the shared cache completion
    // contract. Existing GBuffer and engine lighting consume the canonical hit.
    primary_hits[index]=Hit(voxel_sample(cell,step),0x8c000001u|(material<<8u)|((face+1u)<<28u),rd,t*0.1);
}

@compute @workgroup_size(8,8)
fn mesh_resolve(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=vec2<u32>(p.screen.xy)) || p.settings.z<=0.0 {return;}
    mesh_candidate(id.x+id.y*u32(p.screen.x),textureLoad(mesh_visibility,vec2<i32>(id.xy),0));
}
