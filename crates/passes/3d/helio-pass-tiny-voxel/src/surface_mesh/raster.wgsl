struct Camera { eye:vec4<f32>, right:vec4<f32>, up:vec4<f32>, forward:vec4<f32>, light:vec4<f32> }
@group(0) @binding(0) var<uniform> camera:Camera;
@group(0) @binding(1) var<storage,read> quads:array<Quad>;
struct Vertex {
    @builtin(position) clip:vec4<f32>,
    @location(0) @interpolate(flat) descriptor:u32,
    @location(1) @interpolate(flat) quad:u32,
}
@vertex fn vertex(@builtin(vertex_index) vertex_id:u32,@builtin(instance_index) index:u32)->Vertex {
    let q=quads[index];
    let point=quad_vertex(q,vertex_id);
    if point.w==0.0 {return Vertex(vec4<f32>(0.0,0.0,0.0,1.0),q.origin_face_material,index);}
    let position=point.xyz;
    let delta=position-camera.eye.xyz;
    let distance=dot(delta,camera.forward.xyz);
    let clip=vec4<f32>(dot(delta,camera.right.xyz)/(camera.eye.w*camera.right.w),
        dot(delta,camera.up.xyz)/camera.eye.w,camera.up.w,distance);
    return Vertex(clip,q.origin_face_material,index);
}
@fragment fn identity(vertex:Vertex)->@location(0) vec4<u32> {
    let q=quads[vertex.quad];
    let low=vec3<u32>(q.origin_face_material&63u,(q.origin_face_material>>6u)&63u,(q.origin_face_material>>12u)&63u);
    let face=(q.origin_face_material>>18u)&7u;let axis=face/2u;
    let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    var high=low;high[u]+=q.extent&63u;high[v]+=(q.extent>>6u)&63u;
    // Attribute interpolation inherits rasterized vertex quantization. Recover
    // the hit from the exact integer face plane and the actual pixel ray.
    let size=vec2<f32>(camera.light.w,camera.forward.w);
    let ndc=vertex.clip.xy/size*2.0-vec2<f32>(1.0);
    let rd=normalize(camera.forward.xyz+camera.right.xyz*(ndc.x*camera.eye.w*camera.right.w)
        -camera.up.xyz*(ndc.y*camera.eye.w));
    let distance=(f32(low[axis])-camera.eye[axis])/rd[axis];
    let local=camera.eye.xyz+rd*distance;
    var cell=vec3<u32>(max(floor(local),vec3<f32>(0.0)));
    cell[u]=clamp(cell[u],low[u],high[u]-1u);cell[v]=clamp(cell[v],low[v],high[v]-1u);
    cell[axis]=low[axis]-select(0u,1u,(face&1u)==0u);
    let material=(q.origin_face_material>>21u)&3u;
    return vec4<u32>(cell.x+32u*cell.y+1024u*cell.z,material|((face+1u)<<8u),
        bitcast<u32>(distance),vertex.quad);
}
@fragment fn lit(vertex:Vertex,@builtin(sample_index) sample:u32)->@location(0) vec4<f32> {
    let face=(vertex.descriptor>>18u)&7u;let material=(vertex.descriptor>>21u)&3u;
    var normal=vec3<f32>(0.0);normal[face/2u]=select(1.0,-1.0,(face&1u)!=0u);
    let palette=array<vec3<f32>,4>(vec3<f32>(0.0),vec3<f32>(0.18,0.42,0.055),vec3<f32>(0.35,0.12,0.035),vec3<f32>(0.46,0.49,0.56));
    let irradiance=0.12+0.88*max(0.0,dot(normal,camera.light.xyz));
    return vec4<f32>(palette[material]*irradiance,1.0);
}
