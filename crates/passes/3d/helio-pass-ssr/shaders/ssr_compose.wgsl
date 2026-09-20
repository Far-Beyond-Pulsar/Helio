//!use helio_prelude
@group(0) @binding(0) var<storage,read> cameras: array<Camera,2>;
@group(1) @binding(0) var normals: texture_2d<f32>;
@group(1) @binding(1) var orm: texture_2d<f32>;
@group(1) @binding(2) var emissive: texture_2d<f32>;
@group(1) @binding(3) var depth: texture_depth_2d;
@group(1) @binding(4) var reflection: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p=array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
    return vec4<f32>(p[i],0.0,1.0);
}
@fragment fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p=vec2<i32>(position.xy);
    let d=textureLoad(depth,p,0);
    if d>=1.0 { discard; }
    let uv=position.xy/vec2<f32>(textureDimensions(depth));
    let reflection_dims=textureDimensions(reflection);
    let reflection_pos=uv*vec2<f32>(reflection_dims)-vec2<f32>(0.5);
    let reflection_base=vec2<i32>(floor(reflection_pos));
    let reflection_frac=fract(reflection_pos);
    let reflection_max=vec2<i32>(reflection_dims)-vec2<i32>(1);
    let p00=clamp(reflection_base,vec2<i32>(0),reflection_max);
    let p10=clamp(reflection_base+vec2<i32>(1,0),vec2<i32>(0),reflection_max);
    let p01=clamp(reflection_base+vec2<i32>(0,1),vec2<i32>(0),reflection_max);
    let p11=clamp(reflection_base+vec2<i32>(1),vec2<i32>(0),reflection_max);
    let hit=mix(
        mix(textureLoad(reflection,p00,0),textureLoad(reflection,p10,0),reflection_frac.x),
        mix(textureLoad(reflection,p01,0),textureLoad(reflection,p11,0),reflection_frac.x),
        reflection_frac.y,
    );
    let n=textureLoad(normals,p,0);
    let material=textureLoad(orm,p,0);
    let f0=clamp(vec3<f32>(n.w,material.a,textureLoad(emissive,p,0).a),vec3<f32>(0.0),vec3<f32>(0.999));
    let world=helio_world_from_depth(cameras[0].view_proj_inv,uv,d);
    let v=normalize(cameras[0].position_near.xyz-world);
    let ndv=max(dot(helio_gbuffer_normal(n.xyz),v),0.0);
    let f=f0+(max(vec3<f32>(1.0-material.g),f0)-f0)*pow(1.0-ndv,5.0);
    return vec4<f32>(max(hit.rgb,vec3<f32>(0.0))*clamp(hit.a,0.0,1.0)*f*clamp(material.r,0.0,1.0),0.0);
}
