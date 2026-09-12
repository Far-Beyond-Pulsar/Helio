@group(3) @binding(0) var acc_struct: acceleration_structure;
// Same visibility interface as shadows.wgsl. Both sampling and reconstruction
// use this implementation in RayTraced mode; screen depth never decides a hit.
fn shadow_factor(id: u32, position: vec3<f32>, normal: vec3<f32>, pixel: vec2<f32>, frame: u32) -> f32 {
    let light=lights[id];
    let cast_shadow=select(light.shadow_index!=INVALID_LIGHT,(light._pad&2u)!=0u,(light._pad&1u)!=0u);
    if !cast_shadow { return 1.0; }
    let bias=max(0.0001,max(max(abs(position.x),abs(position.y)),abs(position.z))*0.000002);
    let origin=position+normal*bias;
    let inc=incident(light,origin);
    var distance=globals.ray_settings.x;
    if light.light_type!=0u { distance=length(light.position_range.xyz-origin)-bias; }
    if distance<=bias { return 1.0; }
    var query: ray_query;
    // Opaque triangles, terminate on the first accepted blocker.
    rayQueryInitialize(&query,acc_struct,RayDesc(0x05u,0xffu,bias,distance,origin,inc.direction));
    while rayQueryProceed(&query) {}
    return select(0.0,1.0,rayQueryGetCommittedIntersection(&query).kind==RAY_QUERY_INTERSECTION_NONE);
}
