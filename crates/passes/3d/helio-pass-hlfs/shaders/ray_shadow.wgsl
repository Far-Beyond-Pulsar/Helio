@group(3) @binding(0) var acc_struct: acceleration_structure;
fn rt_shadow(light: GpuLight, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    if light.shadow_index==INVALID_LIGHT { return 1.0; }
    if screen_occluded(light,position,normal) { return 0.0; }
    let origin=position+normal*0.004;
    let inc=incident(light,origin);
    var distance=9999.0;
    if light.light_type!=0u { distance=length(light.position_range.xyz-origin)-0.005; }
    if distance<=0.005 { return 1.0; }
    var query: ray_query;
    rayQueryInitialize(&query,acc_struct,RayDesc(0x01u,0xffu,0.005,distance,origin,inc.direction));
    while rayQueryProceed(&query) {}
    return select(0.0,1.0,rayQueryGetCommittedIntersection(&query).kind==RAY_QUERY_INTERSECTION_NONE);
}
