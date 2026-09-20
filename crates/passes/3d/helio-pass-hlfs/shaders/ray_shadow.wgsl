@group(3) @binding(0) var acc_struct: acceleration_structure;
@group(3) @binding(1) var<storage, read> ray_transmission: array<vec4<f32>>;
// Same visibility interface as shadows.wgsl. Both sampling and reconstruction
// use this implementation in RayTraced mode; screen depth never decides a hit.
fn shadow_factor(id: u32, position: vec3<f32>, normal: vec3<f32>, pixel: vec2<f32>, frame: u32) -> Visibility {
    let light=lights[id];
    let cast_shadow=select(light.shadow_index!=INVALID_LIGHT,(light._pad&2u)!=0u,(light._pad&1u)!=0u);
    if !cast_shadow { return Visibility(1.0); }
    // Perspective depth quantization grows into millimetres of world-space
    // error at distance. A position-magnitude-only offset self-shadows even a
    // flat receiver. Estimate the local error along its normal from one depth
    // ULP, in addition to the floor for transform/traversal rounding.
    let depth=textureLoad(gbuf_depth,vec2<i32>(pixel),0);
    let adjacent_depth=bitcast<f32>(bitcast<u32>(depth)+1u);
    let depth_error=abs(world_position(pixel,adjacent_depth)-position);
    let rounding=max(max(abs(position.x),abs(position.y)),abs(position.z))*0.000002;
    let velocity=textureLoad(gbuf_velocity,vec2<i32>(pixel),0);
    // Corrected G-buffer receivers no longer carry depth-buffer quantization
    // error. Bound the FP16 residual's relative rounding plus transform error;
    // legacy producers retain the depth-ULP bound instead of assuming precision.
    let corrected=globals.has_velocity!=0u && velocity.w==2.0;
    let error=select(dot(abs(normal),depth_error),abs(velocity.z)*0.001,corrected);
    let bias=max(0.0001,error+rounding);
    let origin=position+normal*bias;
    let inc=incident(light,origin);
    var distance=globals.ray_settings.x;
    if light.light_type!=0u { distance=length(light.position_range.xyz-origin)-0.0001; }
    if distance<=0.0001 { return Visibility(1.0); }
    var query: ray_query;
    if !USE_RAY_TRANSMISSION {
        // Preserve the opaque-only early-termination path.
        rayQueryInitialize(&query,acc_struct,RayDesc(0x05u,0xffu,0.0001,distance,origin,inc.direction));
        while rayQueryProceed(&query) {}
        return Visibility(select(0.0,1.0,rayQueryGetCommittedIntersection(&query).kind==RAY_QUERY_INTERSECTION_NONE));
    }
    // Force non-opaque candidates, including opaque BLAS geometry. Each crossed
    // triangle is a thin sheet. Never confirm a transparent hit: traversal must
    // keep visiting both farther and nearer candidates, in unspecified order.
    var throughput=vec3<f32>(1.0);
    rayQueryInitialize(&query,acc_struct,RayDesc(0x02u,0xffu,0.0001,distance,origin,inc.direction));
    while rayQueryProceed(&query) {
        let hit=rayQueryGetCandidateIntersection(&query);
        if hit.instance_index>=arrayLength(&ray_transmission) {
            rayQueryTerminate(&query);
            return Visibility(0.0);
        }
        throughput*=ray_transmission[hit.instance_index].rgb;
        if all(throughput==vec3<f32>(0.0)) {
            rayQueryTerminate(&query);
            return visibility_from_rgb(throughput);
        }
    }
    return visibility_from_rgb(throughput);
}
