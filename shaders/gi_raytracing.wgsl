struct GIParams {
    num_rays: u32,
    max_bounces: u32,
    intensity: f32,
};

var<uniform> params: GIParams;
var acc_struct: acceleration_structure;
var output: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.1415926;

fn random_unit_vector(seed: u32, index: u32) -> vec3<f32> {
    let theta = 2.0 * PI * fract(sin(f32(seed + index) * 12.9898) * 43758.5453);
    let phi = acos(2.0 * fract(sin(f32(seed + index + 1u) * 78.233) * 43758.5453) - 1.0);
    
    return vec3<f32>(
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    );
}

fn trace_gi_ray(origin: vec3<f32>, direction: vec3<f32>, seed: u32) -> vec3<f32> {
    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var ray_origin = origin;
    var ray_dir = direction;
    
    for (var bounce = 0u; bounce < params.max_bounces; bounce++) {
        var rq: ray_query;
        rayQueryInitialize(&rq, acc_struct, RayDesc(
            RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            0xFFu,
            0.001,
            1000.0,
            ray_origin,
            ray_dir
        ));
        rayQueryProceed(&rq);
        
        let intersection = rayQueryGetCommittedIntersection(&rq);
        if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
            radiance += throughput * vec3<f32>(0.2, 0.3, 0.4);
            break;
        }
        
        ray_origin += ray_dir * intersection.t;
        
        let hit_normal = vec3<f32>(0.0, 1.0, 0.0);
        let albedo = vec3<f32>(0.8);
        
        radiance += throughput * albedo * 0.1;
        
        let hemisphere_dir = random_unit_vector(seed, bounce * 2u);
        ray_dir = normalize(hit_normal + hemisphere_dir);
        throughput *= albedo / PI;
        
        if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01) {
            break;
        }
    }
    
    return radiance * params.intensity;
}

@compute @workgroup_size(8, 8)
fn compute_gi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(output);
    if (any(global_id.xy >= target_size)) {
        return;
    }
    
    let seed = global_id.x + global_id.y * target_size.x;
    
    var accumulated_gi = vec3<f32>(0.0);
    
    for (var ray = 0u; ray < params.num_rays; ray++) {
        let ray_dir = random_unit_vector(seed, ray);
        accumulated_gi += trace_gi_ray(vec3<f32>(0.0), ray_dir, seed + ray);
    }
    
    accumulated_gi /= f32(params.num_rays);
    
    textureStore(output, global_id.xy, vec4<f32>(accumulated_gi, 1.0));
}
