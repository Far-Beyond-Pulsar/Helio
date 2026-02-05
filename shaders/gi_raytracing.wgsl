const MAX_BOUNCES: i32 = 3;

struct Parameters {
    cam_position: vec3<f32>,
    depth: f32,
    cam_orientation: vec4<f32>,
    fov: vec2<f32>,
    torus_radius: f32,
    rotation_angle: f32,
};

var<uniform> parameters: Parameters;
var acc_struct: acceleration_structure;
var output: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.1415926;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

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
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(
        RAY_FLAG_NONE,
        0xFFu,
        0.001,
        parameters.depth,
        origin,
        direction
    ));
    rayQueryProceed(&rq);
    
    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
        // Sky gradient
        let t = 0.5 * (direction.y + 1.0);
        return mix(vec3<f32>(1.0), vec3<f32>(0.5, 0.7, 1.0), t);
    }
    
    // Get hit position and normal
    let hit_pos = origin + direction * intersection.t;
    let world_normal = normalize((intersection.object_to_world * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    
    // Material properties based on instance
    var albedo = vec3<f32>(0.8, 0.8, 0.8);
    var roughness = 0.5;
    var metallic = 0.0;
    
    if (intersection.instance_custom_data == 0u) {
        // Red cube - shiny metal
        albedo = vec3<f32>(1.0, 0.3, 0.3);
        roughness = 0.1;
        metallic = 0.9;
    } else if (intersection.instance_custom_data == 1u) {
        // Green sphere - very reflective
        albedo = vec3<f32>(0.3, 1.0, 0.3);
        roughness = 0.05;
        metallic = 1.0;
    } else {
        // Blue plane - matte
        albedo = vec3<f32>(0.3, 0.3, 1.0);
        roughness = 0.9;
        metallic = 0.0;
    }
    
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    
    // SHADOW RAY - check if in shadow
    var shadow_rq: ray_query;
    rayQueryInitialize(&shadow_rq, acc_struct, RayDesc(
        RAY_FLAG_NONE,
        0xFFu,
        0.01,
        100.0,
        hit_pos + world_normal * 0.001,  // Offset to avoid self-intersection
        light_dir
    ));
    rayQueryProceed(&shadow_rq);
    
    let shadow_intersection = rayQueryGetCommittedIntersection(&shadow_rq);
    let in_shadow = shadow_intersection.kind != RAY_QUERY_INTERSECTION_NONE;
    
    // Lighting
    let ambient = vec3<f32>(0.1);
    let diffuse = max(dot(world_normal, light_dir), 0.0) * (1.0 - metallic);
    let shadow_factor = select(1.0, 0.2, in_shadow);
    
    var color = albedo * (ambient + diffuse * shadow_factor);
    
    // REFLECTION RAY for shiny surfaces
    if (metallic > 0.5) {
        let reflect_dir = reflect(direction, world_normal);
        
        var reflect_rq: ray_query;
        rayQueryInitialize(&reflect_rq, acc_struct, RayDesc(
            RAY_FLAG_NONE,
            0xFFu,
            0.001,
            parameters.depth,
            hit_pos + world_normal * 0.001,
            reflect_dir
        ));
        rayQueryProceed(&reflect_rq);
        
        let reflect_intersection = rayQueryGetCommittedIntersection(&reflect_rq);
        if (reflect_intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
            let t = 0.5 * (reflect_dir.y + 1.0);
            let sky = mix(vec3<f32>(1.0), vec3<f32>(0.5, 0.7, 1.0), t);
            color = mix(color, sky, metallic);
        } else {
            // Hit another object - color by what we hit
            var reflect_albedo = vec3<f32>(0.5);
            if (reflect_intersection.instance_custom_data == 0u) {
                reflect_albedo = vec3<f32>(1.0, 0.3, 0.3);
            } else if (reflect_intersection.instance_custom_data == 1u) {
                reflect_albedo = vec3<f32>(0.3, 1.0, 0.3);
            } else {
                reflect_albedo = vec3<f32>(0.3, 0.3, 1.0);
            }
            color = mix(color, reflect_albedo * 0.8, metallic);
        }
    }
    
    return color;
}

@compute @workgroup_size(8, 8)
fn compute_gi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(output);
    if (any(global_id.xy >= target_size)) {
        return;
    }
    
    let half_size = vec2<f32>(target_size) * 0.5;
    let ndc = (vec2<f32>(global_id.xy) + vec2<f32>(0.5) - half_size) / half_size;
    
    let local_dir = vec3<f32>(ndc * tan(parameters.fov * 0.5), 1.0);
    let world_dir = normalize(qrot(parameters.cam_orientation, local_dir));
    
    let seed = global_id.x + global_id.y * target_size.x;
    let color = trace_gi_ray(parameters.cam_position, world_dir, seed);
    
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}

@vertex
fn draw_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(vi & 1u) * 4.0 - 1.0, f32(vi & 2u) * 2.0 - 1.0, 0.0, 1.0);
}

var input: texture_2d<f32>;

@fragment
fn draw_fs(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(input, vec2<i32>(frag_coord.xy), 0);
}
