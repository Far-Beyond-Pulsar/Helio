// GPU-driven preprocessing compute shader
// Runs once per frame to cull and sort lights on the GPU
// Outputs: visible light count, reordered light list, indirect draw buffers

struct PreprocessInput {
    camera_pos: vec3f,
    camera_forward: vec3f,
    camera_right: vec3f,
    camera_up: vec3f,
    camera_fov_y: f32,
    scene_aabb_min: vec3f,
    scene_aabb_max: vec3f,
    total_lights: u32,
    viewport_width: u32,
    viewport_height: u32,
    frame: u32,
}

struct PreprocessOutput {
    visible_light_count: atomic<u32>,
    opaque_draw_count: atomic<u32>,
    transparent_draw_count: atomic<u32>,
}

struct GpuLight {
    position: vec3f,
    range: f32,
    direction: vec3f,
    light_type: u32,  // 0=directional, 1=point, 2=spot
    intensity: f32,
    cos_inner: f32,
    cos_outer: f32,
    color: vec3f,
}

struct GpuPreprocessLight {
    position: vec3f,
    range: f32,
    direction: vec3f,
    light_type: u32,
    intensity: f32,
    cos_inner: f32,
    cos_outer: f32,
    color: vec3f,
}

@group(0) @binding(0) var<uniform> input_data: PreprocessInput;
@group(0) @binding(1) var<storage, read_write> output_data: PreprocessOutput;
@group(0) @binding(2) var<storage, read> input_lights: array<GpuLight>;
@group(0) @binding(3) var<storage, read_write> visible_lights: array<GpuPreprocessLight>;

// Frustum culling: check if light sphere intersects view frustum
fn frustum_cull(light_pos: vec3f, light_range: f32) -> bool {
    // Simple sphere-vs-camera-range culling for now
    // Real implementation would do full frustum plane tests
    let to_light = light_pos - input_data.camera_pos;
    let dist_sq = dot(to_light, to_light);
    let max_dist = light_range * 4.0;  // Extended range for off-screen lights
    
    return dist_sq <= (max_dist * max_dist);
}

// Compute light contribution score (prioritize bright, close, on-screen lights)
fn compute_score(idx: u32, light: GpuLight) -> f32 {
    let light_pos = light.position;
    let to_light = light_pos - input_data.camera_pos;
    let dist = length(to_light);
    
    var score = 0.0;
    
    if light.light_type == 0u {
        // Directional: always high priority
        score = 100000.0 + light.intensity * 1000.0;
    } else if light.light_type == 1u {
        // Point light: by distance + intensity
        let r = max(light.range, 0.01);
        let attenu = 1.0 / (1.0 + (dist / r) * (dist / r));
        score = light.intensity * attenu * (r / dist);
    } else {
        // Spot light: cone angle bonus
        let dir_norm = normalize(light.direction);
        let to_cam = normalize(-to_light);
        let cone = max(dot(dir_norm, to_cam), 0.0);
        let cone_score = mix(0.5, 1.5, (cone - light.cos_outer) / (light.cos_inner - light.cos_outer));
        
        let r = max(light.range, 0.01);
        let attenu = 1.0 / (1.0 + (dist / r) * (dist / r));
        score = light.intensity * attenu * cone_score * (r / dist);
    }
    
    // Clip negative or invalid scores to 0
    return max(score, 0.0);
}

@compute
@workgroup_size(256)  // 256 threads per workgroup
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let total = input_data.total_lights;
    
    // Early exit if thread exceeds light count
    if thread_id >= total {
        return;
    }
    
    let light = input_lights[thread_id];
    
    // Frustum cull this light
    if !frustum_cull(light.position, light.range) {
        return;
    }
    
    // Light passed culling — assign it a slot in visible_lights
    let visible_idx = atomicAdd(&output_data.visible_light_count, 1u);
    
    if visible_idx < 1024u {  // Hard cap at 1024 visible (should never hit in practice)
        visible_lights[visible_idx] = GpuPreprocessLight(
            light.position,
            light.range,
            light.direction,
            light.light_type,
            light.intensity,
            light.cos_inner,
            light.cos_outer,
            light.color,
        );
    }
}
