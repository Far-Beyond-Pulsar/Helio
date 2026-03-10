// GPU-driven preprocessing compute shader
// Runs once per frame to cull and sort lights on the GPU
// Outputs: visible light count, reordered light list, indirect draw buffers

struct PreprocessInput {
    camera_pos:     vec3f,
    _pad0:          u32,
    camera_forward: vec3f,
    _pad1:          u32,
    camera_right:   vec3f,
    _pad2:          u32,
    camera_up:      vec3f,
    camera_fov_y:   f32,
    total_lights:   u32,
    viewport_width: u32,
    viewport_height: u32,
    frame:          u32,
    // Six frustum planes (Gribb-Hartmann).  Each vec4 is (nx, ny, nz, d)
    // with the INSIDE of the frustum satisfying dot(n,p)+d >= 0.
    // Planes are NOT unit-length; the sphere test normalises on the fly.
    frustum_planes: array<vec4f, 6>,
}

struct PreprocessOutput {
    visible_light_count:    atomic<u32>,
    opaque_draw_count:      atomic<u32>,
    transparent_draw_count: atomic<u32>,
}

struct GpuLight {
    position:   vec3f,
    range:      f32,
    direction:  vec3f,
    light_type: u32,   // 0=directional, 1=point, 2=spot
    intensity:  f32,
    cos_inner:  f32,
    cos_outer:  f32,
    color:      vec3f,
}

struct GpuPreprocessLight {
    position:   vec3f,
    range:      f32,
    direction:  vec3f,
    light_type: u32,
    intensity:  f32,
    cos_inner:  f32,
    cos_outer:  f32,
    color:      vec3f,
}

@group(0) @binding(0) var<uniform>            input_data:     PreprocessInput;
@group(0) @binding(1) var<storage, read_write> output_data:    PreprocessOutput;
@group(0) @binding(2) var<storage, read>       input_lights:   array<GpuLight>;
@group(0) @binding(3) var<storage, read_write> visible_lights: array<GpuPreprocessLight>;

// ── Proper sphere-vs-frustum test (bounds-based, NOT centre-point-based) ─────
//
// Returns true if the sphere (centre, radius) intersects the frustum.
// Tests all 6 planes; a sphere is outside if its centre is more than `radius`
// behind any single plane.
fn frustum_cull_sphere(centre: vec3f, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane  = input_data.frustum_planes[i];
        let normal = plane.xyz;
        // Signed distance of the centre (un-normalised).
        // A positive value means the centre is on the inside of this plane.
        let dist = dot(normal, centre) + plane.w;
        // If the centre is more than `radius * |normal|` behind this plane,
        // the entire sphere is outside the frustum.
        if dist < -(radius * length(normal)) {
            return false;
        }
    }
    return true;
}

// Compute light contribution score (prioritise bright, close, on-screen lights)
fn compute_score(light: GpuLight) -> f32 {
    let to_light = light.position - input_data.camera_pos;
    let dist = max(length(to_light), 0.001);

    var score = 0.0;

    if light.light_type == 0u {
        // Directional: always highest priority
        score = 100000.0 + light.intensity * 1000.0;
    } else if light.light_type == 1u {
        // Point light: by distance + intensity
        let r = max(light.range, 0.01);
        let attenu = 1.0 / (1.0 + (dist / r) * (dist / r));
        score = light.intensity * attenu * (r / dist);
    } else {
        // Spot light: cone angle bonus
        let dir_norm  = normalize(light.direction);
        let to_cam    = normalize(-to_light);
        let cos_angle = dot(dir_norm, to_cam);
        let cone_t    = saturate((cos_angle - light.cos_outer) /
                                  max(light.cos_inner - light.cos_outer, 0.001));
        let cone_score = mix(0.5, 1.5, cone_t);
        let r = max(light.range, 0.01);
        let attenu = 1.0 / (1.0 + (dist / r) * (dist / r));
        score = light.intensity * attenu * cone_score * (r / dist);
    }

    return max(score, 0.0);
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let total     = input_data.total_lights;

    if thread_id >= total { return; }

    let light = input_lights[thread_id];

    // Directional lights always pass (they're infinite, no bounding sphere).
    let passes = (light.light_type == 0u) ||
                 frustum_cull_sphere(light.position, light.range);
    if !passes { return; }

    // Light passed culling — atomically claim a slot in visible_lights.
    let visible_idx = atomicAdd(&output_data.visible_light_count, 1u);

    if visible_idx < 1024u {
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
