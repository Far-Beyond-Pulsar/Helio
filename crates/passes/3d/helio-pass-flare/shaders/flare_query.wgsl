// ── Flare Query (Compute) ─────────────────────────────────────────────────────
//
// Reads the light storage buffer and depth texture, projects each flare-enabled
// light to screen space, checks occlusion against the depth buffer, and writes
// a compacted list of visible flares.

@group(0) @binding(0) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(1) var<storage, read_write> flare_queries: array<GpuFlareQuery>;
@group(0) @binding(2) var<storage, read_write> flare_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> cameras: array<Camera, CAMERA_SLOTS>;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var<uniform> flare_uniforms: FlareUniforms;

// must equal CAMERA_SLOTS in libhelio::camera
const CAMERA_SLOTS: u32 = 7u;

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
};

struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
    god_rays_enabled:  u32,
    god_rays_density:  f32,
    god_rays_weight:   f32,
    god_rays_decay:    f32,
    god_rays_exposure: f32,
    flare_enabled:      u32,
    flare_type:         u32,
    flare_intensity:    f32,
    flare_scale:        f32,
    flare_tint_r:       f32,
    flare_tint_g:       f32,
    flare_tint_b:       f32,
    ies_profile_index:    i32,
    light_function_index: i32,
    ies_angle_scale:      f32,
    ies_angle_offset:     f32,
};

struct GpuFlareQuery {
    screen_pos:      vec2<f32>,
    screen_depth:    f32,
    light_intensity: f32,
    light_color:     vec3<f32>,
    light_index:     u32,
};

struct FlareUniforms {
    light_count:   u32,
    max_flares:    u32,
    screen_width:  f32,
    screen_height: f32,
};

@compute @workgroup_size(64)
fn cs_flare_query(@builtin(global_invocation_id) gid: vec3<u32>) {
    let light_idx = gid.x;
    if light_idx >= flare_uniforms.light_count { return; }

    let light = lights[light_idx];
    if light.flare_enabled == 0u { return; }

    let world_pos = vec4<f32>(light.position_range.xyz, 1.0);
    let clip_pos = cameras[0].view_proj * world_pos;
    let ndc = clip_pos.xyz / clip_pos.w;

    if ndc.z < 0.0 || ndc.z > 1.0 { return; }

    let screen_uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if (screen_uv.x < 0.0 || screen_uv.x > 1.0 ||
        screen_uv.y < 0.0 || screen_uv.y > 1.0) { return; }

    let screen_pos = screen_uv * vec2<f32>(flare_uniforms.screen_width, flare_uniforms.screen_height);

    if light.light_type != 0u {
        var depth_sum: f32 = 0.0;
        for (var dy = -1i; dy <= 1i; dy++) {
            for (var dx = -1i; dx <= 1i; dx++) {
                let p = vec2<i32>(i32(screen_pos.x) + dx, i32(screen_pos.y) + dy);
                depth_sum += textureLoad(depth_tex, p, 0);
            }
        }
        let avg_depth = depth_sum / 9.0;
        if ndc.z > avg_depth + 0.0005 { return; }
    }

    let idx = atomicAdd(&flare_count[0], 1u);
    if idx >= flare_uniforms.max_flares { return; }

    let tint = vec3<f32>(light.flare_tint_r, light.flare_tint_g, light.flare_tint_b);
    flare_queries[idx] = GpuFlareQuery(
        screen_pos.xy,
        ndc.z,
        light.color_intensity.w * light.flare_intensity,
        light.color_intensity.xyz * tint,
        light_idx,
    );
}
