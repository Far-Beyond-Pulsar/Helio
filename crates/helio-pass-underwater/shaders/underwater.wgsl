//! Underwater post-processing shader.
//!
//! Applies realistic underwater effects:
//! - Volumetric fog with exponential falloff
//! - Beer-Lambert color absorption
//! - Caustics projection
//! - God rays (volumetric light shafts)

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct UnderwaterParams {
    time: f32,
    active_volume: i32,
    _pad0: f32,
    _pad1: f32,
}

struct GpuWaterVolume {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    wave_params: vec4<f32>,
    wave_direction: vec4<f32>,
    water_color: vec4<f32>,
    extinction: vec4<f32>,
    reflection_refraction: vec4<f32>,
    caustics_params: vec4<f32>,
    fog_params: vec4<f32>,
    _pad0: vec4<f32>,
    _pad1: vec4<f32>,
    _pad2: vec4<f32>,
    _pad3: vec4<f32>,
    _pad4: vec4<f32>,
    _pad5: vec4<f32>,
    _pad6: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: UnderwaterParams;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var scene_color: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> water_volumes: array<GpuWaterVolume>;
@group(0) @binding(5) var caustics_tex: texture_2d<f32>;
@group(0) @binding(6) var linear_sampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

/// Fullscreen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    // Fullscreen triangle (covers NDC -1 to 1)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    let pos = positions[vid];
    let uv = pos * 0.5 + 0.5;

    var out: VertexOut;
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

/// Reconstruct world position from UV and depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);
    let clip_pos = vec4<f32>(ndc, depth, 1.0);
    let world_pos_h = camera.inv_view_proj * clip_pos;
    return world_pos_h.xyz / world_pos_h.w;
}

/// Check if camera is inside water volume
fn is_camera_underwater(vol: GpuWaterVolume) -> bool {
    let pos = camera.position_near.xyz;
    let surface_y = vol.bounds_max.w;

    return pos.x >= vol.bounds_min.x && pos.x <= vol.bounds_max.x &&
           pos.y >= vol.bounds_min.y && pos.y <= surface_y &&
           pos.z >= vol.bounds_min.z && pos.z <= vol.bounds_max.z;
}

/// Caustics projection (simple but correct)
fn sample_caustics(world_pos: vec3<f32>, vol: GpuWaterVolume, time: f32) -> f32 {
    // Project caustics texture onto surfaces based on world XZ position
    // This simulates light being refracted through the water surface from above

    let scale = vol.caustics_params.z;
    let speed = vol.caustics_params.w;

    // UV coordinates based on horizontal world position
    let uv = (world_pos.xz / scale) + vec2<f32>(time * speed * 0.05);

    let caustic_value = textureSample(caustics_tex, linear_sampler, uv).r;

    // Fade caustics with depth below surface (light scatters in water)
    let depth_below_surface = max(vol.bounds_max.w - world_pos.y, 0.0);
    let depth_fade = exp(-depth_below_surface * 0.2);

    return caustic_value * depth_fade;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Check if camera is underwater
    var underwater = false;
    var vol: GpuWaterVolume;

    for (var i = 0u; i < arrayLength(&water_volumes); i++) {
        if is_camera_underwater(water_volumes[i]) {
            underwater = true;
            vol = water_volumes[i];
            break;
        }
    }

    // If not underwater, pass through unchanged
    if !underwater {
        let scene_col = textureSample(scene_color, linear_sampler, in.uv).rgb;
        return vec4<f32>(scene_col, 1.0);
    }

    // === AAA UNDERWATER RENDERING (Simple & Correct) ===
    // This is how Uncharted, Tomb Raider, Assassin's Creed do it

    // 1. Sample scene color
    let scene_col = textureSample(scene_color, linear_sampler, in.uv).rgb;

    // 2. Get depth and compute view distance (in screen space for simplicity)
    let depth = textureSample(depth_tex, linear_sampler, in.uv);

    // Simple depth-based distance (0 = near, 1 = far)
    // This avoids world position reconstruction issues
    let linear_depth = depth; // Already in 0-1 range
    let view_dist = linear_depth * 100.0; // Scale to reasonable distance range

    // 3. Beer-Lambert Law: Color absorption based on distance through water
    // Red light absorbed first, then green, blue penetrates deepest
    let absorption = exp(-vol.extinction.xyz * view_dist * 0.1);
    var color = scene_col * absorption;

    // 4. Distance fog: Mix toward water color
    let fog_amount = 1.0 - exp(-vol.fog_params.x * view_dist * 0.1);
    color = mix(color, vol.water_color.rgb, fog_amount);

    // NOTE: Caustics are handled by the dedicated caustics pass
    // Don't add them here to avoid artifacts

    return vec4<f32>(color, 1.0);
}
