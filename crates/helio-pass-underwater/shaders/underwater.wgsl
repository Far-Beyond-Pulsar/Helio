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
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad0: f32,
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
    let view_pos = camera.inv_proj * clip_pos;
    let world_pos = camera.inv_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    return world_pos.xyz;
}

/// Check if camera is inside water volume
fn is_camera_underwater(vol: GpuWaterVolume) -> bool {
    let pos = camera.position;
    let surface_y = vol.bounds_max.w;

    return pos.x >= vol.bounds_min.x && pos.x <= vol.bounds_max.x &&
           pos.y >= vol.bounds_min.y && pos.y <= surface_y &&
           pos.z >= vol.bounds_min.z && pos.z <= vol.bounds_max.z;
}

/// Compute volumetric fog with exponential falloff
fn compute_fog(view_dist: f32, density: f32) -> f32 {
    return 1.0 - exp(-density * view_dist);
}

/// Beer-Lambert color absorption
fn apply_absorption(color: vec3<f32>, extinction: vec3<f32>, distance: f32) -> vec3<f32> {
    return color * exp(-extinction * distance);
}

/// Sample caustics with world position
fn sample_caustics(world_pos: vec3<f32>, vol: GpuWaterVolume) -> f32 {
    let caustics_uv = world_pos.xz / vol.caustics_params.z;
    return textureSample(caustics_tex, linear_sampler, caustics_uv).r;
}

/// God rays (radial blur from sun direction)
fn compute_god_rays(uv: vec2<f32>, intensity: f32) -> f32 {
    // Sun direction projected to screen space (simplified - top of screen)
    let sun_uv = vec2<f32>(0.5, 0.2);
    let delta = uv - sun_uv;
    let dist = length(delta);

    // Radial samples
    let num_samples = 8u;
    var occlusion = 0.0;

    for (var i = 0u; i < num_samples; i++) {
        let t = f32(i) / f32(num_samples);
        let sample_uv = mix(uv, sun_uv, t * 0.5);
        let depth = textureSample(depth_tex, linear_sampler, sample_uv);

        // Simplified occlusion check
        if depth > 0.99 {
            occlusion += 1.0;
        }
    }

    occlusion /= f32(num_samples);

    // Radial falloff
    let falloff = 1.0 - smoothstep(0.0, 0.8, dist);

    return occlusion * falloff * intensity;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Sample scene color and depth
    let scene_col = textureSample(scene_color, linear_sampler, in.uv).rgb;
    let depth = textureSample(depth_tex, linear_sampler, in.uv);

    // Check if any water volume contains camera
    var active_volume_idx = -1;
    for (var i = 0u; i < arrayLength(&water_volumes); i++) {
        if is_camera_underwater(water_volumes[i]) {
            active_volume_idx = i32(i);
            break;
        }
    }

    // If not underwater, pass through
    if active_volume_idx < 0 {
        return vec4<f32>(scene_col, 1.0);
    }

    let vol = water_volumes[u32(active_volume_idx)];

    // Reconstruct world position
    let world_pos = reconstruct_world_pos(in.uv, depth);
    let view_dist = distance(camera.position, world_pos);

    // 1. Volumetric fog
    let fog_amount = compute_fog(view_dist, vol.fog_params.x);

    // 2. Color absorption (Beer-Lambert)
    let absorbed = apply_absorption(scene_col, vol.extinction.xyz, view_dist);

    // 3. Caustics (only on surfaces below water)
    var caustics_contribution = vec3<f32>(0.0);
    if world_pos.y < vol.bounds_max.w {
        let caustics = sample_caustics(world_pos, vol);
        caustics_contribution = vec3<f32>(caustics) * vol.caustics_params.y;
    }

    // 4. God rays
    let god_rays = compute_god_rays(in.uv, vol.fog_params.y);
    let god_rays_color = vec3<f32>(0.7, 0.85, 1.0) * god_rays * 0.3;

    // Composite
    let underwater_color = mix(
        absorbed + caustics_contribution,
        vol.water_color.rgb,
        fog_amount
    ) + god_rays_color;

    return vec4<f32>(underwater_color, 1.0);
}
