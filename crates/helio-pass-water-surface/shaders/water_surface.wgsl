//! Water surface rendering shader.
//!
//! Implements realistic water surfaces with:
//! - Gerstner wave displacement and normals
//! - Screen-space reflections (SSR)
//! - Screen-space refraction
//! - Fresnel-based blending
//! - Foam generation
//! - Smooth depth fade

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

struct WaterParams {
    time: f32,
    ssr_steps: u32,
    ssr_step_size: f32,
    debug_flags: u32,
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

// Bind group 0
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: WaterParams;

// Bind group 1
@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(2) var scene_color: texture_2d<f32>;
@group(1) @binding(3) var linear_sampler: sampler;

// Bind group 2
@group(2) @binding(0) var<storage, read> water_volumes: array<GpuWaterVolume>;
@group(2) @binding(1) var caustics_tex: texture_2d<f32>;
@group(2) @binding(2) var caustics_sampler: sampler;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) view_ray: vec3<f32>,
    @location(2) @interpolate(flat) volume_idx: u32,
}

// Constants
const PI = 3.14159265359;

/// Generate fullscreen quad vertices
@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) inst: u32
) -> VertexOut {
    // Fullscreen quad vertices (NDC)
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );

    let pos = positions[vid];
    let clip_pos = vec4<f32>(pos, 0.0, 1.0);

    // Reconstruct view ray for ray-marching
    let ndc = vec4<f32>(pos.x, pos.y, 1.0, 1.0);
    let view_pos = camera.inv_proj * ndc;
    let view_ray = (camera.inv_view * vec4<f32>(view_pos.xyz / view_pos.w, 0.0)).xyz;

    var out: VertexOut;
    out.clip_pos = clip_pos;
    out.world_pos = camera.position + view_ray;
    out.view_ray = normalize(view_ray);
    out.volume_idx = inst;
    return out;
}

/// Multi-octave Gerstner wave displacement
fn gerstner_waves(pos: vec2<f32>, time: f32, vol: GpuWaterVolume) -> vec3<f32> {
    let amplitude = vol.wave_params.x;
    let frequency = vol.wave_params.y;
    let speed = vol.wave_params.z;
    let steepness = vol.wave_params.w;
    let direction = normalize(vol.wave_direction.xy);

    var height = 0.0;
    var dx = 0.0;
    var dz = 0.0;

    // 4 octaves
    for (var i = 0u; i < 4u; i++) {
        let scale = pow(0.6, f32(i));
        let freq = frequency * pow(1.7, f32(i));
        let spd = speed * pow(1.3, f32(i));

        let k = freq;
        let a = amplitude * scale;
        let d = direction;
        let q = steepness / (k * a * 4.0);

        let phase = k * dot(d, pos) - spd * time;
        let c = cos(phase);
        let s = sin(phase);

        height += a * s;
        dx += q * a * d.x * c;
        dz += q * a * d.y * c;
    }

    return vec3<f32>(height, dx, dz);
}

/// Calculate water surface normal from wave derivatives
fn calculate_normal(wave_result: vec3<f32>) -> vec3<f32> {
    let dx = wave_result.y;
    let dz = wave_result.z;
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

/// Project world position to screen UV
fn world_to_screen_uv(world_pos: vec3<f32>) -> vec2<f32> {
    let clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = clip_pos.xy / clip_pos.w;
    return ndc * 0.5 + 0.5;
}

/// Reconstruct world position from screen UV and depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);
    let clip_pos = vec4<f32>(ndc, depth, 1.0);
    let view_pos = camera.inv_proj * clip_pos;
    let world_pos = camera.inv_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    return world_pos.xyz;
}

/// Screen-space reflection ray-marching
fn screen_space_reflection(
    origin: vec3<f32>,
    reflect_dir: vec3<f32>,
    max_steps: u32,
    step_size: f32
) -> vec4<f32> {
    var ray_pos = origin;
    let ray_step = reflect_dir * step_size;

    for (var i = 0u; i < max_steps; i++) {
        ray_pos += ray_step;

        let uv = world_to_screen_uv(ray_pos);
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            break; // Ray left screen
        }

        let scene_depth = textureSample(depth_tex, linear_sampler, uv);
        let scene_world = reconstruct_world_pos(uv, scene_depth);
        let ray_depth = distance(camera.position, ray_pos);
        let scene_depth_dist = distance(camera.position, scene_world);

        if ray_depth > scene_depth_dist && ray_depth - scene_depth_dist < step_size * 2.0 {
            // Hit! Sample scene color
            let color = textureSample(scene_color, linear_sampler, uv);
            return vec4<f32>(color.rgb, 1.0);
        }
    }

    // Miss: return sky color
    return vec4<f32>(0.5, 0.7, 1.0, 0.0);
}

/// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

/// Compute foam based on wave steepness
fn compute_foam(steepness: f32, threshold: f32, amount: f32) -> f32 {
    if steepness > threshold {
        let t = (steepness - threshold) / (1.0 - threshold);
        return t * amount;
    }
    return 0.0;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let vol = water_volumes[in.volume_idx];

    // Ray-march to find water surface intersection
    // For simplicity, use constant surface height (can enhance with ray-marching)
    let surface_y = vol.bounds_max.w;
    let t = (surface_y - camera.position.y) / in.view_ray.y;

    if t < 0.0 {
        discard; // Camera above water, ray pointing up
    }

    let surface_pos_flat = camera.position + in.view_ray * t;
    let world_xz = surface_pos_flat.xz;

    // Calculate Gerstner wave displacement
    let wave_result = gerstner_waves(world_xz, params.time, vol);
    let wave_height = wave_result.x;
    let surface_pos = vec3<f32>(world_xz.x, surface_y + wave_height, world_xz.y);

    // Surface normal
    let surface_normal = calculate_normal(wave_result);

    // Screen UV of this fragment
    let screen_uv = world_to_screen_uv(surface_pos);

    // Depth test: check if water surface is in front of scene geometry
    let scene_depth = textureSample(depth_tex, linear_sampler, screen_uv);
    let scene_world = reconstruct_world_pos(screen_uv, scene_depth);
    let water_depth = distance(camera.position, surface_pos);
    let scene_depth_dist = distance(camera.position, scene_world);

    if water_depth > scene_depth_dist {
        discard; // Water behind scene geometry
    }

    // View direction
    let view_dir = normalize(camera.position - surface_pos);

    // 1. Screen-space reflection
    let reflect_dir = reflect(-view_dir, surface_normal);
    let ssr_result = screen_space_reflection(
        surface_pos,
        reflect_dir,
        params.ssr_steps,
        params.ssr_step_size
    );

    // 2. Screen-space refraction (sample scene with distorted UV)
    let refract_dir = refract(-view_dir, surface_normal, 1.0 / 1.33);
    let refract_offset = surface_normal.xz * vol.reflection_refraction.y * 0.05;
    let refract_uv = clamp(screen_uv + refract_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let refract_color = textureSample(scene_color, linear_sampler, refract_uv).rgb;

    // 3. Fresnel blend
    let fresnel_power = vol.reflection_refraction.z;
    let cos_theta = max(dot(view_dir, surface_normal), 0.0);
    let fresnel = fresnel_schlick(cos_theta, 0.02);
    let reflection_strength = vol.reflection_refraction.x;

    let water_surface = mix(
        refract_color,
        ssr_result.rgb,
        fresnel * reflection_strength * ssr_result.a
    );

    // 4. Caustics (sample from tiled caustics texture)
    let caustics_uv = world_xz / vol.caustics_params.z;
    let caustics = textureSample(caustics_tex, caustics_sampler, caustics_uv).r;
    let caustics_contribution = caustics * vol.caustics_params.y * (1.0 - fresnel);

    // 5. Foam
    let steepness = length(vec2<f32>(wave_result.y, wave_result.z));
    let foam = compute_foam(steepness, vol.water_color.w, vol.extinction.w);

    // 6. Depth fade (smooth edges)
    let depth_diff = scene_depth_dist - water_depth;
    let depth_fade = smoothstep(0.0, 2.0, depth_diff);

    // Final color
    let base_color = vol.water_color.rgb;
    var final_color = mix(base_color, water_surface + caustics_contribution, depth_fade);
    final_color = mix(final_color, vec3<f32>(1.0), foam);

    // Alpha based on depth fade
    let alpha = depth_fade * (1.0 - foam * 0.5);

    return vec4<f32>(final_color, alpha);
}
