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
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
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
@group(1) @binding(0) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(1) var scene_color: texture_2d<f32>;
@group(1) @binding(2) var linear_sampler: sampler;

// Bind group 2
@group(2) @binding(0) var<storage, read> water_volumes: array<GpuWaterVolume>;
@group(2) @binding(1) var caustics_tex: texture_2d<f32>;
@group(2) @binding(2) var caustics_sampler: sampler;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) volume_idx: u32,
    @location(2) @interpolate(flat) face_id: u32,
}

// Constants
const PI = 3.14159265359;

/// Generate box vertices for water volume (36 vertices = 6 faces * 6 vertices per quad)
@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) inst: u32
) -> VertexOut {
    let vol = water_volumes[inst];
    let bounds_min = vol.bounds_min.xyz;
    let bounds_max = vol.bounds_max.xyz;
    let surface_y = vol.bounds_max.w;

    // Override bounds_max.y with surface height
    let adjusted_max = vec3<f32>(bounds_max.x, surface_y, bounds_max.z);

    // 6 vertices per face, 6 faces = 36 vertices
    let face_id = vid / 6u;
    let vert_id = vid % 6u;

    // Quad vertex pattern (2 triangles)
    var local_pos: vec3<f32>;

    // Define vertices for each face
    if face_id == 0u {
        // Top face (water surface with waves)
        let quad_verts = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
        );
        let uv = quad_verts[vert_id];
        local_pos = vec3<f32>(
            mix(bounds_min.x, adjusted_max.x, uv.x),
            adjusted_max.y,
            mix(bounds_min.z, adjusted_max.z, uv.y)
        );
    } else if face_id == 1u {
        // Bottom face
        let quad_verts = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
            vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
        );
        let uv = quad_verts[vert_id];
        local_pos = vec3<f32>(
            mix(bounds_min.x, adjusted_max.x, uv.x),
            bounds_min.y,
            mix(bounds_min.z, adjusted_max.z, uv.y)
        );
    } else if face_id == 2u {
        // Front face (+Z)
        let quad_verts = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
        );
        let uv = quad_verts[vert_id];
        local_pos = vec3<f32>(
            mix(bounds_min.x, adjusted_max.x, uv.x),
            mix(bounds_min.y, adjusted_max.y, uv.y),
            adjusted_max.z
        );
    } else if face_id == 3u {
        // Back face (-Z)
        let quad_verts = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
            vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
        );
        let uv = quad_verts[vert_id];
        local_pos = vec3<f32>(
            mix(bounds_min.x, adjusted_max.x, uv.x),
            mix(bounds_min.y, adjusted_max.y, uv.y),
            bounds_min.z
        );
    } else if face_id == 4u {
        // Right face (+X)
        let quad_verts = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
        );
        let uv = quad_verts[vert_id];
        local_pos = vec3<f32>(
            adjusted_max.x,
            mix(bounds_min.y, adjusted_max.y, uv.y),
            mix(bounds_min.z, adjusted_max.z, uv.x)
        );
    } else {
        // Left face (-X)
        let quad_verts = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
            vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
        );
        let uv = quad_verts[vert_id];
        local_pos = vec3<f32>(
            bounds_min.x,
            mix(bounds_min.y, adjusted_max.y, uv.y),
            mix(bounds_min.z, adjusted_max.z, uv.x)
        );
    }

    let clip_pos = camera.view_proj * vec4<f32>(local_pos, 1.0);

    var out: VertexOut;
    out.clip_pos = clip_pos;
    out.world_pos = local_pos;
    out.volume_idx = inst;
    out.face_id = face_id;
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

/// Simplified screen-space reflection (without depth buffer)
fn screen_space_reflection(
    origin: vec3<f32>,
    reflect_dir: vec3<f32>,
    max_steps: u32,
    step_size: f32
) -> vec4<f32> {
    // Simple reflection: offset screen UV based on reflection direction
    let view_space_reflect = (camera.view * vec4<f32>(reflect_dir, 0.0)).xyz;
    let reflect_offset = view_space_reflect.xy * 0.1; // Scale for subtle distortion

    let base_uv = world_to_screen_uv(origin);
    let reflect_uv = clamp(base_uv + reflect_offset, vec2<f32>(0.0), vec2<f32>(1.0));

    // Sample scene color at reflected UV
    let color = textureSample(scene_color, linear_sampler, reflect_uv);

    // Return with full confidence (always hit)
    return vec4<f32>(color.rgb, 1.0);
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
    let surface_y = vol.bounds_max.w;

    var surface_pos: vec3<f32>;
    var surface_normal: vec3<f32>;

    // Top face (face_id == 0): Apply Gerstner waves
    if in.face_id == 0u {
        let world_xz = in.world_pos.xz;
        let wave_result = gerstner_waves(world_xz, params.time, vol);
        let wave_height = wave_result.x;
        surface_pos = vec3<f32>(world_xz.x, surface_y + wave_height, world_xz.y);
        surface_normal = calculate_normal(wave_result);
    } else {
        // Other faces: flat geometry, compute geometric normal
        surface_pos = in.world_pos;

        // Compute face normal based on face_id
        if in.face_id == 1u {
            surface_normal = vec3<f32>(0.0, -1.0, 0.0); // Bottom
        } else if in.face_id == 2u {
            surface_normal = vec3<f32>(0.0, 0.0, 1.0); // Front
        } else if in.face_id == 3u {
            surface_normal = vec3<f32>(0.0, 0.0, -1.0); // Back
        } else if in.face_id == 4u {
            surface_normal = vec3<f32>(1.0, 0.0, 0.0); // Right
        } else {
            surface_normal = vec3<f32>(-1.0, 0.0, 0.0); // Left
        }
    }

    // Screen UV for sampling scene color/effects
    let screen_uv = world_to_screen_uv(in.world_pos);

    // View direction
    let view_dir = normalize(camera.position_near.xyz - surface_pos);

    // Different rendering for top face vs side/bottom faces
    var final_color: vec3<f32>;
    var alpha: f32;

    if in.face_id == 0u {
        // Top face: Water surface with reflection and refraction

        // Calculate fresnel for reflection/refraction mix
        let cos_theta = max(dot(view_dir, surface_normal), 0.0);
        let fresnel = fresnel_schlick(cos_theta, 0.02);

        // 1. Refraction - distort UV based on surface normal
        let normal_distortion = surface_normal.xz * vol.reflection_refraction.y * 0.03;
        let refract_uv = clamp(screen_uv + normal_distortion, vec2<f32>(0.0), vec2<f32>(1.0));
        let refract_color = textureSample(scene_color, linear_sampler, refract_uv).rgb;

        // 2. Reflection - sample scene in reflection direction
        let reflect_dir = reflect(-view_dir, surface_normal);
        let ssr_result = screen_space_reflection(
            surface_pos,
            reflect_dir,
            params.ssr_steps,
            params.ssr_step_size
        );

        // 3. Mix refraction and reflection based on fresnel
        // At grazing angles (fresnel high): more reflection
        // Looking straight down (fresnel low): more refraction
        var water_color = mix(refract_color, ssr_result.rgb, fresnel * ssr_result.a);

        // 4. Add subtle water tint (less intense than before)
        water_color = mix(water_color, vol.water_color.rgb, 0.1);

        // 5. Foam - only on steep wave crests, subtle white highlights
        let world_xz = in.world_pos.xz;
        let wave_result = gerstner_waves(world_xz, params.time, vol);
        let steepness = length(vec2<f32>(wave_result.y, wave_result.z));
        let foam = compute_foam(steepness, vol.water_color.w, vol.extinction.w);

        // Mix foam as a subtle highlight, not pure white
        final_color = mix(water_color, vec3<f32>(0.9, 0.95, 1.0), foam * 0.2);

        // More transparent for better see-through
        alpha = 0.7;
    } else {
        // Side/bottom faces: Transparent water with refraction only

        // Simple refraction for side faces
        let refract_offset = (surface_normal.xy * vol.reflection_refraction.y * 0.015);
        let refract_uv = clamp(screen_uv + refract_offset, vec2<f32>(0.0), vec2<f32>(1.0));
        let refract_color = textureSample(scene_color, linear_sampler, refract_uv).rgb;

        // Apply water color tint based on depth
        let depth_in_water = abs(surface_pos.y - surface_y);
        let absorption = exp(-vol.extinction.rgb * depth_in_water);
        final_color = refract_color * absorption + vol.water_color.rgb * (1.0 - absorption.r) * 0.3;

        // More transparent for side faces
        alpha = 0.4;
    }

    return vec4<f32>(final_color, alpha);
}
