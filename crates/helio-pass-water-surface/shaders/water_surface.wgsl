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

/// Enhanced screen-space reflection with ray marching
fn screen_space_reflection(
    origin: vec3<f32>,
    reflect_dir: vec3<f32>,
    max_steps: u32,
    step_size: f32
) -> vec4<f32> {
    // Project reflection direction to screen space for ray marching
    let view_space_reflect = (camera.view * vec4<f32>(reflect_dir, 0.0)).xyz;

    // Start at surface position
    var ray_pos = origin;
    let base_uv = world_to_screen_uv(origin);

    // Ray march in world space, checking screen-space position
    var best_uv = base_uv;
    var confidence = 0.0;

    for (var i = 0u; i < max_steps; i++) {
        ray_pos += reflect_dir * step_size;

        let test_uv = world_to_screen_uv(ray_pos);

        // Check if ray is still on screen
        if test_uv.x < 0.0 || test_uv.x > 1.0 || test_uv.y < 0.0 || test_uv.y > 1.0 {
            break;
        }

        best_uv = test_uv;
        confidence = 1.0 - (f32(i) / f32(max_steps)); // Fade with distance
    }

    // Sample scene color at best reflection point
    let color = textureSample(scene_color, linear_sampler, best_uv);

    // Edge fade for screen boundaries
    let edge_fade = smoothstep(0.0, 0.1, best_uv.x) * smoothstep(0.0, 0.1, best_uv.y) *
                    smoothstep(1.0, 0.9, best_uv.x) * smoothstep(1.0, 0.9, best_uv.y);

    return vec4<f32>(color.rgb, confidence * edge_fade);
}

/// Fresnel-Schlick approximation (physically-based)
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let cos_clamped = clamp(cos_theta, 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - cos_clamped, 5.0);
}

/// Detail normal generation for fine ripples
fn generate_detail_normals(pos: vec2<f32>, time: f32, scale: f32) -> vec3<f32> {
    let offset = 0.01;

    // Sample multiple noise octaves for fine detail
    let h0 = sin(pos.x * scale + time * 0.5) * cos(pos.y * scale * 1.3 + time * 0.7) * 0.5 +
             sin(pos.x * scale * 2.3 - time * 0.8) * cos(pos.y * scale * 1.7 - time * 0.6) * 0.25;

    let h1 = sin((pos.x + offset) * scale + time * 0.5) * cos((pos.y) * scale * 1.3 + time * 0.7) * 0.5 +
             sin((pos.x + offset) * scale * 2.3 - time * 0.8) * cos((pos.y) * scale * 1.7 - time * 0.6) * 0.25;

    let h2 = sin((pos.x) * scale + time * 0.5) * cos((pos.y + offset) * scale * 1.3 + time * 0.7) * 0.5 +
             sin((pos.x) * scale * 2.3 - time * 0.8) * cos((pos.y + offset) * scale * 1.7 - time * 0.6) * 0.25;

    let dx = h1 - h0;
    let dy = h2 - h0;

    return normalize(vec3<f32>(-dx * 10.0, 1.0, -dy * 10.0));
}

/// Specular highlight (Blinn-Phong)
fn calculate_specular(view_dir: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>, shininess: f32) -> f32 {
    let halfway = normalize(view_dir + light_dir);
    let spec = pow(max(dot(normal, halfway), 0.0), shininess);
    return spec;
}

/// Subsurface scattering approximation
fn subsurface_scattering(view_dir: vec3<f32>, light_dir: vec3<f32>, normal: vec3<f32>, thickness: f32) -> f32 {
    let scatter_dir = light_dir + normal * 0.3;
    let scatter = pow(clamp(dot(view_dir, -scatter_dir), 0.0, 1.0), 4.0) * thickness;
    return scatter;
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
        // === AAA QUALITY WATER SURFACE ===

        // 1. ENHANCED NORMALS: Combine Gerstner waves with detail ripples
        let world_xz = in.world_pos.xz;
        let wave_result = gerstner_waves(world_xz, params.time, vol);
        let detail_normal = generate_detail_normals(world_xz, params.time, 15.0);

        // Blend wave normal with detail normals
        let blended_normal = normalize(surface_normal * 0.7 + detail_normal * 0.3);

        // 2. PHYSICALLY-BASED FRESNEL (water IOR ~1.333)
        let cos_theta = max(dot(view_dir, blended_normal), 0.0);
        let fresnel = fresnel_schlick(cos_theta, 0.02); // F0 for water

        // 3. CHROMATIC ABERRATION REFRACTION
        // Slightly different distortion per RGB channel for realistic dispersion
        let refract_strength = vol.reflection_refraction.y * 0.05;
        let distortion_base = blended_normal.xz * refract_strength;

        let refract_uv_r = clamp(screen_uv + distortion_base * 1.02, vec2<f32>(0.0), vec2<f32>(1.0));
        let refract_uv_g = clamp(screen_uv + distortion_base * 1.00, vec2<f32>(0.0), vec2<f32>(1.0));
        let refract_uv_b = clamp(screen_uv + distortion_base * 0.98, vec2<f32>(0.0), vec2<f32>(1.0));

        let refract_color = vec3<f32>(
            textureSample(scene_color, linear_sampler, refract_uv_r).r,
            textureSample(scene_color, linear_sampler, refract_uv_g).g,
            textureSample(scene_color, linear_sampler, refract_uv_b).b
        );

        // 4. HIGH-QUALITY SCREEN-SPACE REFLECTIONS
        let reflect_dir = reflect(-view_dir, blended_normal);
        let ssr_result = screen_space_reflection(
            surface_pos,
            reflect_dir,
            params.ssr_steps * 2u, // Double the steps for quality
            params.ssr_step_size * 0.5
        );

        // 5. SUBSURFACE SCATTERING (simulated)
        // Fake sun direction - you can expose this in params
        let sun_dir = normalize(vec3<f32>(0.5, 0.8, 0.3));
        let scatter = subsurface_scattering(view_dir, sun_dir, blended_normal, 0.3);
        let scatter_color = vol.water_color.rgb * scatter * 0.5;

        // 6. SPECULAR HIGHLIGHTS (sun reflection on waves)
        let specular = calculate_specular(view_dir, blended_normal, sun_dir, 256.0);
        let specular_color = vec3<f32>(1.0, 0.98, 0.95) * specular * 0.8;

        // 7. MIX REFRACTION AND REFLECTION
        // Use physically-based fresnel for mixing
        var water_color = mix(refract_color, ssr_result.rgb, fresnel * ssr_result.a);

        // Add subsurface scattering
        water_color += scatter_color;

        // Add specular highlights
        water_color += specular_color;

        // 8. WATER COLOR TINT (absorption)
        // Stronger tint at grazing angles (more water to look through)
        let absorption_factor = mix(0.15, 0.35, 1.0 - cos_theta);
        water_color = mix(water_color, vol.water_color.rgb, absorption_factor);

        // 9. ENHANCED FOAM
        let steepness = length(vec2<f32>(wave_result.y, wave_result.z));
        let foam_base = compute_foam(steepness, vol.water_color.w, vol.extinction.w);

        // Add detail foam (fine bubbles)
        let foam_detail = max(sin(world_xz.x * 25.0 + params.time * 2.0) *
                             cos(world_xz.y * 25.0 - params.time * 1.8) * 0.5 + 0.5, 0.0);
        let foam = foam_base + foam_detail * foam_base * 0.3;

        // Foam color with slight blue tint
        let foam_color = vec3<f32>(0.95, 0.97, 1.0);
        final_color = mix(water_color, foam_color, clamp(foam * 0.5, 0.0, 0.8));

        // 10. ALPHA with fresnel-based transparency
        alpha = mix(0.85, 0.98, fresnel);
    } else {
        // === SIDE/BOTTOM FACES: UNDERWATER VIEW ===

        // Add subtle detail normals even to flat faces
        let detail = generate_detail_normals(in.world_pos.xz, params.time, 20.0);
        let perturbed_normal = normalize(surface_normal + detail * 0.1);

        // Chromatic aberration refraction for underwater view
        let refract_strength = vol.reflection_refraction.y * 0.04;
        let distortion_base = perturbed_normal.xy * refract_strength;

        let refract_uv_r = clamp(screen_uv + distortion_base * 1.02, vec2<f32>(0.0), vec2<f32>(1.0));
        let refract_uv_g = clamp(screen_uv + distortion_base * 1.00, vec2<f32>(0.0), vec2<f32>(1.0));
        let refract_uv_b = clamp(screen_uv + distortion_base * 0.98, vec2<f32>(0.0), vec2<f32>(1.0));

        let refract_color = vec3<f32>(
            textureSample(scene_color, linear_sampler, refract_uv_r).r,
            textureSample(scene_color, linear_sampler, refract_uv_g).g,
            textureSample(scene_color, linear_sampler, refract_uv_b).b
        );

        // Depth-based absorption (Beer's law)
        let depth_in_water = abs(surface_pos.y - surface_y) + 0.1;
        let absorption = exp(-vol.extinction.rgb * depth_in_water * 0.8);

        // Apply absorption and add water color
        final_color = refract_color * absorption + vol.water_color.rgb * (1.0 - absorption.r) * 0.6;

        // Subtle subsurface glow
        let sun_dir = normalize(vec3<f32>(0.5, 0.8, 0.3));
        let scatter = subsurface_scattering(view_dir, sun_dir, perturbed_normal, 0.2);
        final_color += vol.water_color.rgb * scatter * 0.3;

        // Transparency based on depth
        alpha = mix(0.75, 0.92, clamp(depth_in_water * 0.5, 0.0, 1.0));
    }

    return vec4<f32>(final_color, alpha);
}
