// Water volume walls (sides + bottom) rendering
//
// Vertex:   Transforms volume box mesh to world space based on water bounds
// Fragment: Applies water color absorption and basic underwater lighting
//
// Bindings (same as surface shaders for compatibility)
//   0 camera           uniform
//   1  water_volumes    storage read
//   2  water_sim        texture_2d<f32>  (unused here, but kept for layout)
//   3  water_samp       sampler
//   4  caustics_tex     texture_2d<f32>
//   5  shared_samp      sampler
//   6  scene_color      texture_2d<f32>
//   7  viewport         uniform vec4f
//   8  depth_texture    texture_depth_2d
//   9  depth_sampler    sampler
//   10 gbuffer_normal   texture_2d<f32>

struct Camera {
    view:           mat4x4f,
    proj:           mat4x4f,
    view_proj:      mat4x4f,
    inv_view_proj:  mat4x4f,
    position_near:  vec4f,
    forward_far:    vec4f,
    jitter_frame:   vec4f,
    prev_view_proj: mat4x4f,
}

struct WaterVolume {
    bounds_min:            vec4f,
    bounds_max:            vec4f,  // w=surface_height
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,
    extinction:            vec4f,
    reflection_refraction: vec4f,
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,
    pad1: vec4f, pad2: vec4f, pad3: vec4f,
}

@group(0) @binding(0) var<uniform>       camera:         Camera;
@group(0) @binding(1) var<storage, read> volumes:        array<WaterVolume>;
@group(0) @binding(2) var water_sim:      texture_2d<f32>;
@group(0) @binding(3) var water_samp:     sampler;
@group(0) @binding(4) var caustics_tex:   texture_2d<f32>;
@group(0) @binding(5) var shared_samp:    sampler;
@group(0) @binding(6) var scene_color:    texture_2d<f32>;
@group(0) @binding(7) var<uniform>        viewport:       vec4f;
@group(0) @binding(8) var depth_texture:  texture_depth_2d;
@group(0) @binding(9) var depth_sampler:  sampler;
@group(0) @binding(10) var gbuffer_normal: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) simUV: vec2f,      // UV for sampling heightfield
    @location(2) fadeAlpha: f32,    // Fade factor for vertices above water
}

// Helper to convert world XZ to simulation UV
fn worldToSimUV(worldXZ: vec2f, bmin: vec3f, bmax: vec3f) -> vec2f {
    return vec2f(
        (worldXZ.x - bmin.x) / (bmax.x - bmin.x),
        (worldXZ.y - bmin.z) / (bmax.z - bmin.z)
    );
}

// Transform unit box vertex to water volume bounds
fn boxToWorld(boxPos: vec3f, bmin: vec3f, bmax: vec3f, surface_h: f32) -> vec3f {
    return vec3f(
        mix(bmin.x, bmax.x, (boxPos.x + 1.0) * 0.5),
        mix(bmin.y, surface_h, (boxPos.y + 1.0) * 0.5),
        mix(bmin.z, bmax.z, (boxPos.z + 1.0) * 0.5),
    );
}

@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol = volumes[0];
    let bmin = vol.bounds_min.xyz;
    let bmax = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;
    
    let worldPos = boxToWorld(position, bmin, bmax, surface_h);
    
    // Sample heightfield to get displaced water surface at this XZ position
    let simUV = worldToSimUV(worldPos.xz, bmin, bmax);
    let heightfield_sample = textureSampleLevel(water_sim, water_samp, simUV, 0.0);
    let displaced_height = surface_h + heightfield_sample.r * (surface_h - bmin.y);
    
    // Fade out vertices that are above the displaced water surface
    let height_diff = displaced_height - worldPos.y;
    let fadeAlpha = smoothstep(-0.2, 0.5, height_diff);  // Fade zone
    
    let clipPos = camera.view_proj * vec4f(worldPos, 1.0);

    var out: VertexOutput;
    out.position = clipPos;
    out.worldPos = worldPos;
    out.simUV = simUV;
    out.fadeAlpha = fadeAlpha;
    return out;
}

// Reconstruct world position from screen UV + depth
fn reconstruct_world_pos(uv: vec2f, depth: f32) -> vec3f {
    let ndc_xy = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let world_h = camera.inv_view_proj * vec4f(ndc_xy, depth, 1.0);
    return world_h.xyz / world_h.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Discard fragments above water surface
    if in.fadeAlpha < 0.01 {
        discard;
    }
    
    let vol = volumes[0];
    let camera_pos = camera.position_near.xyz;
    let bmin = vol.bounds_min.xyz;
    let bmax = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;
    
    // Determine if floor or wall
    let is_floor = abs(in.worldPos.y - bmin.y) < 0.05;
    
    // Calculate geometric normal (what direction this face is facing)
    var geom_normal: vec3f;
    if is_floor {
        geom_normal = vec3f(0.0, 1.0, 0.0);
    } else {
        // For walls, normal points away from water center horizontally
        let center_xz = (bmin.xz + bmax.xz) * 0.5;
        let to_center = vec3f(center_xz.x - in.worldPos.x, 0.0, center_xz.y - in.worldPos.z);
        geom_normal = normalize(-to_center);
    }
    
    // For floor, get detailed water normal from heightfield
    // For walls, use geometric normal with wave perturbation from offset UV
    var water_normal: vec3f;
    var sample_uv: vec2f;
    
    if is_floor {
        // Floor uses exact UV position
        sample_uv = in.simUV;
        // Full water surface normal (5 iterations like surface shader)
        var uv = sample_uv;
        var info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
        for (var i = 0; i < 5; i++) {
            uv += info.ba * 0.005;
            info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
        }
        let ba = vec2f(info.b, info.a);
        water_normal = vec3f(info.b, sqrt(max(0.0, 1.0 - dot(ba, ba))), info.a);
    } else {
        // Walls: offset UV based on Y position to create vertical variation
        let y_norm = (in.worldPos.y - bmin.y) / (surface_h - bmin.y);
        sample_uv = in.simUV + vec2f(0.0, y_norm * 0.3);  // Offset V by height
        
        let info = textureSampleLevel(water_sim, water_samp, sample_uv, 0.0);
        let wave_perturb = vec2f(info.b, info.a) * 0.3;
        water_normal = normalize(geom_normal + vec3f(wave_perturb.x, 0.0, wave_perturb.y));
    }
    
    // Screen-space refraction using appropriate normal
    let screen_uv = in.position.xy * viewport.zw;
    let refract_str = vol.reflection_refraction.y;
    let refract_uv = clamp(screen_uv + water_normal.xz * refract_str,
                          vec2f(0.001), vec2f(0.999));
    
    // Sample background scene
    var scene_sample = textureSampleLevel(scene_color, shared_samp, refract_uv, 0.0).rgb;
    
    // For walls, blend heavily toward water color (they should look like thick water)
    // For floor, use normal refraction
    var refracted: vec3f;
    if is_floor {
        refracted = scene_sample;
    } else {
        // Walls: blend background with solid water color (90% water, 10% scene)
        let solid_water_color = vol.water_color.rgb * 0.8;  // Slightly dimmed water color
        refracted = mix(scene_sample, solid_water_color, 0.9);
    }
    
    // Calculate water thickness for absorption
    let info = textureSampleLevel(water_sim, water_samp, sample_uv, 0.0);
    let water_surface_y = surface_h + info.r * (surface_h - bmin.y);
    
    // For walls, use uniform water thickness (horizontal extent of water volume)
    // For floor, use vertical depth
    var water_thickness: f32;
    if is_floor {
        // Floor: vertical depth below surface
        water_thickness = max(0.0, water_surface_y - in.worldPos.y);
    } else {
        // Walls: use consistent thickness based on water volume size
        // This gives uniform appearance across entire wall height
        let water_extent_x = bmax.x - bmin.x;
        let water_extent_z = bmax.z - bmin.z;
        let avg_extent = (water_extent_x + water_extent_z) * 0.25;  // Average half-width
        water_thickness = avg_extent * 1.5;  // Consistent thickness for walls
    }
    
    // Beer-Lambert absorption - apply water color tint
    // For walls, apply stronger water color to ensure uniform appearance
    let color_strength = select(2.0, 1.0, is_floor);  // Walls get stronger tint
    refracted *= pow(vol.water_color.rgb, vec3f(1.0 / color_strength));
    
    // Additional depth-based absorption (now uniform for walls)
    let extinction = vol.extinction.rgb;
    let absorption_strength = select(1.2, 0.8, is_floor);  // Walls get stronger absorption
    let absorption = exp(-extinction * water_thickness * absorption_strength);
    refracted *= absorption;
    
    // Sample and apply caustics
    let caustics_sample = textureSampleLevel(caustics_tex, shared_samp, sample_uv, 0.0);
    let caustics_intensity = caustics_sample.r * vol.sim_params.y;
    let caustic_strength = select(0.5, 1.0, is_floor);
    refracted += vec3f(caustics_intensity) * caustic_strength * 0.5;
    
    // Calculate alpha with Fresnel-like effect
    let view_dir = normalize(in.worldPos - camera_pos);
    let view_angle = abs(dot(-view_dir, geom_normal));
    let fresnel_factor = pow(1.0 - view_angle, 2.0);
    
    // Base alpha - walls have moderate transparency, floor more opaque
    var alpha = select(0.7, 0.85, is_floor);
    
    // Apply Fresnel - more opaque at grazing angles
    alpha = mix(alpha, 1.0, fresnel_factor * 0.3);
    
    // Apply height-based fade from vertex shader
    alpha *= in.fadeAlpha;
    
    // Distance fade for far geometry
    let view_dist = length(in.worldPos - camera_pos);
    alpha *= smoothstep(50.0, 5.0, view_dist);
    
    return vec4f(refracted, alpha);
}
