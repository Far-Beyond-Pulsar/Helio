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
    @location(1) wallUV: vec2f,     // UV that varies horizontally + vertically
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
    
    // Calculate wall UV that tiles uniformly both horizontally and vertically
    // Use world position divided by a tile size for both U and V
    let tile_size = 2.0;  // Size of tiles in world units
    
    // Use the unit cube position to figure out orientation
    let abs_pos = abs(position);
    var u: f32;
    var v: f32;
    
    // Check which axis is dominant to determine face orientation
    if abs_pos.x >= abs_pos.z {
        // X-aligned wall (East or West face) - use Z for horizontal, Y for vertical
        u = worldPos.z / tile_size;
        v = worldPos.y / tile_size;
    } else {
        // Z-aligned wall (North or South face) - use X for horizontal, Y for vertical
        u = worldPos.x / tile_size;
        v = worldPos.y / tile_size;
    }
    
    let wallUV = vec2f(u, v);
    
    // Sample heightfield to check if above water surface  
    let simUV = worldToSimUV(worldPos.xz, bmin, bmax);
    let heightfield_sample = textureSampleLevel(water_sim, water_samp, simUV, 0.0);
    let displaced_height = surface_h + heightfield_sample.r * (surface_h - bmin.y);
    
    // Fade out vertices that are above the displaced water surface
    let height_diff = displaced_height - worldPos.y;
    let fadeAlpha = smoothstep(-0.2, 0.5, height_diff);
    
    let clipPos = camera.view_proj * vec4f(worldPos, 1.0);

    var out: VertexOutput;
    out.position = clipPos;
    out.worldPos = worldPos;
    out.wallUV = wallUV;
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
    if in.fadeAlpha < 0.01 {
        discard;
    }
    
    let vol = volumes[0];
    let camera_pos = camera.position_near.xyz;
    let bmin = vol.bounds_min.xyz;
    let bmax = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;
    
    // Calculate geometric normal - points away from water center
    let center_xz = (bmin.xz + bmax.xz) * 0.5;
    let to_center = vec3f(center_xz.x - in.worldPos.x, 0.0, center_xz.y - in.worldPos.z);
    let geom_normal = normalize(-to_center);
    
    // Sample water texture with tiling
    let tile_scale = 2.0;
    let sample_uv = in.wallUV * tile_scale;
    let info = textureSampleLevel(water_sim, water_samp, sample_uv, 0.0);
    
    // Perturb normal based on water texture
    let wave_perturb = vec2f(info.b, info.a) * 0.3;
    let water_normal = normalize(geom_normal + vec3f(wave_perturb.x, 0.0, wave_perturb.y));
    
    // Screen-space refraction
    let screen_uv = in.position.xy * viewport.zw;
    let refract_str = vol.reflection_refraction.y;
    let refract_uv = clamp(screen_uv + water_normal.xz * refract_str,
                          vec2f(0.001), vec2f(0.999));
    
    // Sample background through refraction - match surface shader coloring
    let scene_sample = textureSampleLevel(scene_color, shared_samp, refract_uv, 0.0).rgb;
    
    // Apply water color like surface shader does (Beer-Lambert absorption)
    var refracted = scene_sample * vol.water_color.rgb;
    
    // Walls simulate looking through thickness of water - apply stronger absorption than surface
    let water_extent_x = bmax.x - bmin.x;
    let water_extent_z = bmax.z - bmin.z;
    let avg_extent = (water_extent_x + water_extent_z) * 0.25;
    let water_thickness = avg_extent * 2.5;  // Increased thickness
    
    // Additional depth-based absorption for thickness - stronger than before
    let extinction = vol.extinction.rgb;
    let absorption = exp(-extinction * water_thickness * 1.5);  // Increased multiplier
    refracted *= absorption;
    
    // No caustics on walls - they cause white squares
    
    // Calculate alpha with Fresnel
    let view_dir = normalize(in.worldPos - camera_pos);
    let view_angle = abs(dot(-view_dir, geom_normal));
    let fresnel_factor = pow(1.0 - view_angle, 2.0);
    
    var alpha = 0.8;  // Semi-transparent to match water surface
    alpha = mix(alpha, 1.0, fresnel_factor * 0.3);
    // Don't multiply by fadeAlpha - that creates vertical gradient
    
    let view_dist = length(in.worldPos - camera_pos);
    alpha *= smoothstep(50.0, 5.0, view_dist);
    
    return vec4f(refracted, alpha);
}
