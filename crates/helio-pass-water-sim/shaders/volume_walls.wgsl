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
    @location(1) depth: f32,
}

// Transform unit box vertex to water volume bounds
fn boxToWorld(boxPos: vec3f, bmin: vec3f, bmax: vec3f, surface_h: f32) -> vec3f {
    // Box vertices are in [-1, 1] range
    // For X and Z: map to full water volume bounds
    // For Y: map from bounds_min.y to surface_height (NOT bounds_max.y)
    //        The surface shader handles the displaced top surface
    return vec3f(
        mix(bmin.x, bmax.x, (boxPos.x + 1.0) * 0.5),
        mix(bmin.y, surface_h, (boxPos.y + 1.0) * 0.5),  // Stop at undisplaced surface
        mix(bmin.z, bmax.z, (boxPos.z + 1.0) * 0.5),
    );
}

@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol = volumes[0];
    let bmin = vol.bounds_min.xyz;
    let bmax = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;  // Undisplaced water surface level
    
    let worldPos = boxToWorld(position, bmin, bmax, surface_h);
    let clipPos = camera.view_proj * vec4f(worldPos, 1.0);

    var out: VertexOutput;
    out.position = clipPos;
    out.worldPos = worldPos;
    out.depth = clipPos.z / clipPos.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol = volumes[0];
    let camera_pos = camera.position_near.xyz;
    let bmin = vol.bounds_min.xyz;
    let bmax = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;
    
    // Calculate viewing angle (to make walls more transparent when viewed edge-on)
    let view_dir = normalize(camera_pos - in.worldPos);
    let surface_normal_approx = vec3f(0.0, 1.0, 0.0);  // Approximate up direction
    let view_angle = abs(dot(view_dir, surface_normal_approx));
    
    // Distance from camera for fog/absorption
    let view_dist = length(in.worldPos - camera_pos);
    
    // Normalize position within volume [0,1]
    let normPos = (in.worldPos - bmin) / (bmax - bmin);
    
    // Check if we're looking at a vertical wall (X or Z aligned)
    let is_vertical = abs(in.worldPos.y - bmin.y) > 0.01 && abs(in.worldPos.y - surface_h) > 0.01;
    
    // Sample caustics - project from above onto floor and walls
    let caustics_uv = normPos.xz;
    let caustics_sample = textureSample(caustics_tex, shared_samp, caustics_uv);
    let caustics_intensity = caustics_sample.r;
    
    // Base underwater color with depth
    let depth_ratio = normPos.y;  // 0 at bottom, 1 at surface
    let shallow_color = vec3f(0.3, 0.5, 0.6);  // Lighter blue-green near surface
    let deep_color = vec3f(0.05, 0.15, 0.25);   // Darker blue in depths
    var color = mix(deep_color, shallow_color, depth_ratio);
    
    // Apply caustics (stronger on floor, subtle on walls)
    let caustic_strength = select(0.3, 1.0, !is_vertical);  // Weaker on walls
    let caustic_color = vec3f(0.7, 0.85, 1.0) * caustics_intensity * vol.sim_params.y;
    color += caustic_color * caustic_strength;
    
    // Ambient light scattered from surface (more at shallow depths)
    let scatter = vec3f(0.2, 0.3, 0.4) * depth_ratio * 0.5;
    color += scatter;
    
    // Distance fog for depth perception
    let fog_factor = exp(-view_dist * 0.08);
    color *= fog_factor;
    
    // Make walls very transparent - they're just for volumetric effect
    // The surface shader handles the main refraction
    var alpha = 0.15;  // Very subtle
    
    // Increase opacity for bottom floor (more visible)
    alpha = select(alpha, 0.35, !is_vertical);
    
    // Fade out at edges when viewed from grazing angles
    alpha *= smoothstep(0.0, 0.3, view_angle);
    
    // Reduce opacity with distance (distant walls fade away)
    alpha *= smoothstep(30.0, 5.0, view_dist);
    
    return vec4f(color, alpha);
}
