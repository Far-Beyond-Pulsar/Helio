//! Standard bind group declarations
//!
//! This defines the bind groups that ALL shaders use.
//! Must match the bind group layouts defined in bindgroup.rs

// ============================================================================
// Group 0: Global uniforms (per-frame)
// ============================================================================

struct Camera {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
}

struct Globals {
    frame: u32,
    delta_time: f32,
    ambient_intensity: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

// ============================================================================
// Group 1: Material (per-draw)
// ============================================================================

struct Material {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emissive: f32,
    ao: f32,
}

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_map: texture_2d<f32>;
@group(1) @binding(3) var material_sampler: sampler;

// ============================================================================
// Group 2: Lighting (per-scene)
// ============================================================================

struct GpuLight {
    position: vec3<f32>,
    light_type: f32,  // 0=directional, 1=point, 2=spot, 3=rect
    direction: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    intensity: f32,
    inner_angle: f32,
    outer_angle: f32,
    width: f32,
    height: f32,
}

@group(2) @binding(0) var<storage, read> lights: array<GpuLight>;
@group(2) @binding(1) var shadow_atlas: texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;
@group(2) @binding(3) var env_map: texture_cube<f32>;

// ============================================================================
// Group 3: Bindless textures (optional, not used yet)
// ============================================================================

// TODO: Add bindless texture array support when needed
// @group(3) @binding(0) var textures: binding_array<texture_2d<f32>>;

// ============================================================================
// Group 4: Pass-specific storage (per-pass, compute only)
// ============================================================================

// This is defined per-pass as needed
// Example: @group(4) @binding(0) var<storage, read_write> particles: array<Particle>;
