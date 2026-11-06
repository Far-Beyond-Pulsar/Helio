//! Material profiling shader.
//!
//! Renders test patches with specific material properties and measures GPU
//! execution time: the surface (`surface_probe`) a materials kind offered the
//! overlay (`helio_frame_api::diagnostics::SurfaceProbe`), or the overlay's
//! Lambert stand-in, lit by `num_lights` synthetic point lights. The overlay
//! opts this source into every snippet of the probe (its markers prepended).

struct MaterialParams {
    roughness: f32,
    metallic: f32,
    num_lights: u32,
    num_shadow_lights: u32,
}

@group(0) @binding(0) var<uniform> params: MaterialParams;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

/// Fullscreen triangle vertex shader.
@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOut {
    var out: VertexOut;
    
    // Fullscreen triangle
    let x = f32((vertex_idx << 1u) & 2u);
    let y = f32(vertex_idx & 2u);
    
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    
    return out;
}

/// Fragment shader: render test patch with full BRDF calculations.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Fixed surface properties for test patch
    let base_color = vec3<f32>(0.8, 0.8, 0.8);
    let normal = vec3<f32>(0.0, 0.0, 1.0);
    let view_dir = normalize(vec3<f32>(0.0, 0.0, 1.0));
    let world_pos = vec3<f32>(in.uv.x, in.uv.y, 0.0);
    
    var total_light = vec3<f32>(0.0);
    
    // Accumulate lighting from all lights (mimic deferred lighting pass)
    for (var i = 0u; i < params.num_lights; i++) {
        // A synthetic light: spread over a grid above the patch.
        let position = vec3<f32>(f32(i % 4u) - 1.5, f32((i / 4u) % 4u) - 1.5, 2.0);
        let light_dir = normalize(position - world_pos);
        let distance = length(position - world_pos);
        let attenuation = 1.0 / (distance * distance + 1.0);

        let brdf = surface_probe(
            normal,
            view_dir,
            light_dir,
            base_color,
            params.roughness,
            params.metallic,
        );

        total_light += brdf * 10.0 * attenuation;
    }
    
    // Shadow sampling for shadow-casting lights (mimic PCF sampling)
    for (var i = 0u; i < params.num_shadow_lights; i++) {
        // Simulate 4 shadow samples per light (like deferred lighting)
        var shadow_factor = 0.0;
        for (var s = 0u; s < 4u; s++) {
            // Dummy shadow comparison (we just want timing, not correctness)
            let offset = vec2<f32>(f32(s) * 0.01, f32(s) * 0.01);
            let sample_pos = world_pos.xy + offset;
            shadow_factor += step(0.5, fract(sample_pos.x + sample_pos.y));
        }
        shadow_factor /= 4.0;
        total_light *= shadow_factor;
    }
    
    // Ambient IBL approximation
    let ambient = vec3<f32>(0.03) * base_color;
    total_light += ambient;
    
    return vec4<f32>(total_light, 1.0);
}
