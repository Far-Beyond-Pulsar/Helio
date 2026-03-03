//! Depth-only shadow pass shader
//!
//! Renders all shadow casters into a single shadow atlas layer.
//! The light index is carried via @builtin(instance_index), which the CPU
//! sets by issuing one instance per light (draw_indexed(…, light_idx..light_idx+1)).

struct LightMatrix {
    mat: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> light_matrices: array<LightMatrix>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(instance_index) light_idx: u32,
) -> @builtin(position) vec4<f32> {
    return light_matrices[light_idx].mat * vec4<f32>(position, 1.0);
}
