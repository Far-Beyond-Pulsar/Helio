struct Face { value:u32, dynamic:u32, pad0:u32, pad1:u32 }
@group(0) @binding(2) var<uniform> face:Face;
@group(0) @binding(4) var<storage,read> dirty:array<u32>;
/// Depth-clear vertex shader.
///
/// Generates a single full-screen triangle (vertices computed in the shader, no
/// vertex buffer needed) and writes depth = 1.0 (far plane) to every pixel.
/// Used by `ShadowPass` to GPU-clear individual shadow atlas faces before
/// re-rendering movable-object shadow geometry onto them.
///
/// Pipeline state:
///   - DepthCompare: Always  →  overwrites whatever is in the depth buffer.
///   - DepthWrite:   true
///   - Fragment:     none    →  depth write happens automatically.
///   - CullMode:     none    →  the triangle is always front-facing.

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    if face.dynamic != 0u && dirty[face.value] == 0u { return vec4f(2,2,2,1); }
    // Classic "giant triangle" trick: three vertices in clip space whose
    // convex hull covers the entire NDC cube [-1,1]² with z = 1.0 (far plane).
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}
