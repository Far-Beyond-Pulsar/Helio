// shadow.wgsl — depth-only pass for shadow atlas.

struct GpuShadowMatrix {
    view_proj: mat4x4<f32>,
};

// binding 0: storage array of all shadow projection matrices
@group(0) @binding(0) var<storage, read> shadow_matrices: array<GpuShadowMatrix>;
// binding 1: uniform containing the atlas slot index for this draw
@group(0) @binding(1) var<uniform> slot_index: u32;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) _bitan:   f32,
    @location(2) _uv:      vec2<f32>,
    @location(3) _norm:    vec4<f32>,
    @location(4) _tan:     vec4<f32>,
    @location(5) i0: vec4<f32>,
    @location(6) i1: vec4<f32>,
    @location(7) i2: vec4<f32>,
    @location(8) i3: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.i0, in.i1, in.i2, in.i3);
    let vp = shadow_matrices[slot_index].view_proj;
    return vp * model * vec4<f32>(in.position, 1.0);
}
