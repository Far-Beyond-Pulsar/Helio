// Shadow caster pass — depth-only, GPU-driven.
//
// Vertex shader projects world-space geometry into each light's clip space using
// the pre-computed shadow matrix for this face.  There is no fragment stage —
// the rasteriser writes depth automatically.
//
// Design mirrors Unreal Engine 4 "Shadow Depth Pass" and Unity HDRP
// "Shadow Caster Pass": position-only transform, depth-write only,
// front-face culled to eliminate self-shadowing acne.

// ── Types ─────────────────────────────────────────────────────────────────────

// Per-instance world transform.  Must match GpuInstanceData in libhelio (144 bytes).
struct GpuInstanceData {
    transform:    mat4x4<f32>,   // offset   0
    normal_mat_0: vec4<f32>,     // offset  64  (unused in shadow pass)
    normal_mat_1: vec4<f32>,     // offset  80
    normal_mat_2: vec4<f32>,     // offset  96
    bounds:       vec4<f32>,     // offset 112
    mesh_id:      u32,           // offset 128
    material_id:  u32,           // offset 132
    flags:        u32,           // offset 136
    _pad:         u32,           // offset 140
}

// Which shadow atlas face is being rendered this pass.
// Addressed via dynamic uniform buffer offset — one 16-byte slot per face.
struct FaceIndex {
    value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────

// Pre-computed light-space view-projection matrices; one per shadow atlas face.
@group(0) @binding(0) var<storage, read> shadow_matrices: array<mat4x4<f32>>;
// Per-instance world transforms for the entire scene.
@group(0) @binding(1) var<storage, read> instances:       array<GpuInstanceData>;
// Current face selection, updated each pass via dynamic offset into a pre-written buffer.
@group(0) @binding(2) var<uniform>       face:            FaceIndex;

// ── Vertex stage ──────────────────────────────────────────────────────────────

@vertex
fn vs_main(
    @location(0)             position: vec3<f32>,
    @builtin(instance_index) slot:     u32,
) -> @builtin(position) vec4<f32> {
    let world = instances[slot].transform * vec4<f32>(position, 1.0);
    return shadow_matrices[face.value] * world;
}

// No fragment stage: the GPU writes depth automatically for the depth-only pipeline.
