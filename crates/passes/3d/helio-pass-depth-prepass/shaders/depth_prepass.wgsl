//! Depth-only prepass shader.
//!
//! Transforms vertices through the camera view-projection and writes to the depth buffer.
//! No fragment output — depth writes are implicit.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

/// Per-instance GPU data.  Must match `GpuInstanceData` in libhelio.
struct GpuInstanceData {
    transform:     mat4x4<f32>,
    normal_mat_0:  vec4<f32>,
    normal_mat_1:  vec4<f32>,
    normal_mat_2:  vec4<f32>,
    bounds:        vec4<f32>,
    prev_model:    mat4x4<f32>,
    mesh_id:       u32,
    material_id:   u32,
    flags:         u32,
    _pad:          u32,
    skin_vertex_offset: u32,
    bone_palette_offset: u32,
    bone_palette_count: u32,
    _skin_pad: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> instance_data:     array<GpuInstanceData>;
// Per-draw-call-group compacted original instance slots (see IndirectDispatchPass).
@group(0) @binding(2) var<storage, read> compacted_indices: array<u32>;
struct SkinVertex { joints: vec4<u32>, weights: vec4<f32> }
@group(0) @binding(3) var<storage, read> skin_vertices: array<SkinVertex>;
@group(0) @binding(4) var<storage, read> bone_palette: array<mat4x4<f32>>;

@vertex
fn vs_main(
    @location(0)             position:    vec3<f32>,
    @location(2)             _tex_coords: vec2<f32>,  // kept for vertex layout compatibility
    // Per-instance vertex data, offset by the indirect draw's `first_instance`
    // on every backend (`helio_pass_gbuffer::draw_slots`).
    @location(15)            slot:        u32,
    @builtin(vertex_index) vertex_index: u32,
) -> @invariant @builtin(position) vec4<f32> {
    let inst      = instance_data[compacted_indices[slot]];
    var local_position = position;
    if inst.bone_palette_count > 0u {
        let skin = skin_vertices[vertex_index + inst.skin_vertex_offset];
        var blended = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        var total_weight = 0.0;
        for (var influence = 0u; influence < 4u; influence = influence + 1u) {
            let weight = skin.weights[influence];
            let joint = skin.joints[influence];
            if weight > 0.0 && joint < inst.bone_palette_count {
                blended += bone_palette[inst.bone_palette_offset + joint] * weight;
                total_weight += weight;
            }
        }
        if total_weight > 0.00001 {
            local_position = (blended * vec4<f32>(position, 1.0) / total_weight).xyz;
        }
    }
    let world_pos = inst.transform * vec4<f32>(local_position, 1.0);
    // Match G-buffer projection order exactly, including rounding.
    return cameras[0].proj * (cameras[0].view * world_pos);
}
