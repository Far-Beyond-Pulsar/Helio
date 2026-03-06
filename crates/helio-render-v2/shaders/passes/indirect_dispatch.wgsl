/// Compute shader: build GPU-driven indirect draw buffers
///
/// For each visible draw call, writes a DrawIndexedIndirect command to either
/// the opaque or transparent indirect buffer, grouped by material for efficient batching.
///
/// This enables multi-draw-indirect submission: instead of 100+ CPU draw calls per frame,
/// we submit one multi-draw per material (typically 5-20 materials total).

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

/// GPU representation of a DrawCall (uploaded from CPU each frame)
struct GpuDrawCall {
    // Mesh buffer indices (into a registry or unified buffer)
    vertex_offset: u32,
    index_offset: u32,
    index_count: u32,
    vertex_count: u32,
    
    // Material ID for batching (assigned by renderer)
    material_id: u32,
    
    // Flags
    transparent_blend: u32,  // bool packed as u32
    
    // Bounding volume for frustum culling (future optimization)
    bounds_center: vec3f,
    bounds_radius: f32,
}

/// Global camera/scene data
struct GlobalUniforms {
    view_proj: mat4x4f,
    view_proj_inv: mat4x4f,
    camera_position: vec3f,
    time: f32,
    camera_forward: vec3f,
    near_plane: f32,
    camera_right: vec3f,
    far_plane: f32,
    camera_up: vec3f,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

// Input: draw list uploaded from CPU
@group(1) @binding(0) var<storage, read> draw_calls: array<GpuDrawCall>;

// Output: indirect command buffers
@group(1) @binding(1) var<storage, read_write> opaque_indirect: array<DrawIndexedIndirect>;
@group(1) @binding(2) var<storage, read_write> transparent_indirect: array<DrawIndexedIndirect>;

// Output: atomic draw counts
@group(1) @binding(3) var<storage, read_write> opaque_count: atomic<u32>;
@group(1) @binding(4) var<storage, read_write> transparent_count: atomic<u32>;

@compute @workgroup_size(256)
fn build_indirect_buffers(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let draw_idx = gid.x;
    let total_draws = arrayLength(&draw_calls);
    
    if draw_idx >= total_draws {
        return;
    }
    
    let dc = draw_calls[draw_idx];
    
    // Build indirect command
    let cmd = DrawIndexedIndirect(
        dc.index_count,
        1u,  // instance_count
        dc.index_offset,
        i32(dc.vertex_offset),
        0u,  // first_instance
    );
    
    // Write to appropriate buffer based on transparency
    if dc.transparent_blend != 0u {
        let idx = atomicAdd(&transparent_count, 1u);
        transparent_indirect[idx] = cmd;
    } else {
        let idx = atomicAdd(&opaque_count, 1u);
        opaque_indirect[idx] = cmd;
    }
}

