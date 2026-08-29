//! GPU draw call types for indirect rendering.
//!
//! The indirect dispatch compute shader fills an array of `wgpu::util::DrawIndexedIndirect`
//! structs from `GpuDrawCall` templates. The CPU only submits one `multi_draw_indexed_indirect`
//! call — O(1) regardless of scene complexity.

use bytemuck::{Pod, Zeroable};

/// A template draw call that the GPU culling compute uses to emit indirect commands.
///
/// Describes one batched draw — all instances in the batch share the same mesh geometry
/// (index range) and are stored consecutively in the instance buffer starting at
/// `first_instance`.  Identical (mesh, material) pairs are automatically merged into
/// a single `GpuDrawCall` during `Scene::flush()`, enabling hardware instancing.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuDrawCall {
///     index_count:    u32,
///     first_index:    u32,
///     vertex_offset:  i32,
///     first_instance: u32,  // first index into GpuInstance array for this batch
///     instance_count: u32,  // number of consecutive instances in the batch
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuDrawCall {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    /// First index into the `GpuInstance` storage buffer for this instanced batch.
    pub first_instance: u32,
    /// Number of instances in the batch (≥ 1).  Maximises GPU hardware instancing.
    pub instance_count: u32,
}

/// GPU-side indirect draw command (matches `wgpu::util::DrawIndexedIndirectArgs`).
///
/// The culling compute shader writes these. The render pass reads them via
/// `multi_draw_indexed_indirect`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

impl DrawIndexedIndirectArgs {
    /// Creates a culled (invisible) command — instance_count = 0.
    pub const fn culled(
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Self {
        Self {
            index_count,
            instance_count: 0,
            first_index,
            base_vertex,
            first_instance,
        }
    }
}
