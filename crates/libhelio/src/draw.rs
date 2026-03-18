//! GPU draw call types for indirect rendering.
//!
//! The indirect dispatch compute shader fills an array of `wgpu::util::DrawIndexedIndirect`
//! structs from `GpuDrawCall` templates. The CPU only submits one `multi_draw_indexed_indirect`
//! call — O(1) regardless of scene complexity.

use bytemuck::{Pod, Zeroable};

/// A template draw call that the GPU culling compute uses to emit indirect commands.
///
/// This describes one draw — typically one mesh LOD × one material slot.
/// The culling pass writes `DrawIndexedIndirect` from this template when the instance
/// survives frustum + occlusion culling.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuDrawCall {
///     index_count:    u32,
///     first_index:    u32,
///     vertex_offset:  i32,
///     instance_id:    u32,   // index into GpuInstance array
///     _pad:           u32,
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuDrawCall {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub instance_id: u32,
    pub _pad: u32,
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
    pub const fn culled(index_count: u32, first_index: u32, base_vertex: i32, first_instance: u32) -> Self {
        Self {
            index_count,
            instance_count: 0,
            first_index,
            base_vertex,
            first_instance,
        }
    }
}
