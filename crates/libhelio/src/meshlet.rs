//! GPU-side meshlet descriptor for virtual geometry rendering.
//!
//! A meshlet is a small, spatially-coherent cluster of triangles — typically 64 or fewer.
//! The culling compute shader tests each meshlet independently against the view frustum
//! and the backface cone, then emits one `DrawIndexedIndirect` command per visible meshlet.
//! This gives fully GPU-driven O(1) CPU rendering even for meshes with tens of millions
//! of triangles.

use bytemuck::{Pod, Zeroable};

/// Maximum triangles per meshlet.  64 is the canonical value — fits one wavefront on AMD
/// and a full warp pair on NVIDIA.  Change to 128 for higher amortisation cost but fewer
/// draw commands on less-detailed geometry.
pub const MESHLET_MAX_TRIANGLES: u32 = 64;

/// GPU-side descriptor for a single meshlet (a small cluster of triangles). Exactly 64 bytes.
///
/// Stored in a tightly-packed storage buffer and indexed by `global_invocation_id` inside
/// the virtual-geometry culling compute shader.
///
/// # Layout (64 bytes, 16-byte aligned)
/// ```text
///  0..12   center:          vec3<f32>  bounding sphere center (mesh local space)
/// 12..16   radius:          f32        bounding sphere radius
/// 16..28   cone_apex:       vec3<f32>  backface cone apex (mesh local space)
/// 28..32   cone_cutoff:     f32        cos(half-angle); > 1.0 = disable cone cull
/// 32..44   cone_axis:       vec3<f32>  normalised backface cone axis (mesh local)
/// 44..48   lod_error:       f32        reserved for future hierarchical LOD
/// 48..52   first_index:     u32        absolute offset into the global index buffer
/// 52..56   index_count:     u32        number of indices (triangles × 3, ≤ 64 × 3)
/// 56..60   vertex_offset:   i32        base_vertex added to every index by the GPU
/// 60..64   instance_index:  u32        slot in the VG instance buffer for this object
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuMeshletEntry {
    /// Bounding sphere center in mesh-local space.
    pub center: [f32; 3],
    /// Bounding sphere radius (before applying the object's world transform).
    pub radius: f32,

    /// Backface cone apex in mesh-local space (an approximation: the centroid works well).
    pub cone_apex: [f32; 3],
    /// cos(half-angle) of the backface cone.
    /// When the view direction dot this cone faces the opposite direction we can skip drawing.
    /// Set to `2.0` to disable cone culling for this meshlet (nearly-flat or mixed-winding).
    pub cone_cutoff: f32,

    /// Normalised backface cone axis in mesh-local space.
    pub cone_axis: [f32; 3],
    /// Reserved: screen-space error metric for future hierarchical LOD selection.
    pub lod_error: f32,

    /// Absolute byte-index offset into the global index mega-buffer
    /// (= mesh.first_index + offset_within_mesh).
    pub first_index: u32,
    /// Number of indices in this meshlet (= triangles × 3, ≤ `MESHLET_MAX_TRIANGLES × 3`).
    pub index_count: u32,
    /// Base vertex added by the GPU to every index value when drawing.
    /// Equals the mesh's `first_vertex` in the global vertex mega-buffer.
    pub vertex_offset: i32,
    /// Slot in the per-frame VG instance buffer (`GpuInstanceData` array) that
    /// provides the world transform, material ID, etc. for this meshlet's owning object.
    pub instance_index: u32,
}

const _: () = {
    assert!(
        std::mem::size_of::<GpuMeshletEntry>() == 64,
        "GpuMeshletEntry must be exactly 64 bytes"
    );
};
