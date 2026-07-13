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

/// Number of progressive LOD levels stored for every virtual mesh.
pub const VG_LOD_LEVELS: usize = 8;

/// Number of meshlets processed cooperatively by one VG cull workgroup.
pub const VG_CULL_MESHLETS_PER_WORK_ITEM: u32 = 64;

/// GPU-side descriptor for a single meshlet (a small cluster of triangles). Exactly 64 bytes.
///
/// Stored once per virtual mesh in a tightly-packed storage buffer. Virtual
/// objects refer to contiguous per-LOD ranges in this immutable descriptor
/// array, so instances never duplicate meshlet metadata.
///
/// # Layout (64 bytes, 16-byte aligned)
/// ```text
///  0..12   center:          vec3<f32>  bounding sphere center (mesh local space)
/// 12..16   radius:          f32        bounding sphere radius
/// 16..28   cone_apex:       vec3<f32>  backface cone apex (mesh local space)
/// 28..32   cone_cutoff:     f32        cos(half-angle); > 1.0 = disable cone cull
/// 32..44   cone_axis:       vec3<f32>  normalised backface cone axis (mesh local)
/// 44..48   lod_error:       f32        accumulated object-space simplification error
/// 48..52   first_index:     u32        absolute offset into the global index buffer
/// 52..56   index_count:     u32        number of indices (triangles × 3, ≤ 64 × 3)
/// 56..60   vertex_offset:   i32        base_vertex added to every index by the GPU
/// 60..64   instance_index:  u32        reserved; object ownership is stored separately
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
    /// Accumulated object-space simplification error for this meshlet's LOD.
    pub lod_error: f32,

    /// Absolute byte-index offset into the global index mega-buffer
    /// (= mesh.first_index + offset_within_mesh).
    pub first_index: u32,
    /// Number of indices in this meshlet (= triangles × 3, ≤ `MESHLET_MAX_TRIANGLES × 3`).
    pub index_count: u32,
    /// Base vertex added by the GPU to every index value when drawing.
    /// Equals the mesh's `first_vertex` in the global vertex mega-buffer.
    pub vertex_offset: i32,
    /// Reserved for ABI stability. Object ownership is supplied by
    /// [`GpuVgObject`] so descriptors can be shared by every instance.
    pub instance_index: u32,
}

/// GPU-side descriptor for one virtual-geometry object. Exactly 128 bytes.
///
/// The first compute stage assigns one lane to each object for conservative
/// object culling and a single measured-error LOD decision. The second stage
/// expands that selected LOD across immutable [`GpuVgWorkItem`] spans.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVgObject {
    /// Slot in the VG `GpuInstanceData` and `InstanceCullData` arrays.
    pub instance_index: u32,
    /// Number of valid entries in the LOD arrays (zero for invalid geometry).
    pub lod_count: u32,
    /// Largest meshlet count among the object's LODs. This object's exact
    /// contribution to the worst-case indirect draw capacity.
    pub max_meshlet_count: u32,
    /// Per-frame GPU scratch containing selected LOD + 1; zero means culled.
    /// CPU publishers must initialize this to zero. The object-selection stage
    /// overwrites it before any meshlet span reads it.
    pub reserved: u32,

    /// Conservative mesh-local bounding sphere `[center.xyz, radius]`.
    pub local_bounds: [f32; 4],
    /// Accumulated object-space simplification error for each LOD.
    pub lod_errors: [f32; VG_LOD_LEVELS],
    /// First descriptor in the shared meshlet buffer for each LOD.
    pub lod_first_meshlets: [u32; VG_LOD_LEVELS],
    /// Number of descriptors in each LOD range.
    pub lod_meshlet_counts: [u32; VG_LOD_LEVELS],
}

/// Per-visible-draw metadata emitted beside each indirect command. Exactly 16 bytes.
///
/// `DrawIndexedIndirect::first_instance` indexes this array. The draw shader
/// follows `instance_index` to the transform/material array and uses the stable
/// meshlet and LOD identifiers for truthful debug visualisations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVgDraw {
    pub instance_index: u32,
    pub meshlet_index: u32,
    pub lod_level: u32,
    pub reserved: u32,
}

/// Immutable expansion record for the second-stage meshlet cull. Exactly 8 bytes.
///
/// Records are built once with the object layout. Each record covers up to 64
/// meshlets from whichever whole-object LOD the first GPU stage selects. Using
/// fixed spans keeps work bounded while allowing a very large object to occupy
/// many GPU workgroups instead of serialising through one workgroup.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVgWorkItem {
    pub object_index: u32,
    pub local_meshlet_base: u32,
}

const _: () = {
    assert!(
        std::mem::size_of::<GpuMeshletEntry>() == 64,
        "GpuMeshletEntry must be exactly 64 bytes"
    );
    assert!(
        std::mem::size_of::<GpuVgObject>() == 128,
        "GpuVgObject must be exactly 128 bytes"
    );
    assert!(
        std::mem::size_of::<GpuVgDraw>() == 16,
        "GpuVgDraw must be exactly 16 bytes"
    );
    assert!(
        std::mem::size_of::<GpuVgWorkItem>() == 8,
        "GpuVgWorkItem must be exactly 8 bytes"
    );
};

#[cfg(test)]
mod tests {
    use super::{
        GpuMeshletEntry, GpuVgDraw, GpuVgObject, GpuVgWorkItem,
        VG_CULL_MESHLETS_PER_WORK_ITEM, VG_LOD_LEVELS,
    };

    #[test]
    fn gpu_virtual_geometry_layouts_are_stable() {
        assert_eq!(VG_LOD_LEVELS, 8);
        assert_eq!(std::mem::size_of::<GpuMeshletEntry>(), 64);
        assert_eq!(std::mem::size_of::<GpuVgObject>(), 128);
        assert_eq!(std::mem::size_of::<GpuVgDraw>(), 16);
        assert_eq!(std::mem::size_of::<GpuVgWorkItem>(), 8);
        assert_eq!(std::mem::align_of::<GpuVgObject>(), 4);
        assert_eq!(VG_CULL_MESHLETS_PER_WORK_ITEM, 64);
    }
}

