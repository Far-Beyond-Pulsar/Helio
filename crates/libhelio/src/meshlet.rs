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

/// Packed per-meshlet vertex format (48 bytes, matches WGSL `array<GpuMeshletVertex>` stride).
///
/// WGSL `vec3<f32>` has alignment 16, forcing the struct to round up to 48 bytes.
/// Two u32 padding fields keep the Rust and WGSL layouts in sync.
///
/// Stored in a flat storage buffer (`meshlet_vertices`) indexed by
/// `SV_VertexID` (= local index + `meshlet_vertex_offset`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuMeshletVertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords0: [f32; 2],
    pub tex_coords1: [f32; 2],
    pub normal: u32,
    pub tangent: u32,
    /// Padding to 48 bytes to match WGSL array stride (vec3 alignment = 16).
    pub _pad: [u32; 2],
}

/// GPU-side descriptor for a single meshlet (a small cluster of triangles). Exactly 64 bytes.
///
/// Stored once per virtual mesh in a tightly-packed storage buffer. The flat
/// DAG replaces per-LOD ranges; each meshlet independently determines its own
/// LOD via `parent_cluster_id` and `lod_error`.
///
/// # Layout (64 bytes)
/// ```text
///  0..12   center:             vec3<f32>  bounding sphere center (mesh local space)
/// 12..16   radius:             f32        bounding sphere radius
/// 16..28   cone_apex:          vec3<f32>  backface cone apex (mesh local space)
/// 28..32   cone_cutoff:        f32        cos(half-angle); > 1.0 = disable cone cull
/// 32..44   cone_axis:          vec3<f32>  normalised backface cone axis (mesh local)
/// 44..48   lod_error:          f32        accumulated object-space simplification error
/// 48..52   packed_counts:      u32        lo 16 = vertex_count, hi 16 = triangle_count
/// 52..56   meshlet_index_offset: u32     offset in u16 elements into meshlet index stream
/// 56..60   meshlet_vertex_offset: u32    offset in GpuMeshletVertex elements into vertex stream
/// 60..64   parent_cluster_id:  u32        index of parent (coarser LOD) meshlet, or 0xFFFFFFFF for root
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

    /// Packed vertex_count (lo 16 bits) and triangle_count (hi 16 bits).
    /// vertex_count = packed_counts & 0xFFFF, triangle_count = packed_counts >> 16.
    pub packed_counts: u32,
    /// Offset in u16 elements into the flat meshlet index stream (= meshlet_index_stream).
    pub meshlet_index_offset: u32,
    /// Offset in GpuMeshletVertex elements into the flat meshlet vertex stream.
    pub meshlet_vertex_offset: u32,
    /// Index of the parent (coarser LOD) meshlet in the global flat meshlet buffer.
    /// `0xFFFFFFFF` indicates a root meshlet (coarsest LOD) with no parent.
    pub parent_cluster_id: u32,
}

/// GPU-side descriptor for one virtual-geometry object. Exactly 48 bytes.
///
/// The flat DAG scheme eliminates per-LOD arrays. `leaf_meshlet_count` is the
/// number of **DAG leaves** (finest LOD), stored contiguously starting at
/// `first_meshlet`. Coarser meshlets follow in the shared meshlet buffer and are
/// reached only via `parent_cluster_id` walks — they are never cull entry points.
///
/// `emit_flag_base` indexes a per-object slice of the frame's emit-flag buffer
/// so multiple instances sharing the same meshlet descriptors can each claim
/// draws independently. Local flag index = `meshlet_index - first_meshlet`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVgObject {
    /// Slot in the VG `GpuInstanceData` and `InstanceCullData` arrays.
    pub instance_index: u32,
    /// Number of finest-LOD (DAG leaf) meshlets for this object.
    /// Work items and the cull shader only iterate this range.
    pub leaf_meshlet_count: u32,
    /// Global offset into the flat meshlet buffer for this object's first leaf.
    pub first_meshlet: u32,
    /// Total meshlets across all LODs for this object's mesh (leaf + coarser).
    /// Bounds the per-object emit-flag slice.
    pub total_meshlet_count: u32,

    /// Conservative mesh-local bounding sphere `[center.xyz, radius]`.
    pub local_bounds: [f32; 4],

    /// Base index into the per-frame `meshlet_emit_flags` buffer for this object.
    /// Flag slot = `emit_flag_base + (meshlet_index - first_meshlet)`.
    pub emit_flag_base: u32,
    /// Per-frame object visibility written by `cs_select_objects`
    /// (`0` = culled, non-zero = visible).
    pub visible: u32,
    pub _pad0: u32,
    pub _pad1: u32,
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

/// Work item for the second-stage meshlet cull. Exactly 8 bytes.
///
/// Each record covers up to 64 **DAG leaf** meshlets for one object.
/// Leaves walk the parent chain (`parent_cluster_id` + `lod_error`) and emit
/// exactly one selected cluster. Coarser meshlets are never work-item entries.
/// Using fixed spans keeps work bounded while allowing a very large object
/// to occupy many GPU workgroups instead of serialising through one.
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
        std::mem::size_of::<GpuMeshletVertex>() == 48,
        "GpuMeshletVertex must be exactly 48 bytes"
    );
    assert!(
        std::mem::size_of::<GpuVgObject>() == 48,
        "GpuVgObject must be exactly 48 bytes"
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
        GpuMeshletEntry, GpuMeshletVertex, GpuVgDraw, GpuVgObject, GpuVgWorkItem,
        VG_CULL_MESHLETS_PER_WORK_ITEM, VG_LOD_LEVELS,
    };

    #[test]
    fn gpu_virtual_geometry_layouts_are_stable() {
        assert_eq!(VG_LOD_LEVELS, 8);
        assert_eq!(std::mem::size_of::<GpuMeshletEntry>(), 64);
        assert_eq!(std::mem::size_of::<GpuMeshletVertex>(), 48);
        assert_eq!(std::mem::size_of::<GpuVgObject>(), 48);
        assert_eq!(std::mem::size_of::<GpuVgDraw>(), 16);
        assert_eq!(std::mem::size_of::<GpuVgWorkItem>(), 8);
        assert_eq!(std::mem::align_of::<GpuVgObject>(), 4);
        assert_eq!(VG_CULL_MESHLETS_PER_WORK_ITEM, 64);
    }
}

