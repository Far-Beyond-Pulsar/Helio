//! CPU-side virtual geometry: meshlet decomposition for GPU-driven rendering.
//!
//! `meshletize()` splits a mesh into spatially coherent clusters of ≤ 64 triangles
//! called **meshlets**.  The GPU culling compute shader tests these clusters against
//! the view frustum and a backface cone, then emits one `DrawIndexedIndirect` command
//! per visible meshlet.  This gives O(1) CPU work regardless of triangle count.
//!
//! # Algorithm
//! The current implementation uses a greedy sequential partitioning strategy:
//! triangles are grouped into consecutive chunks of `MESHLET_MAX_TRIANGLES`.  This is
//! fast and deterministic.  Future work can upgrade to a meshopt-style scan that also
//! optimises vertex cache locality.

use glam::Vec3;
use libhelio::{GpuMeshletEntry, MESHLET_MAX_TRIANGLES};

use crate::mesh::PackedVertex;

// ─── Handle types ───────────────────────────────────────────────────────────

/// Opaque handle to a virtual mesh uploaded to the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualMeshId(pub(crate) u32);

// ─── Upload / descriptor types ──────────────────────────────────────────────

/// High-resolution mesh for virtual geometry upload.
///
/// The scene splits this into meshlets automatically when you call
/// `Scene::insert_virtual_mesh()`.  Keep the CPU-side data alive only until
/// after `Scene::flush()` returns.
pub struct VirtualMeshUpload {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
}

/// Descriptor for a virtual object (one instance of a `VirtualMeshId`).
#[derive(Debug, Clone, Copy)]
pub struct VirtualObjectDescriptor {
    pub virtual_mesh: VirtualMeshId,
    pub material_id: u32,
    pub transform: glam::Mat4,
    /// World-space bounding sphere `[cx, cy, cz, radius]`.
    pub bounds: [f32; 4],
    pub flags: u32,
}

// ─── Private helpers ────────────────────────────────────────────────────────

/// Decode a packed SNORM8×4 vertex normal to `[f32; 3]`.
#[inline(always)]
fn unpack_normal(packed: u32) -> Vec3 {
    let x = ((packed & 0xFF) as i8) as f32 / 127.0;
    let y = (((packed >> 8) & 0xFF) as i8) as f32 / 127.0;
    let z = (((packed >> 16) & 0xFF) as i8) as f32 / 127.0;
    Vec3::new(x, y, z)
}

/// Ritter's incremental bounding sphere — fast, slightly sub-optimal.
fn ritter_sphere(positions: &[Vec3]) -> (Vec3, f32) {
    if positions.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    // Pick an initial axis: find the points with min/max x.
    let (mut min_pt, mut max_pt) = (positions[0], positions[0]);
    for &p in positions.iter() {
        if p.x < min_pt.x { min_pt = p; }
        if p.x > max_pt.x { max_pt = p; }
    }
    let mut center = (min_pt + max_pt) * 0.5;
    let mut radius = (max_pt - min_pt).length() * 0.5;

    // Expand to include every point.
    for &p in positions.iter() {
        let d = (p - center).length();
        if d > radius {
            let overflow = d - radius;
            radius = (radius + d) * 0.5;  // = old_radius + overflow/2
            center += (p - center).normalize() * (overflow * 0.5);
        }
    }
    (center, radius.max(0.0))
}

/// Compute a conservative backface cone for a set of triangles.
///
/// Returns `(apex, axis, cutoff)` in mesh-local space where:
/// - `apex`    : centroid of the meshlet (a fine approximation)
/// - `axis`    : average triangle normal (points "away" from back faces)
/// - `cutoff`  : `cos(half_angle)`; > 1.0 disables cone culling for this meshlet
fn backface_cone(positions: &[Vec3], indices: &[u32]) -> (Vec3, Vec3, f32) {
    let mut avg_normal = Vec3::ZERO;
    let mut centroid = Vec3::ZERO;
    let mut tri_count = 0.0f32;

    let tri_n = indices.len() / 3;
    for t in 0..tri_n {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let v0 = positions[i0];
        let v1 = positions[i1];
        let v2 = positions[i2];
        let n = (v1 - v0).cross(v2 - v0); // area-weighted normal
        avg_normal += n;
        centroid += v0 + v1 + v2;
        tri_count += 1.0;
    }

    let apex = if tri_count > 0.0 {
        centroid / (tri_count * 3.0)
    } else {
        Vec3::ZERO
    };

    let axis_len = avg_normal.length();
    if axis_len < 1e-6 {
        // Degenerate: disable cone culling.
        return (apex, Vec3::Y, 2.0);
    }
    let axis = avg_normal / axis_len;

    // Find the minimum dot product between any per-vertex normal and the axis.
    // This is the cosine of the maximum half-angle of the cone.
    let mut min_dot = 1.0f32;
    for t in 0..tri_n {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let v0 = positions[i0];
        let v1 = positions[i1];
        let v2 = positions[i2];
        let n_raw = (v1 - v0).cross(v2 - v0);
        let n_len = n_raw.length();
        if n_len < 1e-6 { continue; }
        let n = n_raw / n_len;
        let d = n.dot(axis);
        if d < min_dot { min_dot = d; }
    }

    // cutoff = cos(half_angle + small margin for numerical safety).
    // A meshlet is back-facing when every triangle in it is back-facing, i.e.
    //   dot(view_dir, axis) < cutoff
    // We add a -0.1 margin so borderline cases are not incorrectly culled.
    let cutoff = (min_dot - 0.1).max(-1.0);
    // If the triangles show enough variation (180° spread) cone culling is useless.
    if cutoff <= -1.0 {
        return (apex, axis, 2.0);
    }
    (apex, axis, cutoff)
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Split `(vertices, indices)` into meshlets and return the GPU descriptors.
///
/// `mesh_first_index`  : absolute start of this mesh's index range in the mega-buffer
/// `mesh_first_vertex` : absolute start of this mesh's vertex range in the mega-buffer
///
/// Each returned entry has `first_index` and `vertex_offset` pre-set to the
/// global mega-buffer positions; the caller only needs to set `instance_index` later.
pub fn meshletize(
    vertices: &[PackedVertex],
    indices: &[u32],
    mesh_first_index: u32,
    mesh_first_vertex: u32,
) -> Vec<GpuMeshletEntry> {
    let tri_count = indices.len() / 3;
    if tri_count == 0 {
        return Vec::new();
    }

    let max_tri = MESHLET_MAX_TRIANGLES as usize;
    let meshlet_count = (tri_count + max_tri - 1) / max_tri;
    let mut out = Vec::with_capacity(meshlet_count);

    // Collect all positions once (cheaper than re-indexing per meshlet).
    let positions: Vec<Vec3> = vertices
        .iter()
        .map(|v| Vec3::from(v.position))
        .collect();

    let mut tri = 0;
    while tri < tri_count {
        let tri_end = (tri + max_tri).min(tri_count);
        let local_idx_start = tri * 3;
        let local_idx_end = tri_end * 3;
        let local_indices = &indices[local_idx_start..local_idx_end];

        // Collect unique vertex positions for bounding sphere.
        let meshlet_positions: Vec<Vec3> = local_indices
            .iter()
            .filter_map(|&i| positions.get(i as usize).copied())
            .collect();

        let (center, radius) = ritter_sphere(&meshlet_positions);

        // These are mesh-local indices (values 0..vertex_count).
        let (cone_apex, cone_axis, cone_cutoff) = backface_cone(&positions, local_indices);

        out.push(GpuMeshletEntry {
            center: center.to_array(),
            radius,
            cone_apex: cone_apex.to_array(),
            cone_cutoff,
            cone_axis: cone_axis.to_array(),
            lod_error: 0.0,
            // Global buffer offsets (position in the mega-buffer).
            first_index: mesh_first_index + local_idx_start as u32,
            index_count: (local_idx_end - local_idx_start) as u32,
            vertex_offset: mesh_first_vertex as i32,
            instance_index: 0, // Patched later when building the per-frame VG data.
        });

        tri = tri_end;
    }

    out
}
