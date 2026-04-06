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

use std::collections::HashMap;

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
#[derive(Debug, Clone)]
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
    /// Group membership bitmask.  Use `GroupMask::NONE` for ungrouped objects.
    pub groups: crate::groups::GroupMask,
    /// Movability mode. Defaults to Static when None.
    pub movability: Option<libhelio::Movability>,
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
        if p.x < min_pt.x {
            min_pt = p;
        }
        if p.x > max_pt.x {
            max_pt = p;
        }
    }
    let mut center = (min_pt + max_pt) * 0.5;
    let mut radius = (max_pt - min_pt).length() * 0.5;

    // Expand to include every point.
    for &p in positions.iter() {
        let d = (p - center).length();
        if d > radius {
            let overflow = d - radius;
            radius = (radius + d) * 0.5; // = old_radius + overflow/2
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
        if n_len < 1e-6 {
            continue;
        }
        let n = n_raw / n_len;
        let d = n.dot(axis);
        if d < min_dot {
            min_dot = d;
        }
    }

    // The shader culls when: dot(view_dir, -axis) >= cutoff
    // All triangles are back-facing when: dot(view_dir, axis) <= -sqrt(1 - min_dot²)
    // i.e., cull when dot(view_dir, -axis) >= sqrt(1 - min_dot²)
    // So cutoff = sqrt(1 - min_dot²) + margin (margin makes culling slightly less
    // aggressive, preventing false culls on silhouettes).
    let spread = (1.0_f32 - min_dot * min_dot).max(0.0).sqrt();
    let cutoff = spread + 0.1;
    // If spread >= 0.9 the cone covers nearly all directions; cone culling cannot help.
    if cutoff >= 1.0 {
        return (apex, axis, 2.0);
    }
    (apex, axis, cutoff)
}

// ─── LOD helpers ────────────────────────────────────────────────────────────

/// Simplify a mesh by clustering nearby vertices into a uniform 3-D grid.
///
/// Each grid cell picks the first vertex that falls into it as a representative.
/// Triangles whose three corners collapse to the same cell are degenerate and
/// are discarded.
///
/// `grid_cells` controls resolution: 64 → roughly 25 % of original triangles,
/// 16 → roughly 6 %.  Returns the original mesh unchanged if simplification
/// would reduce the triangle count to zero.
fn vertex_cluster_simplify(
    vertices: &[PackedVertex],
    indices: &[u32],
    grid_cells: u32,
) -> (Vec<PackedVertex>, Vec<u32>) {
    if indices.is_empty() || vertices.is_empty() || grid_cells == 0 {
        return (vertices.to_vec(), indices.to_vec());
    }

    // Bounding box of all vertex positions.
    let mut min_pt = Vec3::from(vertices[0].position);
    let mut max_pt = min_pt;
    for v in vertices {
        let p = Vec3::from(v.position);
        min_pt = min_pt.min(p);
        max_pt = max_pt.max(p);
    }
    let extent = max_pt - min_pt;
    let cell_size = (extent / grid_cells as f32).max(Vec3::splat(1e-6));

    // Map every input vertex to a grid cell; the first vertex in each cell is its
    // representative.
    let mut cell_to_repr: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut vertex_remap: Vec<u32> = Vec::with_capacity(vertices.len());
    for (vi, v) in vertices.iter().enumerate() {
        let p = Vec3::from(v.position);
        let cell = (
            (((p.x - min_pt.x) / cell_size.x) as u32).min(grid_cells - 1),
            (((p.y - min_pt.y) / cell_size.y) as u32).min(grid_cells - 1),
            (((p.z - min_pt.z) / cell_size.z) as u32).min(grid_cells - 1),
        );
        let repr = *cell_to_repr.entry(cell).or_insert(vi as u32);
        vertex_remap.push(repr);
    }

    // Build a compact vertex array from the representatives.
    let mut repr_to_compact: HashMap<u32, u32> = HashMap::new();
    let mut out_vertices: Vec<PackedVertex> = Vec::new();
    for &repr in &vertex_remap {
        if !repr_to_compact.contains_key(&repr) {
            repr_to_compact.insert(repr, out_vertices.len() as u32);
            out_vertices.push(vertices[repr as usize]);
        }
    }

    // Remap indices and remove degenerate triangles.
    let mut out_indices: Vec<u32> = Vec::with_capacity(indices.len());
    let tri_count = indices.len() / 3;
    for t in 0..tri_count {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let c0 = repr_to_compact[&vertex_remap[i0]];
        let c1 = repr_to_compact[&vertex_remap[i1]];
        let c2 = repr_to_compact[&vertex_remap[i2]];
        if c0 == c1 || c1 == c2 || c0 == c2 {
            continue; // degenerate after clustering
        }
        out_indices.push(c0);
        out_indices.push(c1);
        out_indices.push(c2);
    }

    // Fallback: if everything collapsed return the original.
    if out_indices.is_empty() {
        return (vertices.to_vec(), indices.to_vec());
    }

    (out_vertices, out_indices)
}

/// Generate three LOD levels for a mesh using vertex clustering.
///
/// Returns `[(vertices, indices)]` at decreasing detail levels:
/// - index 0: full detail (original mesh)
/// - index 1: medium detail (64-cell grid, ~25 % triangles)
/// - index 2: coarse detail (16-cell grid, ~6 % triangles)
pub fn generate_lod_meshes(
    vertices: &[PackedVertex],
    indices: &[u32],
) -> Vec<(Vec<PackedVertex>, Vec<u32>)> {
    let lod0 = (vertices.to_vec(), indices.to_vec());
    let lod1 = vertex_cluster_simplify(vertices, indices, 64);
    let lod2 = vertex_cluster_simplify(vertices, indices, 16);
    vec![lod0, lod1, lod2]
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Sort the triangles of a mesh by centroid so spatially adjacent triangles are
/// consecutive.  Returns a new index buffer with the same values but reordered
/// by centroid position (X → Z → Y).  This makes the greedy sequential meshlet
/// grouping produce tight, spatially coherent clusters.
pub fn sort_triangles_spatially(vertices: &[PackedVertex], indices: &[u32]) -> Vec<u32> {
    let tri_count = indices.len() / 3;
    if tri_count == 0 {
        return indices.to_vec();
    }
    let positions: Vec<Vec3> = vertices.iter().map(|v| Vec3::from(v.position)).collect();
    let mut tri_order: Vec<u32> = (0..tri_count as u32).collect();
    tri_order.sort_unstable_by(|&a, &b| {
        let centroid = |ti: u32| -> Vec3 {
            let i0 = indices[ti as usize * 3] as usize;
            let i1 = indices[ti as usize * 3 + 1] as usize;
            let i2 = indices[ti as usize * 3 + 2] as usize;
            let p0 = positions.get(i0).copied().unwrap_or(Vec3::ZERO);
            let p1 = positions.get(i1).copied().unwrap_or(Vec3::ZERO);
            let p2 = positions.get(i2).copied().unwrap_or(Vec3::ZERO);
            (p0 + p1 + p2) * (1.0 / 3.0)
        };
        let ca = centroid(a);
        let cb = centroid(b);
        ca.x.partial_cmp(&cb.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ca.z.partial_cmp(&cb.z).unwrap_or(std::cmp::Ordering::Equal))
            .then(ca.y.partial_cmp(&cb.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    tri_order
        .iter()
        .flat_map(|&ti| {
            let base = ti as usize * 3;
            [indices[base], indices[base + 1], indices[base + 2]]
        })
        .collect()
}

/// Split `(vertices, indices)` into meshlets and return the GPU descriptors.
///
/// `mesh_first_index`  : absolute start of this mesh's index range in the mega-buffer
/// `mesh_first_vertex` : absolute start of this mesh's vertex range in the mega-buffer
///
/// **`indices` must already be spatially sorted** (call `sort_triangles_spatially`
/// first and upload those sorted indices to the MeshPool) so that consecutive
/// triangles in each 64-element chunk form a tight bounding sphere.
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
    let positions: Vec<Vec3> = vertices.iter().map(|v| Vec3::from(v.position)).collect();

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
            instance_index: 0, // Patched later during rebuild_vg_buffers.
        });

        tri = tri_end;
    }

    out
}

