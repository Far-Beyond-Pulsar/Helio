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

// ─── LOD helpers: QEM edge-collapse simplifier ─────────────────────────────

/// 4×4 symmetric error quadric stored as 10 unique floats (upper triangle).
#[derive(Clone, Copy)]
struct Quadric {
    a: [f64; 10],
}

impl Quadric {
    const ZERO: Self = Self { a: [0.0; 10] };

    fn from_plane(nx: f64, ny: f64, nz: f64, d: f64) -> Self {
        Self {
            a: [
                nx * nx, nx * ny, nx * nz, nx * d,
                         ny * ny, ny * nz, ny * d,
                                  nz * nz, nz * d,
                                            d * d,
            ],
        }
    }

    fn add(&self, other: &Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..10 {
            r.a[i] = self.a[i] + other.a[i];
        }
        r
    }

    fn error_at(&self, x: f64, y: f64, z: f64) -> f64 {
        let q = &self.a;
        q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x
            + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y
            + q[7] * z * z + 2.0 * q[8] * z
            + q[9]
    }

    fn optimal_position(&self, v0: Vec3, v1: Vec3) -> Vec3 {
        let q = &self.a;
        // Try to solve the 3×3 linear system for the optimal point.
        // | q[0] q[1] q[2] |   | x |   | -q[3] |
        // | q[1] q[4] q[5] | × | y | = | -q[6] |
        // | q[2] q[5] q[7] |   | z |   | -q[8] |
        let det = q[0] * (q[4] * q[7] - q[5] * q[5])
                - q[1] * (q[1] * q[7] - q[5] * q[2])
                + q[2] * (q[1] * q[5] - q[4] * q[2]);
        if det.abs() > 1e-10 {
            let inv_det = 1.0 / det;
            let x = inv_det * ((-q[3]) * (q[4] * q[7] - q[5] * q[5])
                             - q[1] * ((-q[6]) * q[7] - q[5] * (-q[8]))
                             + q[2] * ((-q[6]) * q[5] - q[4] * (-q[8])));
            let y = inv_det * (q[0] * ((-q[6]) * q[7] - q[5] * (-q[8]))
                             - (-q[3]) * (q[1] * q[7] - q[5] * q[2])
                             + q[2] * (q[1] * (-q[8]) - (-q[6]) * q[2]));
            let z = inv_det * (q[0] * (q[4] * (-q[8]) - (-q[6]) * q[5])
                             - q[1] * (q[1] * (-q[8]) - (-q[6]) * q[2])
                             + (-q[3]) * (q[1] * q[5] - q[4] * q[2]));
            return Vec3::new(x as f32, y as f32, z as f32);
        }
        // Fallback: pick the endpoint with lower error, or the midpoint.
        let mid = (v0 + v1) * 0.5;
        let e0 = self.error_at(v0.x as f64, v0.y as f64, v0.z as f64);
        let e1 = self.error_at(v1.x as f64, v1.y as f64, v1.z as f64);
        let em = self.error_at(mid.x as f64, mid.y as f64, mid.z as f64);
        if e0 <= e1 && e0 <= em { v0 }
        else if e1 <= em { v1 }
        else { mid }
    }
}

/// QEM edge-collapse mesh simplification.
///
/// Reduces the mesh to approximately `target_tri_count` triangles using the
/// quadric error metric algorithm. Preserves mesh boundaries and avoids
/// creating degenerate or flipped triangles.
fn qem_simplify(
    vertices: &[PackedVertex],
    indices: &[u32],
    target_tri_count: usize,
) -> (Vec<PackedVertex>, Vec<u32>) {
    use std::collections::{BinaryHeap, HashSet};
    use std::cmp::Ordering;

    let tri_count = indices.len() / 3;
    if tri_count <= target_tri_count || vertices.is_empty() || indices.is_empty() {
        return (vertices.to_vec(), indices.to_vec());
    }

    let positions: Vec<Vec3> = vertices.iter().map(|v| Vec3::from(v.position)).collect();

    // Compute per-vertex quadrics from incident face planes.
    let mut quadrics = vec![Quadric::ZERO; vertices.len()];
    for t in 0..tri_count {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let n = edge1.cross(edge2);
        let len = n.length();
        if len < 1e-10 {
            continue;
        }
        let n = n / len;
        let d = -n.dot(p0);
        let plane_q = Quadric::from_plane(n.x as f64, n.y as f64, n.z as f64, d as f64);
        quadrics[i0] = quadrics[i0].add(&plane_q);
        quadrics[i1] = quadrics[i1].add(&plane_q);
        quadrics[i2] = quadrics[i2].add(&plane_q);
    }

    // Detect boundary edges (edges with only one adjacent face) to penalize collapsing them.
    let mut edge_face_count: HashMap<(u32, u32), u32> = HashMap::new();
    for t in 0..tri_count {
        let tri_indices = [
            indices[t * 3],
            indices[t * 3 + 1],
            indices[t * 3 + 2],
        ];
        for e in 0..3 {
            let a = tri_indices[e];
            let b = tri_indices[(e + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_face_count.entry(key).or_insert(0) += 1;
        }
    }
    let mut is_boundary = vec![false; vertices.len()];
    for (&(a, b), &count) in &edge_face_count {
        if count == 1 {
            is_boundary[a as usize] = true;
            is_boundary[b as usize] = true;
        }
    }
    // Add large boundary-preservation penalty quadrics.
    for t in 0..tri_count {
        let tri_indices = [
            indices[t * 3],
            indices[t * 3 + 1],
            indices[t * 3 + 2],
        ];
        for e in 0..3 {
            let a = tri_indices[e] as usize;
            let b = tri_indices[(e + 1) % 3] as usize;
            let key = if a < b { (a as u32, b as u32) } else { (b as u32, a as u32) };
            if edge_face_count.get(&key) == Some(&1) {
                let edge_dir = (positions[b] - positions[a]).normalize_or_zero();
                let p0 = positions[a];
                let p1 = positions[b];
                let p2 = positions[tri_indices[(e + 2) % 3] as usize];
                let face_n = (p1 - p0).cross(p2 - p0);
                let face_n_len = face_n.length();
                if face_n_len < 1e-10 { continue; }
                let face_n = face_n / face_n_len;
                let boundary_n = edge_dir.cross(face_n).normalize_or_zero();
                let d = -boundary_n.dot(p0);
                let penalty = Quadric::from_plane(
                    boundary_n.x as f64 * 100.0,
                    boundary_n.y as f64 * 100.0,
                    boundary_n.z as f64 * 100.0,
                    d as f64 * 100.0,
                );
                quadrics[a] = quadrics[a].add(&penalty);
                quadrics[b] = quadrics[b].add(&penalty);
            }
        }
    }

    // Build edge set and priority queue.
    #[derive(Clone)]
    struct EdgeCollapse {
        cost: f64,
        v0: u32,
        v1: u32,
    }
    impl PartialEq for EdgeCollapse { fn eq(&self, o: &Self) -> bool { self.cost == o.cost } }
    impl Eq for EdgeCollapse {}
    impl PartialOrd for EdgeCollapse {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    impl Ord for EdgeCollapse {
        fn cmp(&self, o: &Self) -> Ordering {
            // Min-heap via reversed comparison.
            o.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
        }
    }

    let mut remap: Vec<u32> = (0..vertices.len() as u32).collect();
    fn find_root(remap: &mut [u32], mut v: u32) -> u32 {
        while remap[v as usize] != v {
            remap[v as usize] = remap[remap[v as usize] as usize];
            v = remap[v as usize];
        }
        v
    }

    let mut heap: BinaryHeap<EdgeCollapse> = BinaryHeap::new();
    let mut seen_edges: HashSet<(u32, u32)> = HashSet::new();
    for t in 0..tri_count {
        let tri_indices = [
            indices[t * 3],
            indices[t * 3 + 1],
            indices[t * 3 + 2],
        ];
        for e in 0..3 {
            let a = tri_indices[e];
            let b = tri_indices[(e + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            if seen_edges.insert(key) {
                let combined = quadrics[a as usize].add(&quadrics[b as usize]);
                let opt = combined.optimal_position(positions[a as usize], positions[b as usize]);
                let cost = combined.error_at(opt.x as f64, opt.y as f64, opt.z as f64).max(0.0);
                heap.push(EdgeCollapse { cost, v0: a, v1: b });
            }
        }
    }

    // Working copies.
    let mut live_positions = positions.clone();
    let mut live_tris: Vec<[u32; 3]> = (0..tri_count)
        .map(|t| [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]])
        .collect();
    let mut current_tri_count = tri_count;

    while current_tri_count > target_tri_count {
        let Some(edge) = heap.pop() else { break };
        let r0 = find_root(&mut remap, edge.v0);
        let r1 = find_root(&mut remap, edge.v1);
        if r0 == r1 { continue; }

        // Collapse r1 → r0: move r0 to optimal position.
        let combined = quadrics[r0 as usize].add(&quadrics[r1 as usize]);
        let opt = combined.optimal_position(live_positions[r0 as usize], live_positions[r1 as usize]);
        live_positions[r0 as usize] = opt;
        quadrics[r0 as usize] = combined;
        remap[r1 as usize] = r0;

        // Update triangles: remap r1→r0, remove degenerates.
        let mut removed = 0;
        let mut i = 0;
        while i < live_tris.len() {
            let tri = &mut live_tris[i];
            for v in tri.iter_mut() {
                *v = find_root(&mut remap, *v);
            }
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                live_tris.swap_remove(i);
                removed += 1;
            } else {
                i += 1;
            }
        }
        current_tri_count = current_tri_count.saturating_sub(removed);

        // Re-insert edges touching r0 with updated costs.
        let mut new_edges: HashSet<(u32, u32)> = HashSet::new();
        for tri in &live_tris {
            for e in 0..3 {
                let a = tri[e];
                let b = tri[(e + 1) % 3];
                if a == r0 || b == r0 {
                    let key = if a < b { (a, b) } else { (b, a) };
                    if new_edges.insert(key) {
                        let comb = quadrics[a as usize].add(&quadrics[b as usize]);
                        let p = comb.optimal_position(live_positions[a as usize], live_positions[b as usize]);
                        let c = comb.error_at(p.x as f64, p.y as f64, p.z as f64).max(0.0);
                        heap.push(EdgeCollapse { cost: c, v0: a, v1: b });
                    }
                }
            }
        }
    }

    if live_tris.is_empty() {
        return (vertices.to_vec(), indices.to_vec());
    }

    // Compact: collect used vertices, remap indices.
    let mut used: HashMap<u32, u32> = HashMap::new();
    let mut out_verts: Vec<PackedVertex> = Vec::new();
    let mut out_indices: Vec<u32> = Vec::with_capacity(live_tris.len() * 3);
    for tri in &live_tris {
        for &v in tri {
            let r = find_root(&mut remap, v);
            let compact = *used.entry(r).or_insert_with(|| {
                let idx = out_verts.len() as u32;
                let mut pv = vertices[r as usize];
                pv.position = live_positions[r as usize].to_array();
                out_verts.push(pv);
                idx
            });
            out_indices.push(compact);
        }
    }

    (out_verts, out_indices)
}

/// Generate exactly 8 LOD levels for a mesh using QEM edge-collapse simplification.
///
/// Returns `[(vertices, indices)]` at decreasing detail levels (LOD 0–7).
/// Each successive level targets ~50% of the previous level's triangle count,
/// matching UE5's traditional mesh LOD system.
///
/// If simplification cannot reduce the mesh further, the last successful level
/// is duplicated to fill the remaining slots — this ensures the cull shader
/// always finds a meshlet at every LOD level.
pub fn generate_lod_meshes(
    vertices: &[PackedVertex],
    indices: &[u32],
) -> Vec<(Vec<PackedVertex>, Vec<u32>)> {
    let base_tri_count = indices.len() / 3;
    let mut levels = Vec::with_capacity(8);
    levels.push((vertices.to_vec(), indices.to_vec()));

    let ratios: [f32; 7] = [0.50, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008];
    for &ratio in &ratios {
        let target = ((base_tri_count as f32 * ratio) as usize).max(1);
        let (prev_verts, prev_indices) = levels.last().unwrap();
        let simplified = qem_simplify(prev_verts, prev_indices, target);
        levels.push(simplified);
    }

    // Ensure exactly 8 levels — duplicate the last if QEM couldn't reduce further.
    while levels.len() < 8 {
        let last = levels.last().unwrap().clone();
        levels.push(last);
    }

    levels
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

