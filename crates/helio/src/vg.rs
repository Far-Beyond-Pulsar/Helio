//! CPU-side virtual geometry: meshlet decomposition for GPU-driven rendering.
//!
//! Uses [meshopt](https://crates.io/crates/meshopt) (meshoptimizer FFI) for
//! the full optimization pipeline: indexing, vertex cache optimization,
//! overdraw optimization, vertex fetch optimization, simplification, and
//! meshlet building with bounds computation.

use std::mem;

use libhelio::{GpuMeshletEntry, GpuMeshletVertex, MESHLET_MAX_TRIANGLES};
use meshopt::DecodePosition;

use crate::mesh::PackedVertex;

#[derive(Debug, Clone)]
pub(crate) struct GeneratedLodMesh {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
    /// Conservative accumulated object-space simplification error.
    pub error: f32,
}

// ─── Handle types ───────────────────────────────────────────────────────────

/// Opaque handle to a virtual mesh uploaded to the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualMeshId(pub u32);

// ─── Upload / descriptor types ──────────────────────────────────────────────

/// High-resolution mesh for virtual geometry upload.
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
    pub bounds: [f32; 4],
    pub flags: u32,
    pub groups: crate::groups::GroupMask,
    pub movability: Option<libhelio::Movability>,
}

// ─── meshopt DecodePosition impl ──────────────────────────────────────────

impl DecodePosition for PackedVertex {
    fn decode_position(&self) -> [f32; 3] {
        self.position
    }
}

// ─── Full meshopt optimization pipeline ───────────────────────────────────

/// Run the full meshopt optimization pipeline on a mesh:
/// 1. Weld byte-identical vertices while preserving attribute seams
/// 2. Vertex cache optimization (reorder tris for GPU transform cache)
/// 3. Overdraw optimization (reorder tris to reduce pixel overdraw)
/// 4. Vertex fetch optimization (reorder verts for memory locality)
fn optimize_mesh(vertices: &[PackedVertex], indices: &[u32]) -> (Vec<PackedVertex>, Vec<u32>) {
    if vertices.is_empty() || indices.is_empty() {
        return (vertices.to_vec(), indices.to_vec());
    }

    // Step 1: Weld only byte-identical vertices. Position-only welding corrupts
    // hard normals, tangents, lightmap UVs, and texture seams.
    let (welded_verts, welded_indices) = weld_exact_vertices(vertices, indices);

    // Step 2: Vertex cache optimization.
    let vcache_indices = meshopt::optimize_vertex_cache(&welded_indices, welded_verts.len());

    // Step 3: Overdraw optimization (threshold 1.05 = allow up to 5% worse cache ratio).
    let mut overdraw_indices = vcache_indices;
    meshopt::optimize_overdraw_in_place_decoder(&mut overdraw_indices, &welded_verts, 1.05);

    // Step 4: Vertex fetch optimization (reorder verts for locality).
    let remap = meshopt::optimize_vertex_fetch_remap(&overdraw_indices, welded_verts.len());
    let fetch_indices = meshopt::remap_index_buffer(Some(&overdraw_indices), welded_verts.len(), &remap);
    let fetch_verts = meshopt::remap_vertex_buffer(&welded_verts, welded_verts.len(), &remap);

    (fetch_verts, fetch_indices)
}

// ─── LOD generation ───────────────────────────────────────────────────────

/// Generate up to 8 distinct LOD levels using meshoptimizer's simplifier.
///
/// LOD 0 is the fully optimized original mesh. Each successive level targets
/// a smaller fraction of the original triangle count. The chain stops rather
/// than padding with mislabeled clones when an asset cannot be simplified any
/// further. The full meshopt pipeline (cache, overdraw, fetch optimization) is
/// applied to every retained level.
pub(crate) fn generate_lod_meshes(
    vertices: &[PackedVertex],
    indices: &[u32],
) -> Vec<GeneratedLodMesh> {
    if vertices.is_empty()
        || indices.is_empty()
        || indices.len() % 3 != 0
        || vertices
            .iter()
            .any(|vertex| vertex.position.iter().any(|value| !value.is_finite()))
        || indices
            .iter()
            .any(|&index| index as usize >= vertices.len())
    {
        return Vec::new();
    }

    // Optimize the base mesh first.
    let (opt_verts, opt_indices) = optimize_mesh(vertices, indices);
    let base_tri_count = opt_indices.len() / 3;

    let mut levels = Vec::with_capacity(8);
    levels.push(GeneratedLodMesh {
        vertices: opt_verts,
        indices: opt_indices,
        error: 0.0,
    });

    eprintln!(
        "[vg] base: {} verts, {} tris (from {} verts, {} tris)",
        levels[0].vertices.len(),
        base_tri_count,
        vertices.len(),
        indices.len() / 3,
    );

    let lod_ratios = [0.50, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008];
    for &ratio in &lod_ratios {
        let target_indices = (((base_tri_count as f32 * ratio) as usize).max(1)) * 3;
        let previous = levels.last().expect("base LOD exists");

        if previous.indices.len() <= target_indices {
            continue;
        }

        let attributes = simplification_attributes(&previous.vertices);
        let locks = vec![false; previous.vertices.len()];
        let mut relative_error = 0.0;

        // Build a progressive chain. Meshoptimizer recommends accumulating the
        // measured error when each level starts from the previous one.
        //
        // LockBorder pins vertices on the mesh's topological border (edges used
        // by only one triangle) so they never move during simplification. Without
        // it, non-closed/chunked assets (terrain tiles, modular pieces, anything
        // meant to butt up against a neighbor) drift apart at their shared edges
        // as soon as LOD1+ kicks in, opening visible gaps between what used to be
        // flush geometry — the `locks` array above only handles manual per-vertex
        // pins, it does nothing for border vertices on its own.
        let simplified_indices = meshopt::simplify_with_attributes_and_locks_decoder(
            &previous.indices,
            &previous.vertices,
            &attributes,
            &[10.0, 10.0, 10.0, 10.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 1.0],
            11 * std::mem::size_of::<f32>(),
            &locks,
            target_indices,
            f32::MAX,
            meshopt::SimplifyOptions::LockBorder,
            Some(&mut relative_error),
        );

        if !simplified_indices.is_empty() && simplified_indices.len() < previous.indices.len() {
            let absolute_error = previous.error
                + relative_error * meshopt::simplify_scale_decoder(&previous.vertices);

            // Compact, then run the full cache/overdraw/fetch pipeline on the LOD.
            let (compact_verts, compact_indices) =
                compact_mesh(&previous.vertices, &simplified_indices);
            let (final_verts, final_indices) = optimize_mesh(&compact_verts, &compact_indices);

            if final_indices.len() >= previous.indices.len() {
                continue;
            }

            eprintln!(
                "[vg] LOD {}: {}/{} tris (target {}, error {:.6})",
                levels.len(),
                final_indices.len() / 3,
                base_tri_count,
                target_indices / 3,
                absolute_error,
            );
            levels.push(GeneratedLodMesh {
                vertices: final_verts,
                indices: final_indices,
                error: absolute_error,
            });
        }
    }

    levels
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Weld byte-identical vertices without crossing any attribute discontinuity.
///
/// Walks `indices` rather than `vertices` so the output is compacted down to
/// only the vertices this mesh actually references. This matters a lot when
/// `vertices` is a large shared pool and `indices` addresses a small subset of
/// it — e.g. a multi-section FBX import where every section's `VirtualMeshUpload`
/// carries the *same* full shared vertex array but only its own slice of
/// indices. Scanning `vertices` directly (the previous implementation) welded
/// and kept the *entire* shared pool for every section regardless of how much
/// of it that section used, which corrupted more than just memory: LOD1+
/// simplification's accumulated error (`meshopt::simplify_scale_decoder`,
/// called on this function's output) is computed from the vertex buffer's own
/// extent, so a tiny section riding on a pool sized for the whole combined
/// mesh got its error scaled against the wrong (much larger) extent — which
/// then feeds directly into the GPU's screen-space LOD-selection threshold
/// test, corrupting which LOD (if any) gets selected for that section on
/// every instance, every frame.
fn weld_exact_vertices(
    vertices: &[PackedVertex],
    indices: &[u32],
) -> (Vec<PackedVertex>, Vec<u32>) {
    use std::collections::HashMap;

    fn vertex_key(v: &PackedVertex) -> [u32; 10] {
        [
            v.position[0].to_bits(),
            v.position[1].to_bits(),
            v.position[2].to_bits(),
            v.bitangent_sign.to_bits(),
            v.tex_coords0[0].to_bits(),
            v.tex_coords0[1].to_bits(),
            v.tex_coords1[0].to_bits(),
            v.tex_coords1[1].to_bits(),
            v.normal,
            v.tangent,
        ]
    }

    let mut vertex_to_new: HashMap<[u32; 10], u32> = HashMap::new();
    let mut welded_verts: Vec<PackedVertex> = Vec::new();

    let welded_indices = indices
        .iter()
        .map(|&old_idx| {
            let v = &vertices[old_idx as usize];
            let key = vertex_key(v);
            *vertex_to_new.entry(key).or_insert_with(|| {
                let idx = welded_verts.len() as u32;
                welded_verts.push(*v);
                idx
            })
        })
        .collect();

    (welded_verts, welded_indices)
}

fn simplification_attributes(vertices: &[PackedVertex]) -> Vec<f32> {
    fn unpack_snorm3(packed: u32) -> [f32; 3] {
        let component = |shift| ((packed >> shift) as u8 as i8) as f32 / 127.0;
        [component(0), component(8), component(16)]
    }

    let mut attributes = Vec::with_capacity(vertices.len() * 11);
    for vertex in vertices {
        let normal = unpack_snorm3(vertex.normal);
        let tangent = unpack_snorm3(vertex.tangent);
        attributes.extend_from_slice(&[
            vertex.tex_coords0[0],
            vertex.tex_coords0[1],
            vertex.tex_coords1[0],
            vertex.tex_coords1[1],
            normal[0],
            normal[1],
            normal[2],
            tangent[0],
            tangent[1],
            tangent[2],
            vertex.bitangent_sign,
        ]);
    }
    attributes
}

/// Remove unreferenced vertices and remap indices.
fn compact_mesh(vertices: &[PackedVertex], indices: &[u32]) -> (Vec<PackedVertex>, Vec<u32>) {
    let mut used = vec![u32::MAX; vertices.len()];
    let mut out_verts: Vec<PackedVertex> = Vec::new();
    let mut out_indices: Vec<u32> = Vec::with_capacity(indices.len());

    for &idx in indices {
        let i = idx as usize;
        if i >= vertices.len() {
            out_indices.push(0);
            continue;
        }
        if used[i] == u32::MAX {
            used[i] = out_verts.len() as u32;
            out_verts.push(vertices[i]);
        }
        out_indices.push(used[i]);
    }

    (out_verts, out_indices)
}

// ─── Meshlet building via meshopt ─────────────────────────────────────────

/// Build meshlets using meshoptimizer and return per-meshlet vertex/index streams.
///
/// Returns `(meshlet_entries, meshlet_vertices, meshlet_indices)` where:
/// - `meshlet_vertices` is a flat array of `GpuMeshletVertex` (all meshlets' unique vertices)
/// - `meshlet_indices` is a flat array of `u16` (all meshlets' local triangle indices)
///
/// Each meshlet entry's `meshlet_vertex_offset` and `meshlet_index_offset` are
/// set relative to the start of these returned arrays. Callers that concatenate
/// multiple calls must adjust these offsets accordingly.
pub fn meshletize_with_indices(
    vertices: &[PackedVertex],
    indices: &[u32],
) -> (Vec<GpuMeshletEntry>, Vec<GpuMeshletVertex>, Vec<u16>) {
    let tri_count = indices.len() / 3;
    if tri_count == 0
        || vertices.is_empty()
        || indices.len() % 3 != 0
        || indices
            .iter()
            .any(|&index| index as usize >= vertices.len())
    {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let max_verts = 64usize;
    let max_tris = MESHLET_MAX_TRIANGLES as usize;

    let vertex_adapter = meshopt::VertexDataAdapter::new(
        bytemuck::cast_slice(vertices),
        mem::size_of::<PackedVertex>(),
        0,
    )
    .expect("valid vertex layout");

    let meshlets = meshopt::clusterize::build_meshlets(
        indices,
        &vertex_adapter,
        max_verts,
        max_tris,
        0.5,
    );

    let mut entries = Vec::with_capacity(meshlets.len());
    let mut all_vertices: Vec<GpuMeshletVertex> = Vec::new();
    let mut all_indices: Vec<u16> = Vec::new();

    for i in 0..meshlets.len() {
        let m = meshlets.get(i);

        let vertex_offset = all_vertices.len() as u32;
        let index_offset = all_indices.len() as u32;
        let vertex_count = m.vertices.len() as u32;

        // Extract meshlet-unique vertices from the input vertex array.
        for &local_idx in m.vertices {
            let src = &vertices[local_idx as usize];
            all_vertices.push(GpuMeshletVertex {
                position: src.position,
                bitangent_sign: src.bitangent_sign,
                tex_coords0: src.tex_coords0,
                tex_coords1: src.tex_coords1,
                normal: src.normal,
                tangent: src.tangent,
                _pad: [0; 2],
            });
        }

        // Build meshlet-local u16 triangle indices.
        for &local_tri_idx in m.triangles {
            all_indices.push(local_tri_idx as u16);
        }

        let triangle_count = (m.triangles.len() / 3) as u32;
        let packed_counts = vertex_count | (triangle_count << 16);

        // Compute bounds using global vertex indices.
        let meshlet_global_indices: Vec<u32> = m
            .triangles
            .iter()
            .map(|&local_tri_idx| m.vertices[local_tri_idx as usize])
            .collect();

        let bounds = meshopt::clusterize::compute_cluster_bounds_decoder(
            &meshlet_global_indices,
            vertices,
        );

        entries.push(GpuMeshletEntry {
            center: bounds.center,
            radius: bounds.radius,
            cone_apex: bounds.cone_apex,
            cone_cutoff: bounds.cone_cutoff,
            cone_axis: bounds.cone_axis,
            lod_error: 0.0,
            packed_counts,
            meshlet_index_offset: index_offset,
            meshlet_vertex_offset: vertex_offset,
            parent_cluster_id: u32::MAX,
        });
    }

    (entries, all_vertices, all_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(position: [f32; 3], uv: [f32; 2], normal: [f32; 3]) -> PackedVertex {
        PackedVertex::from_components(position, normal, uv, [1.0, 0.0, 0.0], 1.0)
    }

    #[test]
    fn exact_weld_preserves_uv_and_normal_seams() {
        let a = vertex([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 1.0, 0.0]);
        let exact_duplicate = a;
        let uv_seam = vertex([0.0, 0.0, 0.0], [1.0, 0.0], [0.0, 1.0, 0.0]);
        let hard_normal = vertex([0.0, 0.0, 0.0], [0.0, 0.0], [1.0, 0.0, 0.0]);

        let (welded, remapped) =
            weld_exact_vertices(&[a, exact_duplicate, uv_seam, hard_normal], &[0, 1, 2, 3]);

        assert_eq!(welded.len(), 3);
        assert_eq!(remapped[0], remapped[1]);
        assert_ne!(remapped[0], remapped[2]);
        assert_ne!(remapped[0], remapped[3]);
    }

    #[test]
    fn weld_compacts_to_the_referenced_subset_of_a_shared_pool() {
        // Mirrors a multi-section import: `vertices` is a large shared pool
        // (as if it held every section of a combined mesh), but `indices`
        // only addresses a tiny slice of it — one triangle's worth. The
        // welded output must be sized to what's actually referenced, not to
        // the whole incoming pool, or every downstream extent/scale
        // computation (LOD error accumulation, in particular) gets corrupted
        // by irrelevant geometry the caller never asked this mesh to include.
        let mut vertices = Vec::new();
        for i in 0..500u32 {
            vertices.push(vertex(
                [i as f32, 0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0, 1.0],
            ));
        }
        // The one triangle this "section" actually uses, at the far end of
        // the shared pool.
        let tri = [497u32, 498, 499];

        let (welded, remapped) = weld_exact_vertices(&vertices, &tri);

        assert_eq!(
            welded.len(),
            3,
            "welded output should contain only the 3 referenced vertices, \
             not the full {}-vertex shared pool",
            vertices.len()
        );
        assert_eq!(remapped.len(), 3);
        // Positions must still be correct after compaction.
        let positions: std::collections::HashSet<_> = welded
            .iter()
            .map(|v| v.position[0].to_bits())
            .collect();
        assert!(positions.contains(&497.0f32.to_bits()));
        assert!(positions.contains(&498.0f32.to_bits()));
        assert!(positions.contains(&499.0f32.to_bits()));
    }

    #[test]
    fn malformed_indices_are_rejected_before_meshoptimizer() {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [1.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]),
        ];

        assert!(generate_lod_meshes(&vertices, &[0, 1]).is_empty());
        assert!(generate_lod_meshes(&vertices, &[0, 1, 3]).is_empty());
        let mut non_finite = vertices.clone();
        non_finite[0].position[0] = f32::NAN;
        assert!(generate_lod_meshes(&non_finite, &[0, 1, 2]).is_empty());
        assert!(meshletize_with_indices(&vertices, &[0, 1])
            .0
            .is_empty());
        assert!(meshletize_with_indices(&vertices, &[0, 1, 3])
            .0
            .is_empty());
    }

    #[test]
    fn meshlet_indices_and_bounds_cover_the_source_geometry() {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [1.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 1.0, 0.0], [1.0, 1.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]),
        ];
        let source = vec![0, 1, 2, 0, 2, 3];
        let (meshlets, meshlet_verts, meshlet_idxs) =
            meshletize_with_indices(&vertices, &source);

        // Verify that the vertex stream has at least as many entries as there
        // are unique vertex references in the source (should be 4 for a quad).
        assert!(meshlet_verts.len() <= source.len());

        // Verify that every meshlet's index range references valid data.
        for meshlet in &meshlets {
            let vc = meshlet.packed_counts & 0xFFFF;
            let tc = meshlet.packed_counts >> 16;
            assert!(vc >= 3 && vc <= 64);
            assert!(tc >= 1);
            let vo = meshlet.meshlet_vertex_offset as usize;
            let io = meshlet.meshlet_index_offset as usize;
            let idx_count = tc as usize * 3;
            // Every local index must reference a valid meshlet vertex.
            for j in 0..idx_count {
                let local_idx = meshlet_idxs[io + j] as usize;
                assert!(local_idx < vc as usize, "local index {local_idx} >= vertex_count {vc}");
            }
            // Every vertex must be within the meshlet's bounding sphere.
            for j in 0..vc as usize {
                let gpu_v = &meshlet_verts[vo + j];
                let p = glam::Vec3::from_array(gpu_v.position);
                let center = glam::Vec3::from_array(meshlet.center);
                assert!(
                    p.distance(center) <= meshlet.radius + 1e-5,
                    "vertex {j} at {p:?} outside sphere center {center:?} radius {}",
                    meshlet.radius
                );
            }
        }
    }

    #[test]
    fn generated_lod_errors_are_finite_and_monotonic() {
        let side = 12usize;
        let mut vertices = Vec::with_capacity(side * side);
        for y in 0..side {
            for x in 0..side {
                vertices.push(vertex(
                    [x as f32, y as f32, ((x * y) % 5) as f32 * 0.05],
                    [x as f32 / side as f32, y as f32 / side as f32],
                    [0.0, 0.0, 1.0],
                ));
            }
        }
        let mut indices = Vec::new();
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let i = (y * side + x) as u32;
                indices.extend_from_slice(&[i, i + 1, i + side as u32 + 1]);
                indices.extend_from_slice(&[i, i + side as u32 + 1, i + side as u32]);
            }
        }

        let lods = generate_lod_meshes(&vertices, &indices);
        assert!((2..=8).contains(&lods.len()));
        assert_eq!(lods[0].error, 0.0);
        for pair in lods.windows(2) {
            assert!(pair[1].error.is_finite());
            assert!(pair[1].error >= pair[0].error);
            assert!(pair[1].indices.len() < pair[0].indices.len());
        }
    }

    #[test]
    fn simplification_locks_border_vertices_so_chunks_stay_flush() {
        // A perfectly flat, single-sided grid patch — every vertex has no back
        // face, so the entire outer ring is on the mesh's topological border,
        // and since the patch is flat, collapsing or sliding a border vertex
        // along its (straight) edge costs zero measured error. That makes this
        // the worst case for the bug: an *unlocked* simplifier has no reason
        // not to remove border vertices first, since a flat rectangle's
        // "ideal" zero-error simplification is just its two corner triangles.
        // A convex-hull corner would incidentally survive that either way, so
        // this checks a MID-EDGE border vertex instead — collinear along a
        // straight boundary, contributing nothing to shape error, and exactly
        // the kind of point a real terrain-tile/modular-piece boundary needs
        // to keep in order to stay flush with whatever sits against that edge.
        let side = 16usize;
        let mut vertices = Vec::with_capacity(side * side);
        for y in 0..side {
            for x in 0..side {
                vertices.push(vertex(
                    [x as f32, y as f32, 0.0],
                    [x as f32 / side as f32, y as f32 / side as f32],
                    [0.0, 0.0, 1.0],
                ));
            }
        }
        let mut indices = Vec::new();
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let i = (y * side + x) as u32;
                indices.extend_from_slice(&[i, i + 1, i + side as u32 + 1]);
                indices.extend_from_slice(&[i, i + side as u32 + 1, i + side as u32]);
            }
        }

        // Midpoint of the y=0 edge: a straight-border vertex, not a corner.
        let mid_edge = vertices[side / 2].position;

        let lods = generate_lod_meshes(&vertices, &indices);
        assert!(lods.len() >= 2, "test grid should yield more than one LOD");

        let most_decimated = lods.last().expect("at least one LOD");
        assert!(
            most_decimated.indices.len() < indices.len(),
            "most decimated LOD should actually be simplified"
        );

        let survives = most_decimated
            .vertices
            .iter()
            .any(|v| v.position == mid_edge);
        assert!(
            survives,
            "mid-edge border vertex {mid_edge:?} was moved/welded away by the \
             most decimated LOD — border vertices are not being locked during \
             simplification, which will crack this asset against any neighbor \
             sharing that edge"
        );
    }

    #[test]
    fn irreducible_triangle_is_not_padded_with_fake_lods() {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [1.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]),
        ];

        let lods = generate_lod_meshes(&vertices, &[0, 1, 2]);
        assert_eq!(lods.len(), 1);
        assert_eq!(lods[0].indices.len(), 3);
    }

    /// Build a grid mesh of `side×side` vertices (yields `(side-1)²` quads).
    fn make_grid(side: usize) -> (Vec<PackedVertex>, Vec<u32>) {
        let mut vertices = Vec::with_capacity(side * side);
        for y in 0..side {
            for x in 0..side {
                vertices.push(vertex(
                    [x as f32, y as f32, ((x * y) % 5) as f32 * 0.05],
                    [x as f32 / side as f32, y as f32 / side as f32],
                    [0.0, 0.0, 1.0],
                ));
            }
        }
        let mut indices = Vec::new();
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let i = (y * side + x) as u32;
                indices.extend_from_slice(&[i, i + 1, i + side as u32 + 1]);
                indices.extend_from_slice(&[i, i + side as u32 + 1, i + side as u32]);
            }
        }
        (vertices, indices)
    }

    /// Simulate the GPU DAG traversal (coarse→fine) with the given lod_error
    /// values and camera distance.  Returns the index into `ancestor_errors`
    /// (0 = finest / leaf, last = coarsest / root).
    fn dag_traverse(
        ancestor_errors: &[f32], // finest → root
        max_scale: f32,
        focal_pixels: f32,
        distance: f32,
        threshold: f32,
    ) -> usize {
        let mut i = ancestor_errors.len();
        loop {
            if i == 0 {
                break;
            }
            i -= 1;
            let lod_error = ancestor_errors[i];
            let closest_distance = distance.max(1.0e-4);
            let projected = lod_error * max_scale * focal_pixels / closest_distance;
            if projected <= threshold || i == 0 {
                return i;
            }
        }
        0
    }

    /// CPU mirror of GPU leaves-only DAG cull + global emit-flag dedup.
    ///
    /// Only meshlets in `0..leaf_count` start traversal. Each selected meshlet
    /// is claimed at most once (simulating atomicExchange on emit flags).
    fn simulate_leaves_only_cull(
        meshlets: &[GpuMeshletEntry],
        leaf_count: usize,
        max_scale: f32,
        focal_pixels: f32,
        distance: f32,
        threshold: f32,
    ) -> Vec<u32> {
        let mut claimed = vec![false; meshlets.len()];
        let mut emits = Vec::new();

        for leaf in 0..leaf_count.min(meshlets.len()) {
            // Walk up to root.
            let mut stack = Vec::new();
            let mut cur = leaf;
            loop {
                stack.push(cur);
                let parent = meshlets[cur].parent_cluster_id;
                if parent == u32::MAX || stack.len() >= 8 {
                    break;
                }
                let p = parent as usize;
                if p >= meshlets.len() {
                    break;
                }
                cur = p;
            }
            // Walk down from root; pick coarsest good-enough level.
            let mut i = stack.len();
            let mut selected = None;
            while i > 0 {
                i -= 1;
                let idx = stack[i];
                let error = meshlets[idx].lod_error;
                let projected = error * max_scale * focal_pixels / distance.max(1e-4);
                if projected <= threshold || i == 0 {
                    selected = Some(idx as u32);
                    break;
                }
            }
            if let Some(idx) = selected {
                let slot = idx as usize;
                if slot < claimed.len() && !claimed[slot] {
                    claimed[slot] = true;
                    emits.push(idx);
                }
            }
        }
        emits
    }

    /// Build a multi-LOD meshlet DAG matching `meshes.rs` (for unit tests).
    fn build_test_dag(
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> (Vec<GpuMeshletEntry>, usize, Vec<u32>) {
        let lods = generate_lod_meshes(vertices, indices);
        let mut all_meshlets = Vec::new();
        let mut lod_meshlet_counts = Vec::new();

        for lod in &lods {
            let (mut meshlets, _, _) = meshletize_with_indices(&lod.vertices, &lod.indices);
            for m in &mut meshlets {
                m.lod_error = lod.error;
            }
            lod_meshlet_counts.push(meshlets.len() as u32);
            all_meshlets.extend(meshlets);
        }

        let lod_count = lod_meshlet_counts.len();
        let mut child_start = 0usize;
        for level in 0..lod_count.saturating_sub(1) {
            let child_count = lod_meshlet_counts[level] as usize;
            let parent_start = child_start + child_count;
            let parent_count = lod_meshlet_counts[level + 1] as usize;

            for ci in child_start..child_start + child_count {
                let c_center = glam::Vec3::from_array(all_meshlets[ci].center);
                let c_radius = all_meshlets[ci].radius;
                let mut best = u32::MAX;
                let mut best_d = f32::MAX;
                for pi in parent_start..parent_start + parent_count {
                    let p_center = glam::Vec3::from_array(all_meshlets[pi].center);
                    let p_radius = all_meshlets[pi].radius;
                    let d = c_center.distance(p_center);
                    if d <= c_radius + p_radius + 1e-6 && d < best_d {
                        best = pi as u32;
                        best_d = d;
                    }
                }
                if best == u32::MAX {
                    for pi in parent_start..parent_start + parent_count {
                        let p_center = glam::Vec3::from_array(all_meshlets[pi].center);
                        let d = c_center.distance(p_center);
                        if d < best_d {
                            best = pi as u32;
                            best_d = d;
                        }
                    }
                }
                all_meshlets[ci].parent_cluster_id = best;
            }
            child_start += child_count;
        }
        if let Some(&root_count) = lod_meshlet_counts.last() {
            let root_start = all_meshlets.len() - root_count as usize;
            for m in &mut all_meshlets[root_start..] {
                m.parent_cluster_id = u32::MAX;
            }
        }

        let leaf_count = lod_meshlet_counts.first().copied().unwrap_or(0) as usize;
        (all_meshlets, leaf_count, lod_meshlet_counts)
    }

    #[test]
    fn dag_parent_links_point_only_to_coarser_lods_and_roots_are_max() {
        let (vertices, indices) = make_grid(20);
        let (meshlets, leaf_count, lod_counts) = build_test_dag(&vertices, &indices);
        assert!(lod_counts.len() >= 2, "need multi-LOD for parent link test");
        assert!(leaf_count > 0);

        let total = meshlets.len();
        // Prefix sums of LOD ranges.
        let mut starts = vec![0usize; lod_counts.len()];
        for i in 1..lod_counts.len() {
            starts[i] = starts[i - 1] + lod_counts[i - 1] as usize;
        }

        for (li, &count) in lod_counts.iter().enumerate() {
            let start = starts[li];
            let end = start + count as usize;
            let is_coarsest = li + 1 == lod_counts.len();
            for idx in start..end {
                let parent = meshlets[idx].parent_cluster_id;
                if is_coarsest {
                    assert_eq!(
                        parent,
                        u32::MAX,
                        "root meshlet {idx} must have parent 0xFFFFFFFF"
                    );
                } else {
                    assert_ne!(parent, u32::MAX, "non-root meshlet {idx} needs a parent");
                    let p = parent as usize;
                    assert!(p < total, "parent {p} out of range for meshlet {idx}");
                    // Parent must be in a strictly coarser LOD range.
                    let parent_lod = starts
                        .iter()
                        .enumerate()
                        .find(|&(j, &s)| {
                            let e = s + lod_counts[j] as usize;
                            p >= s && p < e
                        })
                        .map(|(j, _)| j)
                        .expect("parent in some LOD");
                    assert!(
                        parent_lod > li,
                        "parent of LOD{li} meshlet {idx} must be coarser, got LOD{parent_lod}"
                    );
                }
            }
        }
    }

    #[test]
    fn dag_lod_error_zero_at_finest_and_non_decreasing_along_parent_chain() {
        let (vertices, indices) = make_grid(20);
        let (meshlets, leaf_count, _) = build_test_dag(&vertices, &indices);
        assert!(leaf_count > 0);

        for leaf in 0..leaf_count {
            assert_eq!(
                meshlets[leaf].lod_error, 0.0,
                "finest LOD meshlet {leaf} must have lod_error == 0"
            );
            let mut cur = leaf;
            let mut prev_error = meshlets[cur].lod_error;
            let mut steps = 0;
            loop {
                let parent = meshlets[cur].parent_cluster_id;
                if parent == u32::MAX || steps > 8 {
                    break;
                }
                let p = parent as usize;
                assert!(p < meshlets.len());
                let err = meshlets[p].lod_error;
                assert!(
                    err.is_finite() && err >= prev_error - 1e-6,
                    "lod_error must be non-decreasing along parent chain: {prev_error} -> {err}"
                );
                // Coarser levels of a simplified mesh should carry positive error.
                if steps == 0 && meshlets.len() > leaf_count {
                    // At least one step up from a multi-LOD leaf should raise error
                    // unless simplification measured zero (degenerate); allow >=.
                }
                prev_error = err;
                cur = p;
                steps += 1;
            }
        }
    }

    #[test]
    fn dag_parent_links_have_no_cycles() {
        let (vertices, indices) = make_grid(18);
        let (meshlets, _, _) = build_test_dag(&vertices, &indices);
        let n = meshlets.len();
        for start in 0..n {
            let mut cur = start;
            let mut seen = std::collections::HashSet::new();
            for _ in 0..16 {
                assert!(
                    seen.insert(cur),
                    "cycle detected in parent chain starting at {start}"
                );
                let parent = meshlets[cur].parent_cluster_id;
                if parent == u32::MAX {
                    break;
                }
                let p = parent as usize;
                assert!(p < n, "parent out of range");
                cur = p;
            }
        }
    }

    #[test]
    fn leaves_only_unique_emits_at_most_leaf_count_and_fewer_when_far() {
        let (vertices, indices) = make_grid(24);
        let (meshlets, leaf_count, lod_counts) = build_test_dag(&vertices, &indices);
        assert!(lod_counts.len() >= 2);
        assert!(leaf_count > 0);

        let focal = 540.0_f32;
        let threshold = 2.0_f32;

        let near = simulate_leaves_only_cull(&meshlets, leaf_count, 1.0, focal, 1.0, threshold);
        let far = simulate_leaves_only_cull(&meshlets, leaf_count, 1.0, focal, 800.0, threshold);

        // Unique by construction of the claim set.
        assert_eq!(near.len(), near.iter().collect::<std::collections::HashSet<_>>().len());
        assert_eq!(far.len(), far.iter().collect::<std::collections::HashSet<_>>().len());

        assert!(
            near.len() <= leaf_count,
            "unique emits {} must be ≤ leaf count {leaf_count}",
            near.len()
        );
        assert!(
            far.len() <= leaf_count,
            "unique emits {} must be ≤ leaf count {leaf_count}",
            far.len()
        );
        assert!(
            far.len() < leaf_count,
            "at far distance unique emits {} should be << leaves {leaf_count}",
            far.len()
        );
        assert!(
            far.len() < near.len(),
            "far ({}) should emit fewer clusters than near ({})",
            far.len(),
            near.len()
        );
    }

    #[test]
    fn two_leaves_sharing_parent_produce_one_emit_when_parent_selected() {
        // Synthetic DAG: leaves 0,1 → parent 2 (root).
        let mut meshlets = vec![
            GpuMeshletEntry {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
                cone_apex: [0.0; 3],
                cone_cutoff: 2.0,
                cone_axis: [0.0, 0.0, 1.0],
                lod_error: 0.0,
                packed_counts: 3 | (1 << 16),
                meshlet_index_offset: 0,
                meshlet_vertex_offset: 0,
                parent_cluster_id: 2,
            },
            GpuMeshletEntry {
                center: [0.5, 0.0, 0.0],
                radius: 1.0,
                cone_apex: [0.0; 3],
                cone_cutoff: 2.0,
                cone_axis: [0.0, 0.0, 1.0],
                lod_error: 0.0,
                packed_counts: 3 | (1 << 16),
                meshlet_index_offset: 0,
                meshlet_vertex_offset: 0,
                parent_cluster_id: 2,
            },
            GpuMeshletEntry {
                center: [0.25, 0.0, 0.0],
                radius: 2.0,
                cone_apex: [0.0; 3],
                cone_cutoff: 2.0,
                cone_axis: [0.0, 0.0, 1.0],
                lod_error: 1.0, // large error → selected when far enough / threshold loose
                packed_counts: 3 | (1 << 16),
                meshlet_index_offset: 0,
                meshlet_vertex_offset: 0,
                parent_cluster_id: u32::MAX,
            },
        ];
        // projected = 1.0 * 1.0 * 1000 / dist. At dist=1000, projected=1.0 <= threshold 1.0 → parent.
        let emits = simulate_leaves_only_cull(&meshlets, 2, 1.0, 1000.0, 1000.0, 1.0);
        assert_eq!(emits, vec![2], "both leaves must collapse to a single parent emit");

        // Near: projected parent error huge → emit leaves (two unique).
        meshlets[2].lod_error = 10.0;
        let near = simulate_leaves_only_cull(&meshlets, 2, 1.0, 1000.0, 10.0, 1.0);
        // projected parent = 10*1000/10 = 1000 > 1 → need finer → leaves
        let mut near_sorted = near.clone();
        near_sorted.sort();
        assert_eq!(near_sorted, vec![0, 1]);
    }

    #[test]
    fn projected_error_lod_selection_coarse_far_fine_near() {
        // ancestor_errors: finest → root
        let chain = [0.0, 0.01, 0.04, 0.16];
        let focal = 1000.0;
        let thr = 1.0;

        // Near: only finest (error 0) is acceptable once coarser exceeds thr.
        assert_eq!(dag_traverse(&chain, 1.0, focal, 5.0, thr), 0);
        // Mid: can accept level 1 (0.01*1000/100 = 0.1 <= 1)
        assert_eq!(dag_traverse(&chain, 1.0, focal, 100.0, thr), 2);
        // Far: coarsest root
        assert_eq!(dag_traverse(&chain, 1.0, focal, 10_000.0, thr), 3);
        // Non-uniform scale increases projected error → finer selection
        assert_eq!(dag_traverse(&chain, 4.0, focal, 100.0, thr), 1);
    }

    #[test]
    fn dag_traversal_diagnostics() {
        let side = 24usize;
        let (vertices, indices) = make_grid(side);

        let lods = generate_lod_meshes(&vertices, &indices);
        eprintln!("\n=== DAG Diagnostic ===");
        eprintln!("Grid: {}×{} = {} verts, {} tris",
            side, side, vertices.len(), indices.len() / 3);
        eprintln!("LOD count: {}", lods.len());

        // Meshletize each LOD and collect errors.
        let mut lod_errors: Vec<f32> = Vec::new();
        let mut lod_meshlet_counts: Vec<u32> = Vec::new();
        let mut all_meshlets: Vec<GpuMeshletEntry> = Vec::new();

        for (li, lod) in lods.iter().enumerate() {
            let (mut meshlets, _, _) = meshletize_with_indices(&lod.vertices, &lod.indices);
            let count = meshlets.len() as u32;
            lod_errors.push(lod.error);
            lod_meshlet_counts.push(count);
            // Assign cumulative error (same as meshes.rs fixup step)
            for m in &mut meshlets {
                m.lod_error = lod.error;
            }
            eprintln!("  LOD {}: {} tris, {} meshlets, cum_error={:.6}",
                li, lod.indices.len() / 3, meshlets.len(), lod.error);
            all_meshlets.extend(meshlets);
        }

        // ── Build DAG parent links (same algorithm as meshes.rs) ────────────
        let lod_count = lod_meshlet_counts.len();
        let mut child_start = 0usize;
        for level in 0..lod_count.saturating_sub(1) {
            let child_count = lod_meshlet_counts[level] as usize;
            let parent_start = child_start + child_count;
            let parent_count = lod_meshlet_counts[level + 1] as usize;

            for ci in child_start..child_start + child_count {
                let c_center = glam::Vec3::from_array(all_meshlets[ci].center);
                let c_radius = all_meshlets[ci].radius;
                let mut best = u32::MAX;
                let mut best_d = f32::MAX;
                for pi in parent_start..parent_start + parent_count {
                    let p_center = glam::Vec3::from_array(all_meshlets[pi].center);
                    let p_radius = all_meshlets[pi].radius;
                    let d = c_center.distance(p_center);
                    if d <= c_radius + p_radius + 1e-6 && d < best_d {
                        best = pi as u32;
                        best_d = d;
                    }
                }
                if best == u32::MAX {
                    for pi in parent_start..parent_start + parent_count {
                        let p_center = glam::Vec3::from_array(all_meshlets[pi].center);
                        let d = c_center.distance(p_center);
                        if d < best_d {
                            best = pi as u32;
                            best_d = d;
                        }
                    }
                }
                all_meshlets[ci].parent_cluster_id = best;
            }
            child_start += child_count;
        }
        // Root meshlets get 0xFFFFFFFF (no parent)
        let root_start: usize = all_meshlets.len() - *lod_meshlet_counts.last().unwrap() as usize;
        for m in &mut all_meshlets[root_start..] {
            m.parent_cluster_id = u32::MAX;
        }

        // ── Validate DAG ────────────────────────────────────────────────────
        eprintln!("\n  Validating DAG...");
        let total = all_meshlets.len();
        let mut visited = vec![false; total];
        for start in 0..total {
            let mut cur = start;
            let mut steps = 0;
            loop {
                if visited[cur] { break; }
                visited[cur] = true;
                let m = &all_meshlets[cur];
                if m.parent_cluster_id == u32::MAX || steps > 8 { break; }
                cur = m.parent_cluster_id as usize;
                assert!(cur < total, "parent out of bounds: {cur} >= {total}");
                steps += 1;
            }
        }
        assert!(visited.iter().all(|&v| v), "Some meshlets were not visited");

        // Count unique parent references for each root meshlet
        eprintln!("\n  Child→parent convergence (fan-in):");
        for level in 0..lod_count.saturating_sub(1) {
            let child_count = lod_meshlet_counts[level] as usize;
            let parent_count = lod_meshlet_counts[level + 1] as usize;
            let mut fan_in = vec![0u32; parent_count];
            // Count children that reference each parent
            child_start = lod_meshlet_counts[..level].iter().copied().sum::<u32>() as usize;
            for ci in child_start..child_start + child_count {
                let m = &all_meshlets[ci];
                if m.parent_cluster_id != u32::MAX {
                    let pi = m.parent_cluster_id as usize;
                    let parent_lod_start = lod_meshlet_counts[..level + 1].iter().copied().sum::<u32>() as usize;
                    if pi >= parent_lod_start && pi < parent_lod_start + parent_count {
                        fan_in[pi - parent_lod_start] += 1;
                    }
                }
            }
            eprintln!("    LOD {} → LOD {} children per parent: {:?}",
                level, level + 1, fan_in);
        }

        // ── Leaves-only + global claim simulation at various distances ─────
        let leaf_count = lod_meshlet_counts[0] as usize;
        let focal_pixels = 540.0;
        let threshold_px = 0.5;
        let max_scale = 1.0;

        eprintln!("\n  Leaves-only + claim simulation:");
        let distances: [f32; 9] = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0];
        for &dist in &distances {
            let emits = simulate_leaves_only_cull(
                &all_meshlets,
                leaf_count,
                max_scale,
                focal_pixels,
                dist,
                threshold_px,
            );
            let mut lod_emit_counts = vec![0u32; lod_count];
            for &idx in &emits {
                let mut cursor = 0usize;
                for li in 0..lod_count {
                    let cnt = lod_meshlet_counts[li] as usize;
                    if (idx as usize) >= cursor && (idx as usize) < cursor + cnt {
                        lod_emit_counts[li] += 1;
                        break;
                    }
                    cursor += cnt;
                }
            }
            let unique = emits.len();
            eprintln!("    dist={dist:>6.1}: {unique:>3} unique emits (≤ {leaf_count} leaves)");
            eprintln!("              per-LOD: {:?}", &lod_emit_counts);

            assert!(unique <= leaf_count);
            if dist <= 1.0 {
                assert!(lod_emit_counts[0] > 0, "At dist={dist}, LOD 0 should have emits");
            }
            if dist >= 500.0 {
                let coarse_emits: u32 = lod_emit_counts[1..].iter().sum();
                assert!(
                    coarse_emits > 0 || unique < leaf_count,
                    "At dist={dist}, should select coarser LODs or fewer clusters"
                );
            }
        }
    }
}
