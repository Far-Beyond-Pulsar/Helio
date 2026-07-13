//! CPU-side virtual geometry: meshlet decomposition for GPU-driven rendering.
//!
//! Uses [meshopt](https://crates.io/crates/meshopt) (meshoptimizer FFI) for
//! the full optimization pipeline: indexing, vertex cache optimization,
//! overdraw optimization, vertex fetch optimization, simplification, and
//! meshlet building with bounds computation.

use std::mem;

use libhelio::{GpuMeshletEntry, MESHLET_MAX_TRIANGLES};
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
pub struct VirtualMeshId(pub(crate) u32);

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

/// Generate exactly 8 LOD levels using meshoptimizer's simplifier.
///
/// LOD 0 is the fully optimized original mesh. Each successive level targets
/// ~50% of the previous level's triangle count. The full meshopt pipeline
/// (cache, overdraw, fetch optimization) is applied to every level.
pub(crate) fn generate_lod_meshes(
    vertices: &[PackedVertex],
    indices: &[u32],
) -> Vec<GeneratedLodMesh> {
    if vertices.is_empty()
        || indices.is_empty()
        || indices.len() % 3 != 0
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
            levels.push(previous.clone());
            continue;
        }

        let attributes = simplification_attributes(&previous.vertices);
        let locks = vec![false; previous.vertices.len()];
        let mut relative_error = 0.0;

        // Build a progressive chain. Meshoptimizer recommends accumulating the
        // measured error when each level starts from the previous one.
        let simplified_indices = meshopt::simplify_with_attributes_and_locks_decoder(
            &previous.indices,
            &previous.vertices,
            &attributes,
            &[10.0, 10.0, 10.0, 10.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 1.0],
            11 * std::mem::size_of::<f32>(),
            &locks,
            target_indices,
            f32::MAX,
            meshopt::SimplifyOptions::None,
            Some(&mut relative_error),
        );

        if simplified_indices.is_empty() {
            levels.push(previous.clone());
        } else {
            let absolute_error = previous.error
                + relative_error * meshopt::simplify_scale_decoder(&previous.vertices);

            // Compact, then run the full cache/overdraw/fetch pipeline on the LOD.
            let (compact_verts, compact_indices) =
                compact_mesh(&previous.vertices, &simplified_indices);
            let (final_verts, final_indices) = optimize_mesh(&compact_verts, &compact_indices);

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

    while levels.len() < 8 {
        let last = levels.last().unwrap().clone();
        levels.push(last);
    }

    levels
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Weld byte-identical vertices without crossing any attribute discontinuity.
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
    let mut remap = vec![0u32; vertices.len()];
    let mut welded_verts: Vec<PackedVertex> = Vec::new();

    for (old_idx, v) in vertices.iter().enumerate() {
        let key = vertex_key(v);
        let new_idx = *vertex_to_new.entry(key).or_insert_with(|| {
            let idx = welded_verts.len() as u32;
            welded_verts.push(*v);
            idx
        });
        remap[old_idx] = new_idx;
    }

    let welded_indices = indices.iter().map(|&i| remap[i as usize]).collect();

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

/// Build meshlets using meshoptimizer and return the reordered index buffer.
///
/// Returns `(meshlet_entries, flat_index_buffer)` — the flat indices are the
/// meshlet-grouped index data ready for upload to the mega-buffer.
pub fn meshletize_with_indices(
    vertices: &[PackedVertex],
    indices: &[u32],
    mesh_first_index: u32,
    mesh_first_vertex: u32,
) -> (Vec<GpuMeshletEntry>, Vec<u32>) {
    let tri_count = indices.len() / 3;
    if tri_count == 0
        || vertices.is_empty()
        || indices.len() % 3 != 0
        || indices
            .iter()
            .any(|&index| index as usize >= vertices.len())
    {
        return (Vec::new(), Vec::new());
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
    let mut flat_indices: Vec<u32> = Vec::new();

    for i in 0..meshlets.len() {
        let m = meshlets.get(i);
        let first_index_offset = flat_indices.len() as u32;

        let mut meshlet_global_indices: Vec<u32> = Vec::with_capacity(m.triangles.len());
        for &local_tri_idx in m.triangles {
            let vertex_slot = m.vertices[local_tri_idx as usize];
            meshlet_global_indices.push(vertex_slot);
            flat_indices.push(vertex_slot);
        }

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
            first_index: mesh_first_index + first_index_offset,
            index_count: meshlet_global_indices.len() as u32,
            vertex_offset: mesh_first_vertex as i32,
            instance_index: 0,
        });
    }

    (entries, flat_indices)
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
    fn malformed_indices_are_rejected_before_meshoptimizer() {
        let vertices = vec![
            vertex([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([1.0, 0.0, 0.0], [1.0, 0.0], [0.0, 0.0, 1.0]),
            vertex([0.0, 1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]),
        ];

        assert!(generate_lod_meshes(&vertices, &[0, 1]).is_empty());
        assert!(generate_lod_meshes(&vertices, &[0, 1, 3]).is_empty());
        assert!(meshletize_with_indices(&vertices, &[0, 1], 0, 0)
            .0
            .is_empty());
        assert!(meshletize_with_indices(&vertices, &[0, 1, 3], 0, 0)
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
        let (meshlets, flattened) = meshletize_with_indices(&vertices, &source, 0, 0);

        assert_eq!(flattened.len(), source.len());
        let mut flattened_sorted = flattened.clone();
        let mut source_sorted = source.clone();
        flattened_sorted.sort_unstable();
        source_sorted.sort_unstable();
        assert_eq!(flattened_sorted, source_sorted);

        for meshlet in &meshlets {
            let first = meshlet.first_index as usize;
            let end = first + meshlet.index_count as usize;
            for &index in &flattened[first..end] {
                let p = glam::Vec3::from_array(vertices[index as usize].position);
                let center = glam::Vec3::from_array(meshlet.center);
                assert!(p.distance(center) <= meshlet.radius + 1e-5);
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
        assert_eq!(lods.len(), 8);
        assert_eq!(lods[0].error, 0.0);
        for pair in lods.windows(2) {
            assert!(pair[1].error.is_finite());
            assert!(pair[1].error >= pair[0].error);
            assert!(pair[1].indices.len() <= pair[0].indices.len());
        }
    }
}
