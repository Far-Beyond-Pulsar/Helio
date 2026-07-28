//! Virtual geometry buffer rebuild and frame data packaging.
//!
//! Meshlet descriptors are stored once per referenced virtual mesh. A separate
//! object array supplies the instance index, conservative local bounds, measured
//! LOD errors, and the descriptor range for each LOD.

use std::collections::HashMap;

use libhelio::{
    GpuMeshletEntry, GpuMeshletVertex, GpuVgObject, GpuVgWorkItem, VgFrameData,
    VG_CULL_MESHLETS_PER_WORK_ITEM,
};

use crate::vg::VirtualMeshId;

use super::super::types::VirtualMeshRecord;

fn append_unique_meshlets(
    referenced_meshes: impl IntoIterator<Item = VirtualMeshId>,
    mesh_records: &HashMap<VirtualMeshId, VirtualMeshRecord>,
    output_meshlets: &mut Vec<GpuMeshletEntry>,
    output_vertices: &mut Vec<GpuMeshletVertex>,
    output_indices: &mut Vec<u16>,
) -> HashMap<VirtualMeshId, u32> {
    let mut bases = HashMap::new();

    for mesh_id in referenced_meshes {
        if bases.contains_key(&mesh_id) {
            continue;
        }
        let Some(record) = mesh_records.get(&mesh_id) else {
            continue;
        };
        if record.meshlets.is_empty() || record.lod_count == 0 {
            continue;
        }

        let meshlet_base = u32::try_from(output_meshlets.len())
            .expect("virtual geometry exceeds the u32 descriptor address space");
        let vertex_base = u32::try_from(output_vertices.len())
            .expect("virtual geometry vertex stream exceeds u32");
        let index_base = u32::try_from(output_indices.len())
            .expect("virtual geometry index stream exceeds u32");

        // Fix up meshlet offsets from record-local to global.
        for m in &record.meshlets {
            let mut fixed = *m;
            fixed.meshlet_vertex_offset = fixed
                .meshlet_vertex_offset
                .checked_add(vertex_base)
                .expect("meshlet_vertex_offset overflow during global fixup");
            fixed.meshlet_index_offset = fixed
                .meshlet_index_offset
                .checked_add(index_base)
                .expect("meshlet_index_offset overflow during global fixup");
            if fixed.parent_cluster_id != u32::MAX {
                fixed.parent_cluster_id = meshlet_base
                    .checked_add(fixed.parent_cluster_id)
                    .expect("parent_cluster_id overflow during global offset fixup");
            }
            output_meshlets.push(fixed);
        }

        output_vertices.extend_from_slice(&record.meshlet_vertices);
        output_indices.extend_from_slice(&record.meshlet_indices);

        bases.insert(mesh_id, meshlet_base);
    }

    bases
}

/// Emit work items covering only DAG leaves (finest LOD meshlets).
///
/// Coarser meshlets must not start their own traversal — they are selected
/// exclusively when a leaf walks up the parent chain. Processing all LODs as
/// entry points causes duplicate emits and wrong LOD mixes.
fn append_work_items(
    object_index: u32,
    leaf_meshlet_count: u32,
    output: &mut Vec<GpuVgWorkItem>,
) {
    for local_meshlet_base in
        (0..leaf_meshlet_count).step_by(VG_CULL_MESHLETS_PER_WORK_ITEM as usize)
    {
        output.push(GpuVgWorkItem {
            object_index,
            local_meshlet_base,
        });
    }
}
impl super::super::Scene {
    /// Returns the immutable mesh descriptors, object-level LOD metadata, and
    /// instance data consumed by the virtual-geometry pass.
    pub fn vg_frame_data(&self) -> Option<VgFrameData<'_>> {
        if self.vg_cpu_objects.is_empty() {
            return None;
        }
        Some(VgFrameData {
            meshlets: bytemuck::cast_slice(&self.vg_cpu_meshlets),
            meshlet_vertices: bytemuck::cast_slice(&self.vg_cpu_meshlet_vertices),
            meshlet_indices: bytemuck::cast_slice(&self.vg_cpu_meshlet_indices),
            objects: bytemuck::cast_slice(&self.vg_cpu_objects),
            instances: bytemuck::cast_slice(&self.vg_cpu_instances),
            work_items: bytemuck::cast_slice(&self.vg_cpu_work_items),
            meshlet_count: u32::try_from(self.vg_cpu_meshlets.len())
                .expect("virtual geometry exceeds the u32 descriptor address space"),
            meshlet_vertex_count: u32::try_from(self.vg_cpu_meshlet_vertices.len())
                .expect("virtual geometry vertex count exceeds u32"),
            meshlet_index_count: u32::try_from(self.vg_cpu_meshlet_indices.len())
                .expect("virtual geometry index count exceeds u32"),
            object_count: u32::try_from(self.vg_cpu_objects.len())
                .expect("virtual geometry exceeds the u32 object address space"),
            work_item_count: u32::try_from(self.vg_cpu_work_items.len())
                .expect("virtual geometry exceeds the u32 work-item address space"),
            emit_flag_count: self
                .vg_cpu_objects
                .last()
                .map(|o| o.emit_flag_base.saturating_add(o.total_meshlet_count))
                .unwrap_or(0),
            buffer_version: self.vg_buffer_version,
            instance_version: self.vg_instance_version,
            instance_dirty_start: self
                .vg_published_instance_dirty_range
                .map_or(0, |(start, _)| u32::try_from(start).expect("VG dirty start exceeds u32")),
            instance_dirty_count: self
                .vg_published_instance_dirty_range
                .map_or(0, |(start, end)| {
                    u32::try_from(end - start).expect("VG dirty count exceeds u32")
                }),
        })
    }

    /// Rebuild the CPU mirrors used by the GPU-driven virtual-geometry pass.
    ///
    /// Each referenced virtual mesh contributes its descriptors exactly once,
    /// regardless of instance count. Each object then points at the shared
    /// per-LOD ranges. The indirect buffer grows dynamically on demand based on
    /// the total unique meshlet count, removing the fixed `max_draw_count` ceiling.
    pub(in crate::scene) fn rebuild_vg_buffers(&mut self) {
        let dense_object_count = self.vg_objects.dense_len();
        self.vg_cpu_meshlets.clear();
        self.vg_cpu_meshlet_vertices.clear();
        self.vg_cpu_meshlet_indices.clear();
        self.vg_cpu_objects.clear();
        self.vg_cpu_instances.clear();
        self.vg_cpu_work_items.clear();
        self.vg_instance_dirty_range = None;
        self.vg_published_instance_dirty_range = None;
        self.vg_cpu_objects.reserve(dense_object_count);
        self.vg_cpu_instances.reserve(dense_object_count);

        let referenced_meshes = (0..dense_object_count)
            .filter_map(|index| self.vg_objects.get_dense(index))
            .map(|object| object.virtual_mesh)
            .collect::<Vec<_>>();
        let mesh_bases = append_unique_meshlets(
            referenced_meshes,
            &self.vg_meshes,
            &mut self.vg_cpu_meshlets,
            &mut self.vg_cpu_meshlet_vertices,
            &mut self.vg_cpu_meshlet_indices,
        );

        let mut emit_flag_base = 0u32;
        for dense_index in 0..dense_object_count {
            let Some(object) = self.vg_objects.get_dense(dense_index) else {
                continue;
            };
            let Some(mesh) = self.vg_meshes.get(&object.virtual_mesh) else {
                continue;
            };
            let Some(&mesh_base) = mesh_bases.get(&object.virtual_mesh) else {
                continue;
            };

            let instance_index = u32::try_from(self.vg_cpu_instances.len())
                .expect("virtual geometry exceeds the u32 instance address space");
            let object_index = u32::try_from(self.vg_cpu_objects.len())
                .expect("virtual geometry exceeds the u32 object address space");

            self.vg_cpu_instances.push(object.instance);
            // Leaves-only entry: work items cover leaf_meshlet_count. Coarser
            // meshlets live after the leaf range and are reached via parent walks.
            // emit_flag_base gives each object a private claim slice so instances
            // sharing meshlet descriptors still draw independently.
            self.vg_cpu_objects.push(GpuVgObject {
                instance_index,
                leaf_meshlet_count: mesh.leaf_meshlet_count,
                first_meshlet: mesh_base,
                total_meshlet_count: mesh.total_meshlet_count,
                local_bounds: mesh.local_bounds,
                emit_flag_base,
                visible: 0,
                _pad0: 0,
                _pad1: 0,
            });
            emit_flag_base = emit_flag_base
                .checked_add(mesh.total_meshlet_count)
                .expect("emit flag base overflow");
            append_work_items(
                object_index,
                mesh.leaf_meshlet_count,
                &mut self.vg_cpu_work_items,
            );
        }

        self.vg_buffer_version = self.vg_buffer_version.wrapping_add(1);
        eprintln!(
            "[vg] rebuild: {} objects, {} unique meshlets, {} work spans, {} meshlet verts, {} meshlet idxs",
            self.vg_cpu_objects.len(),
            self.vg_cpu_meshlets.len(),
            self.vg_cpu_work_items.len(),
            self.vg_cpu_meshlet_vertices.len(),
            self.vg_cpu_meshlet_indices.len(),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use libhelio::{GpuMeshletEntry, GpuMeshletVertex};

    use super::{append_unique_meshlets, append_work_items, VirtualMeshId, VirtualMeshRecord};

    fn meshlet(meshlet_vertex_offset: u32, meshlet_index_offset: u32) -> GpuMeshletEntry {
        GpuMeshletEntry {
            center: [0.0; 3],
            radius: 1.0,
            cone_apex: [0.0; 3],
            cone_cutoff: 2.0,
            cone_axis: [0.0, 1.0, 0.0],
            lod_error: 0.0,
            packed_counts: 3 | (1 << 16), // vertex_count=3, triangle_count=1
            meshlet_index_offset,
            meshlet_vertex_offset,
            parent_cluster_id: u32::MAX,
        }
    }

    fn record(
        meshlets: Vec<GpuMeshletEntry>,
        vertices: Vec<GpuMeshletVertex>,
        indices: Vec<u16>,
    ) -> VirtualMeshRecord {
        let count = meshlets.len() as u32;
        VirtualMeshRecord {
            meshlets,
            meshlet_vertices: vertices,
            meshlet_indices: indices,
            local_bounds: [0.0, 0.0, 0.0, 1.0],
            lod_count: 1,
            total_meshlet_count: count,
            leaf_meshlet_count: count,
            ref_count: 0,
        }
    }

    fn dummy_vertex() -> GpuMeshletVertex {
        GpuMeshletVertex {
            position: [0.0; 3],
            bitangent_sign: 1.0,
            tex_coords0: [0.0; 2],
            tex_coords1: [0.0; 2],
            normal: 0,
            tangent: 0,
            _pad: [0; 2],
        }
    }

    #[test]
    fn repeated_instances_share_one_descriptor_copy() {
        let first = VirtualMeshId(3);
        let second = VirtualMeshId(7);
        // Record `first` has 2 meshlets, each with 3 vertices → 6 verts, 6 idxs
        let verts_a = vec![
            dummy_vertex(), dummy_vertex(), dummy_vertex(),
            dummy_vertex(), dummy_vertex(), dummy_vertex(),
        ];
        let idxs_a = vec![0u16, 1, 2, 3, 4, 5];
        // Record `second` has 1 meshlet with 1 vertex → 1 vert, 3 idxs
        let verts_b = vec![dummy_vertex()];
        let idxs_b = vec![0u16, 0, 0];
        let records = HashMap::from([
            (first, record(vec![meshlet(0, 0), meshlet(3, 3)], verts_a, idxs_a)),
            (second, record(vec![meshlet(0, 0)], verts_b, idxs_b)),
        ]);
        let mut output_meshlets = Vec::new();
        let mut output_vertices = Vec::new();
        let mut output_indices = Vec::new();

        let bases = append_unique_meshlets(
            [first, first, second, first, second],
            &records,
            &mut output_meshlets,
            &mut output_vertices,
            &mut output_indices,
        );

        assert_eq!(output_meshlets.len(), 3);
        assert_eq!(bases[&first], 0);
        assert_eq!(bases[&second], 2);
        // First meshlet of `first` — vertex offset stays 0
        assert_eq!(output_meshlets[0].meshlet_vertex_offset, 0);
        // Second meshlet of `first` — vertex offset 3
        assert_eq!(output_meshlets[1].meshlet_vertex_offset, 3);
        // Meshlet of `second` — combined vertex offset = 6 (3+3)
        assert_eq!(output_meshlets[2].meshlet_vertex_offset, 6);
        // Vertex stream: 6+1 = 7 vertices
        assert_eq!(output_vertices.len(), 7);
        // Index stream: 6+3 = 9 indices
        assert_eq!(output_indices.len(), 9);
    }

    #[test]
    fn work_items_cover_exact_fixed_meshlet_spans() {
        let mut output = Vec::new();

        append_work_items(3, 0, &mut output);
        append_work_items(5, 1, &mut output);
        append_work_items(7, 64, &mut output);
        append_work_items(11, 65, &mut output);
        append_work_items(13, 130, &mut output);

        assert_eq!(
            output
                .iter()
                .map(|item| (item.object_index, item.local_meshlet_base))
                .collect::<Vec<_>>(),
            [
                (5, 0),
                (7, 0),
                (11, 0),
                (11, 64),
                (13, 0),
                (13, 64),
                (13, 128),
            ]
        );
    }

    #[test]
    fn work_items_cover_only_leaf_count_not_total_meshlets() {
        // Multi-LOD mesh: 100 leaves + 50 coarser = 150 total. Work items must
        // only span the 100 leaves so coarser clusters are not cull entry points.
        let mut output = Vec::new();
        let leaf_count = 100u32;
        append_work_items(0, leaf_count, &mut output);
        let covered: u32 = output
            .iter()
            .map(|item| {
                let remaining = leaf_count.saturating_sub(item.local_meshlet_base);
                remaining.min(libhelio::VG_CULL_MESHLETS_PER_WORK_ITEM)
            })
            .sum();
        assert_eq!(covered, leaf_count);
        assert!(
            output
                .iter()
                .all(|item| item.local_meshlet_base < leaf_count)
        );
        // Must NOT generate bases into the coarser range (100..150).
        assert!(output.iter().all(|item| item.local_meshlet_base < 100));
    }

    #[test]
    fn parent_cluster_id_global_fixup_maps_into_valid_range() {
        // Two meshes each with a child→parent pair (record-local parents 1 and 0).
        let mut child_a = meshlet(0, 0);
        child_a.parent_cluster_id = 1; // local parent index within record
        let mut parent_a = meshlet(3, 3);
        parent_a.parent_cluster_id = u32::MAX;

        let mut child_b = meshlet(0, 0);
        child_b.parent_cluster_id = 1;
        let mut parent_b = meshlet(3, 3);
        parent_b.parent_cluster_id = u32::MAX;

        let first = VirtualMeshId(1);
        let second = VirtualMeshId(2);
        let verts = vec![
            dummy_vertex(),
            dummy_vertex(),
            dummy_vertex(),
            dummy_vertex(),
            dummy_vertex(),
            dummy_vertex(),
        ];
        let idxs = vec![0u16, 1, 2, 3, 4, 5];
        let records = HashMap::from([
            (
                first,
                record(vec![child_a, parent_a], verts.clone(), idxs.clone()),
            ),
            (second, record(vec![child_b, parent_b], verts, idxs)),
        ]);

        let mut out_m = Vec::new();
        let mut out_v = Vec::new();
        let mut out_i = Vec::new();
        let bases = append_unique_meshlets(
            [first, second],
            &records,
            &mut out_m,
            &mut out_v,
            &mut out_i,
        );

        assert_eq!(out_m.len(), 4);
        let base_a = bases[&first];
        let base_b = bases[&second];
        assert_eq!(base_a, 0);
        assert_eq!(base_b, 2);

        // Child of mesh A: local parent 1 → global 0+1 = 1
        assert_eq!(out_m[0].parent_cluster_id, 1);
        assert_eq!(out_m[1].parent_cluster_id, u32::MAX);
        // Child of mesh B: local parent 1 → global 2+1 = 3
        assert_eq!(out_m[2].parent_cluster_id, 3);
        assert_eq!(out_m[3].parent_cluster_id, u32::MAX);

        for m in &out_m {
            if m.parent_cluster_id != u32::MAX {
                assert!(
                    (m.parent_cluster_id as usize) < out_m.len(),
                    "parent {} out of global range 0..{}",
                    m.parent_cluster_id,
                    out_m.len()
                );
            }
        }
    }
}
