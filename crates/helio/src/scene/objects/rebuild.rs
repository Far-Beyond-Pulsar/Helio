//! GPU buffer rebuild for automatic instancing.
//!
//! This module contains the core logic for reconstructing GPU instance, AABB, draw call,
//! indirect, and visibility buffers from the CPU-side object arena, automatically grouping
//! objects with the same mesh + material into instanced draw calls.

use helio_core::{DrawIndexedIndirectArgs, GpuDrawCall, GpuInstanceAabb, GpuInstanceData};

use super::super::helpers::object_is_visible;
use super::super::types::StaticMeshRenderInput;

impl super::super::Scene {
    /// Rebuild the regular-object GPU buffers from a transient SceneDB-owned
    /// snapshot. No static-mesh entity data is retained by Helio between calls.
    pub fn rebuild_static_mesh_instances(&mut self, inputs: &[StaticMeshRenderInput]) {
        let mut order: Vec<usize> = (0..inputs.len()).collect();
        order.sort_by_key(|&i| {
            let input = &inputs[i];
            let (class, graph_hash) = self
                .materials
                .get(input.material)
                .map(|m| (m.gpu.material_class, m.graph_hash))
                .unwrap_or((0, 0));
            (
                class,
                graph_hash,
                input.mesh_key,
                input.instance.material_id,
            )
        });

        let mut instances = Vec::with_capacity(inputs.len());
        let mut aabbs = Vec::with_capacity(inputs.len());
        let mut draw_calls = Vec::new();
        let mut indirect = Vec::new();
        let mut visibility = Vec::with_capacity(inputs.len());
        let mut group_keys = Vec::new();
        let mut group_transparent = Vec::new();
        let mut group_forward = Vec::new();

        let mut i = 0;
        while i < order.len() {
            let first = &inputs[order[i]];
            let material = self.materials.get(first.material);
            let class = material.map(|m| m.gpu.material_class).unwrap_or(0);
            let graph_hash = material.map(|m| m.graph_hash).unwrap_or(0);
            let key = (first.mesh_key, first.instance.material_id);
            let group_start = instances.len() as u32;
            let index_count = first.draw.index_count;
            let first_index = first.draw.first_index;
            let vertex_offset = first.draw.vertex_offset;

            while i < order.len() {
                let input = &inputs[order[i]];
                if (input.mesh_key, input.instance.material_id) != key {
                    break;
                }
                instances.push(input.instance);
                aabbs.push(input.aabb);
                visibility.push(if object_is_visible(input.groups, self.group_hidden) {
                    1
                } else {
                    0
                });
                i += 1;
            }

            let instance_count = instances.len() as u32 - group_start;
            draw_calls.push(GpuDrawCall {
                index_count,
                first_index,
                vertex_offset,
                first_instance: group_start,
                instance_count,
            });
            indirect.push(DrawIndexedIndirectArgs {
                index_count,
                instance_count,
                first_index,
                base_vertex: vertex_offset,
                first_instance: group_start,
            });
            group_keys.push((class, graph_hash));
            group_transparent.push(
                material
                    .map(|m| (m.gpu.flags & libhelio::FLAG_TRANSPARENT_ONLY) != 0)
                    .unwrap_or(false),
            );
            group_forward.push(
                material
                    .map(|m| (m.gpu.flags & libhelio::FLAG_FORWARD_SHADING) != 0)
                    .unwrap_or(false),
            );
        }

        let mut opaque_ranges = Vec::new();
        let mut transparent_ranges = Vec::new();
        let mut forward_ranges = Vec::new();
        let mut group_index = 0;
        while group_index < group_keys.len() {
            let key = group_keys[group_index];
            let start = group_index as u32;
            let first_group = group_index;
            while group_index < group_keys.len() && group_keys[group_index] == key {
                group_index += 1;
            }
            let range = (key.0, key.1, start, (group_index - first_group) as u32);
            if group_forward[first_group] {
                forward_ranges.push(range);
            } else if group_transparent[first_group] {
                transparent_ranges.push(range);
            } else {
                opaque_ranges.push(range);
            }
        }

        self.gpu_scene.material_class_ranges = opaque_ranges;
        self.gpu_scene.transparent_material_class_ranges = transparent_ranges;
        self.gpu_scene.forward_material_class_ranges = forward_ranges;
        self.gpu_scene.instances.set_data(instances);
        self.gpu_scene.aabbs.set_data(aabbs);
        self.gpu_scene.draw_calls.set_data(draw_calls.clone());
        self.gpu_scene.indirect.set_data(indirect);
        self.gpu_scene.visibility.set_data(visibility);
        self.gpu_scene
            .compacted_indices
            .set_data(vec![0; inputs.len()]);
        self.gpu_scene
            .compacted_indices_2
            .set_data(vec![0; inputs.len()]);

        let mut static_indirect = Vec::new();
        let mut movable_indirect = Vec::new();
        let mut slot = 0u32;
        for input in &order.iter().map(|&i| &inputs[i]).collect::<Vec<_>>() {
            let entry = DrawIndexedIndirectArgs {
                index_count: input.draw.index_count,
                instance_count: 1,
                first_index: input.draw.first_index,
                base_vertex: input.draw.vertex_offset,
                first_instance: slot,
            };
            if input.movability.can_move() {
                movable_indirect.push(entry);
            } else {
                static_indirect.push(entry);
            }
            slot += 1;
        }
        self.gpu_scene.shadow_static_draw_count = static_indirect.len() as u32;
        self.gpu_scene.shadow_movable_draw_count = movable_indirect.len() as u32;
        self.gpu_scene
            .shadow_static_indirect
            .set_data(static_indirect);
        self.gpu_scene
            .shadow_movable_indirect
            .set_data(movable_indirect);
        self.gpu_scene.static_objects_generation =
            self.gpu_scene.static_objects_generation.wrapping_add(1);
        // `flush()` must not immediately replace this transient frame with
        // the legacy persistent-object arena's contents.
        self.objects_dirty = false;
    }

    /// Rebuilds GPU buffers with automatic instancing.
    ///
    /// Sorts objects by (mesh_id, material_id) and groups consecutive objects with
    /// the same key into instanced draw calls. This reduces draw call count and improves
    /// GPU cache hit rates — all automatically, no user input required.
    ///
    /// Called from `flush()` when `objects_dirty` is true.
    ///
    /// # Algorithm
    ///
    /// 1. Build sort order: indices [0..N) sorted by (mesh_id, material_id)
    /// 2. Iterate in sorted order, grouping by (mesh_id, material_id):
    ///    - Allocate contiguous GPU slots for each group
    ///    - Create one draw call per group with `instance_count = group_size`
    /// 3. Update ObjectRecords with new GPU slots
    /// 4. Build visibility buffer in sorted order
    /// 5. Upload all buffers to GPU
    ///
    /// # Performance
    ///
    /// - CPU cost: O(N log N) sort + O(N) buffer rebuild
    /// - GPU cost: O(N) buffer uploads (5 buffers)
    /// - Memory: O(N) temporary vectors
    ///
    /// # Draw Calls
    ///
    /// Generates D draw calls (where D = number of unique (mesh, material) pairs).
    /// For a scene with:
    /// - 10,000 objects
    /// - 50 unique meshes
    /// - 100 unique materials
    ///
    /// This could reduce draw calls from 10,000 to ~500, depending on mesh/material distribution.
    ///
    /// # GPU Cache Coherency
    ///
    /// By sorting objects, we ensure that:
    /// - Objects using the same mesh are drawn consecutively (vertex cache hits)
    /// - Objects using the same material are drawn consecutively (texture cache hits)
    /// - GPU can efficiently batch vertex fetches and texture samples
    pub(in crate::scene) fn rebuild_instance_buffers(&mut self) {
        let n = self.objects.dense_len();
        if n == 0 {
            self.gpu_scene.instances.set_data(Vec::new());
            self.gpu_scene.aabbs.set_data(Vec::new());
            self.gpu_scene.draw_calls.set_data(Vec::new());
            self.gpu_scene.indirect.set_data(Vec::new());
            self.gpu_scene.visibility.set_data(Vec::new());
            self.gpu_scene.compacted_indices.set_data(Vec::new());
            self.gpu_scene.compacted_indices_2.set_data(Vec::new());
            self.gpu_scene.material_class_ranges.clear();
            self.gpu_scene.transparent_material_class_ranges.clear();
            self.gpu_scene.forward_material_class_ranges.clear();
            return;
        }

        // Build a sort order over the dense array indices, grouped by
        // (material_class, graph_hash, mesh_id, material_id) so that contiguous
        // draw groups share both class and graph_hash, letting each range use a
        // single PSO.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| {
            let r = self.objects.get_dense(i).unwrap();
            let (class, graph_hash) = self
                .materials
                .get(r.material)
                .map(|m| (m.gpu.material_class, m.graph_hash))
                .unwrap_or((0, 0));
            (
                class,
                graph_hash,
                r.instance.mesh_id,
                r.instance.material_id,
            )
        });

        let mut instances: Vec<GpuInstanceData> = Vec::with_capacity(n);
        let mut aabbs: Vec<GpuInstanceAabb> = Vec::with_capacity(n);
        let mut draw_calls: Vec<GpuDrawCall> = Vec::new();
        let mut indirect: Vec<DrawIndexedIndirectArgs> = Vec::new();
        let mut visibility: Vec<u32> = Vec::with_capacity(n);
        // Track the new GPU slot assigned to each dense-array entry.
        let mut gpu_slots: Vec<u32> = vec![0u32; n];
        // Track the (material_class, graph_hash) of each draw group for range building.
        // group_transparent tracks whether the group's material has FLAG_TRANSPARENT_ONLY.
        // group_forward tracks whether the group's material has FLAG_FORWARD_SHADING.
        let mut group_keys: Vec<(u32, u64)> = Vec::new();
        let mut group_transparent: Vec<bool> = Vec::new();
        let mut group_forward: Vec<bool> = Vec::new();

        let group_hidden = self.group_hidden;

        let mut i = 0;
        while i < order.len() {
            let r0 = self.objects.get_dense(order[i]).unwrap();
            let (class, graph_hash) = self
                .materials
                .get(r0.material)
                .map(|m| (m.gpu.material_class, m.graph_hash))
                .unwrap_or((0, 0));
            let key = (r0.instance.mesh_id, r0.instance.material_id);
            let group_start = instances.len() as u32;
            let (index_count, first_index, vertex_offset) = (
                r0.draw.index_count,
                r0.draw.first_index,
                r0.draw.vertex_offset,
            );

            // Consume all objects in this group.
            while i < order.len() {
                let r = self.objects.get_dense(order[i]).unwrap();
                if (r.instance.mesh_id, r.instance.material_id) != key {
                    break;
                }
                gpu_slots[order[i]] = instances.len() as u32;
                instances.push(r.instance);
                aabbs.push(r.aabb);
                visibility.push(if object_is_visible(r.groups, group_hidden) {
                    1u32
                } else {
                    0u32
                });
                i += 1;
            }

            let instance_count = instances.len() as u32 - group_start;
            draw_calls.push(GpuDrawCall {
                index_count,
                first_index,
                vertex_offset,
                first_instance: group_start,
                instance_count,
            });
            indirect.push(DrawIndexedIndirectArgs {
                index_count,
                instance_count,
                first_index,
                base_vertex: vertex_offset,
                first_instance: group_start,
            });
            group_keys.push((class, graph_hash));
            // Determine transparency and forward-shading from the material flags
            let is_transparent = self
                .materials
                .get(r0.material)
                .map(|m| (m.gpu.flags & libhelio::FLAG_TRANSPARENT_ONLY) != 0)
                .unwrap_or(false);
            let is_forward = self
                .materials
                .get(r0.material)
                .map(|m| (m.gpu.flags & libhelio::FLAG_FORWARD_SHADING) != 0)
                .unwrap_or(false);
            group_transparent.push(is_transparent);
            group_forward.push(is_forward);
        }

        // Build material class ranges from consecutive draw groups with the same
        // (class, graph_hash) so each range can use a single PSO.
        // Split into opaque, transparent, and forward-shaded ranges.
        let mut opaque_ranges: Vec<(u32, u64, u32, u32)> = Vec::new();
        let mut transparent_ranges: Vec<(u32, u64, u32, u32)> = Vec::new();
        let mut forward_ranges: Vec<(u32, u64, u32, u32)> = Vec::new();
        let mut gi = 0;
        while gi < group_keys.len() {
            let (class, graph_hash) = group_keys[gi];
            let start = gi as u32;
            let mut count = 0u32;
            while gi < group_keys.len() && group_keys[gi] == (class, graph_hash) {
                count += 1;
                gi += 1;
            }
            // Check if any group in this range is forward-shaded (takes priority)
            let is_forward = group_forward[gi - count as usize];
            if is_forward {
                forward_ranges.push((class, graph_hash, start, count));
            } else {
                // Check if any group in this range is transparent
                let is_transparent = group_transparent[gi - count as usize];
                if is_transparent {
                    transparent_ranges.push((class, graph_hash, start, count));
                } else {
                    opaque_ranges.push((class, graph_hash, start, count));
                }
            }
        }
        log::info!("[Scene] rebuilt material_class_ranges: opaque={:?} transparent={:?} forward={:?} ({} objects, {} draw groups)",
            opaque_ranges, transparent_ranges, forward_ranges, n, draw_calls.len());
        self.gpu_scene.material_class_ranges = opaque_ranges;
        self.gpu_scene.transparent_material_class_ranges = transparent_ranges;
        self.gpu_scene.forward_material_class_ranges = forward_ranges;

        // Patch each ObjectRecord with its new GPU slot so that in-frame
        // `update_object_transform` / `update_object_bounds` can update in-place.
        for (di, &slot) in gpu_slots.iter().enumerate() {
            if let Some(r) = self.objects.get_dense_mut(di) {
                r.gpu_slot = slot;
                r.draw.first_instance = slot;
            }
        }

        log::debug!(
            "rebuild_instance_buffers: {} objects → {} draw groups ({} instanced)",
            n,
            draw_calls.len(),
            n - draw_calls.len()
        );

        // Sized only to keep the GPU buffer's capacity in step with `instances` —
        // content is fully overwritten by IndirectDispatchPass every frame, so the
        // zeros here are never read.
        let compacted_indices_capacity = vec![0u32; instances.len()];
        let compacted_indices_2_capacity = vec![0u32; instances.len()];

        self.gpu_scene.instances.set_data(instances);
        self.gpu_scene.aabbs.set_data(aabbs);
        self.gpu_scene.draw_calls.set_data(draw_calls);
        self.gpu_scene.indirect.set_data(indirect);
        self.gpu_scene.visibility.set_data(visibility);
        self.gpu_scene
            .compacted_indices
            .set_data(compacted_indices_capacity);
        self.gpu_scene
            .compacted_indices_2
            .set_data(compacted_indices_2_capacity);
        self.rebuild_shadow_partition_buffers();
    }

    /// Builds the shadow-specific partitioned instance + indirect buffers.
    ///
    /// Separates objects by movability into two groups:
    /// - Static/Stationary → `shadow_static_instances` + `shadow_static_indirect`
    /// - Movable           → `shadow_movable_instances` + `shadow_movable_indirect`
    ///
    /// Each group has its own 0-based instance indices so the shadow passes can
    /// render them independently with separate atlases (Unreal-style static+dynamic split).
    ///
    /// When `static_objects_dirty` is `true`, `static_objects_generation` is incremented
    /// to signal the ShadowPass to re-render the static shadow atlas.
    pub(in crate::scene) fn rebuild_shadow_partition_buffers(&mut self) {
        let n = self.objects.dense_len();

        // Build two INDIRECT call lists — one per mobility class.
        // first_instance in each entry is the object's dense_index into the main
        // `instances` buffer, so transforms stay in sync with update_object_transform.
        // DO NOT copy instance data into separate buffers — that causes stale shadows.
        let mut static_indirect: Vec<DrawIndexedIndirectArgs> = Vec::new();
        let mut movable_indirect: Vec<DrawIndexedIndirectArgs> = Vec::new();

        for i in 0..n {
            let r = self.objects.get_dense(i).unwrap();
            // Use the object's actual first_instance (its slot in the main instances buffer).
            let entry = DrawIndexedIndirectArgs {
                index_count: r.draw.index_count,
                instance_count: 1,
                first_index: r.draw.first_index,
                base_vertex: r.draw.vertex_offset,
                first_instance: r.draw.first_instance,
            };
            if r.movability.can_move() {
                movable_indirect.push(entry);
            } else {
                static_indirect.push(entry);
            }
        }

        let static_draw_count = static_indirect.len() as u32;
        let movable_draw_count = movable_indirect.len() as u32;

        // Bump static generation if the static set was modified
        if self.static_objects_dirty {
            self.gpu_scene.static_objects_generation += 1;
            self.static_objects_dirty = false;
        }

        self.gpu_scene.shadow_static_draw_count = static_draw_count;
        self.gpu_scene.shadow_movable_draw_count = movable_draw_count;

        self.gpu_scene
            .shadow_static_indirect
            .set_data(static_indirect);
        self.gpu_scene
            .shadow_movable_indirect
            .set_data(movable_indirect);

        log::debug!(
            "rebuild_shadow_partition_buffers: {} static + {} movable shadow draws",
            static_draw_count,
            movable_draw_count,
        );
    }
}
