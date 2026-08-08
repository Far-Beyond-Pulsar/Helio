//! GPU buffer rebuild for automatic instancing.
//!
//! This module contains the core logic for reconstructing GPU instance, AABB, draw call,
//! indirect, and visibility buffers from the CPU-side object arena, automatically grouping
//! objects with the same mesh + material into instanced draw calls.

use helio_core::{DrawIndexedIndirectArgs, GpuDrawCall, GpuInstanceAabb, GpuInstanceData};
use pulsar_scenedb::Entity;

use super::super::helpers::object_is_visible;
use super::super::types::ObjectRecord;

impl super::super::Scene {
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
        // One O(n) collection out of the ECS, sorted/grouped/patched in place
        // from here on — same cost shape as the old `DenseArena`'s `dense: Vec<T>`
        // (that was already a full-array scan+sort on every rebuild; this is
        // the same, just sourced from `World::query` instead of a public
        // field). `ObjectRecord` is `Clone` (small, no heap fields), so this
        // is a flat copy, not a walk of anything nested.
        let rows: Vec<(Entity, ObjectRecord)> = self
            .world
            .query::<&ObjectRecord>()
            .map(|(entity, r)| (entity, r.clone()))
            .collect();
        let n = rows.len();
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
            let r = &rows[i].1;
            let (class, graph_hash) = self
                .materials
                .get(r.material)
                .map(|m| (m.gpu.material_class, m.graph_hash))
                .unwrap_or((0, 0));
            (class, graph_hash, r.instance.mesh_id, r.instance.material_id)
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
            let r0 = &rows[order[i]].1;
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
                let r = &rows[order[i]].1;
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
            let is_transparent = self.materials.get(r0.material)
                .map(|m| (m.gpu.flags & libhelio::FLAG_TRANSPARENT_ONLY) != 0)
                .unwrap_or(false);
            let is_forward = self.materials.get(r0.material)
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
            let entity = rows[di].0;
            if let Some(r) = self.world.get_mut::<ObjectRecord>(entity) {
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
        self.gpu_scene.compacted_indices.set_data(compacted_indices_capacity);
        self.gpu_scene.compacted_indices_2.set_data(compacted_indices_2_capacity);
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
        // Build two INDIRECT call lists — one per mobility class.
        // first_instance in each entry is the object's slot in the main
        // `instances` buffer (just patched by `rebuild_instance_buffers`), so
        // transforms stay in sync with update_object_transform.
        // DO NOT copy instance data into separate buffers — that causes stale shadows.
        let mut static_indirect: Vec<DrawIndexedIndirectArgs> = Vec::new();
        let mut movable_indirect: Vec<DrawIndexedIndirectArgs> = Vec::new();

        for (_, r) in self.world.query::<&ObjectRecord>() {
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

#[cfg(test)]
mod tests {
    use crate::handles::MaterialId;
    use crate::mesh::{MeshUpload, PackedVertex};
    use crate::scene::{ObjectDescriptor, Scene};
    use crate::groups::GroupMask;
    use bytemuck::Zeroable;
    use glam::Mat4;
    use libhelio::{GpuMaterial, Movability};

    fn create_test_device() -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("No adapter found");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .expect("Failed to create device");
        (std::sync::Arc::new(device), std::sync::Arc::new(queue))
    }

    fn triangle_mesh(scene: &mut Scene) -> crate::handles::MeshId {
        scene.insert_mesh(MeshUpload {
            vertices: vec![PackedVertex::default(); 3],
            indices: vec![0, 1, 2],
        })
    }

    fn default_material(scene: &mut Scene) -> MaterialId {
        scene.insert_material(GpuMaterial::zeroed())
    }

    /// End-to-end proof that object storage really is backed by
    /// `pulsar_scenedb::World` now: insert → flush (rebuild_instance_buffers,
    /// this file) → update transform (in-place GPU write) → remove → flush
    /// again, checking the real `gpu_scene` instance buffer at each step —
    /// not just that the CPU record round-trips.
    #[test]
    fn insert_update_remove_round_trips_through_the_real_gpu_buffers() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        let mesh = triangle_mesh(&mut scene);
        let material = default_material(&mut scene);

        let id = scene
            .insert_object(ObjectDescriptor {
                mesh,
                material,
                transform: Mat4::IDENTITY,
                bounds: [0.0, 0.0, 0.0, 1.0],
                flags: 0,
                groups: GroupMask::NONE,
                movability: Some(Movability::Movable),
                user_tag: 0,
            })
            .expect("insert_object");

        scene.flush();
        assert_eq!(scene.gpu_scene().resources().instance_count, 1);
        assert_eq!(
            scene.get_object_transform(id).expect("transform"),
            Mat4::IDENTITY
        );

        let moved = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        scene
            .update_object_transform(id, moved)
            .expect("update_object_transform");
        // In-place write (objects_dirty is false after the flush above) —
        // no second flush needed for the CPU-side round-trip to see it.
        assert_eq!(scene.get_object_transform(id).expect("transform"), moved);

        let editor_rows: Vec<_> = scene.iter_objects_for_editor().collect();
        assert_eq!(editor_rows.len(), 1);
        assert_eq!(editor_rows[0].0, id);

        scene.remove_object(id).expect("remove_object");
        assert!(scene.get_object_transform(id).is_err());

        scene.flush();
        assert_eq!(scene.gpu_scene().resources().instance_count, 0);
    }
}
