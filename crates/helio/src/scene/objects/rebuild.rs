//! GPU buffer rebuild operations for both persistent and optimized modes.
//!
//! This module contains the core logic for reconstructing GPU instance, AABB, draw call,
//! indirect, and visibility buffers from the CPU-side object arena.

use helio_v3::{DrawIndexedIndirectArgs, GpuDrawCall, GpuInstanceAabb, GpuInstanceData};

use super::super::helpers::object_is_visible;

impl super::super::Scene {
    /// Optimizes the scene layout for cache coherency and GPU instancing.
    ///
    /// Sorts objects by (mesh, material) and groups consecutive objects with
    /// the same key into instanced draw calls. This significantly improves
    /// rendering performance but disables O(1) add/remove operations until
    /// the next topology change.
    ///
    /// # When to Call
    ///
    /// Call this after bulk object insertion (e.g., level load, loading screen)
    /// when you want maximum rendering performance.
    ///
    /// # Performance
    ///
    /// - **Cost:** O(N log N) sort + O(N) buffer rebuild (one-time)
    /// - **Benefit:** Reduced draw calls (instanced batching), better GPU cache utilization
    /// - **Trade-off:** Next add/remove will revert to persistent mode
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Load level objects
    /// for object_desc in level.objects {
    ///     scene.insert_object(object_desc)?;
    /// }
    ///
    /// // Optimize layout once before gameplay
    /// scene.optimize_scene_layout();
    ///
    /// // Now render loop benefits from optimal batching
    /// loop {
    ///     scene.render(...);
    /// }
    /// ```
    pub fn optimize_scene_layout(&mut self) {
        if self.objects.dense_len() == 0 {
            return;
        }

        self.rebuild_instance_buffers_optimized();
        self.objects_layout_optimized = true;
        self.objects_dirty = false;

        log::info!(
            "Scene layout optimized: {} objects",
            self.objects.dense_len()
        );
    }

    /// Persistent slot path: rebuilds GPU buffers without sorting.
    ///
    /// Each object gets one draw call (instance_count = 1).
    /// GPU slot = dense_index for O(1) add/remove operations.
    ///
    /// Called from `flush()` when `objects_dirty` is true and `objects_layout_optimized` is false.
    ///
    /// # Algorithm
    ///
    /// 1. Allocate vectors for N objects (where N = dense_len)
    /// 2. Linear iteration: for each dense index i:
    ///    - Push instance data at slot i
    ///    - Push AABB at slot i
    ///    - Create draw call: `first_instance = i, instance_count = 1`
    ///    - Create indirect args with same parameters
    /// 3. Build visibility buffer: 1 if visible, 0 if hidden
    /// 4. Update ObjectRecords with GPU slot = dense_index
    /// 5. Upload all buffers to GPU
    ///
    /// # Performance
    ///
    /// - CPU cost: O(N) linear iteration + O(N) allocations
    /// - GPU cost: O(N) buffer uploads (5 buffers: instances, aabbs, draws, indirect, visibility)
    /// - Memory: O(N) temporary vectors
    ///
    /// # Draw Calls
    ///
    /// Generates N draw calls (one per object). This is acceptable because:
    /// - GPU-driven indirect rendering handles many small draws efficiently
    /// - Frustum culling happens on GPU (only visible draws are executed)
    /// - Users can call `optimize_scene_layout()` for batching when needed
    pub(in crate::scene) fn rebuild_instance_buffers_persistent(&mut self) {
        let n = self.objects.dense_len();
        if n == 0 {
            self.gpu_scene.instances.set_data(Vec::new());
            self.gpu_scene.aabbs.set_data(Vec::new());
            self.gpu_scene.draw_calls.set_data(Vec::new());
            self.gpu_scene.indirect.set_data(Vec::new());
            self.gpu_scene.visibility.set_data(Vec::new());
            return;
        }

        let mut instances = Vec::with_capacity(n);
        let mut aabbs = Vec::with_capacity(n);
        let mut draw_calls = Vec::with_capacity(n);
        let mut indirect = Vec::with_capacity(n);

        // Linear iteration: each object gets slot = dense_index
        for i in 0..n {
            let r = self.objects.get_dense(i).unwrap();
            instances.push(r.instance);
            aabbs.push(r.aabb);

            // One draw call per object
            draw_calls.push(GpuDrawCall {
                index_count: r.draw.index_count,
                first_index: r.draw.first_index,
                vertex_offset: r.draw.vertex_offset,
                first_instance: i as u32,
                instance_count: 1,
            });

            indirect.push(DrawIndexedIndirectArgs {
                index_count: r.draw.index_count,
                instance_count: 1,
                first_index: r.draw.first_index,
                base_vertex: r.draw.vertex_offset,
                first_instance: i as u32,
            });
        }

        // Build visibility
        let group_hidden = self.group_hidden;
        let visibility: Vec<u32> = (0..n)
            .map(|i| {
                let r = self.objects.get_dense(i).unwrap();
                if object_is_visible(r.groups, group_hidden) {
                    1u32
                } else {
                    0u32
                }
            })
            .collect();

        // Update ObjectRecords with GPU slots
        for i in 0..n {
            if let Some(r) = self.objects.get_dense_mut(i) {
                r.gpu_slot = i as u32;
                r.draw.first_instance = i as u32;
            }
        }

        self.gpu_scene.instances.set_data(instances);
        self.gpu_scene.aabbs.set_data(aabbs);
        self.gpu_scene.draw_calls.set_data(draw_calls);
        self.gpu_scene.indirect.set_data(indirect);
        self.gpu_scene.visibility.set_data(visibility);

        log::debug!(
            "rebuild_instance_buffers_persistent: {} objects → {} draws",
            n,
            n
        );
    }

    /// Optimized path: sorts objects by (mesh_id, material_id) for cache coherency.
    ///
    /// Groups consecutive objects with the same (mesh, material) into instanced draw calls
    /// with `instance_count > 1`. This reduces draw call count and improves GPU cache hit rates.
    ///
    /// Called from `flush()` when `objects_dirty` is true and `objects_layout_optimized` is true,
    /// or when explicitly invoked via `optimize_scene_layout()`.
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
    /// This could reduce draw calls from 10,000 (persistent) to ~500 (optimized),
    /// depending on mesh/material distribution.
    ///
    /// # GPU Cache Coherency
    ///
    /// By sorting objects, we ensure that:
    /// - Objects using the same mesh are drawn consecutively (vertex cache hits)
    /// - Objects using the same material are drawn consecutively (texture cache hits)
    /// - GPU can efficiently batch vertex fetches and texture samples
    pub(in crate::scene) fn rebuild_instance_buffers_optimized(&mut self) {
        let n = self.objects.dense_len();
        if n == 0 {
            self.gpu_scene.instances.set_data(Vec::new());
            self.gpu_scene.aabbs.set_data(Vec::new());
            self.gpu_scene.draw_calls.set_data(Vec::new());
            self.gpu_scene.indirect.set_data(Vec::new());
            self.gpu_scene.visibility.set_data(Vec::new());
            return;
        }

        // Build a sort order over the dense array indices, grouped by (mesh_id, material_id).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| {
            let r = self.objects.get_dense(i).unwrap();
            (r.instance.mesh_id, r.instance.material_id)
        });

        let mut instances: Vec<GpuInstanceData> = Vec::with_capacity(n);
        let mut aabbs: Vec<GpuInstanceAabb> = Vec::with_capacity(n);
        let mut draw_calls: Vec<GpuDrawCall> = Vec::new();
        let mut indirect: Vec<DrawIndexedIndirectArgs> = Vec::new();
        // Track the new GPU slot assigned to each dense-array entry.
        let mut gpu_slots: Vec<u32> = vec![0u32; n];

        let mut i = 0;
        while i < order.len() {
            let r0 = self.objects.get_dense(order[i]).unwrap();
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
        }

        // Patch each ObjectRecord with its new GPU slot so that in-frame
        // `update_object_transform` / `update_object_bounds` can update in-place.
        for (di, &slot) in gpu_slots.iter().enumerate() {
            if let Some(r) = self.objects.get_dense_mut(di) {
                r.gpu_slot = slot;
                r.draw.first_instance = slot;
            }
        }

        log::debug!(
            "rebuild_instance_buffers_optimized: {} objects → {} draw groups",
            n,
            draw_calls.len()
        );

        // Build visibility buffer: 0 = hidden (any group is hidden), 1 = visible.
        let group_hidden = self.group_hidden;
        let visibility: Vec<u32> = order
            .iter()
            .map(|&di| {
                let r = self.objects.get_dense(di).unwrap();
                if object_is_visible(r.groups, group_hidden) {
                    1u32
                } else {
                    0u32
                }
            })
            .collect();

        self.gpu_scene.instances.set_data(instances);
        self.gpu_scene.aabbs.set_data(aabbs);
        self.gpu_scene.draw_calls.set_data(draw_calls);
        self.gpu_scene.indirect.set_data(indirect);
        self.gpu_scene.visibility.set_data(visibility);
    }
}

