//! Virtual geometry buffer rebuild and frame data packaging.
//!
//! Handles CPU-side buffer reconstruction and provides frame data views for the renderer.

use libhelio::VgFrameData;

impl super::super::Scene {
    /// Returns a view into the CPU-side meshlet/instance buffers for VG rendering.
    ///
    /// Returns `None` if there are no virtual geometry objects in the scene.
    ///
    /// # Returns
    /// - `Some(VgFrameData)` if there are VG objects
    /// - `None` if the scene has no VG objects
    ///
    /// # Frame Data Contents
    ///
    /// The returned [`VgFrameData`] contains:
    /// - `meshlets`: Flat array of all meshlet descriptors (for all instances)
    /// - `instances`: Array of instance data (transforms, bounds, material IDs)
    /// - `meshlet_count`: Total number of meshlets
    /// - `instance_count`: Total number of instances
    /// - `buffer_version`: Monotonically increasing version number
    ///
    /// # Buffer Version
    ///
    /// The `buffer_version` increments whenever VG buffers are rebuilt. The renderer
    /// uses this to detect when GPU buffers need to be re-uploaded.
    ///
    /// # Performance
    /// - CPU cost: O(1) - returns references to existing buffers
    /// - GPU cost: None (this is a read-only view)
    /// - Memory: No allocations
    ///
    /// # Example
    /// ```ignore
    /// if let Some(vg_data) = scene.vg_frame_data() {
    ///     // Upload to GPU if version changed
    ///     if vg_data.buffer_version != last_version {
    ///         upload_vg_buffers(vg_data);
    ///         last_version = vg_data.buffer_version;
    ///     }
    /// }
    /// ```
    pub fn vg_frame_data(&self) -> Option<VgFrameData<'_>> {
        if self.vg_cpu_meshlets.is_empty() {
            return None;
        }
        Some(VgFrameData {
            meshlets: bytemuck::cast_slice(&self.vg_cpu_meshlets),
            instances: bytemuck::cast_slice(&self.vg_cpu_instances),
            meshlet_count: self.vg_cpu_meshlets.len() as u32,
            instance_count: self.vg_cpu_instances.len() as u32,
            buffer_version: self.vg_buffer_version,
        })
    }

    /// Rebuild CPU-side VG buffers from the current VG object set.
    ///
    /// This assigns each VG object a contiguous `instance_index` slot and patches the
    /// `GpuMeshletEntry::instance_index` field in all meshlets owned by that object.
    ///
    /// Called from `flush()` when `vg_objects_dirty` is true.
    ///
    /// # Algorithm
    ///
    /// 1. Clear CPU-side meshlet and instance buffers
    /// 2. For each VG object:
    ///    - Assign sequential instance_index
    ///    - Push instance data to instance buffer
    ///    - Clone all meshlets from the object's virtual mesh
    ///    - Patch each meshlet's instance_index to reference this instance
    ///    - Append meshlets to flat meshlet buffer
    /// 3. Increment buffer_version to signal re-upload needed
    ///
    /// # Performance
    /// - CPU cost: O(N) over VG objects + O(M) over total meshlets
    /// - GPU cost: None (GPU upload happens in renderer)
    /// - Memory: Rebuilds two vectors (meshlets and instances)
    ///
    /// # Buffer Layout
    ///
    /// **Meshlet buffer (flat array):**
    /// ```text
    /// [obj0_meshlet0, obj0_meshlet1, ..., obj1_meshlet0, obj1_meshlet1, ...]
    /// ```
    ///
    /// **Instance buffer:**
    /// ```text
    /// [obj0_instance, obj1_instance, obj2_instance, ...]
    /// ```
    ///
    /// Each meshlet references its owning instance via `instance_index`.
    ///
    /// # Example Output
    ///
    /// For 2 VG objects:
    /// - Object 0: 100 meshlets
    /// - Object 1: 150 meshlets
    ///
    /// Result:
    /// - `vg_cpu_instances.len() = 2`
    /// - `vg_cpu_meshlets.len() = 250`
    /// - Object 0's meshlets have `instance_index = 0`
    /// - Object 1's meshlets have `instance_index = 1`
    pub(in crate::scene) fn rebuild_vg_buffers(&mut self) {
        let instance_count = self.vg_objects.dense_len();
        self.vg_cpu_instances.clear();
        self.vg_cpu_meshlets.clear();
        self.vg_cpu_instances.reserve(instance_count);

        for i in 0..instance_count {
            let Some(obj) = self.vg_objects.get_dense(i) else {
                continue;
            };
            let instance_index = self.vg_cpu_instances.len() as u32;
            self.vg_cpu_instances.push(obj.instance);

            let Some(mesh_record) = self.vg_meshes.get(&obj.virtual_mesh) else {
                continue;
            };
            for mut meshlet in mesh_record.meshlets.iter().copied() {
                meshlet.instance_index = instance_index;
                self.vg_cpu_meshlets.push(meshlet);
            }
        }

        self.vg_buffer_version = self.vg_buffer_version.wrapping_add(1);
        log::debug!(
            "rebuild_vg_buffers: {} VG objects → {} meshlets",
            instance_count,
            self.vg_cpu_meshlets.len()
        );
    }
}

