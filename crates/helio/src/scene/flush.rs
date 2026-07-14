//! Flush (GPU upload) orchestration for the scene.
//!
//! This module contains the [`flush`](Scene::flush) method which synchronises all
//! pending CPU-side changes to GPU buffers, and the dirty-range tracking associated
//! with flush operations.

use bytemuck::Zeroable;
use libhelio::GpuShadowMatrix;

use crate::scene::Scene;

/// FNV-1a hash over f32 bit patterns. Used for per-caster shadow dirty tracking.
/// Hashing bit patterns (not float values) ensures NaN and -0.0 are handled consistently.
#[inline]
fn fnv1a_f32s(vals: &[f32]) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut h = OFFSET;
    for &v in vals {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

#[inline]
fn quantize_f32s<const N: usize>(vals: [f32; N], quantum: f32) -> [f32; N] {
    vals.map(|value| (value / quantum).round() * quantum)
}

impl Scene {
    /// Flush pending changes to GPU buffers.
    ///
    /// This method:
    /// 1. Assigns shadow atlas base layers to shadow-casting lights
    /// 2. Flushes mesh pool uploads (vertex/index data)
    /// 3. Flushes material texture buffer uploads
    /// 4. Rebuilds object instance buffers if dirty (persistent or optimized mode)
    /// 5. Rebuilds virtual geometry buffers if dirty
    /// 6. Flushes all GPU scene buffers (instances, draws, indirect, visibility, etc.)
    ///
    /// # Performance
    ///
    /// **Clean state (no topology changes):**
    /// - CPU cost: O(1) - only shadow index assignment
    /// - GPU cost: O(lights) shadow index updates
    ///
    /// **Dirty state (topology changed):**
    /// - CPU cost: O(N) for persistent rebuild, O(N log N) for optimized rebuild
    /// - GPU cost: O(N) buffer uploads for all object data
    ///
    /// # Shadow Management
    ///
    /// Automatically assigns shadow atlas layers to shadow-casting lights:
    /// - Maximum 42 shadow casters (42 × 6 = 252 atlas layers)
    /// - 6 slots per light (point = 6 faces, directional = 4 cascades + 2 padding, spot = 1 + 5 padding)
    /// - Lights beyond the cap have shadows disabled automatically
    ///
    /// # When to Call
    ///
    /// Call `flush()` after all scene modifications for the frame, before rendering:
    /// ```ignore
    /// // Modify scene
    /// scene.insert_object(desc)?;
    /// scene.update_object_transform(id, transform)?;
    /// scene.hide_group(group_id);
    ///
    /// // Flush changes
    /// scene.flush();
    ///
    /// // Render
    /// renderer.render(&scene, target)?;
    /// ```
    pub fn flush(&mut self) {
        // Assign shadow atlas slots to the highest-importance shadow-casting lights.
        //
        // Problem with sequential assignment: the first N lights inserted always win the
        // 42-caster budget, regardless of how far away or how dim they are. A bright
        // close light inserted after slot 42 is full gets no shadow.
        //
        // Solution — two-phase importance selection:
        //   Phase 1: Score every shadow-requesting light by VIEW-INDEPENDENT importance:
        //              intensity × range²
        //            Directional lights always score ∞ (global, never culled).
        //            Sort descending → top 42 are the frame's active casters.
        //   Phase 2: Re-sort the WINNERS by their GPU buffer index (stable secondary key).
        //            Same lights that were in budget last frame keep the same atlas slots,
        //            preventing slot churn from minor score fluctuations. Only new entrants
        //            and exits cause slot reassignment (and thus dirty-gen bumps).
        //
        // IMPORTANT: Camera distance is intentionally NOT used in scoring. Using camera
        // distance causes the budget to reshuffle every frame the camera moves, which
        // triggers shadow atlas re-renders (expensive with many draw calls). The budget
        // should only change when lights are added/removed or their properties change.
        {
            const MAX_SHADOW_CASTERS: usize = 42;
            const FACES_PER_LIGHT: u32 = 6;
            let light_count = self.gpu_scene.lights.len();

            // Phase 1: score and select the top MAX_SHADOW_CASTERS.
            let mut scored: Vec<(f32, usize)> = Vec::with_capacity(light_count);
            for i in 0..light_count {
                let light = self.gpu_scene.lights.0.as_slice()[i];
                if light.shadow_index == u32::MAX {
                    continue; // user explicitly disabled shadows on this light
                }
                let score = if light.light_type == 0 {
                    // Directional: infinite range, always highest priority.
                    f32::MAX
                } else {
                    let range = light.position_range[3].max(0.001);
                    // intensity × range² — view-independent, stable across camera moves.
                    // Larger/brighter lights win the budget regardless of camera position.
                    light.color_intensity[3] * (range * range)
                };
                scored.push((score, i));
            }

            // Sort descending by importance to determine which lights win the budget.
            scored.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            let winner_count = scored.len().min(MAX_SHADOW_CASTERS);

            // Phase 2: re-sort winners by their buffer index (stable secondary key).
            // Lights that stay in budget from frame to frame retain the same atlas slot,
            // keeping per-caster dirty gens stable and avoiding spurious re-renders.
            scored[..winner_count].sort_unstable_by_key(|&(_, i)| i);

            // Assign atlas slots to winners; disable everything else.
            let mut next_layer: u32 = 0;
            for (rank, &(_, i)) in scored.iter().enumerate() {
                let light = self.gpu_scene.lights.0.as_slice()[i];
                if rank < MAX_SHADOW_CASTERS {
                    let mut assigned = light;
                    assigned.shadow_index = next_layer;
                    self.gpu_scene.lights.update(i, assigned);
                    next_layer += FACES_PER_LIGHT;
                } else {
                    let mut disabled = light;
                    disabled.shadow_index = u32::MAX;
                    self.gpu_scene.lights.update(i, disabled);
                }
            }
            let needed = (next_layer as usize).max(1);
            if self.gpu_scene.shadow_matrices.len() != needed {
                self.gpu_scene
                    .shadow_matrices
                    .set_data(vec![GpuShadowMatrix::zeroed(); needed]);
            }
        }

        // ── Per-caster shadow dirty tracking ─────────────────────────────────
        // Compute a content hash per shadow caster. Each hash covers:
        //   • The caster light's own geometry (position, range, direction).
        //   • All movable objects whose bounding sphere overlaps the light's range.
        // Directional lights always include every movable object (infinite range).
        // Casters whose hash differs from last frame bump their dirty gen counter;
        // ShadowPass then re-renders only those casters' atlas faces.
        {
            let light_count = self.gpu_scene.lights.len();
            let mut new_hashes = [0u64; 42];

            // Pass 1: hash each shadow-casting light's geometry into its slot.
            for i in 0..light_count {
                let light = self.gpu_scene.lights.0.as_slice()[i];
                if light.shadow_index == u32::MAX {
                    continue;
                }
                let slot = (light.shadow_index / 6) as usize;
                if slot >= 42 {
                    continue;
                }
                let base_hash = fnv1a_f32s(&light.position_range)
                    ^ fnv1a_f32s(&light.direction_outer)
                    ^ (light.light_type as u64).wrapping_mul(2654435761);
                // Directional CSM depends on the camera frustum, but the GPU matrix pass
                // already texel-snaps cascade placement. Mirror that coarseness here so
                // sub-texel camera motion does not thrash the cached shadow atlas.
                new_hashes[slot] = if light.light_type == 0 {
                    const DIRECTIONAL_CAMERA_SNAP_METERS: f32 = 0.25;
                    const DIRECTIONAL_FORWARD_SNAP: f32 = 1.0 / 1024.0;

                    let snapped_cam_pos = quantize_f32s(
                        self.gpu_scene.camera.position(),
                        DIRECTIONAL_CAMERA_SNAP_METERS,
                    );
                    let snapped_cam_forward =
                        quantize_f32s(self.gpu_scene.camera.forward(), DIRECTIONAL_FORWARD_SNAP);

                    base_hash ^ fnv1a_f32s(&snapped_cam_pos) ^ fnv1a_f32s(&snapped_cam_forward)
                } else {
                    base_hash
                };
            }

            // Write light-geometry hash to per_caster_dirty_gen.
            // ShadowPass detects light movement each frame by comparing this value.
            for slot in 0..42usize {
                self.gpu_scene.per_caster_dirty_gen[slot] = new_hashes[slot];
            }
        }

        let queue = self.gpu_scene.queue.clone();
        self.mesh_pool.flush(&queue);
        self.material_textures.flush(&queue);
        // Rebuild instanced draw lists when the object set has changed.
        if self.objects_dirty {
            if self.objects_layout_optimized {
                self.rebuild_instance_buffers_optimized();
            } else {
                self.rebuild_instance_buffers_persistent();
            }
            self.objects_dirty = false;
            // Full rebuild already called rebuild_shadow_partition_buffers().
            self.shadow_partition_dirty = false;
        }
        // Persistent-mode delta inserts/removes bypass the full rebuild, so shadow
        // partition indirect buffers need an explicit rebuild here.
        if self.shadow_partition_dirty {
            self.rebuild_shadow_partition_buffers();
            self.shadow_partition_dirty = false;
        }
        // Rebuild virtual geometry CPU buffers when VG topology or transforms changed.
        if self.vg_objects_dirty {
            self.rebuild_vg_buffers();
            self.vg_objects_dirty = false;
        }
        self.gpu_scene.flush();
    }
}
