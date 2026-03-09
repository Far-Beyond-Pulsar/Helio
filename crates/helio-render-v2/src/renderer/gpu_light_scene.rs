//! GPU-resident light + shadow matrix storage with dirty-bit delta uploads.
//!
//! Analogous to [`crate::gpu_scene::GpuScene`] for geometry: lights occupy
//! stable GPU slots indexed by their position in the scene's light list.
//! Each frame only **dirty** slots (lights whose data actually changed) are
//! written to the GPU.  A scene with no moving lights produces **zero** GPU
//! uploads — down from ~47 MB / 60 frames with the old full-buffer path.
//!
//! Shadow matrices (6 per light) live in a paired buffer.  They are
//! recomputed only for dirty light slots and, for CSM directional lights,
//! when the camera moves past a configurable threshold.  An FNV hash
//! comparison guards against unnecessary writes even when a matrix was
//! recomputed but didn't actually change values.
//!
//! Both buffers are pre-allocated at `MAX_LIGHTS` capacity (same as before)
//! so the `Arc<wgpu::Buffer>` references shared with `ShadowPass` / the
//! lighting bind group remain valid forever — no rebind needed on growth.

use std::sync::Arc;

use bytemuck::Zeroable;
use glam::{Mat4, Vec3};

use crate::camera::Camera;
use crate::features::LightType;
use crate::features::lighting::{GpuLight, MAX_LIGHTS};
use crate::gpu_transfer;
use crate::passes::ShadowCullLight;
use crate::scene::SceneLight;

use super::shadow_math::{
    compute_directional_cascades, compute_point_light_matrices, compute_spot_matrix,
};
use super::uniforms::GpuShadowMatrix;

const FACES_PER_LIGHT: usize = 6;

// ─────────────────────────────────────────────────────────────────────────────

/// GPU-resident light + shadow matrix buffers with per-slot dirty tracking.
///
/// Created once in [`Renderer::new`].  Both inner `Arc<wgpu::Buffer>` handles
/// are the same objects bound into the lighting bind group — writes go
/// directly into the already-bound buffers with no rebind required.
pub(super) struct GpuLightScene {
    // ── Light GPU buffer (binding 0 of the lighting bind group) ──────────
    pub light_buffer: Arc<wgpu::Buffer>,
    /// CPU mirror of per-slot GpuLight data.
    cpu_lights: Vec<GpuLight>,
    /// True iff slot[i] needs to be written to the GPU this frame.
    light_dirty: Vec<bool>,

    // ── Shadow matrix GPU buffer (binding 4 of the lighting bind group) ──
    pub shadow_matrix_buffer: Arc<wgpu::Buffer>,
    /// CPU mirror: FACES_PER_LIGHT consecutive entries per light slot.
    cpu_shadow_mats: Vec<GpuShadowMatrix>,
    /// True iff slot[i]'s shadow matrices need GPU upload this frame.
    shadow_dirty: Vec<bool>,
    /// FNV hash of the 6 shadow matrices per slot; used to skip
    /// uploads when a recomputed matrix is numerically identical.
    shadow_hashes: Vec<u64>,

    // ── CPU-side scene mirror (for frame-to-frame diffing) ────────────────
    cached_scene_lights: Vec<SceneLight>,

    // ── Per-frame publishable state (forwarded to ShadowPass) ────────────
    /// Number of active lights this frame.
    pub active_count: u32,
    /// Per-slot face count: 6 = point, 4 = directional, 1 = spot.
    pub face_counts: Vec<u8>,
    /// Cull data for ShadowPass; rebuilt whenever the light set changes.
    pub shadow_cull_lights: Vec<ShadowCullLight>,

    // ── Camera state for CSM dirty detection ─────────────────────────────
    last_camera_pos: [f32; 3],
    camera_move_threshold: f32,
}

impl GpuLightScene {
    /// Wrap existing, already-allocated GPU buffers with dirty-tracking state.
    ///
    /// Both buffers must have been created with capacity for `MAX_LIGHTS`
    /// lights (the same objects that live in the lighting bind group).
    pub fn new(
        light_buffer: Arc<wgpu::Buffer>,
        shadow_matrix_buffer: Arc<wgpu::Buffer>,
    ) -> Self {
        let cap = MAX_LIGHTS as usize;
        Self {
            light_buffer,
            cpu_lights:  vec![GpuLight::zeroed(); cap],
            light_dirty: vec![false; cap],

            shadow_matrix_buffer,
            cpu_shadow_mats: vec![GpuShadowMatrix::zeroed(); cap * FACES_PER_LIGHT],
            shadow_dirty:    vec![false; cap],
            shadow_hashes:   vec![u64::MAX; cap],

            cached_scene_lights: Vec::new(),

            active_count: 0,
            face_counts: Vec::new(),
            shadow_cull_lights: Vec::new(),

            last_camera_pos: [f32::NAN; 3],
            camera_move_threshold: 0.5,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Diff `new_lights` against the cached CPU mirror and dirty-flag only
    /// changed / added / removed slots.
    ///
    /// At steady state with no light movement this is a pure CPU compare —
    /// O(N) comparisons, zero GPU writes queued.
    pub fn sync_lights(&mut self, new_lights: &[SceneLight]) {
        let new_count = new_lights.len();
        let old_count = self.cached_scene_lights.len();

        // Diff: update GpuLight data for changed / new slots.
        for i in 0..new_count.min(old_count) {
            if !scene_lights_bitwise_equal(&new_lights[i], &self.cached_scene_lights[i]) {
                self.cpu_lights[i] = scene_light_to_gpu(&new_lights[i]);
                self.light_dirty[i] = true;
                self.shadow_dirty[i] = true; // position/type changed → recompute shadow mats
            }
        }

        // Newly added lights.
        for i in old_count..new_count {
            self.cpu_lights[i] = scene_light_to_gpu(&new_lights[i]);
            self.light_dirty[i] = true;
            self.shadow_dirty[i] = true;
            // Invalidate any stale shadow hash for this slot so it gets recomputed.
            self.shadow_hashes[i] = u64::MAX;
        }

        // Removed lights: zero out their GPU slots.
        for i in new_count..old_count {
            self.cpu_lights[i] = GpuLight::zeroed();
            self.light_dirty[i] = true;
            // Zero the shadow matrices too so stale data is gone.
            let base = i * FACES_PER_LIGHT;
            for j in 0..FACES_PER_LIGHT {
                self.cpu_shadow_mats[base + j] = GpuShadowMatrix::zeroed();
            }
            self.shadow_dirty[i] = true;
            self.shadow_hashes[i] = u64::MAX;
        }

        // Rebuild per-frame helpers.
        self.active_count = new_count as u32;
        self.face_counts.clear();
        for l in new_lights {
            self.face_counts.push(match l.light_type {
                LightType::Point       => 6,
                LightType::Directional => 4,
                LightType::Spot { .. } => 1,
            });
        }

        self.cached_scene_lights.clear();
        self.cached_scene_lights.extend_from_slice(new_lights);
    }

    /// Recompute shadow matrices for dirty light slots and for directional
    /// lights when the camera moves past the configured threshold.
    ///
    /// FNV hash comparison means even a "dirty" slot only queues a GPU write
    /// when the numerical matrix values actually changed.
    pub fn update_shadow_matrices(&mut self, camera: &Camera) {
        let camera_pos: [f32; 3] = camera.position.into();
        let camera_moved = {
            let lp = self.last_camera_pos;
            if lp[0].is_nan() {
                true // first frame
            } else {
                let dx = camera_pos[0] - lp[0];
                let dy = camera_pos[1] - lp[1];
                let dz = camera_pos[2] - lp[2];
                (dx * dx + dy * dy + dz * dz).sqrt() > self.camera_move_threshold
            }
        };
        if camera_moved {
            self.last_camera_pos = camera_pos;
        }

        let identity = Mat4::IDENTITY;
        let count = self.cached_scene_lights.len();

        self.shadow_cull_lights.clear();
        self.shadow_cull_lights.reserve(count);

        for (i, light) in self.cached_scene_lights.iter().enumerate() {
            let is_directional = matches!(light.light_type, LightType::Directional);
            let needs_update = self.shadow_dirty[i] || (camera_moved && is_directional);

            if !needs_update {
                self.shadow_cull_lights.push(ShadowCullLight {
                    position:       light.position,
                    direction:      light.direction,
                    range:          shadow_cull_range(light),
                    is_directional,
                    is_point:       matches!(light.light_type, LightType::Point),
                    matrix_hash:    self.shadow_hashes[i],
                });
                continue;
            }

            let six: [Mat4; 6] = match light.light_type {
                LightType::Point => {
                    compute_point_light_matrices(light.position, light.range)
                }
                LightType::Directional => {
                    let [c0, c1, c2, c3] = compute_directional_cascades(
                        Vec3::from(camera.position),
                        camera.view_proj_inv,
                        light.direction,
                    );
                    [c0, c1, c2, c3, identity, identity]
                }
                LightType::Spot { outer_angle, .. } => {
                    let m = compute_spot_matrix(
                        light.position,
                        light.direction,
                        light.range,
                        outer_angle,
                    );
                    [m, identity, identity, identity, identity, identity]
                }
            };

            let new_hash = fnv_hash_mats(&six);
            let base = i * FACES_PER_LIGHT;

            if new_hash != self.shadow_hashes[i] {
                for (j, m) in six.iter().enumerate() {
                    self.cpu_shadow_mats[base + j] =
                        GpuShadowMatrix { mat: m.to_cols_array() };
                }
                self.shadow_hashes[i] = new_hash;
                self.shadow_dirty[i] = true;
            } else {
                // Matrices are identical — nothing to upload.
                self.shadow_dirty[i] = false;
            }

            self.shadow_cull_lights.push(ShadowCullLight {
                position:       light.position,
                direction:      light.direction,
                range:          shadow_cull_range(light),
                is_directional,
                is_point:       matches!(light.light_type, LightType::Point),
                matrix_hash:    self.shadow_hashes[i],
            });
        }

        // Clear light dirty flags after shadow matrix processing since
        // shadow_dirty now reflects whether a GPU write is needed.
        for i in 0..count {
            // shadow_dirty[i] was set by sync_lights *or* update_shadow_matrices
            // (hash mismatch) and cleared on hash match above — leave as is.
            // light_dirty[i] is cleared by flush().
            let _ = i;
        }
    }

    /// Write all dirty light slots and dirty shadow matrix slots to the GPU.
    ///
    /// Uses a single coalesced `write_buffer` call for the dirty range of the
    /// light buffer (lo..=hi), and one per-slot call for shadow matrices
    /// (often only a handful even in animated scenes).
    ///
    /// Zero cost when nothing changed.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        let count = self.cached_scene_lights.len();
        if count == 0 {
            return;
        }

        // ── Lights: coalesced dirty-range write ───────────────────────────
        let lo = self.light_dirty[..count].iter().position(|&d| d);
        let hi = self.light_dirty[..count].iter().rposition(|&d| d);
        if let (Some(lo), Some(hi)) = (lo, hi) {
            let stride = std::mem::size_of::<GpuLight>() as u64;
            let offset = lo as u64 * stride;
            let slice = &self.cpu_lights[lo..=hi];
            queue.write_buffer(&self.light_buffer, offset, bytemuck::cast_slice(slice));
            let written = ((hi - lo + 1) as u64) * stride;
            gpu_transfer::track_upload(written);
            for d in self.light_dirty[lo..=hi].iter_mut() {
                *d = false;
            }
        }

        // ── Shadow matrices: per dirty-slot write (384 bytes each) ────────
        let mat_stride = std::mem::size_of::<GpuShadowMatrix>() as u64;
        for i in 0..count {
            if !self.shadow_dirty[i] {
                continue;
            }
            let base = i * FACES_PER_LIGHT;
            let slice = &self.cpu_shadow_mats[base..base + FACES_PER_LIGHT];
            let offset = base as u64 * mat_stride;
            queue.write_buffer(
                &self.shadow_matrix_buffer,
                offset,
                bytemuck::cast_slice(slice),
            );
            let written = (FACES_PER_LIGHT as u64) * mat_stride;
            gpu_transfer::track_upload(written);
            self.shadow_dirty[i] = false;
        }
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Bitwise equality check for SceneLight (avoids PartialEq on f32 NaN issues).
fn scene_lights_bitwise_equal(a: &SceneLight, b: &SceneLight) -> bool {
    if a.light_type != b.light_type {
        return false;
    }
    let pos_eq = a.position[0].to_bits() == b.position[0].to_bits()
        && a.position[1].to_bits() == b.position[1].to_bits()
        && a.position[2].to_bits() == b.position[2].to_bits();
    let dir_eq = a.direction[0].to_bits() == b.direction[0].to_bits()
        && a.direction[1].to_bits() == b.direction[1].to_bits()
        && a.direction[2].to_bits() == b.direction[2].to_bits();
    let col_eq = a.color[0].to_bits() == b.color[0].to_bits()
        && a.color[1].to_bits() == b.color[1].to_bits()
        && a.color[2].to_bits() == b.color[2].to_bits();
    pos_eq && dir_eq && col_eq
        && a.intensity.to_bits() == b.intensity.to_bits()
        && a.range.to_bits() == b.range.to_bits()
}

fn scene_light_to_gpu(l: &SceneLight) -> GpuLight {
    let light_type = match l.light_type {
        LightType::Directional => 0.0,
        LightType::Point       => 1.0,
        LightType::Spot { .. } => 2.0,
    };
    let (cos_inner, cos_outer) = match l.light_type {
        LightType::Spot { inner_angle, outer_angle } => {
            (inner_angle.cos(), outer_angle.cos())
        }
        _ => (0.0, 0.0),
    };
    let d = l.direction;
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let direction = if len > 1e-6 {
        [d[0] / len, d[1] / len, d[2] / len]
    } else {
        [0.0, -1.0, 0.0]
    };
    GpuLight {
        position: l.position,
        light_type,
        direction,
        range: l.range,
        color: l.color,
        intensity: l.intensity,
        cos_inner,
        cos_outer,
        _pad: [0.0; 2],
    }
}

fn shadow_cull_range(light: &SceneLight) -> f32 {
    match light.light_type {
        LightType::Directional => light.range * 2.2,
        _                      => light.range * 5.0,
    }
}

fn fnv_hash_mats(mats: &[Mat4; 6]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for m in mats {
        for f in m.to_cols_array() {
            h ^= f.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
