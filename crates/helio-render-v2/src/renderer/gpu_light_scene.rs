//! GPU-resident light + shadow matrix storage with dirty-bit delta uploads.
//!
//! ## Architecture
//!
//! Lights are managed via a **persistent proxy map** — identical to how
//! `GpuScene` manages mesh objects.  `add_light` / `remove_light` /
//! `update_light` are O(1) and set a dirty flag on the affected slot.
//! At steady state (no light changes) the per-frame cost is:
//!
//! * `update_shadow_matrices` — O(N) matrix recompute, GPU write only when
//!   the FNV hash changed (zero uploads for static lights at fixed camera).
//! * `flush` — single coalesced `write_buffer` over the dirty range; no-op
//!   when nothing is dirty.
//!
//! The legacy `sync_lights(&[SceneLight])` snapshot API is kept as a thin
//! compatibility shim and delegates to the persistent path.
//!
//! ## Buffer layout
//!
//! Lights occupy a **dense array** indexed 0..active_count.  When a light is
//! removed via `remove_light`, the last active light is swapped into the freed
//! slot (tombstone-free swap-remove) so the array stays dense and `face_counts`
//! / `shadow_cull_lights` remain contiguous.  The swapped light's slot is
//! marked dirty so the GPU sees the updated position.
//!
//! Both buffers are pre-allocated at `MAX_LIGHTS` capacity so the
//! `Arc<wgpu::Buffer>` references shared with `ShadowPass` / the lighting
//! bind group remain valid forever — no rebind needed.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::Zeroable;

use crate::camera::Camera;
use crate::features::LightType;
use crate::features::lighting::{GpuLight, MAX_LIGHTS};
use crate::gpu_transfer;
use crate::passes::ShadowCullLight;
use crate::scene::{LightId, SceneLight};

// Shadow math functions now on GPU (shadow_matrices.wgsl)

// ─────────────────────────────────────────────────────────────────────────────

/// Internal record for one persistent light slot.
struct LightProxy {
    /// Index into the dense arrays (`cpu_lights`, `face_counts`, etc.).
    index: u32,
    /// CPU mirror — kept for shadow matrix recomputation and compat shim.
    data: SceneLight,
}

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

    // ── GPU-driven shadow matrix computation buffers ──────────────────────
    /// GPU storage buffer: per-light dirty flags (u32 bitmask, 0 or 1)
    pub shadow_dirty_buffer: Arc<wgpu::Buffer>,
    /// CPU mirror: True iff slot[i]'s shadow matrices need recompute.
    shadow_dirty: Vec<bool>,

    /// GPU storage buffer: per-light FNV hashes written by shadow_matrices.wgsl (never read back)
    pub shadow_hash_buffer: Arc<wgpu::Buffer>,
    /// Monotonic generation counter per light slot.
    /// Incremented on every `shadow_dirty[i] = true` transition so the shadow
    /// pass cache key changes without any CPU hash computation or GPU readback.
    shadow_generation: Vec<u64>,

    // ── Persistent proxy map ──────────────────────────────────────────────
    /// LightId → proxy (index into dense arrays + CPU data mirror).
    proxies: HashMap<u32, LightProxy>,
    /// Monotonically increasing id counter (never reuses ids).
    next_light_id: u32,

    // ── Dense CPU mirror (same layout as GPU buffer) ──────────────────────
    /// SceneLight data in slot order — used by update_shadow_matrices,
    /// sync_lights, and flush_scene_state (directional sun direction).
    pub(super) cached_scene_lights: Vec<SceneLight>,

    // ── Per-frame publishable state (forwarded to ShadowPass) ────────────
    /// Number of active lights.
    pub active_count: u32,
    /// Per-slot face count: 6 = point, 4 = directional, 1 = spot.
    /// Rebuilt only on structural changes (add/remove), not every frame.
    pub face_counts: Vec<u8>,
    /// Cull data for ShadowPass; rebuilt every frame in update_shadow_matrices
    /// (stores current shadow hash for skip-logic).
    pub shadow_cull_lights: Vec<ShadowCullLight>,

    // ── Structural change flag ────────────────────────────────────────────
    /// Set whenever a light is added or removed.  Callers gate expensive
    /// work (face_counts clone into Mutex, etc.) on this flag and clear it.
    pub structure_changed: bool,

    // ── Camera state for CSM dirty detection ─────────────────────────────
    last_camera_pos: [f32; 3],
    camera_move_threshold: f32,
}

impl GpuLightScene {
    /// Wrap existing, already-allocated GPU buffers with dirty-tracking state.
    pub fn new(
        light_buffer: Arc<wgpu::Buffer>,
        shadow_matrix_buffer: Arc<wgpu::Buffer>,
        shadow_dirty_buffer: Arc<wgpu::Buffer>,
        shadow_hash_buffer: Arc<wgpu::Buffer>,
    ) -> Self {
        let cap = MAX_LIGHTS as usize;
        Self {
            light_buffer,
            cpu_lights:  vec![GpuLight::zeroed(); cap],
            light_dirty: vec![false; cap],

            shadow_matrix_buffer,

            shadow_dirty_buffer,
            shadow_dirty:    vec![false; cap],

            shadow_hash_buffer,
            shadow_generation: vec![0u64; cap],

            proxies: HashMap::new(),
            next_light_id: 1, // 0 reserved for INVALID

            cached_scene_lights: Vec::new(),

            active_count: 0,
            face_counts: Vec::new(),
            shadow_cull_lights: Vec::new(),

            structure_changed: false,

            last_camera_pos: [f32::NAN; 3],
            camera_move_threshold: 0.5,
        }
    }

    // ── Persistent API ────────────────────────────────────────────────────

    /// Register a new light.  Returns a stable [`LightId`] valid until
    /// [`remove_light`] is called.  O(1).
    pub fn add_light(&mut self, light: SceneLight) -> LightId {
        assert!(
            self.active_count < MAX_LIGHTS as u32,
            "GpuLightScene: MAX_LIGHTS ({}) exceeded", MAX_LIGHTS
        );

        let id = LightId(self.next_light_id);
        self.next_light_id += 1;

        let index = self.active_count as usize;
        self.active_count += 1;

        // Grow dense arrays if needed (they start at MAX_LIGHTS capacity so
        // this is just a push into pre-allocated space).
        if index >= self.cached_scene_lights.len() {
            self.cached_scene_lights.push(light.clone());
        } else {
            self.cached_scene_lights[index] = light.clone();
        }

        self.cpu_lights[index] = scene_light_to_gpu(&light);
        self.light_dirty[index] = true;
        self.shadow_dirty[index] = true;
        self.shadow_generation[index] += 1;

        // Rebuild face_counts (structural change).
        self.face_counts.push(light_face_count(&light));

        self.proxies.insert(id.0, LightProxy { index: index as u32, data: light });
        self.structure_changed = true;

        id
    }

    /// Remove a light by id.  The last active light is swap-removed into the
    /// freed slot so the array stays dense.  O(1).
    pub fn remove_light(&mut self, id: LightId) {
        let proxy = match self.proxies.remove(&id.0) {
            Some(p) => p,
            None => return,
        };

        let slot = proxy.index as usize;
        let last = (self.active_count - 1) as usize;

        if slot != last {
            // Swap the last light into this slot.
            let last_light = self.cached_scene_lights[last].clone();
            self.cached_scene_lights[slot] = last_light.clone();
            self.cpu_lights[slot] = scene_light_to_gpu(&last_light);
            self.light_dirty[slot] = true;

            // Carry forward the moved light's generation and increment so the
            // shadow cache sees a change for both the vacated and filled slot.
            self.shadow_generation[slot] = self.shadow_generation[last].wrapping_add(1);
            self.shadow_dirty[slot] = true;

            self.face_counts[slot] = self.face_counts[last];

            // Find which proxy pointed at `last` and update its index.
            for proxy in self.proxies.values_mut() {
                if proxy.index as usize == last {
                    proxy.index = slot as u32;
                    break;
                }
            }
        }

        // Zero out the now-unused last slot on the GPU.
        self.cpu_lights[last] = GpuLight::zeroed();
        self.light_dirty[last] = true;
        self.shadow_dirty[last] = true;
        self.shadow_generation[last] = self.shadow_generation[last].wrapping_add(1);

        self.cached_scene_lights.truncate(last);
        self.face_counts.truncate(last);
        self.active_count -= 1;
        self.structure_changed = true;
    }

    /// Update all parameters of an existing light.  O(1).
    pub fn update_light(&mut self, id: LightId, light: SceneLight) {
        let proxy = match self.proxies.get_mut(&id.0) {
            Some(p) => p,
            None => return,
        };
        let slot = proxy.index as usize;
        proxy.data = light.clone();
        self.cached_scene_lights[slot] = light.clone();
        self.cpu_lights[slot] = scene_light_to_gpu(&light);
        self.light_dirty[slot] = true;
        self.shadow_dirty[slot] = true;
        self.shadow_generation[slot] += 1;
    }

    /// Update only the world-space position of a light (e.g. moving point
    /// light).  Cheaper than a full `update_light` if other params are fixed.
    pub fn move_light(&mut self, id: LightId, position: [f32; 3]) {
        let proxy = match self.proxies.get_mut(&id.0) {
            Some(p) => p,
            None => return,
        };
        let slot = proxy.index as usize;
        proxy.data.position = position;
        self.cached_scene_lights[slot].position = position;
        self.cpu_lights[slot].position = position;
        self.light_dirty[slot] = true;
        self.shadow_dirty[slot] = true;
        self.shadow_generation[slot] += 1;
    }

    /// Update color + intensity of a light without touching its transform.
    pub fn set_light_params(&mut self, id: LightId, color: [f32; 3], intensity: f32) {
        let proxy = match self.proxies.get_mut(&id.0) {
            Some(p) => p,
            None => return,
        };
        let slot = proxy.index as usize;
        proxy.data.color = color;
        proxy.data.intensity = intensity;
        self.cached_scene_lights[slot].color = color;
        self.cached_scene_lights[slot].intensity = intensity;
        self.cpu_lights[slot].color = color;
        self.cpu_lights[slot].intensity = intensity;
        self.light_dirty[slot] = true;
        // No shadow_dirty — shadow matrices don't depend on color/intensity.
    }

    // ── Compatibility shim ────────────────────────────────────────────────

    /// Diff `new_lights` against the cached CPU mirror and dirty-flag only
    /// changed / added / removed slots.
    ///
    /// This is the legacy snapshot API kept for backward compatibility.
    /// New code should use `add_light` / `remove_light` / `update_light`.
    pub fn sync_lights(&mut self, new_lights: &[SceneLight]) {
        let new_count = new_lights.len();
        let old_count = self.cached_scene_lights.len();

        // Diff: update GpuLight data for changed slots.
        for i in 0..new_count.min(old_count) {
            if !scene_lights_bitwise_equal(&new_lights[i], &self.cached_scene_lights[i]) {
                self.cached_scene_lights[i] = new_lights[i].clone();
                self.cpu_lights[i] = scene_light_to_gpu(&new_lights[i]);
                self.light_dirty[i] = true;
                self.shadow_dirty[i] = true;
                self.shadow_generation[i] += 1;
            }
        }

        // Newly added lights.
        for i in old_count..new_count {
            self.cached_scene_lights.push(new_lights[i].clone());
            self.cpu_lights[i] = scene_light_to_gpu(&new_lights[i]);
            self.light_dirty[i] = true;
            self.shadow_dirty[i] = true;
            self.shadow_generation[i] += 1;
        }

        // Removed lights: zero out their GPU slots.
        for i in new_count..old_count {
            self.cpu_lights[i] = GpuLight::zeroed();
            self.light_dirty[i] = true;
            self.shadow_dirty[i] = true;
            self.shadow_generation[i] += 1;
        }
        self.cached_scene_lights.truncate(new_count);

        let old_active = self.active_count;
        self.active_count = new_count as u32;

        // Rebuild face_counts only when the light set changed structurally.
        if new_count != old_count as usize || old_active != self.active_count {
            self.face_counts.clear();
            for l in new_lights {
                self.face_counts.push(light_face_count(l));
            }
            self.structure_changed = true;
        }
    }

    // ── Shadow matrices (GPU-driven) ──────────────────────────────────────

    /// Mark shadow matrices dirty for lights that need updates.
    /// - Always dirty: lights with shadow_dirty[i] = true (position/direction changed)
    /// - Conditional dirty: directional lights when camera moves (CSM depends on camera frustum)
    ///
    /// Also builds shadow_cull_lights list for ShadowPass culling.
    ///
    /// NOTE: Shadow matrices are now computed on GPU by ShadowMatrixPass.
    /// This method no longer performs CPU matrix computation.
    pub fn update_shadow_matrices(&mut self, camera: &Camera) -> bool {
        let camera_pos: [f32; 3] = camera.position.into();
        let camera_moved = {
            let lp = self.last_camera_pos;
            if lp[0].is_nan() {
                true
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

        let count = self.active_count as usize;

        // Mark directional lights dirty when camera moves (CSM depends on camera frustum)
        if camera_moved {
            for (i, light) in self.cached_scene_lights[..count].iter().enumerate() {
                if matches!(light.light_type, LightType::Directional) {
                    self.shadow_dirty[i] = true;
                    self.shadow_generation[i] += 1;
                }
            }
        }

        // Build shadow_cull_lights for ShadowPass.
        // shadow_generation[i] is incremented every time shadow_dirty[i] is set,
        // so it serves as a cheap GPU-state-driven cache key — no hash, no readback.
        self.shadow_cull_lights.clear();
        self.shadow_cull_lights.reserve(count);
        for (i, light) in self.cached_scene_lights[..count].iter().enumerate() {
            self.shadow_cull_lights.push(ShadowCullLight {
                position:       light.position,
                direction:      light.direction,
                range:          shadow_cull_range(light),
                is_directional: matches!(light.light_type, LightType::Directional),
                is_point:       matches!(light.light_type, LightType::Point),
                matrix_hash:    self.shadow_generation[i],
            });
        }

        camera_moved
    }

    /// Upload shadow dirty flags to GPU buffer for ShadowMatrixPass compute shader.
    /// Zero cost when no lights are dirty.
    pub fn upload_shadow_dirty_flags(&self, queue: &wgpu::Queue) {
        let count = self.active_count as usize;
        if count == 0 { return; }

        // Check if any light is dirty
        if !self.shadow_dirty[..count].iter().any(|&d| d) {
            return;
        }

        // Convert bool to u32 (0 or 1) for GPU
        let dirty_u32: Vec<u32> = self.shadow_dirty[..count]
            .iter()
            .map(|&d| if d { 1u32 } else { 0u32 })
            .collect();

        queue.write_buffer(
            &self.shadow_dirty_buffer,
            0,
            bytemuck::cast_slice(&dirty_u32),
        );
        gpu_transfer::track_upload((count * 4) as u64);
    }

    /// Accessor for shadow dirty buffer (used by ShadowMatrixPass)
    pub fn shadow_dirty_buffer(&self) -> &wgpu::Buffer {
        &self.shadow_dirty_buffer
    }

    /// Accessor for shadow hash buffer (used by ShadowMatrixPass)
    pub fn shadow_hash_buffer(&self) -> &wgpu::Buffer {
        &self.shadow_hash_buffer
    }

    /// Clear shadow dirty flags after GPU compute has processed them.
    /// Call after ShadowMatrixPass executes.
    pub fn clear_shadow_dirty_flags(&mut self) {
        let count = self.active_count as usize;
        for i in 0..count {
            self.shadow_dirty[i] = false;
        }
    }

    /// Enumerate all active light IDs together with their [`SceneLight`] data.
    ///
    /// Used by the editor-mode billboard management in [`Renderer::set_editor_mode`].
    pub(super) fn iter_lights(&self) -> impl Iterator<Item = (LightId, &SceneLight)> {
        self.proxies.iter().map(|(&raw_id, proxy)| {
            let light = &self.cached_scene_lights[proxy.index as usize];
            (LightId(raw_id), light)
        })
    }

    /// Write dirty light slots to the GPU.
    /// Zero cost when nothing changed.
    ///
    /// NOTE: Shadow matrices are now computed on GPU by ShadowMatrixPass.
    /// This method no longer uploads shadow matrix data.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        let count = self.active_count as usize;
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

        // Shadow matrices are now computed and uploaded by GPU compute shader (ShadowMatrixPass).
        // No CPU → GPU upload needed here.
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

fn light_face_count(l: &SceneLight) -> u8 {
    match l.light_type {
        LightType::Point       => 6,
        LightType::Directional => 4,
        LightType::Spot { .. } => 1,
    }
}

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

pub(super) fn scene_light_to_gpu(l: &SceneLight) -> GpuLight {
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

// Legacy hash function — no longer used (shadow matrices computed on GPU).
// Kept for potential legacy sync_lights compatibility.
#[allow(dead_code)]
fn fnv_hash_mats(mats: &[glam::Mat4; 6]) -> u32 {
    let mut h: u32 = 0x811c9dc5;  // FNV-1a 32-bit offset
    for m in mats {
        for f in m.to_cols_array() {
            let bits = f.to_bits();
            h ^= (bits & 0xFF) as u32;
            h = h.wrapping_mul(0x01000193);  // FNV-1a 32-bit prime
            h ^= ((bits >> 8) & 0xFF) as u32;
            h = h.wrapping_mul(0x01000193);
            h ^= ((bits >> 16) & 0xFF) as u32;
            h = h.wrapping_mul(0x01000193);
            h ^= ((bits >> 24) & 0xFF) as u32;
            h = h.wrapping_mul(0x01000193);
        }
    }
    h
}
