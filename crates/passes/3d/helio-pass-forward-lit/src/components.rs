//! SceneDB-owned light primitive.
//!
//! Mirrors the particles plugin's `CoronaEmitterRow` recipe exactly: the
//! pass that consumes a kind of scene content owns that content's schema.
//! SceneDB owns the packed CPU column and its GPU projection; this crate
//! never holds a second, renderer-local copy of "what lights exist."
//!
//! Unlike corona (which still needs a CPU-supplied count each frame),
//! [`LightRow`]'s buffer is read with a *fixed* iteration count
//! ([`MAX_LIGHTS`], its registered capacity) directly out of
//! `ctx.scene_buffers` (`BufferKey::of("scene_lights")`) — see this pass's
//! `prepare()`/`execute()`. No renderer method creates, updates, or binds
//! anything light-specific: the frontend does a plain `World::insert`, and
//! this pass resolves the resulting buffer by key, generically, the same way
//! it already resolves every other declared resource. No per-frame CPU touch
//! at all, not even a live-count read — unused rows are `Zeroable`-default
//! (`color_intensity` all zero), so iterating the full capacity every frame
//! contributes nothing extra.
//!
//! Shadows: `shadow_index` is authored as a request, `u32::MAX` for none and
//! anything else (0 by convention) for a shadow map. `ShadowMatrixPass`
//! allocates atlas slots on the GPU (`shadow_casters.wgsl`, Helio#246): the
//! most important requesting lights (directional first, then
//! `intensity × range²`) get `shadow_index = 6 × slot` written into their GPU
//! rows, the rest `u32::MAX`. It re-runs only when these rows change, so
//! there is no per-frame CPU work and nothing to do while lights are idle.

/// Initial capacity of the `"scene_lights"` buffer: `World`'s own
/// auto-registration capacity, and the capacity frontends register it with.
/// The buffer grows as lights are added; passes iterate its live
/// `BufferHandle::row_capacity()` (unused rows are zeroed), never this
/// constant.
pub const MAX_LIGHTS: u32 = pulsar_scenedb::gpu::world_mirror::DEFAULT_AUTO_REGISTER_CAPACITY;

/// A placed light, authored as a SceneDB component. Field-for-field
/// identical to [`crate::GpuLight`] (enforced by the size/align test
/// below) so conversion is a zero-cost `bytemuck::cast`, matching
/// `CoronaEmitterRow`'s relationship to `GpuCoronaEmitter`.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct LightRow {
    pub position_range: [f32; 4],
    pub direction_outer: [f32; 4],
    pub color_intensity: [f32; 4],
    pub shadow_index: u32,
    pub light_type: u32,
    pub inner_angle: f32,
    pub _pad: u32,
    pub god_rays_enabled: u32,
    pub god_rays_density: f32,
    pub god_rays_weight: f32,
    pub god_rays_decay: f32,
    pub god_rays_exposure: f32,
    pub flare_enabled: u32,
    pub flare_type: u32,
    pub flare_intensity: f32,
    pub flare_scale: f32,
    pub flare_tint_r: f32,
    pub flare_tint_g: f32,
    pub flare_tint_b: f32,
    pub ies_profile_index: i32,
    pub light_function_index: i32,
    pub ies_angle_scale: f32,
    pub ies_angle_offset: f32,
}

// SAFETY: a `#[repr(C)]` bytemuck `Pod` row: no padding bytes, any bit
// pattern valid. Lets an engine's SceneDB component embed it.
unsafe impl pulsar_scenedb::Pod for LightRow {}

/// The buffer the renderer reads [`LightRow`]s from. Engines produce it
/// (a SceneDB component stored under this key, or a scene derivation).
pub const SCENE_LIGHTS: helio_core::BufferContract<LightRow> =
    helio_core::BufferContract::new("scene_lights");

impl From<crate::GpuLight> for LightRow {
    fn from(value: crate::GpuLight) -> Self {
        bytemuck::cast(value)
    }
}

impl From<LightRow> for crate::GpuLight {
    fn from(value: LightRow) -> Self {
        bytemuck::cast(value)
    }
}

#[cfg(test)]
mod tests {
    use super::LightRow;

    #[test]
    fn scene_record_matches_light_gpu_abi() {
        assert_eq!(
            std::mem::size_of::<LightRow>(),
            std::mem::size_of::<crate::GpuLight>()
        );
        assert_eq!(
            std::mem::align_of::<LightRow>(),
            std::mem::align_of::<crate::GpuLight>()
        );
    }
}
