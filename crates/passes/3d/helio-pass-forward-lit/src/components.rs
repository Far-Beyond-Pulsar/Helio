//! SceneDB-owned light primitive.
//!
//! Mirrors `helio-pass-corona::CoronaEmitterComponent`'s recipe exactly: the
//! pass that consumes a kind of scene content owns that content's schema.
//! SceneDB owns the packed CPU column and its GPU projection; this crate
//! never holds a second, renderer-local copy of "what lights exist."
//!
//! Unlike corona (which still needs a CPU-supplied count each frame),
//! [`LightComponent`]'s buffer is read with a *fixed* iteration count
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
//! What this does NOT yet cover: `Scene::flush()`'s CPU-side shadow-atlas
//! importance scoring (`self.lights`/`self.gpu_scene.lights.0`) is a
//! separate, pre-existing, unconditionally-per-frame CPU loop that predates
//! SceneDB integration entirely and is out of scope here — tracked as a
//! Helio issue. Lights sourced through this buffer do not participate in
//! dynamic shadow-caster selection; author them with `shadow_index =
//! u32::MAX` until that system is made SceneDB-aware.

use pulsar_scenedb_derive::SceneStore;

/// Fixed capacity of the `"scene_lights"` buffer. Passes/frontends binding
/// this buffer iterate exactly this many rows, always — see the module doc
/// for why that needs no per-frame count at all.
///
/// Matches `World`'s own auto-registration capacity exactly: `LightComponent`
/// registers no buffer up front (see the module doc's `#[gpu(layout =
/// packed)]` auto-registration) — the first `World::insert` of a
/// `LightComponent` registers `"scene_lights"` at this same capacity.
pub const MAX_LIGHTS: u32 = pulsar_scenedb::gpu::world_mirror::DEFAULT_AUTO_REGISTER_CAPACITY;

/// A placed light, authored as a SceneDB component. Field-for-field
/// identical to [`libhelio::GpuLight`] (enforced by the size/align test
/// below) so conversion is a zero-cost `bytemuck::cast`, matching
/// `CoronaEmitterComponent`'s relationship to `GpuCoronaEmitter`.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "scene_lights")]
pub struct LightComponent {
    #[gpu]
    pub position_range: [f32; 4],
    #[gpu]
    pub direction_outer: [f32; 4],
    #[gpu]
    pub color_intensity: [f32; 4],
    #[gpu]
    pub shadow_index: u32,
    #[gpu]
    pub light_type: u32,
    #[gpu]
    pub inner_angle: f32,
    #[gpu]
    pub _pad: u32,
    #[gpu]
    pub god_rays_enabled: u32,
    #[gpu]
    pub god_rays_density: f32,
    #[gpu]
    pub god_rays_weight: f32,
    #[gpu]
    pub god_rays_decay: f32,
    #[gpu]
    pub god_rays_exposure: f32,
    #[gpu]
    pub flare_enabled: u32,
    #[gpu]
    pub flare_type: u32,
    #[gpu]
    pub flare_intensity: f32,
    #[gpu]
    pub flare_scale: f32,
    #[gpu]
    pub flare_tint_r: f32,
    #[gpu]
    pub flare_tint_g: f32,
    #[gpu]
    pub flare_tint_b: f32,
    #[gpu]
    pub ies_profile_index: i32,
    #[gpu]
    pub light_function_index: i32,
    #[gpu]
    pub ies_angle_scale: f32,
    #[gpu]
    pub ies_angle_offset: f32,
}

impl From<libhelio::GpuLight> for LightComponent {
    fn from(value: libhelio::GpuLight) -> Self {
        bytemuck::cast(value)
    }
}

impl From<LightComponent> for libhelio::GpuLight {
    fn from(value: LightComponent) -> Self {
        bytemuck::cast(value)
    }
}

#[cfg(test)]
mod tests {
    use super::LightComponent;

    #[test]
    fn scene_record_matches_light_gpu_abi() {
        assert_eq!(
            std::mem::size_of::<LightComponent>(),
            std::mem::size_of::<libhelio::GpuLight>()
        );
        assert_eq!(
            std::mem::align_of::<LightComponent>(),
            std::mem::align_of::<libhelio::GpuLight>()
        );
    }
}
