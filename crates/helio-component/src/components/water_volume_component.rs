//! Water volume component (Phase D, Pulsar-Native#558) — heightfield-sim
//! water rendering (waves, reflections, caustics, underwater fog).
//!
//! The component authors a `size` centred on its owner rather than world
//! AABB corners, so the owner's transform stays the only placement. It
//! derives a `WaterVolumeSourceRow` in the owner's space
//! (`environment_rows`); the renderer's environment join places it with the
//! owner's transform and packs placed volumes into the leading rows of
//! `"water_volumes"`, which the water simulation, surface and caustics read
//! (Pulsar-Native#1035, Phase 4).
//!
//! The water pass simulates with pass-wide dynamics (`WaterSimPass`'s wave
//! spring, damping, scale and wind setters). The per-volume `wave_spring`,
//! `wave_damping`, `wave_scale` and wind fields are carried in the row but
//! the simulation does not read them.
//!
//! Not covered here: `WaterHitboxDescriptor`. Read its own doc before
//! assuming it belongs alongside this component — it explicitly records an
//! object's *previous* AABB and *current* AABB from frame to frame (`old_min`/
//! `old_max`/`new_min`/`new_max`) to drive splash displacement. That's
//! per-frame runtime state computed from something else's movement, not
//! data a level designer authors once and places — it needs to be driven by
//! physics/movement code reacting to objects entering a water volume, which
//! is a different kind of integration than "one component, one purpose"
//! placement. Deferred, not overlooked.

use engine_class_derive::{engine_class, register_world_component};
use helio_pass_water_sim::GpuWaterVolume;
use pulsar_reflection::{ComponentRuntimeBehavior, ComponentRuntimeContext, RuntimeComponentOwner};
use serde::{Deserialize, Serialize};

pub const WATER_VOLUME_CLASS_NAME: &str = "WaterVolumeComponent";

/// Heightfield-simulation water volume, centered on the owning object.
#[engine_class(category = "Rendering", clone, debug, serialize, deserialize)]
#[category("Bounds", category_color = "#3AA0FF")]
#[category("Waves", category_color = "#3AA0FF")]
#[category("Wind", category_color = "#3AA0FF")]
#[category("Visual", category_color = "#2FA88A")]
#[category("Reflection", category_color = "#2FA88A")]
#[category("Caustics", category_color = "#C79A3E")]
#[category("Underwater", category_color = "#7C6FD1")]
#[category("Shadow", category_color = "#7C6FD1")]
#[category("Lighting", category_color = "#D18F6F")]
pub struct WaterVolumeComponent {
    #[property]
    pub enabled: bool,

    // ── Bounds ──────────────────────────────────────────────────────────
    /// Full AABB extent (not half-extent) in each axis, centered on the
    /// owning object's position.
    #[property(category = "Bounds")]
    pub size: [f32; 3],
    /// Water surface height, as a Y offset from the owning object's
    /// position (0 = surface sits exactly at the object's own Y).
    #[property(category = "Bounds")]
    pub surface_height_offset: f32,

    // ── Waves ───────────────────────────────────────────────────────────
    /// Peak wave displacement in metres, above and below the rest height.
    /// Clamped by the shader to the headroom within `size`, so waves can
    /// never leave the volume.
    #[property(min = 0.0, max = 20.0, step = 0.05, category = "Waves")]
    pub wave_amplitude: f32,
    #[property(min = 0.0, max = 10.0, step = 0.01, category = "Waves")]
    pub wave_frequency: f32,
    #[property(min = 0.0, max = 20.0, step = 0.05, category = "Waves")]
    pub wave_speed: f32,
    // `[f32; 2]` doesn't implement `Reflectable` (only `[f32; 3]`/`[f32; 4]`
    // do, per pulsar_reflection's built-in prims) -- split into two plain
    // f32 fields rather than reaching for a 3-element array with an unused
    // component, which would be a wrong shape for what this actually is.
    /// Primary wave direction X (XZ plane). Need not be normalized together
    /// with `wave_direction_z`.
    #[property(category = "Waves")]
    pub wave_direction_x: f32,
    /// Primary wave direction Z (XZ plane).
    #[property(category = "Waves")]
    pub wave_direction_z: f32,
    /// 0.0 = sine wave, 1.0 = sharp peaks.
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Waves")]
    pub wave_steepness: f32,
    /// Restoring-force spring constant toward the mean height. ~1.0 feels
    /// fluid; ~2.0 feels jelly-like.
    #[property(min = 0.5, max = 2.0, step = 0.01, category = "Waves")]
    pub wave_spring: f32,
    /// Per-step energy damping (0..1). Closer to 1.0 = waves linger; closer
    /// to 0.9 = waves die quickly.
    #[property(min = 0.0, max = 1.0, step = 0.001, category = "Waves")]
    pub wave_damping: f32,
    /// Spatial scale of gust impulses on the heightfield. 1.0 = default;
    /// 0.25 = fine ripples; 2.0 = large swells.
    #[property(min = 0.05, max = 10.0, step = 0.05, category = "Waves")]
    pub wave_scale: f32,

    // ── Wind ────────────────────────────────────────────────────────────
    /// Wind direction X, world XZ space. `[0, 0]` (both X and Z) = calm
    /// water. Same `[f32; 2]`-isn't-`Reflectable` reasoning as
    /// `wave_direction_x`/`_z` above.
    #[property(category = "Wind")]
    pub wind_direction_x: f32,
    /// Wind direction Z, world XZ space.
    #[property(category = "Wind")]
    pub wind_direction_z: f32,
    /// 0 = calm, ~1 = gentle ripples, ~5 = choppy.
    #[property(min = 0.0, max = 10.0, step = 0.05, category = "Wind")]
    pub wind_strength: f32,

    // ── Visual ──────────────────────────────────────────────────────────
    /// Base water color (deep water).
    #[property(category = "Visual")]
    pub water_color: [f32; 3],
    /// RGB absorption per meter depth (Beer-Lambert).
    #[property(category = "Visual")]
    pub extinction: [f32; 3],
    /// Wave steepness threshold to spawn foam.
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Visual")]
    pub foam_threshold: f32,
    #[property(min = 0.0, max = 5.0, step = 0.01, category = "Visual")]
    pub foam_amount: f32,

    // ── Reflection / refraction ─────────────────────────────────────────
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Reflection")]
    pub reflection_strength: f32,
    /// 1.0 is physically plausible; 0.0 disables distortion.
    #[property(min = 0.0, max = 2.0, step = 0.01, category = "Reflection")]
    pub refraction_strength: f32,
    /// Higher = sharper Fresnel falloff.
    #[property(min = 0.1, max = 20.0, step = 0.1, category = "Reflection")]
    pub fresnel_power: f32,
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Reflection")]
    pub fresnel_min: f32,
    /// Index of refraction. 1.333 for water.
    #[property(min = 1.0, max = 2.5, step = 0.001, category = "Reflection")]
    pub ior: f32,
    #[property(category = "Reflection")]
    pub ssr_enabled: bool,
    // `u32` doesn't implement `Reflectable` (only `i32`/`i64`/`u64` do,
    // per pulsar_reflection's built-in prims) -- `i32` here, clamped to
    // non-negative when building the descriptor.
    #[property(min = 1, max = 256, category = "Reflection")]
    pub ssr_steps: i32,
    #[property(min = 0.001, max = 1.0, step = 0.001, category = "Reflection")]
    pub ssr_step_size: f32,
    #[property(min = 0.001, max = 1.0, step = 0.001, category = "Reflection")]
    pub ssr_thickness: f32,

    // ── Caustics ────────────────────────────────────────────────────────
    #[property(category = "Caustics")]
    pub caustics_enabled: bool,
    #[property(min = 0.0, max = 10.0, step = 0.05, category = "Caustics")]
    pub caustics_intensity: f32,
    #[property(min = 0.1, max = 50.0, step = 0.1, category = "Caustics")]
    pub caustics_scale: f32,
    #[property(min = 0.0, max = 10.0, step = 0.05, category = "Caustics")]
    pub caustics_speed: f32,

    // ── Underwater ──────────────────────────────────────────────────────
    #[property(min = 0.0, max = 1.0, step = 0.001, category = "Underwater")]
    pub fog_density: f32,
    #[property(min = 0.0, max = 10.0, step = 0.05, category = "Underwater")]
    pub god_rays_intensity: f32,
    /// Effective water density for fog.
    #[property(min = 0.0, max = 1.0, step = 0.001, category = "Underwater")]
    pub density: f32,

    // ── Shadow ──────────────────────────────────────────────────────────
    /// Rim light intensity for pool walls.
    #[property(min = 0.0, max = 5.0, step = 0.05, category = "Shadow")]
    pub shadow_rim: f32,
    /// 0.0 = no hitbox shadow, 1.0 = full shadow under a hitbox.
    #[property(min = 0.0, max = 1.0, step = 0.01, category = "Shadow")]
    pub shadow_hitbox: f32,
    #[property(min = 0.0, max = 5.0, step = 0.05, category = "Shadow")]
    pub shadow_ao: f32,

    // ── Lighting ────────────────────────────────────────────────────────
    /// Sun / dominant directional light direction, world space. Need not be
    /// normalized.
    #[property(category = "Lighting")]
    pub sun_direction: [f32; 3],
}

impl Default for WaterVolumeComponent {
    /// Mirrors `WaterVolumeDescriptor::ocean()`'s values.
    fn default() -> Self {
        Self {
            enabled: true,
            size: [200.0, 60.0, 200.0],
            surface_height_offset: 0.0,
            wave_amplitude: 0.5,
            wave_frequency: 0.3,
            wave_speed: 1.5,
            wave_direction_x: 1.0,
            wave_direction_z: 0.0,
            wave_steepness: 0.5,
            wave_spring: 1.2,
            wave_damping: 0.985,
            wave_scale: 1.0,
            wind_direction_x: 0.0,
            wind_direction_z: 0.0,
            wind_strength: 0.0,
            water_color: [0.0, 0.2, 0.4],
            extinction: [0.1, 0.05, 0.02],
            foam_threshold: 0.8,
            foam_amount: 0.6,
            reflection_strength: 0.8,
            refraction_strength: 1.0,
            fresnel_power: 5.0,
            fresnel_min: 0.1,
            ior: 1.333,
            ssr_enabled: true,
            ssr_steps: 32,
            ssr_step_size: 0.05,
            ssr_thickness: 0.02,
            caustics_enabled: true,
            caustics_intensity: 1.5,
            caustics_scale: 5.0,
            caustics_speed: 0.5,
            fog_density: 0.03,
            god_rays_intensity: 1.0,
            density: 0.03,
            shadow_rim: 1.0,
            shadow_hitbox: 0.0,
            shadow_ao: 1.0,
            sun_direction: [0.5, 1.0, 0.5],
        }
    }
}

impl WaterVolumeComponent {
    /// The water volume row for an unrotated, unscaled owner at the origin.
    /// The renderer places it with the owner's transform.
    pub fn local_gpu(&self) -> GpuWaterVolume {
        let [sx, sy, sz] = self.size;
        GpuWaterVolume {
            bounds_min: [-sx * 0.5, -sy * 0.5, -sz * 0.5, 0.0],
            bounds_max: [sx * 0.5, sy * 0.5, sz * 0.5, self.surface_height_offset],
            wave_params: [
                self.wave_amplitude,
                self.wave_frequency,
                self.wave_speed,
                self.wave_steepness,
            ],
            wave_direction: [self.wave_direction_x, self.wave_direction_z, 0.0, 0.0],
            water_color: [
                self.water_color[0],
                self.water_color[1],
                self.water_color[2],
                self.foam_threshold,
            ],
            extinction: [
                self.extinction[0],
                self.extinction[1],
                self.extinction[2],
                self.foam_amount,
            ],
            reflection_refraction: [
                self.reflection_strength,
                self.refraction_strength,
                self.fresnel_power,
                0.0,
            ],
            caustics_params: [
                self.caustics_enabled as u32 as f32,
                self.caustics_intensity,
                self.caustics_scale,
                self.caustics_speed,
            ],
            fog_params: [self.fog_density, self.god_rays_intensity, 0.0, 0.0],
            sim_params: [
                self.ior,
                self.caustics_intensity,
                self.fresnel_min,
                self.density,
            ],
            shadow_params: [self.shadow_rim, self.shadow_hitbox, self.shadow_ao, 0.0],
            sun_direction: [
                self.sun_direction[0],
                self.sun_direction[1],
                self.sun_direction[2],
                0.0,
            ],
            ssr_params: [
                self.ssr_enabled as u32 as f32,
                self.ssr_steps.max(0) as f32,
                self.ssr_step_size,
                self.ssr_thickness,
            ],
            sim_dynamics: [self.wave_spring, self.wave_damping, 0.0, 0.0],
            wind_params: [
                self.wind_direction_x,
                self.wind_direction_z,
                self.wind_strength,
                0.0,
            ],
            _pad6: [0.0; 4],
        }
    }
}

#[register_world_component]
impl ComponentRuntimeBehavior for WaterVolumeComponent {
    const CLASS_NAME: &'static str = WATER_VOLUME_CLASS_NAME;

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // The component reaches the water passes through its derived
        // source row (`environment_rows`); there is nothing to sync.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_behavior_has_the_reflected_component_name() {
        assert_eq!(
            <WaterVolumeComponent as ComponentRuntimeBehavior>::CLASS_NAME,
            WATER_VOLUME_CLASS_NAME
        );
    }

    #[test]
    fn the_local_row_is_centred_on_the_owner() {
        let component = WaterVolumeComponent {
            size: [10.0, 4.0, 10.0],
            surface_height_offset: 1.0,
            ..Default::default()
        };

        let row = component.local_gpu();

        assert_eq!(&row.bounds_min[0..3], &[-5.0, -2.0, -5.0]);
        assert_eq!(&row.bounds_max[0..3], &[5.0, 2.0, 5.0]);
        assert_eq!(row.bounds_max[3], 1.0);
    }
}
