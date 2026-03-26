//! Sky and atmosphere types.

use bytemuck::{Pod, Zeroable};

/// Per-frame sky uniforms. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SkyUniforms {
    /// Sun direction (xyz, normalized world space) + sun angular radius (w)
    pub sun_direction: [f32; 4],
    /// Sun irradiance (xyz linear) + exposure (w)
    pub sun_color: [f32; 4],
    /// Rayleigh scattering coefficient (xyz) + Mie asymmetry (w)
    pub rayleigh_mie: [f32; 4],
}

/// Volumetric cloud properties. This struct is provided by a "volumetric_clouds" actor.
#[derive(Debug, Clone, Copy)]
pub struct VolumetricClouds {
    /// Global cover amount (0.0..1.0)
    pub coverage: f32,
    /// Cloud density (thickness)
    pub density: f32,
    /// Base altitude (world units)
    pub base: f32,
    /// Top altitude (world units)
    pub top: f32,
    /// Horizontal wind in world X direction
    pub wind_x: f32,
    /// Horizontal wind in world Z direction
    pub wind_z: f32,
    /// Global wind speed multiplier
    pub speed: f32,
    /// Skylight intensity contribution from clouds
    pub skylight_intensity: f32,
}

impl Default for VolumetricClouds {
    fn default() -> Self {
        Self {
            coverage: 0.0,
            density: 0.0,
            base: 0.0,
            top: 0.0,
            wind_x: 0.0,
            wind_z: 0.0,
            speed: 0.0,
            skylight_intensity: 0.0,
        }
    }
}

/// Sky state passed to passes that need sky information.
#[derive(Debug, Clone, Copy)]
pub struct SkyContext {
    /// Whether a sky (atmosphere/skybox) is present
    pub has_sky: bool,
    /// Whether sky parameters changed this frame (LUT needs rebuild)
    pub sky_state_changed: bool,
    /// Ambient sky color (approximation for non-sky areas)
    pub sky_color: [f32; 3],
    /// Optional volumetric cloud properties from a `volumetric_clouds` actor.
    pub clouds: Option<VolumetricClouds>,
}

/// Scene actor representing a sky system (atmospheric sky + optional clouds).
#[derive(Debug, Clone, Copy)]
pub struct SkyActor {
    pub context: SkyContext,
}

impl SkyActor {
    pub fn new(context: SkyContext) -> Self {
        Self { context }
    }
}

impl Default for SkyContext {
    fn default() -> Self {
        Self {
            has_sky: false,
            sky_state_changed: false,
            sky_color: [0.1, 0.1, 0.15],
            clouds: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sky_context_default_is_no_sky() {
        let ctx = SkyContext::default();
        assert!(!ctx.has_sky);
        assert!(ctx.clouds.is_none());
    }

    #[test]
    fn sky_context_has_sky_flag_respected() {
        let ctx = SkyContext {
            has_sky: true,
            sky_state_changed: true,
            sky_color: [0.3, 0.4, 0.5],
            clouds: None,
        };
        assert!(ctx.has_sky);
        assert_eq!(ctx.sky_color, [0.3, 0.4, 0.5]);
    }
}

