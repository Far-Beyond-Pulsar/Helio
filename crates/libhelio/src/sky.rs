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
    /// When true, clouds have infinite horizontal extent and tile across the entire skydome.
    pub infinite_extent: bool,
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
            infinite_extent: false,
        }
    }
}

impl VolumetricClouds {
    /// Create a new volumetric clouds descriptor with boundless (infinite) extent.
    /// Covers the entire skydome horizon via tiled weather + dome shell.
    pub fn boundless(coverage: f32, density: f32) -> Self {
        Self {
            coverage: coverage.clamp(0.0, 1.0),
            density: density.clamp(0.0, 2.0),
            base: 1200.0,
            top: 2000.0,
            wind_x: 0.8,
            wind_z: 0.2,
            speed: 1.2,
            skylight_intensity: 0.25,
            infinite_extent: true,
        }
    }

    /// Enable infinite horizontal extent (covers entire skydome when enabled).
    pub fn with_infinite_extent(mut self, enabled: bool) -> Self {
        self.infinite_extent = enabled;
        self
    }

    /// Set coverage (0..1).
    pub fn with_coverage(mut self, v: f32) -> Self {
        self.coverage = v.clamp(0.0, 1.0);
        self
    }
    /// Set density (thickness).
    pub fn with_density(mut self, v: f32) -> Self {
        self.density = v.clamp(0.0, 2.0);
        self
    }
    /// Set altitude range [base, top] in world units.
    pub fn with_altitude(mut self, base: f32, top: f32) -> Self {
        self.base = base;
        self.top = top.max(base + 10.0);
        self
    }
    /// Set wind (x,z) and speed multiplier.
    pub fn with_wind(mut self, x: f32, z: f32, speed: f32) -> Self {
        self.wind_x = x;
        self.wind_z = z;
        self.speed = speed;
        self
    }
    /// Enable/disable boundless skydome tiling fluently.
    pub fn boundless_enabled(mut self, enabled: bool) -> Self {
        self.infinite_extent = enabled;
        self
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
    context: SkyContext,
}

impl SkyActor {
    /// Create a sky actor in indoors mode (no sky, custom ambient color).
    pub fn indoor(sky_color: [f32; 3]) -> Self {
        let mut ctx = SkyContext::default();
        ctx.sky_color = sky_color;
        SkyActor { context: ctx }
    }

    /// Create a default sky actor (no sky, default ambient color).
    pub fn new() -> Self {
        SkyActor {
            context: SkyContext::default(),
        }
    }

    /// Enable atmospheric sky rendering with this sky color.
    pub fn with_sky_color(mut self, sky_color: [f32; 3]) -> Self {
        self.context.has_sky = true;
        self.context.sky_state_changed = true;
        self.context.sky_color = sky_color;
        self
    }

    /// Set the ambient fallback color when sky is not present.
    pub fn with_ambient_color(mut self, sky_color: [f32; 3]) -> Self {
        self.context.sky_color = sky_color;
        self
    }

    /// Attach volumetric clouds to this sky actor.
    ///
    /// Automatically enables sky rendering when clouds are added.
    pub fn with_clouds(mut self, clouds: VolumetricClouds) -> Self {
        self.context.clouds = Some(clouds);
        self.context.has_sky = true;
        self.context.sky_state_changed = true;
        self
    }

    /// Clears volumetric clouds from this sky actor.
    pub fn without_clouds(mut self) -> Self {
        self.context.clouds = None;
        self
    }

    /// Get a copy of the internal sky context for renderer use.
    pub fn context(&self) -> SkyContext {
        self.context
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

impl Default for SkyActor {
    fn default() -> Self {
        SkyActor::new()
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
