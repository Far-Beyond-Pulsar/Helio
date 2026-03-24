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

/// Sky state passed to passes that need sky information.
#[derive(Debug, Clone, Copy)]
pub struct SkyContext {
    /// Whether a sky (atmosphere/skybox) is present
    pub has_sky: bool,
    /// Whether sky parameters changed this frame (LUT needs rebuild)
    pub sky_state_changed: bool,
    /// Ambient sky color (approximation for non-sky areas)
    pub sky_color: [f32; 3],
}

impl Default for SkyContext {
    fn default() -> Self {
        Self {
            has_sky: false,
            sky_state_changed: false,
            sky_color: [0.1, 0.1, 0.15],
        }
    }
}

