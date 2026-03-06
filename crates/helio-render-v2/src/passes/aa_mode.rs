//! Anti-aliasing configuration and modes

/// Anti-aliasing modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiAliasingMode {
    /// No anti-aliasing
    None,
    /// Fast Approximate Anti-Aliasing (FXAA)
    Fxaa,
    /// Subpixel Morphological Anti-Aliasing (SMAA)
    Smaa,
    /// Temporal Anti-Aliasing (TAA)
    Taa,
    /// Multisample Anti-Aliasing (MSAA)
    Msaa(MsaaSamples),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsaaSamples {
    X2 = 2,
    X4 = 4,
    X8 = 8,
}

impl MsaaSamples {
    pub fn as_u32(&self) -> u32 {
        *self as u32
    }
}

impl Default for AntiAliasingMode {
    fn default() -> Self {
        Self::None
    }
}
