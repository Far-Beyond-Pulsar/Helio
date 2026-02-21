/// Quality presets for radiance cascades global illumination
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GIQuality {
    /// Low quality: 2 cascades, 64³ probes, 32 rays per probe
    Low,
    /// Medium quality: 3 cascades, 96³ probes, 64 rays per probe (default)
    Medium,
    /// High quality: 4 cascades, 128³ probes, 128 rays per probe
    High,
}

impl GIQuality {
    /// Number of cascades for this quality level
    pub fn cascade_count(&self) -> u32 {
        match self {
            GIQuality::Low => 2,
            GIQuality::Medium => 3,
            GIQuality::High => 4,
        }
    }

    /// Probe grid resolution (number of probes per axis)
    pub fn probe_resolution(&self) -> u32 {
        match self {
            GIQuality::Low => 64,
            GIQuality::Medium => 96,
            GIQuality::High => 128,
        }
    }

    /// Number of rays to shoot per probe during radiance injection
    pub fn rays_per_probe(&self) -> u32 {
        match self {
            GIQuality::Low => 32,
            GIQuality::Medium => 64,
            GIQuality::High => 128,
        }
    }
}

impl Default for GIQuality {
    fn default() -> Self {
        GIQuality::Medium
    }
}

/// Integration mode with existing shadow system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMode {
    /// GI works alongside shadows (both contribute to lighting)
    Hybrid = 0,
    /// GI replaces shadows entirely
    Replacement = 1,
}

impl Default for IntegrationMode {
    fn default() -> Self {
        IntegrationMode::Hybrid
    }
}

/// Configuration builder for radiance cascades GI
#[derive(Debug, Clone)]
pub struct RadianceCascadesConfig {
    pub quality: GIQuality,
    pub integration_mode: IntegrationMode,
    /// 0.0 = fresh injection every frame, 0.9 = 90% history (smoother but lags dynamic lights)
    pub temporal_blend_factor: f32,
    pub cascade_spacing_factor: f32,
    pub enable_2d_distant_cascades: bool,
    /// Overall GI brightness multiplier
    pub gi_intensity: f32,
    /// Minimum ambient radiance added to every probe (prevents fully black unlit areas)
    pub ambient: f32,
}

impl Default for RadianceCascadesConfig {
    fn default() -> Self {
        Self {
            quality: GIQuality::Medium,
            integration_mode: IntegrationMode::Hybrid,
            temporal_blend_factor: 0.0, // fresh each frame; no convergence lag
            cascade_spacing_factor: 2.0,
            enable_2d_distant_cascades: true,
            gi_intensity: 1.0,
            ambient: 0.04,
        }
    }
}

impl RadianceCascadesConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the quality preset
    pub fn with_quality(mut self, quality: GIQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Set the integration mode
    pub fn with_integration_mode(mut self, mode: IntegrationMode) -> Self {
        self.integration_mode = mode;
        self
    }

    /// Set the GI intensity multiplier
    pub fn with_gi_intensity(mut self, intensity: f32) -> Self {
        self.gi_intensity = intensity.max(0.0);
        self
    }

    /// Set the temporal blend factor (0.0-1.0)
    pub fn with_temporal_blend(mut self, blend: f32) -> Self {
        self.temporal_blend_factor = blend.clamp(0.0, 1.0);
        self
    }

    /// Set the cascade spacing factor
    pub fn with_cascade_spacing(mut self, spacing: f32) -> Self {
        self.cascade_spacing_factor = spacing.max(1.0);
        self
    }

    /// Set the ambient radiance floor (prevents pitch-black unlit areas)
    pub fn with_ambient(mut self, ambient: f32) -> Self {
        self.ambient = ambient.max(0.0);
        self
    }
}
