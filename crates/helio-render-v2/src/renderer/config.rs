//! Renderer configuration.

use crate::features::FeatureRegistry;
use crate::passes::{SsaoConfig, AntiAliasingMode, TaaConfig};

/// Main renderer configuration
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
    pub features: FeatureRegistry,
    pub enable_ssao: bool,
    pub ssao_config: SsaoConfig,
    pub aa_mode: AntiAliasingMode,
    pub taa_config: TaaConfig,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat, features: FeatureRegistry) -> Self {
        Self {
            width,
            height,
            surface_format,
            features,
            enable_ssao: false,
            ssao_config: SsaoConfig::default(),
            aa_mode: AntiAliasingMode::None,
            taa_config: TaaConfig::default(),
        }
    }

    /// Enable SSAO with default settings
    pub fn with_ssao(mut self) -> Self {
        self.enable_ssao = true;
        self
    }

    /// Enable SSAO with custom settings
    pub fn with_ssao_config(mut self, config: SsaoConfig) -> Self {
        self.enable_ssao = true;
        self.ssao_config = config;
        self
    }

    /// Set anti-aliasing mode
    pub fn with_aa(mut self, mode: AntiAliasingMode) -> Self {
        self.aa_mode = mode;
        self
    }

    /// Set TAA config (only used if aa_mode is TAA)
    pub fn with_taa_config(mut self, config: TaaConfig) -> Self {
        self.taa_config = config;
        self
    }
}
