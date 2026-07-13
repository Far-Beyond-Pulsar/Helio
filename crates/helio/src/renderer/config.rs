use crate::material::MAX_TEXTURES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum PerfOverlayMode {
    #[default]
    Disabled = 0,
    PassOverdraw = 1,
    ShaderComplexity = 2,
    TileLightCount = 3,
    PassOutput = 4,
}

pub fn required_wgpu_features(adapter_features: wgpu::Features) -> wgpu::Features {
    #[cfg(not(target_arch = "wasm32"))]
    let required =
        wgpu::Features::TEXTURE_BINDING_ARRAY |
        wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING |
        wgpu::Features::INDIRECT_FIRST_INSTANCE;
    #[cfg(target_arch = "wasm32")]
    let required = wgpu::Features::INDIRECT_FIRST_INSTANCE;
    let optional =
        wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | // compacted indirect count buffer
        wgpu::Features::TIMESTAMP_QUERY | // GPU profiling timestamp queries
        wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS | // GPU profiling timestamps via encoder
        wgpu::Features::VERTEX_WRITABLE_STORAGE;
    required | (adapter_features & optional)
}

#[cfg(test)]
mod tests {
    use super::required_wgpu_features;

    #[test]
    fn indirect_first_instance_is_required_even_when_adapter_does_not_report_it() {
        assert!(
            required_wgpu_features(wgpu::Features::empty())
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
        );
    }

    #[test]
    fn unsupported_optional_features_are_not_requested() {
        let requested = required_wgpu_features(wgpu::Features::empty());
        assert!(!requested.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT));
        assert!(!requested.contains(wgpu::Features::TIMESTAMP_QUERY));
    }
}

pub fn required_wgpu_limits(adapter_limits: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: (MAX_TEXTURES as u32)
            .min(adapter_limits.max_sampled_textures_per_shader_stage),
        max_samplers_per_shader_stage: (MAX_TEXTURES as u32)
            .min(adapter_limits.max_samplers_per_shader_stage),
        ..adapter_limits
    }
}

/// Global Illumination configuration (dual-tier: RC near, ambient far).
#[derive(Debug, Clone, Copy)]
pub struct GiConfig {
    /// Radiance Cascades volume radius around camera (world units).
    /// GI within this radius uses RC, outside uses cheap ambient fallback.
    /// Default: 80.0 (near-field quality like Unreal Lumen).
    pub rc_radius: f32,
    /// Fade margin for smooth RC→ambient transition (world units).
    /// Default: 20.0 (soft blend zone).
    pub rc_fade_margin: f32,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self {
            rc_radius: 80.0,
            rc_fade_margin: 20.0,
        }
    }
}

impl GiConfig {
    pub fn ambient_only() -> Self {
        Self {
            rc_radius: 0.0,
            rc_fade_margin: 0.0,
        }
    }

    pub fn large_radius(radius: f32) -> Self {
        Self {
            rc_radius: radius,
            rc_fade_margin: radius * 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
    pub gi_config: GiConfig,
    pub shadow_quality: libhelio::ShadowQuality,
    pub debug_mode: u32,
    pub render_scale: f32,
    pub perf_overlay_mode: PerfOverlayMode,
    /// Resolution of each shadow atlas face (width × height). Default 1024.
    /// Higher values improve shadow quality at the cost of VRAM (N² scaling).
    pub shadow_atlas_size: u32,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            width,
            height,
            surface_format,
            gi_config: GiConfig::default(),
            shadow_quality: libhelio::ShadowQuality::Medium,
            debug_mode: 0,
            render_scale: 0.75,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            shadow_atlas_size: 1024,
        }
    }

    pub fn with_gi_config(mut self, gi_config: GiConfig) -> Self {
        self.gi_config = gi_config;
        self
    }

    pub fn with_shadow_quality(mut self, quality: libhelio::ShadowQuality) -> Self {
        self.shadow_quality = quality;
        self
    }

    pub fn with_render_scale(mut self, scale: f32) -> Self {
        self.render_scale = scale.clamp(0.25, 1.0);
        self
    }

    pub fn with_perf_overlay_mode(mut self, mode: PerfOverlayMode) -> Self {
        self.perf_overlay_mode = mode;
        self
    }

    pub fn internal_width(&self) -> u32 {
        (((self.width as f32) * self.render_scale).ceil() as u32).max(1)
    }

    pub fn internal_height(&self) -> u32 {
        (((self.height as f32) * self.render_scale).ceil() as u32).max(1)
    }
}
