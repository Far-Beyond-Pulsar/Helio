use crate::material::MAX_TEXTURES;

pub fn required_wgpu_features(adapter_features: wgpu::Features) -> wgpu::Features {
    #[cfg(not(target_arch = "wasm32"))]
    let required =
        wgpu::Features::TEXTURE_BINDING_ARRAY |
        wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    #[cfg(target_arch = "wasm32")]
    let required = wgpu::Features::empty();
    let optional =
        wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | // compacted indirect count buffer
        wgpu::Features::SHADER_PRIMITIVE_INDEX | // @builtin(primitive_index) in fs
        wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS; // GPU profiling timestamps via encoder
    required | (adapter_features & optional)
}

pub fn required_wgpu_limits(adapter_limits: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: MAX_TEXTURES as u32,
        max_samplers_per_shader_stage: MAX_TEXTURES as u32,
        ..adapter_limits
    }
}

/// Global Illumination configuration (AAA dual-tier: RC near, ambient far).
#[derive(Debug, Clone, Copy)]
pub struct GiConfig {
    /// Radiance Cascades volume radius around camera (world units).
    /// GI within this radius uses RC, outside uses cheap ambient fallback.
    /// Default: 80.0 (AAA near-field quality like Unreal Lumen).
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

    pub fn internal_width(&self) -> u32 {
        (((self.width as f32) * self.render_scale).ceil() as u32).max(1)
    }

    pub fn internal_height(&self) -> u32 {
        (((self.height as f32) * self.render_scale).ceil() as u32).max(1)
    }
}
