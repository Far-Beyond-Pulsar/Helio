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
    let required = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    #[cfg(target_arch = "wasm32")]
    let required = wgpu::Features::INDIRECT_FIRST_INSTANCE;
    let mut optional = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | // compacted indirect count buffer
        wgpu::Features::TIMESTAMP_QUERY | // GPU profiling timestamp queries
        wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS | // GPU profiling timestamps via encoder
        wgpu::Features::VERTEX_WRITABLE_STORAGE;
    // Request ray tracing if available (native only, requires Vulkan)
    #[cfg(not(target_arch = "wasm32"))]
    {
        if adapter_features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            optional |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        }
    }
    required | (adapter_features & optional)
}

/// Acknowledgement token for the experimental features [`required_wgpu_features`] asks for.
///
/// wgpu gates every `EXPERIMENTAL_*` feature behind this second opt-in: naming one in
/// `required_features` without also passing an enabled token fails device creation with
/// `ExperimentalFeaturesNotEnabled`, no matter what the adapter supports. The two must
/// therefore travel together, which is why this is derived from `required_wgpu_features`
/// rather than hardcoded — it stays correct if that function starts requesting a
/// different experimental feature.
///
/// Pass it alongside the features:
///
/// ```ignore
/// adapter.request_device(&wgpu::DeviceDescriptor {
///     required_features: required_wgpu_features(adapter.features()),
///     required_limits: required_wgpu_limits(adapter.limits()),
///     experimental_features: required_experimental_features(adapter.features()),
///     ..Default::default()
/// })
/// ```
///
/// Returns a disabled token when nothing experimental is requested, so a device that
/// does not need them is not opted in.
pub fn required_experimental_features(adapter_features: wgpu::Features) -> wgpu::ExperimentalFeatures {
    let requested = required_wgpu_features(adapter_features);
    if requested.intersects(wgpu::Features::all_experimental_mask()) {
        // SAFETY: wgpu asks callers to acknowledge that experimental features may
        // contain soundness bugs reachable from otherwise-safe code, and to report
        // any found. The only experimental feature requested here is
        // EXPERIMENTAL_RAY_QUERY, and only when the adapter reports support for it.
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::{required_wgpu_features, RendererConfig};

    #[test]
    fn indirect_first_instance_is_required_even_when_adapter_does_not_report_it() {
        assert!(required_wgpu_features(wgpu::Features::empty())
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE));
    }

    #[test]
    fn unsupported_optional_features_are_not_requested() {
        let requested = required_wgpu_features(wgpu::Features::empty());
        assert!(!requested.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT));
        assert!(!requested.contains(wgpu::Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn renderer_config_never_exposes_an_empty_extent() {
        let config = RendererConfig::new(0, 0, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!((config.width, config.height), (1, 1));
        assert_eq!((config.internal_width(), config.internal_height()), (1, 1));
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
    /// Maximum number of allocated shadow-map array layers. Each realtime light
    /// reserves six consecutive faces. A capacity of 32 supports five lights
    /// while keeping the two 1024px browser atlases to 256 MiB total.
    pub shadow_face_capacity: u32,
    /// Enable the screen-space reflection pass. Default `false`.
    ///
    /// SSR traces the Hi-Z buffer per pixel and is one of the most expensive
    /// passes in the graph, particularly on Metal. `DeferredLightPass` binds a
    /// 1×1 black fallback when the pass is absent, so disabling it costs only
    /// the reflection contribution.
    pub enable_ssr: bool,
    /// Enable the planar reflection pass. Default `false`.
    ///
    /// Planar reflections re-render the scene mirrored across each reflection
    /// plane, so cost scales with scene complexity times plane count. As with
    /// SSR, `DeferredLightPass` falls back to a 1×1 black texture.
    pub enable_planar_reflections: bool,
    /// Enable environment-cubemap indirect specular. Default `true`.
    ///
    /// This is the *base* reflection layer in deferred lighting — SSR and planar
    /// reflections only composite on top of it, so gating those two does not
    /// remove reflections from the image; this flag does.
    ///
    /// Defaults ON because turning it off is not a mild change: metals have no
    /// diffuse term at all (`kD = (1 - F)(1 - metallic)` is zero at
    /// `metallic = 1`), so the cubemap is their *only* light source and they go
    /// pure black without it. It is also a poor performance lever — one cubemap
    /// sample per pixel, versus SSR's per-pixel Hi-Z trace.
    pub enable_environment_reflections: bool,

    /// Temporal Super-Resolution quality preset.
    ///
    /// When `Some`, the graph builder replaces the simple bilinear upscale in
    /// `TaaPass` with a full [`TsrPass`](helio_pass_tsr::TsrPass).  The preset
    /// controls the neighbourhood tap count and temporal accumulation rate.
    ///
    /// Setting this automatically adjusts the recommended `render_scale`:
    /// call [`with_tsr_quality`](Self::with_tsr_quality) instead of setting
    /// this field directly so that `render_scale` is kept consistent.
    ///
    /// `None` (default) keeps the existing TAA-only path.
    pub tsr_quality: Option<helio_pass_tsr::TsrQuality>,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            surface_format,
            gi_config: GiConfig::default(),
            shadow_quality: libhelio::ShadowQuality::Medium,
            debug_mode: 0,
            render_scale: 0.75,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            shadow_atlas_size: 1024,
            shadow_face_capacity: 32,
            enable_ssr: false,
            enable_planar_reflections: false,
            enable_environment_reflections: true,
            tsr_quality: None,
        }
    }

    /// Enable/disable environment-cubemap indirect specular.
    pub fn with_environment_reflections(mut self, enabled: bool) -> Self {
        self.enable_environment_reflections = enabled;
        self
    }

    /// Enable/disable the screen-space reflection pass.
    pub fn with_ssr(mut self, enabled: bool) -> Self {
        self.enable_ssr = enabled;
        self
    }

    /// Enable/disable the planar reflection pass.
    pub fn with_planar_reflections(mut self, enabled: bool) -> Self {
        self.enable_planar_reflections = enabled;
        self
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

    pub fn with_shadow_face_capacity(mut self, capacity: u32) -> Self {
        self.shadow_face_capacity = capacity.clamp(1, 256);
        self
    }

    /// Enable Temporal Super-Resolution with the given quality preset.
    ///
    /// This sets [`tsr_quality`](Self::tsr_quality) and automatically adjusts
    /// [`render_scale`](Self::render_scale) to the preset's recommended value.
    ///
    /// When TSR is enabled the graph builder inserts a [`TsrPass`](helio_pass_tsr::TsrPass)
    /// in place of the simple bilinear upscale, and FXAA / SMAA are disabled
    /// (TSR's temporal accumulation provides superior anti-aliasing).
    ///
    /// ## Example
    /// ```rust,ignore
    /// let config = RendererConfig::new(1920, 1080, surface_format)
    ///     .with_tsr_quality(TsrQuality::Quality);
    /// ```
    pub fn with_tsr_quality(mut self, quality: helio_pass_tsr::TsrQuality) -> Self {
        self.render_scale = quality.render_scale();
        self.tsr_quality  = Some(quality);
        self
    }

    /// Disable Temporal Super-Resolution, reverting to the TAA-only upscale path.
    pub fn without_tsr(mut self) -> Self {
        self.tsr_quality = None;
        self
    }

    pub fn internal_width(&self) -> u32 {
        (((self.width as f32) * self.render_scale).ceil() as u32).max(1)
    }

    pub fn internal_height(&self) -> u32 {
        (((self.height as f32) * self.render_scale).ceil() as u32).max(1)
    }
}
