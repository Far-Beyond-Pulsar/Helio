use crate::material::MAX_TEXTURES;
use helio_mats::BINDLESS_MATERIAL_FEATURES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// G-buffer → deferred lighting (current default)
    #[default]
    Deferred,
    /// Forward-lit pass for all opaque geometry, no G-buffer.
    /// Transparent pass still runs on top. Suitable for VR stereo.
    ForwardOpaque,
    /// Everything forward — both opaque and transparent use forward passes.
    /// Suitable for WebGPU low-end or simple scenes.
    ForwardOnly,
}

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

/// Select the best available surface texture format for the given HDR output mode.
///
/// * `Ldr` — prefers an sRGB format (`Rgba8UnormSrgb` / `Bgra8UnormSrgb`)
/// * `Hdr10` — prefers `Rgba16Float` (PQ encoded in shader; system handles display)
/// * `ScRgb` — prefers `Rgba16Float` (linear float, Windows HDR / macOS EDR)
/// * `Passthrough` — prefers `Rgba16Float` (raw HDR float)
///
/// Falls back to the first available format if no preferred format is found.
pub fn select_hdr_surface_format(
    caps: &wgpu::SurfaceCapabilities,
    mode: helio_pass_postprocess::HdrOutputMode,
) -> wgpu::TextureFormat {
    let preferred = match mode {
        helio_pass_postprocess::HdrOutputMode::Ldr => {
            caps.formats.iter().find(|f| f.is_srgb()).copied()
        }
        helio_pass_postprocess::HdrOutputMode::Hdr10
        | helio_pass_postprocess::HdrOutputMode::ScRgb
        | helio_pass_postprocess::HdrOutputMode::Passthrough => caps
            .formats
            .iter()
            .find(|f| **f == wgpu::TextureFormat::Rgba16Float)
            .copied()
            .or_else(|| {
                caps.formats
                    .iter()
                    .find(|f| **f == wgpu::TextureFormat::Rgba32Float)
                    .copied()
            }),
    };
    preferred.unwrap_or(caps.formats[0])
}

/// Instance flags Helio recommends for the host's `wgpu::Instance`
/// (Helio#308).
///
/// Release builds turn off wgpu's optional validation. Its default keeps
/// `VALIDATION_INDIRECT_CALL` on in release, which validates every indirect
/// draw on the CPU inside `CommandEncoder::finish`; Helio's indirect args
/// all come from its own culling passes. Debug builds keep full validation.
/// `WGPU_*` environment variables override either
/// (`WGPU_VALIDATION_INDIRECT_CALL=1` re-enables indirect validation).
pub fn recommended_instance_flags() -> wgpu::InstanceFlags {
    if cfg!(debug_assertions) {
        wgpu::InstanceFlags::debugging()
    } else {
        wgpu::InstanceFlags::empty()
    }
    .with_env()
}

/// The adapter's features minus those its driver advertises but can't run
/// Helio's pipelines with. Pass this, rather than `adapter.features()`, to
/// [`required_wgpu_features`] and [`required_experimental_features`].
///
/// Software Vulkan (lavapipe / llvmpipe, and CPU adapters in general)
/// reports `EXPERIMENTAL_RAY_QUERY`, but its shader compiler crashes building
/// ray-query pipelines and the device is lost
/// (Pulsar-Native#840). Without the feature the default graph uses its
/// non-ray-traced fallbacks.
pub fn usable_adapter_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let mut features = adapter.features();
    if !ray_queries_usable(&adapter.get_info()) {
        features.remove(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    }
    features
}

/// Whether ray queries on this adapter run Helio's ray-query pipelines.
pub fn ray_queries_usable(info: &wgpu::AdapterInfo) -> bool {
    let software = |text: &str| {
        let text = text.to_ascii_lowercase();
        text.contains("llvmpipe") || text.contains("lavapipe")
    };
    !(info.device_type == wgpu::DeviceType::Cpu || software(&info.name) || software(&info.driver))
}

/// 64-bit integers and 64-bit atomic min/max, for the virtual-geometry software raster.
pub const VIRTUAL_GEOMETRY_INT64_FEATURES: wgpu::Features =
    wgpu::Features::SHADER_INT64.union(wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX);

pub fn required_wgpu_features(adapter_features: wgpu::Features) -> wgpu::Features {
    let required = wgpu::Features::INDIRECT_FIRST_INSTANCE;
    let mut optional = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | // compacted indirect count buffer
        wgpu::Features::TIMESTAMP_QUERY | // GPU profiling timestamp queries
        wgpu::Features::CONSERVATIVE_RASTERIZATION | // optional exact-face visibility accelerator
        wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS | // GPU profiling timestamps via encoder
        wgpu::Features::VERTEX_WRITABLE_STORAGE | wgpu::Features::RG11B10UFLOAT_RENDERABLE |
        wgpu::Features::PIPELINE_CACHE | // persistent driver-validated pipeline cache
        wgpu::Features::SHADER_INT64 | // hardware wide multiplies (voxel planet fixed-point math)
        wgpu::Features::POLYGON_MODE_LINE; // wireframe debug view
                                           // Virtual geometry's software raster uses a 64-bit atomicMax; both features
                                           // are needed together, and without them it falls back to a tiled kernel.
    if adapter_features.contains(VIRTUAL_GEOMETRY_INT64_FEATURES) {
        optional |= VIRTUAL_GEOMETRY_INT64_FEATURES;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if adapter_features.contains(BINDLESS_MATERIAL_FEATURES) {
        optional |= BINDLESS_MATERIAL_FEATURES;
    }
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
pub fn required_experimental_features(
    adapter_features: wgpu::Features,
) -> wgpu::ExperimentalFeatures {
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
    use super::{ray_queries_usable, required_wgpu_features, RendererConfig};
    use helio_mats::BINDLESS_MATERIAL_FEATURES;

    fn adapter(name: &str, driver: &str, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.into(),
            driver: driver.into(),
            device_type,
            ..wgpu::AdapterInfo::new(wgpu::DeviceType::Other, wgpu::Backend::Vulkan)
        }
    }

    #[test]
    fn software_vulkan_gets_no_ray_queries() {
        use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu};
        assert!(!ray_queries_usable(&adapter(
            "llvmpipe (LLVM 20.1.2, 256 bits)",
            "llvmpipe",
            Cpu
        )));
        assert!(!ray_queries_usable(&adapter("lavapipe", "", IntegratedGpu)));
        assert!(!ray_queries_usable(&adapter("Some CPU device", "", Cpu)));
        assert!(ray_queries_usable(&adapter(
            "NVIDIA GeForce RTX 4080",
            "NVIDIA",
            DiscreteGpu
        )));
    }

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
        assert!(!requested.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION));
    }

    #[test]
    fn conservative_rasterization_is_requested_only_when_available() {
        assert!(
            required_wgpu_features(wgpu::Features::CONSERVATIVE_RASTERIZATION)
                .contains(wgpu::Features::CONSERVATIVE_RASTERIZATION)
        );
    }

    #[test]
    fn complete_material_binding_array_capability_is_requested_together() {
        let requested = required_wgpu_features(BINDLESS_MATERIAL_FEATURES);
        assert!(requested.contains(BINDLESS_MATERIAL_FEATURES));
    }

    #[test]
    fn partial_material_binding_array_capability_is_not_requested() {
        let requested = required_wgpu_features(wgpu::Features::TEXTURE_BINDING_ARRAY);
        assert!(!requested.intersects(BINDLESS_MATERIAL_FEATURES));
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
        // Clamped to 4 GiB - 1 because wgpu-core's indirect-draw validation asserts
        // `max_buffer_size <= u32::MAX` while building its helper pipelines, and that
        // assertion runs at *device creation*:
        //
        //   wgpu-core/src/indirect_validation/draw.rs:72
        //   assertion failed: limits.max_buffer_size <= u32::MAX as u64
        //
        // Every other limit here is taken straight from the adapter, and on a GPU
        // reporting more than 4 GiB of addressable buffer that spread is a hard panic
        // before a single frame is drawn. Helio allocates nothing near this ceiling — the
        // largest single buffer in the engine is the foliage blade arena at a few hundred
        // MiB — so lowering it costs nothing real.
        max_buffer_size: adapter_limits.max_buffer_size.min(u32::MAX as u64),
        ..adapter_limits
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    /// The output's format. The graph writes display-linear colour: for SDR
    /// output give an `*Srgb` format (it encodes on write, and so does every
    /// pass drawn after post-processing); a plain UNORM target stores the
    /// linear values, two stops dark in the mid-tones when shown as sRGB.
    pub surface_format: wgpu::TextureFormat,
    /// Colour the default graphs' `Background` pass writes into `pre_aa` before any geometry.
    ///
    /// Defaults to opaque black, matching the retired hard-coded clear. Alpha is preserved by the
    /// FXAA, fog-composite and post-process passes, so `[0.0; 4]` leaves the scene background
    /// transparent for a compositor (GPUI's `paint_surface` blends the dma-buf with alpha).
    pub background_color: [f32; 4],
    pub shadow_quality: helio_pass_shadow_matrix::ShadowQuality,
    pub debug_mode: u32,
    pub render_scale: f32,
    pub perf_overlay_mode: PerfOverlayMode,
    /// Legacy custom-graph setting. Default graphs use `shadow_budget.max_resolution`.
    pub shadow_atlas_size: u32,
    /// Legacy custom-graph setting. Default graphs use packed residency and `shadow_budget`.
    pub shadow_face_capacity: u32,
    /// GPU tile residency, atlas memory, and per-frame rendering limits.
    pub shadow_budget: helio_pass_shadow_matrix::ShadowBudget,
    /// Enable the screen-space reflection pass. Default `false`.
    ///
    /// SSR traces the Hi-Z buffer per pixel and is one of the most expensive
    /// passes in the graph, particularly on Metal. `DeferredLightPass` binds a
    /// 1×1 black fallback when the pass is absent, so disabling it costs only
    /// the reflection contribution.
    pub enable_ssr: bool,
    /// Enable the foliage passes. Default `true`.
    ///
    /// This is the outermost of three independent zero-cost guarantees, and the only one
    /// that removes the passes from the graph entirely. The other two are runtime: with
    /// the passes present but no foliage types registered, `Scene` leaves the `foliage`
    /// frame slot unwritten and both passes early-out; with types registered but the ring
    /// empty, the rasteriser issues four `draw_indirect` calls with zero instances.
    ///
    /// Defaults ON precisely because those runtime guarantees make an unplanted scene free
    /// — a scene with no SceneDB `foliage_types` column pays nothing for this being true.
    pub enable_foliage: bool,
    /// Densest foliage ground in tufts per square metre, or `None` for the default (8).
    /// Sizes the bands' candidate grids at graph-build time: a type's `density` (tufts a
    /// square metre) above this is thinned to it.
    pub foliage_tufts_per_m2: Option<f32>,
    /// Blades a tuft averages, or `None` for the default (8): sizes the near band's
    /// slabs (a record per blade).
    pub foliage_blades_per_tuft: Option<f32>,
    /// Foliage quality preset: the default range, the density multiplier, the near LOD
    /// distance's scale and the cull granularity. Default [`FoliageQuality::Medium`].
    ///
    /// [`FoliageQuality::Medium`]: helio_pass_foliage_place::FoliageQuality::Medium
    pub foliage_quality: helio_pass_foliage_place::FoliageQuality,
    /// Metres foliage reaches, or `None` for the quality preset's. Bands of tiles
    /// (`helio_pass_foliage_place::plan_bands`) carry it out there at a cost that grows
    /// with the number of bands, one per doubling of the range.
    pub foliage_range: Option<f32>,
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
    /// When `Some`, the graph builder replaces `FxaaPass` (and the old bilinear
    /// TAA upscale) with a full [`TsrPass`](helio_pass_tsr::TsrPass).  The preset
    /// controls the neighbourhood tap count and temporal accumulation rate.
    ///
    /// Setting this automatically adjusts the recommended `render_scale`:
    /// call [`with_tsr_quality`](Self::with_tsr_quality) instead of setting
    /// this field directly so that `render_scale` is kept consistent.
    ///
    /// `None` (default) keeps the existing FXAA-only path.
    pub tsr_quality: Option<helio_pass_tsr::TsrQuality>,

    /// HDR display output mode. Default `Ldr`.
    pub hdr_output_mode: helio_pass_postprocess::HdrOutputMode,
    pub render_mode: RenderMode,
    /// Enable the experimental multiview render-graph path. This is a low-level
    /// graph setting and does not discover or start OpenXR. For application
    /// level automatic XR startup with desktop fallback, use
    /// [`HelioGraphicsConfig::vr_support`](crate::HelioGraphicsConfig::vr_support)
    /// before creating the scene GPU mirror and renderer. Default `false`.
    pub enable_xr: bool,

    /// Enable the portal duplicate-draw passes (`PortalCullPass`/`PortalInstancePass`).
    /// Default `true`.
    ///
    /// Sublevels have no separate pass to gate — a sublevel-tagged instance is
    /// drawn through the *existing* GBuffer/shadow pipeline (see
    /// `helio_pass_object_batch::coordinate_space`), so there is nothing here for a scene
    /// with no sublevels to pay for. Portals are the one part of this
    /// mechanism with real fixed GPU allocations (~10 MB, see
    /// `helio-pass-portal-cull`'s module docs) and a per-frame dispatch, so
    /// unlike sublevels they get an actual off switch: a scene that never
    /// authors no SceneDB `portal_views` column can skip paying for it by setting this `false`.
    pub enable_portals: bool,

    /// The world-space GI probes' finest spacing (metres; each coarser
    /// level doubles it). `None` (default): the graph's choice, 1 m over
    /// voxel terrain and 0.5 m in mesh scenes, whose rooms, pillars and
    /// furniture are a few metres apart.
    pub gi_probe_spacing: Option<f32>,
    /// Opaque id of the virtual-geometry runtime (`helio_vg_stream::VgRuntime::handle_id`)
    /// the default deferred graph builds its virtual-geometry passes for, or
    /// `None` (the default) for classic rendering only. The passes are added
    /// only when the device supports them (`helio_default_passes::virtual_geometry_supported`);
    /// otherwise the graph is exactly the classic one.
    pub virtual_geometry: Option<u64>,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            surface_format,
            background_color: [0.0, 0.0, 0.0, 1.0],
            shadow_quality: helio_pass_shadow_matrix::ShadowQuality::Medium,
            debug_mode: 0,
            render_scale: 0.75,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            shadow_atlas_size: 1024,
            shadow_face_capacity: 32,
            shadow_budget: helio_pass_shadow_matrix::ShadowBudget::default(),
            enable_ssr: false,
            enable_foliage: true,
            foliage_tufts_per_m2: None,
            foliage_blades_per_tuft: None,
            foliage_quality: helio_pass_foliage_place::FoliageQuality::Medium,
            foliage_range: None,
            enable_planar_reflections: false,
            enable_environment_reflections: true,
            tsr_quality: None,
            hdr_output_mode: helio_pass_postprocess::HdrOutputMode::Ldr,
            render_mode: RenderMode::Deferred,
            enable_xr: false,
            enable_portals: true,
            gi_probe_spacing: None,
            virtual_geometry: None,
        }
    }

    /// Build the default deferred graph's virtual-geometry passes for `runtime`
    /// (when the device supports them).
    pub fn with_virtual_geometry(mut self, runtime: &helio_vg_stream::VgRuntime) -> Self {
        self.virtual_geometry = Some(runtime.handle_id());
        // `VG_DEBUG_VIEW` selects a virtual-geometry debug view at creation (tests,
        // headless captures); an explicit `debug_mode` is left alone.
        if self.debug_mode == 0 {
            if let Some(view) = helio_vg_format::debug_view::from_env() {
                self.debug_mode = view.id();
            }
        }
        self
    }

    /// Set the default graphs' scene clear colour. Use alpha 0 for a transparent background.
    pub fn with_background_color(mut self, color: [f32; 4]) -> Self {
        self.background_color = color;
        self
    }

    /// Enable/disable environment-cubemap indirect specular.
    pub fn with_environment_reflections(mut self, enabled: bool) -> Self {
        self.enable_environment_reflections = enabled;
        self
    }

    /// The GI probes' finest spacing (metres; [`Self::gi_probe_spacing`]).
    pub fn with_gi_probe_spacing(mut self, metres: f32) -> Self {
        self.gi_probe_spacing = Some(metres);
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

    pub fn with_shadow_quality(mut self, quality: helio_pass_shadow_matrix::ShadowQuality) -> Self {
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

    pub fn with_render_mode(mut self, mode: RenderMode) -> Self {
        self.render_mode = mode;
        self
    }

    pub fn with_hdr_output_mode(mut self, mode: helio_pass_postprocess::HdrOutputMode) -> Self {
        self.hdr_output_mode = mode;
        self
    }

    pub fn with_shadow_budget(mut self, budget: helio_pass_shadow_matrix::ShadowBudget) -> Self {
        self.shadow_budget = budget;
        self
    }

    /// Legacy custom-graph setting; use `with_shadow_budget` for default graphs.
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
        self.tsr_quality = Some(quality);
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
