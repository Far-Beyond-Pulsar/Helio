//! Helio's shipped pipelines and what they are made of.
//!
//! - [`Preset`]: the shipped pipelines as node graphs made from a renderer
//!   config ([`presets`]): `deferred`, `hlfs`, `forward`, `simple`, ….
//! - [`features`]: the feature list they build from (the spine and every
//!   feature crate's nodes), the renderer's own nodes and the config bridge.
//! - [`graphs`]: node registries, the editor's template, the pin probe.
//! - [`pipelines`]: the presets as named pipelines for a host's registry.
//! - Config helpers: [`with_virtual_geometry`], [`config_with_host_settings`].
//!
//! The presets name no feature beyond the feature list: each feature declares
//! what it needs of the others (GI's terrain tracer, TSR's stochastic mask)
//! through settings in its own preparer.

pub mod features;
pub mod graphs;
pub mod pipelines;
pub mod presets;
pub mod virtual_geometry;
pub use helio_pass_vg::{
    virtual_geometry_supported, VgInputsPass, VgStreamBeginPass, VgStreamEndPass,
};
pub use helio_vg_format::debug_view as vg_debug_view;
pub use presets::{
    deferred, deferred_fxaa, forward, hlfs, hlfs_fxaa, registry, simple, Preset, PresetGraph,
};
pub use virtual_geometry::with_virtual_geometry;

use helio::RendererConfig;
use helio_pass_billboard::BillboardPass;
use helio_pass_corona::CoronaPass;
use helio_pass_debug_overlay::DebugOverlayPass;
use helio_pass_decal::DecalPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_flare::LensFlarePass;
use helio_pass_foliage_gbuffer::FoliageGBufferPass;
use helio_pass_foliage_place::FoliagePlacePass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_gi::{GiMeshTracePass, GiPass, GiResolvePass};
use helio_pass_hiz::HiZBuildPass;
use helio_pass_hlfs::HlfsPass;
use helio_pass_indirect_dispatch::IndirectDispatchPass;
use helio_pass_lens::LensPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_object_batch::ObjectBatchPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_perf_overlay::{
    PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayPass,
};
use helio_pass_planar_reflection::PlanarReflectionPass;
use helio_pass_portal_cull::PortalCullPass;
use helio_pass_portal_instances::{PortalEditorOverlayPass, PortalInstancePass, PortalMaskPass};
use helio_pass_postprocess::{FogCompositePass, PostProcessPass, PostProcessVolumeBlendPass};
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_dirty::ShadowDirtyPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_sky::{AtmosphereCompositePass, AtmospherePass, CloudTracePass};
use helio_pass_sprite_batch::SpriteBatchPass;
use helio_pass_sprite_cull::SpriteCullPass;
use helio_pass_ssr::SsrPass;
use helio_pass_tsr::TsrPass;
use helio_pass_volumetric_fog::VolumetricFogPass;
use helio_pass_water_sim::WaterSimPass;
use helio_spine::BackgroundPass;

/// Which of this crate's passes shader hot reload may swap on their own
/// (see [`helio_core::SwapPolicy`] for what a swap does and why a policy is
/// needed).
///
/// Derived by auditing every node constructor (the spine's and each
/// feature's) for handles the nodes create and share between passes. Those passes form a group and
/// are swapped together:
///
/// * shadows: `ShadowMatrixPass` + `ShadowDirtyPass` (dirty-flag buffer),
///   `ShadowDirtyPass` + `ShadowPass` (face dirty and geometry count buffers)
/// * `HiZBuildPass` + `OcclusionCullPass` (Hi-Z sampler)
/// * `PortalCullPass` + `PortalInstancePass` (portal output buffers)
/// * `FoliagePlacePass` + `FoliageGBufferPass` (blade arena, tile table,
///   visible blades, indirect buffer)
/// * every perf-overlay pass (one shared `PerfOverlayShared`)
/// * `SpriteCullPass` + `SpriteBatchPass` (draw order and indirect buffers)
///
/// Everything else listed as independent takes only renderer-owned handles
/// (camera, debug camera and cull-stats buffers, debug-draw state, SceneDB,
/// the debug-overlay state) and plain configuration; those are the same objects
/// in the live and the replacement graph. `GBufferPass`, `ForwardLitPass`
/// and `TransparentPass` are left out on purpose: they build part of their
/// WGSL at runtime and keep material/template registrations, so edits to them
/// rebuild the whole graph. So are passes an engine's features add, which
/// this crate cannot vouch for.
///
/// Dropping the replacement's unswapped passes has no side effects beyond the
/// ones a resize already has: none of these constructors spawns a thread,
/// registers into a global or shared registry, or writes a file, and the
/// replacement graph's pipeline cache has no persistence path.
pub(crate) fn default_swap_policy() -> helio_core::SwapPolicy {
    use helio_core::graph::type_name_of as name;
    helio_core::SwapPolicy::new()
        .independent::<ObjectBatchPass>()
        .independent::<IndirectDispatchPass>()
        .independent::<BackgroundPass>()
        .independent::<AtmospherePass>()
        .independent::<GiPass>()
        .independent::<GiMeshTracePass>()
        .independent::<GiResolvePass>()
        .independent::<CloudTracePass>()
        .independent::<AtmosphereCompositePass>()
        .independent::<LightCullPass>()
        .independent::<DecalPass>()
        .independent::<SsrPass>()
        .independent::<helio_pass_ssr::SsrCompositePass>()
        .independent::<PlanarReflectionPass>()
        .independent::<DeferredLightPass>()
        .independent::<HlfsPass>()
        .independent::<PortalMaskPass>()
        .independent::<PortalEditorOverlayPass>()
        .independent::<BillboardPass>()
        .independent::<CoronaPass>()
        .independent::<WaterSimPass>()
        .independent::<helio::DebugDrawPass>()
        .independent::<DebugOverlayPass>()
        .independent::<PostProcessVolumeBlendPass>()
        .independent::<VolumetricFogPass>()
        .independent::<FogCompositePass>()
        .independent::<TsrPass>()
        .independent::<FxaaPass>()
        .independent::<LensPass>()
        .independent::<LensFlarePass>()
        .independent::<PostProcessPass>()
        .group_types(&[
            name::<ShadowMatrixPass>(),
            name::<ShadowDirtyPass>(),
            name::<ShadowPass>(),
        ])
        .group_types(&[name::<HiZBuildPass>(), name::<OcclusionCullPass>()])
        .group_types(&[name::<PortalCullPass>(), name::<PortalInstancePass>()])
        .group_types(&[name::<FoliagePlacePass>(), name::<FoliageGBufferPass>()])
        .group_types(&[
            name::<PerfOverlayAnalyzerPass>(),
            name::<PerfOverlayCostAnalyzerPass>(),
            name::<PerfOverlayPass>(),
        ])
        .group_types(&[name::<SpriteCullPass>(), name::<SpriteBatchPass>()])
}

pub(crate) fn convert_perf_mode(
    mode: helio::PerfOverlayMode,
) -> helio_pass_perf_overlay::PerfOverlayMode {
    use helio::PerfOverlayMode as H;
    use helio_pass_perf_overlay::PerfOverlayMode as P;
    match mode {
        H::Disabled => P::Disabled,
        H::PassOverdraw => P::PassOverdraw,
        H::ShaderComplexity => P::ShaderComplexity,
        H::TileLightCount => P::TileLightCount,
        H::PassOutput => P::PassOutput,
    }
}

/// A renderer config with a host's [`helio::helio_settings::RENDERER`]
/// values: the renderer's own, and the default features' settings
/// (foliage). Engines build their config with this rather than
/// `RendererConfig::with_settings`, which knows no feature.
pub fn config_with_host_settings(
    config: RendererConfig,
    settings: &helio::helio_settings::Resolved,
) -> RendererConfig {
    let mut config = config.with_settings(settings);
    let foliage = config
        .features
        .get_or_default::<helio_pass_foliage_place::FoliageSettings>()
        .with_host_settings(settings);
    config.features.insert(foliage);
    config
        .features
        .edit::<helio_pass_gi::GiSettings>()
        .probe_spacing = settings.optional("gi_probe_spacing").map(|m| m as f32);
    config
}
