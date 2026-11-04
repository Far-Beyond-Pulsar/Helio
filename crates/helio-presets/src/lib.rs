//! Helio's shipped pipelines and what they are made of.
//!
//! - [`Preset`]: the shipped pipelines as node graphs made from a renderer
//!   config ([`presets`]): `deferred`, `hlfs`, `forward`, `simple`, ….
//! - [`features`]: the feature list they build from (the spine and every
//!   feature crate's nodes), the renderer's own nodes and the config bridge.
//! - [`graphs`]: node registries, the editor's template, the pin probe.
//! - [`pipelines`]: the presets as named pipelines for a host's registry.
//!
//! The presets name no feature beyond the feature list: each feature declares
//! what it needs of the others (GI's terrain tracer, TSR's stochastic mask)
//! through settings in its own preparer.

pub mod features;
pub mod graphs;
pub mod pipelines;
pub mod presets;
pub use presets::{
    deferred, deferred_fxaa, forward, hlfs, hlfs_fxaa, registry, simple, Preset, PresetGraph,
};

use helio_pass_billboard::BillboardPass;
use helio_pass_debug_overlay::DebugOverlayPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_hlfs::HlfsPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_perf_overlay::{
    PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayPass,
};
use helio_pass_planar_reflection::PlanarReflectionPass;
use helio_pass_postprocess::PostProcessPass;
use helio_pass_ssr::SsrPass;
use helio_pass_tsr::TsrPass;
use helio_spine::BackgroundPass;

/// Which of this crate's passes shader hot reload may swap on their own
/// (see [`helio_core::SwapPolicy`] for what a swap does and why a policy is
/// needed).
///
/// Derived by auditing every node constructor (the spine's and each
/// feature's) for handles the nodes create and share between passes. Those passes form a group and
/// are swapped together:
///
/// * every perf-overlay pass (one shared `PerfOverlayShared`)
///
/// Everything else listed as independent takes only renderer-owned handles
/// (camera, debug camera and cull-stats buffers, debug-draw state, SceneDB,
/// the debug-overlay state) and plain configuration; those are the same objects
/// in the live and the replacement graph. Left out, so an edit to them
/// rebuilds the whole graph: the shadow views' `ShadowMatrixPass` (its light
/// dirty flags are shared with what casts into the views) and the Hi-Z
/// pyramid (its sampler is shared with an object kind's occlusion culling),
/// and every pass an engine's features add (an object kind's batches,
/// culling and drawers), which this crate cannot vouch for.
///
/// Dropping the replacement's unswapped passes has no side effects beyond the
/// ones a resize already has: none of these constructors spawns a thread,
/// registers into a global or shared registry, or writes a file, and the
/// replacement graph's pipeline cache has no persistence path.
pub(crate) fn default_swap_policy() -> helio_core::SwapPolicy {
    use helio_core::graph::type_name_of as name;
    helio_core::SwapPolicy::new()
        .independent::<BackgroundPass>()
        .independent::<helio_spine::UnlitPass>()
        .independent::<LightCullPass>()
        .independent::<SsrPass>()
        .independent::<helio_pass_ssr::SsrCompositePass>()
        .independent::<PlanarReflectionPass>()
        .independent::<DeferredLightPass>()
        .independent::<HlfsPass>()
        .independent::<BillboardPass>()
        .independent::<helio::DebugDrawPass>()
        .independent::<DebugOverlayPass>()
        .independent::<TsrPass>()
        .independent::<FxaaPass>()
        .independent::<PostProcessPass>()
        .group_types(&[
            name::<PerfOverlayAnalyzerPass>(),
            name::<PerfOverlayCostAnalyzerPass>(),
            name::<PerfOverlayPass>(),
        ])
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
