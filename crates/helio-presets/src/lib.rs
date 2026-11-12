//! Helio's shipped pipelines and what they are made of.
//!
//! - [`Preset`]: the shipped pipelines as node graphs made from a renderer
//!   config ([`presets`]): `deferred`, `forward`, and the pipelines lighting models bring.
//! - [`features`]: the feature list they build from (the spine and every
//!   feature crate's nodes), the renderer's own nodes and the config bridge.
//! - [`graphs`]: node registries, a preset as a laid-out graph asset, the pin
//!   probe.
//!
//! Helio registers no pipeline and has no default one: a host decides what it
//! offers. Pulsar's `pulsar.default_pipelines` plugin makes its `.rendergraph`
//! assets from these presets ([`graphs::graph_asset`]) for the plugins a project
//! enables.
//!
//! The presets name no feature beyond the feature list: each feature declares
//! what it needs of the others (GI's terrain tracer, TSR's stochastic mask)
//! through settings in its own preparer.

pub mod features;
pub mod graphs;
pub mod presets;
pub use presets::{deferred, deferred_fxaa, forward, registry, Preset, PresetGraph};

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
