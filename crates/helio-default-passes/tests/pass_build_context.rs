use helio::{PassBuildContext, PassGraphBuilderFn};

// These assignments are compile-time ABI coverage: every default graph entry
// point must remain directly usable with RendererBuilder::with_pass_build_context.
#[test]
fn default_graph_builders_match_pass_build_context_abi() {
    let builders: [PassGraphBuilderFn; 8] = [
        Box::new(helio_default_passes::presets::deferred),
        Box::new(helio_default_passes::presets::deferred),
        Box::new(helio_default_passes::presets::deferred_fxaa),
        Box::new(helio_default_passes::presets::hlfs),
        Box::new(helio_default_passes::presets::hlfs_fxaa),
        Box::new(helio_default_passes::presets::forward),
        Box::new(helio_default_passes::presets::forward),
        Box::new(helio_default_passes::presets::simple),
    ];
    let _ = builders;
}

#[allow(dead_code)]
fn context_is_the_public_shape(_: PassBuildContext<'_>) {}
