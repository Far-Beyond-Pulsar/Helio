use helio::{PassBuildContext, PassGraphBuilderFn};
use helio_default_graphs::{
    build_default_graph_external_with_context, build_default_graph_with_context,
    build_forward_only_graph_with_context, build_forward_opaque_graph_with_context,
    build_fxaa_graph_with_context, build_fxaa_hlfs_graph_with_context,
    build_hlfs_graph_with_context, build_simple_graph_with_context,
};

// These assignments are compile-time ABI coverage: every default graph entry
// point must remain directly usable with RendererBuilder::with_pass_build_context.
#[test]
fn default_graph_builders_match_pass_build_context_abi() {
    let builders: [PassGraphBuilderFn; 8] = [
        Box::new(build_default_graph_with_context),
        Box::new(build_default_graph_external_with_context),
        Box::new(build_fxaa_graph_with_context),
        Box::new(build_hlfs_graph_with_context),
        Box::new(build_fxaa_hlfs_graph_with_context),
        Box::new(build_forward_opaque_graph_with_context),
        Box::new(build_forward_only_graph_with_context),
        Box::new(build_simple_graph_with_context),
    ];
    let _ = builders;
}

#[allow(dead_code)]
fn context_is_the_public_shape(_: PassBuildContext<'_>) {}
