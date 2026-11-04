//! The debug buffer views as a feature.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::BufferViewPass;

/// Depth and normal debug views ([`crate`]).
pub struct BufferViewFeature;

impl Feature for BufferViewFeature {
    fn id(&self) -> &'static str {
        "helio.buffer_view"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(pass_node(
            "buffer_view",
            "Output: Debug Buffer View",
            "Passes: Output",
            "Depth and normal debug views, over the final image. Inert unless one is selected.",
            |build| {
                let pass = BufferViewPass::new(build.env.device, build.env.frame.surface_format);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        ));
    }
}
