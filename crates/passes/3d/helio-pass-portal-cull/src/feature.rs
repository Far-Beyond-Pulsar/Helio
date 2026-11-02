//! Portal culling as a feature: the node the portal draws read their buffers
//! from (`helio_pass_portal_instances::PortalFeature`).

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};

use crate::PortalCullPass;

/// Portal culling ([`crate`]).
pub struct PortalCullFeature;

impl Feature for PortalCullFeature {
    fn id(&self) -> &'static str {
        "helio.portal_cull"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.node(
            pass_node(
                "portal_cull",
                "Portals: Cull",
                "Passes: Scene Prep",
                "Culls portals and compacts the ones to draw. Portal Instance reads its buffers.",
                |build| {
                    let pass = PortalCullPass::new(build.env.device);
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_marker_after(),
        );
    }
}
