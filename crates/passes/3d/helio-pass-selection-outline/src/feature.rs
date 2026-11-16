//! The selection outline as a feature: it opens the selection mask phase
//! (clears the mask every kind then draws its selected geometry into) and
//! draws the outline over the final image.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
use helio_frame_api::{FrameNode, Phase, Slot};

use crate::{SelectionMaskClearPass, SelectionOutlinePass};

/// The editor's selection outline ([`crate`]).
pub struct SelectionOutlineFeature;

impl Feature for SelectionOutlineFeature {
    fn id(&self) -> &'static str {
        "helio.selection_outline"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.consumes(helio_frame_api::selection::OBJECT_SELECTED);
        registrar.node(
            pass_node(
                "selection_mask_clear",
                "Selection: Clear Mask",
                "Passes: Output",
                "Clears the selection mask the selected objects are drawn into; opens the selection mask phase. Editor only.",
                |build| {
                    build.graph.add_pass(Box::new(SelectionMaskClearPass::default()));
                    Ok(())
                },
            )
            .opens(Phase::SelectionMask, 0),
        );
        registrar.node(
            pass_node(
                "selection_outline",
                "Output: Selection Outline",
                "Passes: Output",
                "Outlines the selected objects over the final image, dimmed where the scene hides them. Editor only.",
                |build| {
                    let pass =
                        SelectionOutlinePass::new(build.env.device, build.env.frame.surface_format);
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            // Under the sprites, the performance overlay and the debug lines
            // (the gizmo), which are drawn over it.
            .in_slot(Slot::Display, 5),
        );
    }
}
