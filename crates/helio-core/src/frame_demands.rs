//! Resources a frame's consumers need produced, declared before any pass of
//! the frame executes.
//!
//! Passes run in graph order, so a producer (HiZ) executes before the
//! consumers whose needs depend on this frame's scene (WaterSim reflects only
//! while water exists). Before executing anything, the graph calls every
//! pass's [`RenderPass::declare_frame_demands`](crate::RenderPass::declare_frame_demands)
//! and publishes the result under [`FRAME_DEMANDS`]; producers then skip
//! optional outputs nobody asked for.

use crate::{ResourceKey, ResourceRegistry};

/// Registry key of the frame's `&FrameDemands`.
pub const FRAME_DEMANDS: &str = "frame_demands";

/// Names of optional resources some pass needs this frame.
#[derive(Debug, Default)]
pub struct FrameDemands {
    names: Vec<&'static str>,
}

impl FrameDemands {
    /// Ask for `name` to be produced this frame.
    pub fn demand(&mut self, name: &'static str) {
        if !self.contains(name) {
            self.names.push(name);
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|&demanded| demanded == name)
    }

    pub fn clear(&mut self) {
        self.names.clear();
    }
}

/// Whether `name` must be produced this frame: true when some pass demanded
/// it, and also when the graph published no demands at all (a host or path
/// that does not collect them), so producers keep their previous behaviour.
pub fn is_demanded(registry: &ResourceRegistry<'_>, name: &str) -> bool {
    registry
        .get::<&FrameDemands>(ResourceKey::new(FRAME_DEMANDS))
        .is_none_or(|demands| demands.contains(name))
}
