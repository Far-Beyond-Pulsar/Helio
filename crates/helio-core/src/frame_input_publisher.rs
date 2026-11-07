//! Frame inputs a feature supplies.
//!
//! A host input (`helio_frame_api::host_input`) is a resource the passes
//! read that no pass writes: it is in the frame's registry before the first
//! pass runs. Most come from the host (the render environment, the camera).
//! Some belong to a kind instead: the materials' texture table is built from
//! the scene's texture store, and the host knows no material. Such a kind's
//! node adds a [`FrameInputPublisher`] to the graph when it builds
//! ([`crate::RenderGraph::add_frame_input_publisher`]); the graph updates
//! every publisher once a frame and publishes what it holds with the
//! passes' own frame inputs. A publisher is not a pass: it records no GPU
//! work and has no place in the schedule.
//!
//! How others do it: Bevy's render-world extraction runs each plugin's
//! extract and prepare systems before the render graph, so a material
//! plugin builds its bind groups there and the graph only reads them; Unity
//! SRP renderer features set global resources from `OnCameraSetup`. Here the
//! same step is a publisher the graph owns, so the renderer facade never
//! learns what it publishes.

use crate::{GpuCameraUniforms, ResourceRegistry, Result};

/// What a publisher may use to refresh itself, once a frame.
pub struct FrameInputContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// This frame's camera.
    pub camera: &'a GpuCameraUniforms,
    pub frame_num: u64,
}

/// Resources a feature puts in the frame's registry before any pass runs.
pub trait FrameInputPublisher: Send {
    /// A name for diagnostics.
    fn name(&self) -> &'static str;

    /// Refresh what [`Self::publish`] hands out (called once a frame, before
    /// any publisher publishes). An error fails the frame, as a pass's does.
    fn update(&mut self, ctx: &FrameInputContext<'_>) -> Result<()>;

    /// Write this frame's inputs, borrowed from `self`.
    fn publish<'a>(&'a self, frame: &mut ResourceRegistry<'a>);
}
