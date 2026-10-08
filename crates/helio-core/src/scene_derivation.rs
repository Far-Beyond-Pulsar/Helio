//! Scene derivations: GPU work that turns the frontend's authored scene
//! buffers into the derived buffers passes read, before the graph runs.
//!
//! The frontend's SceneDB mirrors authored component values (a mesh
//! instance's geometry ranges, its owner link, the owner's transform, ...).
//! Passes read inputs shaped for drawing (one object row per drawable
//! section, one light row per light). A derivation joins the former into the
//! latter on the GPU, each frame the inputs change, and publishes its
//! outputs into the frame's [`SceneBufferProjection`] under the keys passes
//! already resolve. No CPU iteration over scene entities is involved, and a
//! derivation keeps no CPU copy of the scene.
//!
//! Derivation outputs are execution data: owned by the derivation, rebuilt
//! from their inputs, never written by the frontend. Their
//! [`BufferHandle::content_generation`] moves exactly when the derivation
//! rewrote them, so passes that skip unchanged inputs keep working.

use crate::{BufferHandle, BufferKey, SceneBufferProjection};
use std::sync::Arc;

/// What a derivation sees: the device and queue, and the frame's scene
/// buffers, including the outputs of derivations that ran before it.
pub struct SceneDerivationContext<'a> {
    pub device: &'a Arc<wgpu::Device>,
    pub queue: &'a Arc<wgpu::Queue>,
    pub inputs: &'a SceneBufferProjection,
}

/// The buffers a derivation publishes this frame, and whether it recorded
/// GPU work into the encoder it was given.
#[derive(Default)]
pub struct SceneDerivationOutput {
    pub buffers: Vec<(BufferKey, BufferHandle)>,
    pub recorded: bool,
}

/// GPU work deriving scene buffers from other scene buffers. See the
/// module doc.
pub trait SceneDerivation: Send {
    fn name(&self) -> &'static str;

    /// Record whatever work this frame needs into `encoder` and return the
    /// buffers to publish. Publish them every frame, also when nothing was
    /// recorded: the projection is rebuilt each frame, and a key that is
    /// missing for a frame reads as "no such data" to every pass.
    fn derive(
        &mut self,
        ctx: &SceneDerivationContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> SceneDerivationOutput;
}

/// Run `derivations` in order against `projection`, publishing each one's
/// outputs into it (replacing a frontend buffer under the same key).
/// Returns the command buffer to submit before the graph, if any
/// derivation recorded work.
pub fn run_scene_derivations(
    derivations: &mut [Box<dyn SceneDerivation>],
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    projection: &mut SceneBufferProjection,
) -> Option<wgpu::CommandBuffer> {
    if derivations.is_empty() {
        return None;
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Scene Derivations"),
    });
    let mut recorded = false;
    for derivation in derivations.iter_mut() {
        let output = derivation.derive(
            &SceneDerivationContext {
                device,
                queue,
                inputs: projection,
            },
            &mut encoder,
        );
        recorded |= output.recorded;
        for (key, handle) in output.buffers {
            projection.insert(key, handle);
        }
    }
    recorded.then(|| encoder.finish())
}
