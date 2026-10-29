//! Which passes a frame runs: the graph follows the scene.
//!
//! A graph is built from every pass a pipeline can use, but a scene uses few
//! of them: a voxel planet has no water simulation, portals, decals or sprites,
//! and nothing asks for the performance overlay. Each pass says whether it has
//! work ([`RenderPass::is_member`](crate::RenderPass::is_member)); the graph
//! asks every pass once per frame and schedules only the members. A pass that
//! leaves is kept (its configuration and persistent state stay) but is not
//! prepared, executed or given graph textures until it joins again, and it
//! allocates its content resources only while it is a member.
//!
//! Scene content is SceneDB rows. Hosts register component columns up front,
//! so a buffer's presence says nothing; [`SceneContent::holds_rows`] reads the
//! rows back whenever SceneDB reports new contents and answers whether any row
//! is live. One reader per buffer key serves every pass that asks (the four
//! portal passes share `portal_views`).

use std::collections::HashMap;

use crate::{BufferKey, SceneBufferLiveness, SceneBufferProjection};

/// The graph's two resolutions. `internal` is what the scene renders at (and
/// `PassContext::width`/`height`); `output` is what the frame is presented
/// at. They differ when temporal upscaling renders below the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderSize {
    pub internal: [u32; 2],
    pub output: [u32; 2],
}

impl RenderSize {
    /// Both resolutions `width` x `height`.
    pub fn uniform(width: u32, height: u32) -> Self {
        let size = [width.max(1), height.max(1)];
        Self {
            internal: size,
            output: size,
        }
    }

    pub fn new(internal: [u32; 2], output: [u32; 2]) -> Self {
        Self {
            internal: internal.map(|v| v.max(1)),
            output: output.map(|v| v.max(1)),
        }
    }
}

/// Whether a SceneDB buffer holds live rows, answered without waiting.
#[derive(Default)]
struct RowReader {
    liveness: SceneBufferLiveness,
    /// The frame the reader was last updated in.
    updated: Option<u64>,
}

/// What a pass may ask about the scene when deciding whether it is a member
/// of the frame (see [`RenderPass::is_member`](crate::RenderPass::is_member)).
pub struct SceneContent<'a> {
    device: Option<&'a wgpu::Device>,
    queue: Option<&'a wgpu::Queue>,
    scene_buffers: Option<&'a SceneBufferProjection>,
    readers: &'a mut HashMap<&'static str, RowReader>,
    frame: u64,
}

impl SceneContent<'_> {
    /// Whether SceneDB buffer `key` holds any live row (any non-zero byte;
    /// removed rows are zeroed).
    ///
    /// Answers from the latest readback: a buffer whose contents changed is
    /// read again in the background and keeps its previous answer until the
    /// new one arrives (a frame or two), and a buffer not yet read counts as
    /// empty. So content that appears in an empty scene joins its passes a
    /// frame or two later, and a pass with content never blinks out while its
    /// rows are re-read. False while the scene has no such buffer, and before
    /// the first frame (no scene yet).
    pub fn holds_rows(&mut self, key: &'static str) -> bool {
        let (Some(device), Some(queue), Some(buffers)) =
            (self.device, self.queue, self.scene_buffers)
        else {
            return false;
        };
        let Some(handle) = buffers.get(BufferKey::of(key)) else {
            return false;
        };
        let reader = self.readers.entry(key).or_default();
        if reader.updated != Some(self.frame) {
            reader.updated = Some(self.frame);
            reader.liveness.update(device, queue, Some(handle));
        }
        reader
            .liveness
            .answer(handle)
            .or_else(|| reader.liveness.last_answer())
            .unwrap_or(false)
    }

    /// The scene's GPU buffers, when there is a scene.
    pub fn scene_buffers(&self) -> Option<&SceneBufferProjection> {
        self.scene_buffers
    }
}

/// The graph-owned readers [`SceneContent`] answers from.
#[derive(Default)]
pub(crate) struct SceneRows {
    readers: HashMap<&'static str, RowReader>,
}

impl SceneRows {
    /// A view of the scene for frame `frame`; `None` buffers (graph lock,
    /// before any frame) make every content question answer "none".
    pub(crate) fn content<'a>(
        &'a mut self,
        device: Option<&'a wgpu::Device>,
        queue: Option<&'a wgpu::Queue>,
        scene_buffers: Option<&'a SceneBufferProjection>,
        frame: u64,
    ) -> SceneContent<'a> {
        SceneContent {
            device,
            queue,
            scene_buffers,
            readers: &mut self.readers,
            frame,
        }
    }
}
