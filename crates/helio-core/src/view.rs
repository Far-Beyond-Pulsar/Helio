//! Views: the cameras a frame renders.
//!
//! A frame renders its **primary** view (the host's camera, or an XR eye)
//! and the **captures** its features ask for: a capture renders into the
//! requester's own target, lit (a planar reflection; UE's SceneCaptures and
//! planar reflections, HDRP's probe cameras). What a rigid map can show (a
//! portal, a sublevel) is not a view: it is drawn into the frame's own
//! G-buffer as more occurrences of each drawer's content, through the frame's
//! coordinate spaces and the portals' chains.
//!
//! A pass asks for views at a pass boundary ([`crate::RenderPass::renders_views`]):
//! the graph submits what it recorded so far (so a view reads this frame's
//! shadows, object batches and probes), then calls the pass's
//! [`crate::RenderPass::render_views`] with a [`ViewContext`]. Each request runs
//! a **view graph**: the frame's own members for that kind of view, made by
//! the graph's builder ([`ViewGraphSource`]; the preset composes it as it
//! composes the primary graph), retained per request key so its passes,
//! pipelines and history survive frames. A view graph starts from the
//! requester's resources (the host's inputs and what once-per-frame passes
//! published) and allocates only what it writes itself, except what the
//! request seeds (its colour target as the scene colour).
//!
//! A capture keeps no history of the frame's (no temporal resolve runs in
//! one, `helio_presets`), and the frame's exposure maps what it captured: a
//! capture adapts no exposure of its own.
//!
//! Every drawer draws every view it is built in with its own program: it
//! reads the view's facts from the [`ViewDescription`] its graph was built
//! for (winding, clip) and the [`GpuCameraUniforms`] of the view (camera,
//! clip plane).

use std::sync::{Arc, Mutex};

use crate::GpuCameraUniforms;

/// What kind of view a graph renders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ViewKind {
    /// The host's camera: the frame's own graph.
    #[default]
    Primary,
    /// Rendered into the requester's target, lit (scene prep through the
    /// lighting).
    Capture,
}

/// What a view is, as the graph that renders it was built for. A setting of
/// every node's build environment (`helio_frame_api::view::view_of`), so a
/// drawer bakes the view's winding and clip into its pipelines once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ViewDescription {
    pub kind: ViewKind,
    /// The view's camera reverses handedness (a planar reflection): front
    /// faces wind clockwise.
    pub mirrored: bool,
    /// The view's camera carries a clip plane (`GpuCameraUniforms::clip_plane`)
    /// drawers must honour.
    pub clipped: bool,
    /// How many views deep the view is (the primary is 0).
    pub depth: u32,
}

impl ViewDescription {
    pub const PRIMARY: Self = Self {
        kind: ViewKind::Primary,
        mirrored: false,
        clipped: false,
        depth: 0,
    };

    pub fn is_primary(&self) -> bool {
        self.kind == ViewKind::Primary
    }

    /// The front face a drawer's pipelines use for this view.
    pub fn front_face(&self) -> wgpu::FrontFace {
        if self.mirrored {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        }
    }

    /// The flags of [`GpuCameraUniforms::view_info`] `y`.
    pub fn flags(&self) -> u32 {
        (self.mirrored as u32) * VIEW_MIRRORED | (self.clipped as u32) * VIEW_CLIPPED
    }
}

/// `view_info.y`: the view is mirrored (the prelude's `HELIO_VIEW_MIRRORED`).
pub const VIEW_MIRRORED: u32 = 1;
/// `view_info.y`: the view has a clip plane (`HELIO_VIEW_CLIPPED`).
pub const VIEW_CLIPPED: u32 = 2;

/// How many captures a frame may render ([`ViewContext::render_view`]
/// refuses past it; the requester shows its fallback). A quality setting of
/// the host (`helio_settings`), defaulting to [`ViewBudget::default`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewBudget {
    /// The deepest capture: 1 lets the primary view's reflections render,
    /// 2 a reflection seen in a reflection.
    pub max_depth: u32,
    /// Secondary views per frame, all depths together.
    pub max_views: u32,
}

impl Default for ViewBudget {
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_views: 16,
        }
    }
}

/// One capture a pass asks for ([`ViewContext::render_view`]).
pub struct ViewRequest<'r> {
    /// Stable per requester and capture (a reflection's plane): the view
    /// graph is retained under it.
    pub key: u64,
    /// The view's camera reverses handedness.
    pub mirrored: bool,
    /// The view's camera, clip plane included (`clip_plane` nonzero makes
    /// the view clipped).
    pub camera: GpuCameraUniforms,
    pub width: u32,
    pub height: u32,
    /// The capture's lit colour target.
    pub target: &'r wgpu::TextureView,
    /// The capture's depth.
    pub depth: &'r wgpu::TextureView,
    /// The texture behind [`Self::depth`] (what copies it: the view's Hi-Z).
    pub depth_texture: &'r wgpu::Texture,
    /// Texture views the view graph takes under a name instead of allocating
    /// them (the colour target as the frame's scene colour).
    pub seeds: &'r [(&'static str, &'r wgpu::TextureView)],
}

/// What a view graph is built for ([`ViewGraphSource::build`]).
pub struct ViewBuild<'b> {
    pub device: &'b Arc<wgpu::Device>,
    pub queue: &'b Arc<wgpu::Queue>,
    pub description: ViewDescription,
    pub width: u32,
    pub height: u32,
    /// The view's own camera buffer (two cameras, as the primary's).
    pub camera_buffer: &'b wgpu::Buffer,
    /// The frame's statistics, so a view's draws count into the frame's.
    pub stats: &'b crate::FrameStats,
}

/// Builds the graph a kind of view runs: the graph builder's (the preset
/// composes it from the frame's members). Returns the graph unlocked; the
/// requester.s graph imports what the request seeds and locks it.
pub trait ViewGraphSource: Send + Sync {
    fn build(&self, build: &ViewBuild<'_>) -> Result<crate::RenderGraph, String>;
}

/// The frame's captures, shared by every graph that renders in the frame:
/// the budget and what was spent of it.
pub struct ViewFrame {
    pub(crate) budget: ViewBudget,
    pub(crate) frame: u64,
    pub(crate) views_rendered: u32,
}

impl ViewFrame {
    pub(crate) fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            budget: ViewBudget::default(),
            frame: u64::MAX,
            views_rendered: 0,
        }))
    }

    /// Starts frame `frame` (once, whichever graph sees it first).
    pub(crate) fn begin(&mut self, frame: u64) {
        if self.frame != frame {
            self.frame = frame;
            self.views_rendered = 0;
        }
    }
}

/// What [`crate::RenderPass::render_views`] gets: the requesting view's frame
/// and an encoder for the pass's own work around its captures (a
/// reflection's mips after it). Work recorded before a [`Self::render_view`] is
/// submitted ahead of the view; what follows the last view is submitted
/// when the call returns.
pub struct ViewContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// The requesting view's camera, frame and resources.
    pub camera: &'a wgpu::Buffer,
    pub camera_data: &'a GpuCameraUniforms,
    pub world_origin: glam::DVec3,
    pub scene_buffers: &'a crate::SceneBufferProjection,
    pub registry: &'a crate::ResourceRegistry<'a>,
    pub frame_num: u64,
    pub width: u32,
    pub height: u32,
    /// The requesting view's target and depth.
    pub target: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    /// What the requesting view is.
    pub description: ViewDescription,
    pub(crate) encoder: Option<wgpu::CommandEncoder>,
    pub(crate) runner: &'a mut dyn ViewRunner,
}

impl ViewContext<'_> {
    /// An encoder for the pass's own work around its captures.
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        let device = self.device;
        self.encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("View requests"),
            })
        })
    }

    /// Whether the frame's budget allows one more view below this one.
    pub fn can_render(&self) -> bool {
        self.runner.can_render(&self.description)
    }

    /// Renders `request` now: submits the work recorded so far, runs the
    /// view's graph (its own submission) and returns whether it ran. `false`
    /// when the frame's budget ([`ViewBudget`]) is spent or the view would be
    /// too deep: the requester shows its fallback.
    pub fn render_view(&mut self, request: &ViewRequest<'_>) -> crate::Result<bool> {
        if !self.can_render() {
            return Ok(false);
        }
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit([encoder.finish()]);
        }
        let parent = ViewParent {
            queue: self.queue,
            world_origin: self.world_origin,
            scene_buffers: self.scene_buffers,
            registry: self.registry,
            frame_num: self.frame_num,
            description: self.description,
        };
        self.runner.run(&parent, request)
    }

    pub(crate) fn finish(mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit([encoder.finish()]);
        }
    }
}

/// The requesting view, as a view runner sees it.
pub(crate) struct ViewParent<'a> {
    pub queue: &'a wgpu::Queue,
    pub world_origin: glam::DVec3,
    pub scene_buffers: &'a crate::SceneBufferProjection,
    pub registry: &'a crate::ResourceRegistry<'a>,
    pub frame_num: u64,
    pub description: ViewDescription,
}

/// The graph that runs a pass's view requests.
pub(crate) trait ViewRunner {
    fn can_render(&self, requester: &ViewDescription) -> bool;
    fn run(&mut self, parent: &ViewParent<'_>, request: &ViewRequest<'_>) -> crate::Result<bool>;
}
