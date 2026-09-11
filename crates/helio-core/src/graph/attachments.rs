//! Symbolic attachment declarations, resolved against the executor's own
//! resource registry (the graph texture pool, plus this frame's swapchain
//! target/depth) instead of each pass wiring up `wgpu::TextureView`s itself.
//!
//! A pass that only needs "the depth buffer, loaded and stored" or "the
//! named transient `pre_aa`, cleared to black" can describe that intention
//! with an [`AttachmentSlot`] and a [`ColorAttachmentIntent`] /
//! [`DepthAttachmentIntent`], and let [`resolve_attachment_view`] look up
//! the physical view at `render_pass_descriptor()` time — the executor
//! still owns `begin_render_pass`/`end_render_pass` exactly as it does
//! today. Passes that need full control keep building
//! `wgpu::RenderPassColorAttachment` by hand as before; this is an additive
//! convenience layered on [`RenderPass::render_pass_descriptor`], not a
//! replacement for it.
//!
//! [`RenderPass::render_pass_descriptor`]: crate::RenderPass::render_pass_descriptor

use super::GraphTexturePool;

/// A symbolic reference to a render-pass attachment. The executor's
/// resource registry (the [`GraphTexturePool`]) resolves this to a
/// physical `wgpu::TextureView` for the *current* frame — the pass never
/// touches a `wgpu::Texture` or the pool's allocation bookkeeping directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttachmentSlot {
    /// The frame's color target — the `target` argument threaded through
    /// `render_pass_descriptor`/`PassContext::target` (the swapchain view,
    /// or whatever the host renderer passed to `RenderGraph::execute`).
    Target,
    /// The frame's depth/stencil target — `PassContext::depth`. Usually a
    /// graph-owned resource named `"depth"`, so this also resolves via the
    /// pool when the caller-supplied `depth` view isn't the one wanted.
    Depth,
    /// A graph-owned named transient (e.g. `"pre_aa"`, `"ssao"`), as
    /// declared via `RenderPass::declare_resources`. Resolves to `None` if
    /// nothing by that name was allocated this frame — a pass folding that
    /// through `?` degrades to "skip this attachment" rather than a panic,
    /// since the executor may legitimately have aliased or elided it.
    Named(&'static str),
}

/// Resolves an [`AttachmentSlot`] to the physical view backing it this
/// frame. `target`/`depth` are exactly the arguments already passed to
/// `render_pass_descriptor`; `pool` is the executor's texture registry
/// (`PassContext::resource_pool`, or threaded through directly when
/// building the descriptor outside `execute()`).
pub fn resolve_attachment_view<'a>(
    slot: AttachmentSlot,
    target: &'a wgpu::TextureView,
    depth: &'a wgpu::TextureView,
    pool: &'a GraphTexturePool,
) -> Option<&'a wgpu::TextureView> {
    match slot {
        AttachmentSlot::Target => Some(target),
        AttachmentSlot::Depth => Some(pool.get_view("depth").unwrap_or(depth)),
        AttachmentSlot::Named(name) => pool.get_view(name),
    }
}

/// Looks up the runtime `wgpu::TextureFormat` backing a slot this frame —
/// used to build the [`PipelineFormatKey`](super::PipelineFormatKey) a pass
/// looks its pipeline up with, so a cached pipeline can never silently drift
/// from the format the executor actually bound.
///
/// `AttachmentSlot::Target` resolves to `None`: the swapchain surface isn't
/// pool-owned, so its format isn't tracked here — pass it in directly from
/// wherever the host renderer keeps its surface configuration (it's usually
/// static per-surface, unlike a resizable/aliased named transient).
pub fn attachment_format(slot: AttachmentSlot, pool: &GraphTexturePool) -> Option<wgpu::TextureFormat> {
    match slot {
        AttachmentSlot::Target => None,
        AttachmentSlot::Depth => pool.get_texture("depth").map(|t| t.format()),
        AttachmentSlot::Named(name) => pool.get_texture(name).map(|t| t.format()),
    }
}

/// Describes one color attachment's resolution and load/store intent,
/// keyed by symbolic [`AttachmentSlot`] rather than a raw view.
#[derive(Debug, Clone, Copy)]
pub struct ColorAttachmentIntent {
    pub slot: AttachmentSlot,
    pub load: wgpu::LoadOp<wgpu::Color>,
    pub store: wgpu::StoreOp,
}

impl ColorAttachmentIntent {
    pub fn new(slot: AttachmentSlot, load: wgpu::LoadOp<wgpu::Color>, store: wgpu::StoreOp) -> Self {
        Self { slot, load, store }
    }

    /// Resolves this intent to a `wgpu::RenderPassColorAttachment`, or
    /// `None` if the slot's backing view isn't available this frame.
    pub fn resolve<'a>(
        &self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        pool: &'a GraphTexturePool,
    ) -> Option<wgpu::RenderPassColorAttachment<'a>> {
        let view = resolve_attachment_view(self.slot, target, depth, pool)?;
        Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: self.load,
                store: self.store,
            },
        })
    }
}

/// Describes the depth/stencil attachment's resolution and load/store
/// intent, keyed by symbolic [`AttachmentSlot`] rather than a raw view.
#[derive(Debug, Clone, Copy)]
pub struct DepthAttachmentIntent {
    pub slot: AttachmentSlot,
    pub depth_load: wgpu::LoadOp<f32>,
    pub depth_store: wgpu::StoreOp,
}

impl DepthAttachmentIntent {
    pub fn new(slot: AttachmentSlot, depth_load: wgpu::LoadOp<f32>, depth_store: wgpu::StoreOp) -> Self {
        Self {
            slot,
            depth_load,
            depth_store,
        }
    }

    pub fn resolve<'a>(
        &self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        pool: &'a GraphTexturePool,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment<'a>> {
        let view = resolve_attachment_view(self.slot, target, depth, pool)?;
        Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: self.depth_load,
                store: self.depth_store,
            }),
            stencil_ops: None,
        })
    }
}
