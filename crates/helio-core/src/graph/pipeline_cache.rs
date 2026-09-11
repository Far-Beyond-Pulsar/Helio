//! Dynamic-rendering pipeline cache keyed by runtime attachment formats.
//!
//! A pass's `wgpu::RenderPipeline` bakes in the color/depth formats of the
//! attachments it will be drawn into. Under the executor's dynamic-rendering
//! model those formats aren't fixed at pass-construction time — the executor
//! can retarget, resize, alias, or reformat a named transient between
//! frames (quality-setting changes, HDR toggles, editor/game-mode target
//! swaps) without the pass re-initializing. Rebuilding a pipeline on every
//! such change is fine; rebuilding it every *frame* is not.
//!
//! [`PipelineFormatCache`] closes that gap: a pass asks for "the pipeline
//! for these formats" every frame, and only pays for a real
//! `Device::create_render_pipeline` on the first frame a given format
//! combination is seen. One cache lives on [`RenderGraph`](super::RenderGraph)
//! and is threaded into every [`PassContext`](crate::PassContext) as
//! `ctx.pipeline_cache`, so passes never need their own per-format map.
//!
//! Uses a `RefCell`, not a `Mutex`: `RenderGraph::execute` drives every pass
//! sequentially on a single thread, so cache access is never contended —
//! only bookkeeping across frames matters, and a `RefCell`'s borrow check
//! costs nothing an atomic lock would. This matches helio-core's stated
//! "zero locks in the render path" guarantee (see the crate-level docs).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Identifies one pipeline variant: which pass owns it, which attachment
/// formats it must match, and an opaque discriminator for passes that keep
/// more than one pipeline per format set (e.g. opaque vs. alpha-blend, or a
/// MSAA sample count).
///
/// Built via [`attachment_format`](super::attachment_format) against the
/// slots a pass actually binds, so a stale key can never alias two
/// genuinely different attachment layouts onto the same cache entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineFormatKey {
    /// The owning pass's `RenderPass::name()`. Names are unique and
    /// `'static` per the trait contract, so this alone partitions the cache
    /// per pass without needing a `TypeId`.
    pub pass: &'static str,
    /// One entry per color attachment, in attachment order.
    pub color_formats: Vec<wgpu::TextureFormat>,
    /// Depth/stencil attachment format, if the pipeline has one.
    pub depth_format: Option<wgpu::TextureFormat>,
    /// Pass-defined discriminator distinguishing multiple pipelines that
    /// would otherwise share the same formats (default 0).
    pub variant: u64,
}

impl PipelineFormatKey {
    pub fn new(
        pass: &'static str,
        color_formats: impl Into<Vec<wgpu::TextureFormat>>,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self {
            pass,
            color_formats: color_formats.into(),
            depth_format,
            variant: 0,
        }
    }

    /// Distinguishes a second pipeline for the same pass and formats (e.g.
    /// an alpha-blend variant of an otherwise-identical opaque pipeline).
    pub fn with_variant(mut self, variant: u64) -> Self {
        self.variant = variant;
        self
    }
}

/// Executor-owned cache mapping [`PipelineFormatKey`] to a built pipeline.
#[derive(Default)]
pub struct PipelineFormatCache {
    entries: RefCell<HashMap<PipelineFormatKey, Arc<wgpu::RenderPipeline>>>,
}

impl PipelineFormatCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the cached pipeline for `key`, building it with `build` on a
    /// miss. `build` runs at most once per distinct key — subsequent frames
    /// whose resolved attachment formats hash to the same key reuse the same
    /// `Arc<wgpu::RenderPipeline>` at the cost of one hash-map lookup.
    ///
    /// Pass this the exact key you already used to look the format up via
    /// [`attachment_format`](super::attachment_format), so a mismatch
    /// between "what the executor resolved" and "what the pipeline was
    /// built for" can't happen.
    pub fn get_or_create(
        &self,
        key: PipelineFormatKey,
        build: impl FnOnce() -> wgpu::RenderPipeline,
    ) -> Arc<wgpu::RenderPipeline> {
        if let Some(existing) = self.entries.borrow().get(&key) {
            return existing.clone();
        }
        // Built outside the borrow: `build()` may itself want to read other
        // graph state that could re-enter the cache (unlikely, but a panic
        // from a held RefCell borrow is a worse failure mode than a
        // redundant pipeline build on a racing miss, which can't happen
        // anyway since access is single-threaded).
        let pipeline = Arc::new(build());
        self.entries.borrow_mut().insert(key, pipeline.clone());
        pipeline
    }

    /// Drops every cached pipeline. The executor does *not* call this on
    /// resize (resizing changes extents, not formats) — only call it
    /// yourself after a shader/layout hot-reload invalidates pipelines out
    /// from under their format keys.
    pub fn clear(&self) {
        self.entries.borrow_mut().clear();
    }

    /// Number of distinct pipeline variants currently cached (debug/profiling).
    pub fn len(&self) -> usize {
        self.entries.borrow().len()
    }
}
