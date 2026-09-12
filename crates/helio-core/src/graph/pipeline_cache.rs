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
//! The cache uses a reader/writer lock because independent passes may record
//! concurrently. Hits take a shared read lock; misses publish under the write
//! lock. Phase 10 prewarming keeps the normal frame path on the hit branch.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

/// Pass-local opaque identifier for an executor-owned pipeline recipe.
///
/// Handles deliberately have no renderer-wide meaning. A pass may use any
/// stable values it owns, and the executor keeps one registry per pass.
pub type PipelineHandle = u32;

/// One host-declared attachment format combination that a graph may reach.
/// The executor never guesses formats from adapter capabilities; the host
/// owns this small, explicit enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineFormatSet {
    pub color_formats: Vec<wgpu::TextureFormat>,
    pub depth_format: Option<wgpu::TextureFormat>,
}

impl PipelineFormatSet {
    pub fn new(
        color_formats: impl Into<Vec<wgpu::TextureFormat>>,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self {
            color_formats: color_formats.into(),
            depth_format,
        }
    }
}

/// Explicit shader-variable to resource-name override used by reflected
/// binding generation. Normal bindings use the shader variable name directly.
#[derive(Clone, Default)]
pub struct BindingOverrideBuilder {
    entries: HashMap<&'static str, &'static str>,
}

impl BindingOverrideBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn map(&mut self, shader_variable: &'static str, resource_name: &'static str) {
        assert!(
            self.entries
                .insert(shader_variable, resource_name)
                .is_none(),
            "duplicate binding override for '{shader_variable}'"
        );
    }

    pub fn resolve<'a>(&self, shader_variable: &'a str) -> std::borrow::Cow<'a, str> {
        if let Some(resource) = self.entries.get(shader_variable).copied() {
            return std::borrow::Cow::Borrowed(resource);
        }
        // Shader authors commonly prefix resource variables with their
        // binding kind (`t_`, `s_`, `b_`, or `u_`). Strip only that explicit
        // vocabulary; arbitrary names remain untouched.
        if shader_variable.len() > 2 && matches!(&shader_variable[..2], "t_" | "s_" | "b_" | "u_") {
            return std::borrow::Cow::Borrowed(&shader_variable[2..]);
        }
        std::borrow::Cow::Borrowed(shader_variable)
    }
}

#[cfg(test)]
mod binding_tests {
    use super::{
        BindingOverrideBuilder, PipelineFormatCache, PipelineFormatKey, PipelineFormatSet,
    };

    fn assert_sync<T: Sync>() {}

    #[test]
    fn overrides_are_explicit_and_default_to_name_matching() {
        let mut bindings = BindingOverrideBuilder::new();
        bindings.map("t_history", "taa_history");
        assert_eq!(bindings.resolve("t_history").as_ref(), "taa_history");
        assert_eq!(bindings.resolve("depth").as_ref(), "depth");
        assert_eq!(bindings.resolve("t_pre_aa").as_ref(), "pre_aa");
    }

    #[test]
    fn pipeline_cache_is_safe_to_share_with_recording_workers() {
        assert_sync::<PipelineFormatCache>();
    }

    #[test]
    fn host_format_sets_replace_only_attachment_formats() {
        let key = PipelineFormatKey::new(
            "pass",
            [wgpu::TextureFormat::Rgba8Unorm],
            Some(wgpu::TextureFormat::Depth32Float),
        )
        .with_variant(9);
        let set = PipelineFormatSet::new(
            [wgpu::TextureFormat::Rgba16Float],
            Some(wgpu::TextureFormat::Depth24Plus),
        );
        let variant = key.with_formats(&set);
        assert_eq!(variant.pass, key.pass);
        assert_eq!(variant.variant, 9);
        assert_eq!(variant.color_formats, set.color_formats);
        assert_eq!(variant.depth_format, set.depth_format);
    }
}

/// A pipeline recipe declared by a render pass and consumed by the executor.
pub struct PipelineRecipe {
    pub handle: PipelineHandle,
    pub key: PipelineFormatKey,
    pub build: Arc<
        dyn Fn(
                &wgpu::Device,
                &PipelineFormatKey,
                Option<&wgpu::PipelineCache>,
            ) -> wgpu::RenderPipeline
            + Send
            + Sync,
    >,
}

/// Collects the pipelines a pass wants the executor to build before execute.
#[derive(Default)]
pub struct PipelineRecipeBuilder {
    recipes: Vec<PipelineRecipe>,
}

impl PipelineRecipeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a format-keyed pipeline recipe. The closure runs only on a cache
    /// miss, never in the pass's per-frame execute path.
    pub fn add(
        &mut self,
        handle: PipelineHandle,
        key: PipelineFormatKey,
        build: impl Fn(
                &wgpu::Device,
                &PipelineFormatKey,
                Option<&wgpu::PipelineCache>,
            ) -> wgpu::RenderPipeline
            + Send
            + Sync
            + 'static,
    ) {
        self.recipes.push(PipelineRecipe {
            handle,
            key,
            build: Arc::new(build),
        });
    }

    pub fn is_empty(&self) -> bool {
        self.recipes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.recipes.len()
    }

    pub(crate) fn into_recipes(self) -> Vec<PipelineRecipe> {
        self.recipes
    }
}

/// Pipelines resolved for one pass before that pass executes.
#[derive(Default)]
pub struct PipelineRegistry {
    entries: HashMap<PipelineHandle, Arc<wgpu::RenderPipeline>>,
}

impl PipelineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert(&mut self, handle: PipelineHandle, pipeline: Arc<wgpu::RenderPipeline>) {
        assert!(
            self.entries.insert(handle, pipeline).is_none(),
            "duplicate pipeline handle {handle} in one render pass"
        );
    }

    /// Returns the ready pipeline for a pass-local handle.
    pub fn get(&self, handle: PipelineHandle) -> Option<&wgpu::RenderPipeline> {
        self.entries.get(&handle).map(Arc::as_ref)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

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

    pub fn with_formats(&self, formats: &PipelineFormatSet) -> Self {
        Self {
            pass: self.pass,
            color_formats: formats.color_formats.clone(),
            depth_format: formats.depth_format,
            variant: self.variant,
        }
    }
}

/// Executor-owned cache mapping [`PipelineFormatKey`] to a built pipeline.
#[derive(Default)]
pub struct PipelineFormatCache {
    entries: Arc<RwLock<HashMap<PipelineFormatKey, Arc<wgpu::RenderPipeline>>>>,
    /// Serializes only cache-miss construction. Hits never take this lock;
    /// the second lookup after acquiring it guarantees one build per key even
    /// when independent recording workers request the same variant together.
    build_lock: Arc<Mutex<()>>,
    pending: Arc<Mutex<HashSet<PipelineFormatKey>>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    driver_cache: Option<Arc<wgpu::PipelineCache>>,
    persistent_path: Option<std::path::PathBuf>,
}

impl PipelineFormatCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a cache backed by wgpu's driver-validated pipeline cache when
    /// the adapter exposes the feature. Invalid or foreign blobs are safely
    /// ignored by wgpu's fallback mode.
    pub fn with_persistent_path(
        device: &wgpu::Device,
        path: impl Into<std::path::PathBuf>,
    ) -> Self {
        let path = path.into();
        let data = std::fs::read(&path).ok();
        let driver_cache = if device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            Some(Arc::new(unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("Helio Pipeline Cache"),
                    data: data.as_deref(),
                    fallback: true,
                })
            }))
        } else {
            None
        };
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            build_lock: Arc::new(Mutex::new(())),
            pending: Arc::new(Mutex::new(HashSet::new())),
            workers: Mutex::new(Vec::new()),
            driver_cache,
            persistent_path: Some(path),
        }
    }

    /// Creates a driver-backed in-memory cache without persistence.
    pub fn with_device(device: &wgpu::Device) -> Self {
        let mut cache = Self::default();
        if device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            cache.driver_cache = Some(Arc::new(unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("Helio Pipeline Cache"),
                    data: None,
                    fallback: true,
                })
            }));
        }
        cache
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
        if let Some(existing) = self
            .entries
            .read()
            .expect("pipeline cache poisoned")
            .get(&key)
        {
            return existing.clone();
        }
        let _build_guard = self
            .build_lock
            .lock()
            .expect("pipeline build lock poisoned");
        if let Some(existing) = self
            .entries
            .read()
            .expect("pipeline cache poisoned")
            .get(&key)
        {
            return existing.clone();
        }
        let pipeline = Arc::new(build());
        let mut entries = self.entries.write().expect("pipeline cache poisoned");
        entries
            .entry(key)
            .or_insert_with(|| pipeline.clone())
            .clone()
    }

    /// Recipe-aware variant that supplies the optional driver cache to the
    /// pipeline constructor. Existing manual callers should continue using
    /// [`Self::get_or_create`].
    pub fn get_or_create_with_driver_cache(
        &self,
        key: PipelineFormatKey,
        build: impl FnOnce(Option<&wgpu::PipelineCache>) -> wgpu::RenderPipeline,
    ) -> Arc<wgpu::RenderPipeline> {
        if let Some(existing) = self
            .entries
            .read()
            .expect("pipeline cache poisoned")
            .get(&key)
        {
            return existing.clone();
        }
        let _build_guard = self
            .build_lock
            .lock()
            .expect("pipeline build lock poisoned");
        if let Some(existing) = self
            .entries
            .read()
            .expect("pipeline cache poisoned")
            .get(&key)
        {
            return existing.clone();
        }
        let pipeline = Arc::new(build(self.driver_cache.as_deref()));
        let mut entries = self.entries.write().expect("pipeline cache poisoned");
        entries
            .entry(key)
            .or_insert_with(|| pipeline.clone())
            .clone()
    }

    /// Returns a ready pipeline immediately, or schedules construction on a
    /// background worker and returns `None`. Pending keys are published before
    /// spawning the worker, so concurrent recording workers cannot enqueue
    /// duplicate builds and the frame thread never waits for a miss.
    pub fn try_get_or_schedule<F>(
        &self,
        key: PipelineFormatKey,
        build: F,
    ) -> Option<Arc<wgpu::RenderPipeline>>
    where
        F: FnOnce(Option<&wgpu::PipelineCache>) -> wgpu::RenderPipeline + Send + 'static,
    {
        if let Some(existing) = self
            .entries
            .read()
            .expect("pipeline cache poisoned")
            .get(&key)
        {
            return Some(existing.clone());
        }
        if !self
            .pending
            .lock()
            .expect("pipeline pending lock poisoned")
            .insert(key.clone())
        {
            return None;
        }
        let entries = Arc::clone(&self.entries);
        let build_lock = Arc::clone(&self.build_lock);
        let pending = Arc::clone(&self.pending);
        let driver_cache = self.driver_cache.clone();
        let pass_name = key.pass;
        let worker = std::thread::spawn(move || {
            let _build_guard = build_lock.lock().expect("pipeline build lock poisoned");
            if !entries
                .read()
                .expect("pipeline cache poisoned")
                .contains_key(&key)
            {
                let pipeline = Arc::new(build(driver_cache.as_deref()));
                entries
                    .write()
                    .expect("pipeline cache poisoned")
                    .insert(key.clone(), pipeline);
            }
            pending
                .lock()
                .expect("pipeline pending lock poisoned")
                .remove(&key);
        });
        self.workers
            .lock()
            .expect("pipeline worker lock poisoned")
            .push(worker);
        eprintln!(
            "Helio pipeline cache miss scheduled in background: {}",
            pass_name
        );
        None
    }

    pub fn is_pending(&self, key: &PipelineFormatKey) -> bool {
        self.pending
            .lock()
            .expect("pipeline pending lock poisoned")
            .contains(key)
    }

    /// Drops every cached pipeline. The executor does *not* call this on
    /// resize (resizing changes extents, not formats) — only call it
    /// yourself after a shader/layout hot-reload invalidates pipelines out
    /// from under their format keys.
    pub fn clear(&self) {
        self.entries
            .write()
            .expect("pipeline cache poisoned")
            .clear();
    }

    /// Number of distinct pipeline variants currently cached (debug/profiling).
    pub fn len(&self) -> usize {
        self.entries.read().expect("pipeline cache poisoned").len()
    }

    pub fn has_driver_cache(&self) -> bool {
        self.driver_cache.is_some()
    }

    /// Persists the opaque driver blob atomically. The blob is only useful on
    /// the same compatible adapter; wgpu performs that validation on load.
    pub fn persist(&self) -> std::io::Result<()> {
        let Some(path) = &self.persistent_path else {
            return Ok(());
        };
        let Some(data) = self
            .driver_cache
            .as_ref()
            .and_then(|cache| cache.get_data())
        else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, data)?;
        std::fs::rename(tmp, path)
    }
}

impl Drop for PipelineFormatCache {
    fn drop(&mut self) {
        if let Ok(workers) = self.workers.get_mut() {
            for worker in workers.drain(..) {
                let _ = worker.join();
            }
        }
        let _ = self.persist();
    }
}
