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
//! lock. Pipelines that must not be built in a frame at all are compiled off
//! it ([`crate::compile`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

struct DriverCacheRecord {
    cache: Weak<wgpu::PipelineCache>,
    path: Option<std::path::PathBuf>,
    persist_lock: Arc<Mutex<()>>,
}

static DRIVER_CACHES: OnceLock<Mutex<HashMap<usize, DriverCacheRecord>>> = OnceLock::new();

fn shared_driver_cache(
    device: &wgpu::Device,
    requested_path: Option<std::path::PathBuf>,
) -> (
    Option<Arc<wgpu::PipelineCache>>,
    Option<std::path::PathBuf>,
    Option<Arc<Mutex<()>>>,
) {
    if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return (None, requested_path, None);
    }

    // All graphs and pass constructors using one Device share this cache.
    // Keep a weak registry entry so closing a renderer releases its Device.
    let device_key = device as *const wgpu::Device as usize;
    let caches = DRIVER_CACHES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut caches = caches
        .lock()
        .expect("driver pipeline cache registry poisoned");
    // The registry is process-wide and its key is a raw Device address. Drop
    // dead records so sequential device lifetimes do not accumulate entries.
    caches.retain(|_, record| record.cache.strong_count() > 0);

    if let Some(record) = caches.get(&device_key) {
        if let Some(cache) = record.cache.upgrade() {
            // A graph opts into persistence before adding passes. If an
            // in-memory graph cache exists, replace it with the loaded cache.
            let upgrade_to_persistent = record.path.is_none() && requested_path.is_some();
            if !upgrade_to_persistent {
                return (
                    Some(cache),
                    record.path.clone().or(requested_path),
                    Some(Arc::clone(&record.persist_lock)),
                );
            }
        }
    }

    let path = requested_path
        .or_else(|| std::env::var_os("PULSAR_PIPELINE_CACHE").map(std::path::PathBuf::from));
    let data = path.as_ref().and_then(|path| std::fs::read(path).ok());
    let cache = Arc::new(unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("Helio Pipeline Cache"),
            data: data.as_deref(),
            fallback: true,
        })
    });
    let persist_lock = caches
        .get(&device_key)
        .map(|record| Arc::clone(&record.persist_lock))
        .unwrap_or_else(|| Arc::new(Mutex::new(())));
    caches.insert(
        device_key,
        DriverCacheRecord {
            cache: Arc::downgrade(&cache),
            path: path.clone(),
            persist_lock: Arc::clone(&persist_lock),
        },
    );
    (Some(cache), path, Some(persist_lock))
}

/// The wgpu cache shared by all pipeline construction for this device.
pub(crate) fn driver_cache_for_device(device: &wgpu::Device) -> Option<Arc<wgpu::PipelineCache>> {
    let path = std::env::var_os("PULSAR_PIPELINE_CACHE").map(std::path::PathBuf::from);
    shared_driver_cache(device, path).0
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
    use super::{BindingOverrideBuilder, PipelineFormatCache};

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
}

/// Executor-owned cache mapping [`PipelineFormatKey`] to a built pipeline.
#[derive(Default)]
pub struct PipelineFormatCache {
    entries: Arc<RwLock<HashMap<PipelineFormatKey, Arc<wgpu::RenderPipeline>>>>,
    /// Serializes only cache-miss construction. Hits never take this lock;
    /// the second lookup after acquiring it guarantees one build per key even
    /// when independent recording workers request the same variant together.
    build_lock: Arc<Mutex<()>>,
    driver_cache: Option<Arc<wgpu::PipelineCache>>,
    persistent_path: Option<std::path::PathBuf>,
    persist_lock: Option<Arc<Mutex<()>>>,
}

const MAX_PIPELINE_VARIANTS: usize = 512;

fn trim_pipeline_entries(
    entries: &mut HashMap<PipelineFormatKey, Arc<wgpu::RenderPipeline>>,
    preserve: &PipelineFormatKey,
) {
    while entries.len() > MAX_PIPELINE_VARIANTS {
        let Some(old_key) = entries.keys().find(|key| *key != preserve).cloned() else {
            break;
        };
        entries.remove(&old_key);
    }
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
        let (driver_cache, persistent_path, persist_lock) = shared_driver_cache(device, Some(path));
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            build_lock: Arc::new(Mutex::new(())),
            driver_cache,
            persistent_path,
            persist_lock,
        }
    }

    /// Creates a driver-backed in-memory cache without persistence.
    pub fn with_device(device: &wgpu::Device) -> Self {
        let path = std::env::var_os("PULSAR_PIPELINE_CACHE").map(std::path::PathBuf::from);
        let (driver_cache, persistent_path, persist_lock) = shared_driver_cache(device, path);
        let mut cache = Self::default();
        cache.driver_cache = driver_cache;
        cache.persistent_path = persistent_path;
        cache.persist_lock = persist_lock;
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
        let result = entries
            .entry(key.clone())
            .or_insert_with(|| pipeline.clone())
            .clone();
        trim_pipeline_entries(&mut entries, &key);
        result
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

    /// Drops every cached pipeline owned by the pass named `name`, so the next
    /// request builds from the pass's current recipe. Other passes' entries
    /// are untouched. Returns how many cached pipelines were removed.
    ///
    /// For a pass swapped in place (shader hot reload): its pipelines are keyed
    /// by pass name, so without this the new instance would be handed the old
    /// instance's pipelines.
    pub fn invalidate_pass(&self, name: &'static str) -> usize {
        let mut entries = self.entries.write().expect("pipeline cache poisoned");
        let before = entries.len();
        entries.retain(|key, _| key.pass != name);
        before - entries.len()
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
        let _persist_guard = self.persist_lock.as_ref().map(|lock| {
            lock.lock()
                .expect("driver pipeline cache persist lock poisoned")
        });
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
        // Per-process temp name: several game instances may share one cache file.
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, data)?;
        std::fs::rename(tmp, path)
    }
}

impl Drop for PipelineFormatCache {
    fn drop(&mut self) {
        let _ = self.persist();
    }
}
