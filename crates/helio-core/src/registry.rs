//! Per-frame transient resource registry.
//!
//! `ResourceRegistry` holds borrowed references to the transient textures/buffers
//! that the `RenderGraph` owns. These are passed into `PassContext` and
//! `PrepareContext` so passes can read outputs of earlier passes without any
//! allocation or locking.
//!
//! Generic: an open, typed map keyed by caller-declared [`ResourceKey`]s. The
//! core never names a specific resource — that is exactly what makes it
//! impossible for a new pass's resource to require editing this crate.

use std::collections::{HashMap, HashSet};

/// A generic fixed-arity group of borrowed texture views.
///
/// The registry stores groups by caller-owned ResourceKey values. The core
/// knows only that a group has an arity and ordered views; it does not
/// define or interpret any concrete render-pass bundle.
#[derive(Clone, Copy)]
pub struct ViewGroup<'a, const N: usize> {
    pub views: [&'a wgpu::TextureView; N],
}

/// Debug-tracked resource slot.
///
/// In debug builds, records which pass wrote the value so we can detect
/// when a pass reads a resource that no prior pass wrote this frame.
/// In release builds, compiles down to a plain `Option<T>` with zero overhead.
#[derive(Clone, Copy)]
pub struct Tracked<T> {
    value: Option<T>,
    #[cfg(debug_assertions)]
    written_by: Option<&'static str>,
}

impl<T: Copy> Tracked<T> {
    /// Creates an empty (unwritten) slot.
    pub const fn empty() -> Self {
        Self {
            value: None,
            #[cfg(debug_assertions)]
            written_by: None,
        }
    }

    /// Creates a slot with a pre-set value (no writer recorded).
    /// Used for renderer-provided fields that are available from the start.
    pub const fn with_value(value: T) -> Self {
        Self {
            value: Some(value),
            #[cfg(debug_assertions)]
            written_by: None,
        }
    }

    /// Writes a value, recording the writer pass name in debug builds.
    pub fn write(&mut self, value: T, _pass_name: &'static str) {
        self.value = Some(value);
        #[cfg(debug_assertions)]
        {
            self.written_by = Some(_pass_name);
        }
    }

    /// Reads the value. Panics in debug builds if the slot was never written
    /// (i.e. no prior pass called `write` on it this frame).
    ///
    /// Returns `None` when the slot was explicitly written but set to `None`
    /// (the panics only fire for unwritten slots, not empty-but-written ones).
    pub fn read(&self, _reader_pass: &'static str) -> Option<T> {
        #[cfg(debug_assertions)]
        if self.written_by.is_none() && self.value.is_some() {
            // This state shouldn't happen if write() is always used,
            // but just in case, don't panic if there's actually a value.
        }
        #[cfg(debug_assertions)]
        if self.value.is_none() && self.written_by.is_none() {
            panic!(
                "[RenderGraph] pass '{reader}' read resource that was never written this frame",
                reader = _reader_pass
            );
        }
        self.value
    }

    /// Reads without debug tracking (for optional resources that legitimately
    /// may be `None`, e.g. `full_res_depth`).
    pub fn get(&self) -> Option<T> {
        self.value
    }

    /// Returns true if this slot was written this frame (debug builds only).
    /// Always returns `true` in release builds.
    #[inline]
    pub fn was_written(&self) -> bool {
        #[cfg(debug_assertions)]
        {
            self.written_by.is_some()
        }
        #[cfg(not(debug_assertions))]
        {
            self.value.is_some()
        }
    }
}

impl<T> Tracked<T> {
    /// Returns `true` if the slot has a value (regardless of tracking state).
    pub fn is_some(&self) -> bool {
        self.value.is_some()
    }

    /// Returns `true` if the slot has no value.
    pub fn is_none(&self) -> bool {
        self.value.is_none()
    }

    /// Converts to `Option<&T>`.
    pub fn as_ref(&self) -> Option<&T> {
        self.value.as_ref()
    }
}

/// A typed handle to an open per-frame resource slot.
///
/// Resource keys are declared by the crate that owns the resource. The
/// registry only uses the key's name to find a slot; the type marker keeps
/// reads and writes statically typed at the call site.
#[derive(Clone, Copy)]
pub struct ResourceKey<T> {
    name: &'static str,
    type_tag: fn() -> &'static str,
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T> ResourceKey<T> {
    /// Creates a key for a named resource slot.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            type_tag: resource_type_tag::<T>,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the stable declaration name used by the registry.
    pub const fn name(self) -> &'static str {
        self.name
    }
}

fn resource_type_tag<T>() -> &'static str {
    std::any::type_name::<T>()
}

trait ErasedResourceSlot: Send + Sync {
    fn type_tag(&self) -> &'static str;
    fn self_ptr(&self) -> *const ();
    fn self_mut_ptr(&mut self) -> *mut ();
    fn has_value(&self) -> bool;
    fn reset_tracking(&mut self, writer: &'static str);
}

struct TypedResourceSlot<T> {
    value: Option<T>,
    #[cfg(debug_assertions)]
    written_by: Option<&'static str>,
}

impl<T> TypedResourceSlot<T> {
    fn empty() -> Self {
        Self {
            value: None,
            #[cfg(debug_assertions)]
            written_by: None,
        }
    }
}

impl<T: Send + Sync> ErasedResourceSlot for TypedResourceSlot<T> {
    fn type_tag(&self) -> &'static str {
        resource_type_tag::<T>()
    }

    fn self_ptr(&self) -> *const () {
        self as *const Self as *const ()
    }

    fn self_mut_ptr(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }

    fn has_value(&self) -> bool {
        self.value.is_some()
    }

    fn reset_tracking(&mut self, writer: &'static str) {
        #[cfg(debug_assertions)]
        {
            self.written_by = self.value.as_ref().map(|_| writer);
        }
        #[cfg(not(debug_assertions))]
        let _ = writer;
    }
}

/// Open, typed per-frame resource storage.
///
/// This registry has no closed list of resource fields. Pass crates can
/// declare new [`ResourceKey`] values without editing `helio-core`.
pub struct ResourceRegistry<'a> {
    slots: HashMap<&'static str, Box<dyn ErasedResourceSlot + 'a>>,
    bindings: HashMap<String, wgpu::BindingResource<'a>>,
    graph_bindings: HashSet<String>,
}

impl<'a> ResourceRegistry<'a> {
    /// Creates an empty registry for a frame.
    pub fn empty() -> Self {
        Self {
            slots: HashMap::new(),
            bindings: HashMap::new(),
            graph_bindings: HashSet::new(),
        }
    }

    /// Publishes a GPU resource for the generic reflected-binding contract.
    pub fn write_binding(
        &mut self,
        name: impl Into<String>,
        resource: wgpu::BindingResource<'a>,
        _writer: &'static str,
    ) {
        self.bindings.insert(name.into(), resource);
    }

    /// Publishes a graph-owned texture view into the reflected-binding projection.
    pub fn write_texture_binding(
        &mut self,
        name: impl Into<String>,
        view: &wgpu::TextureView,
        writer: &'static str,
    ) {
        let name = name.into();
        let resource = unsafe {
            std::mem::transmute::<wgpu::BindingResource<'_>, wgpu::BindingResource<'a>>(
                wgpu::BindingResource::TextureView(view),
            )
        };
        self.write_binding(name.clone(), resource, writer);
        self.graph_bindings.insert(name);
    }

    /// Returns a graph-routed texture view by name.
    pub fn texture_binding(&self, name: &str) -> Option<&'a wgpu::TextureView> {
        match self.bindings.get(name)? {
            wgpu::BindingResource::TextureView(view) => Some(*view),
            _ => None,
        }
    }

    /// Removes graph-generated bindings before the next execution.
    pub fn clear_graph_bindings(&mut self) {
        for name in self.graph_bindings.drain() {
            self.bindings.remove(&name);
        }
    }

    /// Returns a reflected-binding resource by its shader/resource name.
    pub fn binding(&self, name: &str) -> Option<wgpu::BindingResource<'a>> {
        self.bindings.get(name).cloned()
    }

    /// Routes a graph-owned named view through the same open typed slot path
    /// used by pass-owned ResourceKey declarations.
    pub fn route_named_texture(
        &mut self,
        name: &str,
        view: &wgpu::TextureView,
        writer: &'static str,
    ) {
        self.write_texture_binding(name, view, writer);
    }
    /// Writes a value and records its writer in debug builds.
    pub fn write<T: Copy + Send + Sync + 'a>(
        &mut self,
        key: ResourceKey<T>,
        value: T,
        writer: &'static str,
    ) {
        let slot = self
            .slots
            .entry(key.name)
            .or_insert_with(|| Box::new(TypedResourceSlot::<T>::empty()));
        assert_eq!(
            slot.type_tag(),
            (key.type_tag)(),
            "resource key '{}' was used with multiple value types",
            key.name
        );

        // The type tag check above establishes that this is the matching
        // TypedResourceSlot<T>. The registry owns the erased value, so the
        // cast is local and does not expose an untyped API to callers.
        let typed = unsafe { &mut *(slot.self_mut_ptr() as *mut TypedResourceSlot<T>) };
        typed.value = Some(value);
        #[cfg(debug_assertions)]
        {
            typed.written_by = Some(writer);
        }
    }

    /// Reads a value, panicking in debug builds if it was never written.
    pub fn read<T: Copy + Send + Sync + 'a>(
        &self,
        key: ResourceKey<T>,
        reader: &'static str,
    ) -> Option<T> {
        let Some(slot) = self.slots.get(key.name) else {
            #[cfg(debug_assertions)]
            panic!(
                "[RenderGraph] pass '{}' read resource '{}' that was never written this frame",
                reader, key.name
            );
            #[cfg(not(debug_assertions))]
            return None;
        };
        assert_eq!(
            slot.type_tag(),
            (key.type_tag)(),
            "resource key '{}' was used with multiple value types",
            key.name
        );
        let typed = unsafe { &*(slot.self_ptr() as *const TypedResourceSlot<T>) };
        #[cfg(debug_assertions)]
        if !typed.has_value() {
            panic!(
                "[RenderGraph] pass '{}' read resource '{}' that was never written this frame",
                reader, key.name
            );
        }
        typed.value
    }

    /// Reads a value without debug tracking for legitimately optional slots.
    pub fn get<T: Copy + Send + Sync + 'a>(&self, key: ResourceKey<T>) -> Option<T> {
        let slot = self.slots.get(key.name)?;
        assert_eq!(
            slot.type_tag(),
            (key.type_tag)(),
            "resource key '{}' was used with multiple value types",
            key.name
        );
        let typed = unsafe { &*(slot.self_ptr() as *const TypedResourceSlot<T>) };
        typed.value
    }

    /// Returns whether a slot was written during this frame.
    pub fn contains(&self, name: &str) -> bool {
        self.slots.get(name).is_some_and(|slot| slot.has_value())
    }

    pub fn was_written<T: Send + Sync>(&self, key: ResourceKey<T>) -> bool {
        let Some(slot) = self.slots.get(key.name) else {
            return false;
        };
        assert_eq!(
            slot.type_tag(),
            (key.type_tag)(),
            "resource key '{}' was used with multiple value types",
            key.name
        );
        #[cfg(debug_assertions)]
        {
            let typed = unsafe { &*(slot.self_ptr() as *const TypedResourceSlot<T>) };
            return typed.written_by.is_some();
        }
        #[cfg(not(debug_assertions))]
        {
            slot.has_value()
        }
    }

    /// Re-seeds tracking for values carried into the next frame.
    pub fn reset_tracking(&mut self, writer: &'static str) {
        for slot in self.slots.values_mut() {
            slot.reset_tracking(writer);
        }
    }

    pub fn write_texture_view(
        &mut self,
        key: ResourceKey<&'a wgpu::TextureView>,
        view: &'a wgpu::TextureView,
        writer: &'static str,
    ) {
        self.write(key, view, writer);
    }

    pub fn texture_view(
        &self,
        key: ResourceKey<&'a wgpu::TextureView>,
    ) -> Option<&'a wgpu::TextureView> {
        self.get(key).or_else(|| self.texture_binding(key.name()))
    }

    pub fn read_texture_view(
        &self,
        key: ResourceKey<&'a wgpu::TextureView>,
        reader: &'static str,
    ) -> Option<&'a wgpu::TextureView> {
        // Checked as plain (non-panicking) lookups first: a graph-routed
        // internal attachment (`write_color`, populated via
        // `route_named_texture`/`write_texture_binding`) lives only in
        // `bindings`, never in the typed `slots` map, so calling the
        // debug-panicking `read()` directly here would fire before ever
        // reaching the `bindings` fallback. Only fall through to `read()`
        // once both non-panicking paths have missed, so the "never written
        // this frame" debug diagnostic still fires for a genuinely absent key.
        if let Some(view) = self.get(key) {
            return Some(view);
        }
        if let Some(view) = self.texture_binding(key.name()) {
            return Some(view);
        }
        self.read(key, reader)
    }

    pub fn write_buffer(
        &mut self,
        key: ResourceKey<&'a wgpu::Buffer>,
        buffer: &'a wgpu::Buffer,
        writer: &'static str,
    ) {
        self.write(key, buffer, writer);
    }

    pub fn buffer(&self, key: ResourceKey<&'a wgpu::Buffer>) -> Option<&'a wgpu::Buffer> {
        self.get(key)
    }

    pub fn read_buffer(
        &self,
        key: ResourceKey<&'a wgpu::Buffer>,
        reader: &'static str,
    ) -> Option<&'a wgpu::Buffer> {
        self.read(key, reader)
    }

    pub fn write_sampler(
        &mut self,
        key: ResourceKey<&'a wgpu::Sampler>,
        sampler: &'a wgpu::Sampler,
        writer: &'static str,
    ) {
        self.write(key, sampler, writer);
    }

    pub fn sampler(&self, key: ResourceKey<&'a wgpu::Sampler>) -> Option<&'a wgpu::Sampler> {
        self.get(key)
    }

    pub fn read_sampler(
        &self,
        key: ResourceKey<&'a wgpu::Sampler>,
        reader: &'static str,
    ) -> Option<&'a wgpu::Sampler> {
        self.read(key, reader)
    }

    pub fn write_view_group<const N: usize>(
        &mut self,
        key: ResourceKey<ViewGroup<'a, N>>,
        views: [&'a wgpu::TextureView; N],
        writer: &'static str,
    ) {
        self.write(key, ViewGroup { views }, writer);
    }

    pub fn view_group<const N: usize>(
        &self,
        key: ResourceKey<ViewGroup<'a, N>>,
    ) -> Option<ViewGroup<'a, N>> {
        self.get(key)
    }

    pub fn read_view_group<const N: usize>(
        &self,
        key: ResourceKey<ViewGroup<'a, N>>,
        reader: &'static str,
    ) -> Option<ViewGroup<'a, N>> {
        self.read(key, reader)
    }
}
impl<'a> Default for ResourceRegistry<'a> {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod resource_registry_tests {
    use super::{ResourceKey, ResourceRegistry};

    const VALUE: ResourceKey<u32> = ResourceKey::new("test_value");
    const OPTIONAL: ResourceKey<u64> = ResourceKey::new("optional_value");

    #[test]
    fn slots_are_created_on_first_write() {
        let mut registry = ResourceRegistry::empty();

        assert_eq!(registry.get(OPTIONAL), None);
        assert!(!registry.was_written(VALUE));

        registry.write(VALUE, 42, "test_writer");

        assert_eq!(registry.get(VALUE), Some(42));
        assert_eq!(registry.read(VALUE, "test_reader"), Some(42));
        assert!(registry.was_written(VALUE));
    }

    #[test]
    fn reset_tracking_keeps_values_available() {
        let mut registry = ResourceRegistry::empty();
        registry.write(VALUE, 7, "test_writer");
        registry.reset_tracking("Renderer");

        assert_eq!(registry.get(VALUE), Some(7));
        assert!(registry.was_written(VALUE));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "never written this frame")]
    fn required_reads_fail_for_missing_slots() {
        let registry = ResourceRegistry::empty();
        let _ = registry.read(VALUE, "test_reader");
    }
}
