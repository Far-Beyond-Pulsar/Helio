//! GPU objects kept across frames for passes that would otherwise make them
//! every frame.
//!
//! Every wgpu object is a driver object (a Vulkan buffer or descriptor set,
//! a D3D12 heap range) that the driver tracks until the GPU is done with it:
//! objects made and dropped each frame cost CPU in submit and poll, and grow
//! driver memory. What a pass binds is almost always the same from frame to
//! frame, so a [`GpuCache`] keeps:
//!
//! - bind groups by their layout and the resources they bind (wgpu objects
//!   compare by identity): a frame that binds the same buffers and views as
//!   the last one gets the same bind group back;
//! - texture views by texture and descriptor, so bind groups over them are
//!   the same too;
//! - small immutable buffers (no `COPY_DST`, not mappable) by contents and
//!   usage: the placeholder a pass binds when it has nothing to upload is the
//!   same buffer every frame.
//!
//! A cache belongs to what binds through it (a chunk world, a pass), on that
//! owner's device, and goes with it: nothing outlives its owner or device.
//! The owner ends its frames ([`GpuCache::end_frame`]) where its frames
//! happen, and clears the cache where it replaces what it binds (resize,
//! pool growth), so a replaced texture or buffer goes with its bind groups
//! at once. An entry unused for [`KEEP_FRAMES`] frames is dropped too, a
//! bound on anything else. Entries hold what they name, so an id is never
//! reused while an entry compares against it.
//!
//! [`ReadbackPool`]: `MAP_READ` buffers handed out by size and returned when
//! dropped, for readbacks consumed in order.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// Frames an unused entry is kept: longer than anything bound periodically
/// (ping-ponged targets, a level traced every 30 frames). Owners drop what
/// they replace at once ([`GpuCache::clear`] on resize or pool growth), so
/// this only bounds what an owner forgets to.
pub const KEEP_FRAMES: u64 = 120;
/// Entries past which the least recently used are dropped even between
/// frames (an owner that never ends a frame).
const MAX_ENTRIES: usize = 4096;
/// The largest buffer [`GpuCache::buffer_init`] shares by contents.
const MAX_SHARED_BYTES: usize = 4096;

#[derive(Clone, PartialEq, Eq, Hash)]
struct Binding {
    buffer: wgpu::Buffer,
    offset: u64,
    size: Option<wgpu::BufferSize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Bound {
    Buffer(Binding),
    Buffers(Vec<Binding>),
    Sampler(wgpu::Sampler),
    Samplers(Vec<wgpu::Sampler>),
    View(wgpu::TextureView),
    Views(Vec<wgpu::TextureView>),
    Tlas(wgpu::Tlas),
    Tlases(Vec<wgpu::Tlas>),
}

#[derive(PartialEq, Eq, Hash)]
struct GroupKey {
    layout: wgpu::BindGroupLayout,
    entries: Vec<(u32, Bound)>,
}

#[derive(PartialEq, Eq, Hash)]
struct ViewKey {
    texture: wgpu::Texture,
    format: Option<wgpu::TextureFormat>,
    dimension: Option<wgpu::TextureViewDimension>,
    usage: Option<wgpu::TextureUsages>,
    aspect: wgpu::TextureAspect,
    mips: (u32, Option<u32>),
    layers: (u32, Option<u32>),
}

#[derive(PartialEq, Eq, Hash)]
struct BufferKey {
    usage: wgpu::BufferUsages,
    contents: Vec<u8>,
}

struct Entry<V> {
    value: V,
    used: u64,
}

#[derive(Default)]
struct Caches {
    frame: u64,
    groups: HashMap<GroupKey, Entry<wgpu::BindGroup>>,
    views: HashMap<ViewKey, Entry<wgpu::TextureView>>,
    buffers: HashMap<BufferKey, Entry<wgpu::Buffer>>,
    created: Created,
}

/// Objects a cache created (its misses), for tests and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Created {
    pub bind_groups: u64,
    pub views: u64,
    pub buffers: u64,
}

/// Bind groups, views and small immutable buffers kept across the frames of
/// one owner (see the module doc).
#[derive(Default)]
pub struct GpuCache {
    caches: Mutex<Caches>,
}

fn get<K: Hash + Eq, V: Clone>(
    map: &mut HashMap<K, Entry<V>>,
    frame: u64,
    key: K,
    make: impl FnOnce() -> V,
) -> (V, bool) {
    if let Some(entry) = map.get_mut(&key) {
        entry.used = frame;
        return (entry.value.clone(), false);
    }
    if map.len() >= MAX_ENTRIES {
        // The least recently used half goes.
        let mut used: Vec<u64> = map.values().map(|e| e.used).collect();
        let (_, median, _) = used.select_nth_unstable(map.len() / 2);
        let cut = *median;
        map.retain(|_, e| e.used > cut);
    }
    let value = make();
    map.insert(key, Entry { value: value.clone(), used: frame });
    (value, true)
}

impl GpuCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Caches> {
        self.caches.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The bind group `desc` describes: the cached one when the same layout
    /// binds the same resources, else a new one (cached). Bind groups over
    /// external textures are never cached.
    pub fn bind_group(&self, device: &wgpu::Device, desc: &wgpu::BindGroupDescriptor<'_>) -> wgpu::BindGroup {
        let binding = |b: &wgpu::BufferBinding<'_>| Binding {
            buffer: b.buffer.clone(),
            offset: b.offset,
            size: b.size,
        };
        let entries: Option<Vec<(u32, Bound)>> = desc
            .entries
            .iter()
            .map(|entry| {
                let bound = match &entry.resource {
                    wgpu::BindingResource::Buffer(b) => Bound::Buffer(binding(b)),
                    wgpu::BindingResource::BufferArray(bs) => Bound::Buffers(bs.iter().map(binding).collect()),
                    wgpu::BindingResource::Sampler(s) => Bound::Sampler((*s).clone()),
                    wgpu::BindingResource::SamplerArray(ss) => {
                        Bound::Samplers(ss.iter().map(|s| (*s).clone()).collect())
                    }
                    wgpu::BindingResource::TextureView(v) => Bound::View((*v).clone()),
                    wgpu::BindingResource::TextureViewArray(vs) => {
                        Bound::Views(vs.iter().map(|v| (*v).clone()).collect())
                    }
                    wgpu::BindingResource::AccelerationStructure(t) => Bound::Tlas((*t).clone()),
                    wgpu::BindingResource::AccelerationStructureArray(ts) => {
                        Bound::Tlases(ts.iter().map(|t| (*t).clone()).collect())
                    }
                    // External textures (and anything newer) are not cached.
                    _ => return None,
                };
                Some((entry.binding, bound))
            })
            .collect();
        let Some(entries) = entries else {
            return device.create_bind_group(desc);
        };
        let key = GroupKey {
            layout: desc.layout.clone(),
            entries,
        };
        let mut caches = self.lock();
        let frame = caches.frame;
        let (group, made) = get(&mut caches.groups, frame, key, || device.create_bind_group(desc));
        caches.created.bind_groups += u64::from(made);
        group
    }

    /// The whole-texture view of `texture` (`view_with` the default
    /// descriptor).
    pub fn view(&self, texture: &wgpu::Texture) -> wgpu::TextureView {
        self.view_with(texture, &Default::default())
    }

    /// The view `desc` describes of `texture`: cached by the texture and the
    /// descriptor (its label is the first view's).
    pub fn view_with(&self, texture: &wgpu::Texture, desc: &wgpu::TextureViewDescriptor<'_>) -> wgpu::TextureView {
        let key = ViewKey {
            texture: texture.clone(),
            format: desc.format,
            dimension: desc.dimension,
            usage: desc.usage,
            aspect: desc.aspect,
            mips: (desc.base_mip_level, desc.mip_level_count),
            layers: (desc.base_array_layer, desc.array_layer_count),
        };
        let mut caches = self.lock();
        let frame = caches.frame;
        let (view, made) = get(&mut caches.views, frame, key, || texture.create_view(desc));
        caches.created.views += u64::from(made);
        view
    }

    /// A buffer holding `desc.contents`: immutable ones (neither `COPY_DST`
    /// nor mappable) up to 4 KiB are shared by contents and usage, others are
    /// new.
    pub fn buffer_init(&self, device: &wgpu::Device, desc: &wgpu::util::BufferInitDescriptor<'_>) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let writable = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE;
        if desc.usage.intersects(writable) || desc.contents.len() > MAX_SHARED_BYTES {
            return device.create_buffer_init(desc);
        }
        let key = BufferKey {
            usage: desc.usage,
            contents: desc.contents.to_vec(),
        };
        let mut caches = self.lock();
        let frame = caches.frame;
        let (buffer, made) = get(&mut caches.buffers, frame, key, || device.create_buffer_init(desc));
        caches.created.buffers += u64::from(made);
        buffer
    }

    /// The owner's frame ended: entries unused for [`KEEP_FRAMES`] frames
    /// are dropped, releasing what they held.
    pub fn end_frame(&self) {
        let mut caches = self.lock();
        caches.frame += 1;
        let frame = caches.frame;
        let alive = |used: u64| frame - used <= KEEP_FRAMES;
        caches.groups.retain(|_, e| alive(e.used));
        caches.views.retain(|_, e| alive(e.used));
        caches.buffers.retain(|_, e| alive(e.used));
    }

    /// Drop every entry now (the owner replaced what it binds wholesale).
    pub fn clear(&self) {
        let mut caches = self.lock();
        caches.groups.clear();
        caches.views.clear();
        caches.buffers.clear();
    }

    /// Objects created so far.
    pub fn created(&self) -> Created {
        self.lock().created
    }

    /// Entries held now.
    pub fn len(&self) -> usize {
        let caches = self.lock();
        caches.groups.len() + caches.views.len() + caches.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `MAP_READ | COPY_DST` buffers by power-of-two size, handed out as
/// [`PooledBuffer`]s that come back to the pool when dropped (unmapped).
/// Clones share the pool.
#[derive(Clone)]
pub struct ReadbackPool {
    label: &'static str,
    free: Arc<Mutex<HashMap<u64, Vec<wgpu::Buffer>>>>,
}

/// Free buffers kept per size: readbacks in flight at once.
const POOL_PER_SIZE: usize = 8;

impl ReadbackPool {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            free: Default::default(),
        }
    }

    /// A buffer of at least `size` bytes.
    pub fn get(&self, device: &wgpu::Device, size: u64) -> PooledBuffer {
        let size = size.max(wgpu::MAP_ALIGNMENT).next_power_of_two();
        let reused = self
            .free
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get_mut(&size)
            .and_then(Vec::pop);
        let buffer = reused.unwrap_or_else(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        PooledBuffer {
            buffer: Some(buffer),
            pool: self.free.clone(),
            state: Default::default(),
        }
    }
}

/// A [`ReadbackPool`] buffer: back to the pool (unmapped) when dropped.
/// Map it with [`map_read`](Self::map_read), which tracks whether it is
/// mapped (wgpu does not say, and unmapping an unmapped buffer is an error).
pub struct PooledBuffer {
    buffer: Option<wgpu::Buffer>,
    pool: Arc<Mutex<HashMap<u64, Vec<wgpu::Buffer>>>>,
    state: Arc<std::sync::atomic::AtomicU8>,
}

const IDLE: u8 = 0;
const PENDING: u8 = 1;
const MAPPED: u8 = 2;

impl PooledBuffer {
    /// Map the whole buffer for reading; `ready` is set once it is mapped.
    pub fn map_read(&self, ready: Arc<std::sync::atomic::AtomicBool>) {
        use std::sync::atomic::Ordering;
        self.state.store(PENDING, Ordering::Release);
        let state = self.state.clone();
        self.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let mapped = result.is_ok();
            state.store(if mapped { MAPPED } else { IDLE }, Ordering::Release);
            if mapped {
                ready.store(true, Ordering::Release);
            }
        });
    }

    /// Unmap it if mapped (shadows `wgpu::Buffer::unmap`, which errors on an
    /// unmapped buffer).
    pub fn unmap(&self) {
        use std::sync::atomic::Ordering;
        if self
            .state
            .compare_exchange(MAPPED, IDLE, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            (**self).unmap();
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("held until dropped")
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        // A buffer whose map is still pending is not reused; a mapped one
        // is unmapped, and the next copy into it is queued after the last.
        match self.state.load(std::sync::atomic::Ordering::Acquire) {
            PENDING => return,
            MAPPED => buffer.unmap(),
            _ => {}
        }
        let mut free = self.pool.lock().unwrap_or_else(|p| p.into_inner());
        let list = free.entry(buffer.size()).or_default();
        if list.len() < POOL_PER_SIZE {
            list.push(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    fn storage(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn group(cache: &GpuCache, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        cache.bind_group(
            device,
            &wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            },
        )
    }

    fn allocated(device: &wgpu::Device) -> Option<u64> {
        device.generate_allocator_report().map(|r| r.total_allocated_bytes)
    }

    #[test]
    fn frames_binding_the_same_resources_share_one_bind_group() {
        let Some((device, _queue)) = gpu() else { return };
        let cache = GpuCache::new();
        let layout = layout(&device);
        let (a, b) = (storage(&device, 64), storage(&device, 64));
        let first = group(&cache, &device, &layout, &a);
        for _ in 0..20 {
            assert_eq!(group(&cache, &device, &layout, &a), first);
            cache.end_frame();
        }
        assert_ne!(group(&cache, &device, &layout, &b), first);
        assert_eq!(cache.created().bind_groups, 2);
        // The same contents, the same immutable buffer.
        let init = |contents: &[u8]| {
            cache.buffer_init(
                &device,
                &wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                },
            )
        };
        assert_eq!(init(&[1, 2, 3, 4]), init(&[1, 2, 3, 4]));
        assert_ne!(init(&[1, 2, 3, 4]), init(&[4, 3, 2, 1]));
    }

    /// A resource its owner dropped is released a few frames later.
    #[test]
    fn replaced_resources_are_released_after_a_few_frames() {
        let Some((device, _queue)) = gpu() else { return };
        let Some(before) = allocated(&device) else { return };
        let cache = GpuCache::new();
        let layout = layout(&device);
        let big = storage(&device, 64 << 20);
        group(&cache, &device, &layout, &big);
        drop(big);
        assert!(allocated(&device).unwrap() >= before + (64 << 20), "the cache keeps it for now");
        for _ in 0..=KEEP_FRAMES {
            cache.end_frame();
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        assert!(cache.is_empty());
        assert!(allocated(&device).unwrap() < before + (1 << 20), "released once the entry expired");
    }

    /// An owner that replaces what it binds clears the cache: the old
    /// resources go at once, not after the keep window.
    #[test]
    fn a_cleared_cache_releases_at_once() {
        let Some((device, _queue)) = gpu() else { return };
        let Some(before) = allocated(&device) else { return };
        let cache = GpuCache::new();
        let layout = layout(&device);
        let big = storage(&device, 64 << 20);
        group(&cache, &device, &layout, &big);
        drop(big);
        cache.clear();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        assert!(allocated(&device).unwrap() < before + (1 << 20));
    }

    /// Devices made and dropped one after another in one process, each with
    /// its owners' caches (a test suite's worlds): nothing outlives them.
    #[test]
    fn caches_go_with_their_owners_and_devices() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        for _ in 0..4 {
            let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
                return;
            };
            let Ok((device, _queue)) = pollster::block_on(adapter.request_device(&Default::default())) else {
                return;
            };
            let cache = GpuCache::new();
            let layout = layout(&device);
            let buffer = storage(&device, 16 << 20);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 512, height: 512, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            for _ in 0..8 {
                group(&cache, &device, &layout, &buffer);
                cache.view(&texture);
                cache.end_frame();
            }
            // Cache, resources and device dropped, in no particular order.
        }
        let Some(report) = instance.generate_report() else { return };
        let hub = &report.hub;
        let live = [
            ("buffers", hub.buffers.num_allocated),
            ("textures", hub.textures.num_allocated),
            ("texture views", hub.texture_views.num_allocated),
            ("bind groups", hub.bind_groups.num_allocated),
            ("devices", hub.devices.num_allocated),
        ];
        assert!(live.iter().all(|(_, n)| *n == 0), "still alive: {live:?}");
    }

    #[test]
    fn readback_buffers_come_back_to_the_pool() {
        let Some((device, queue)) = gpu() else { return };
        let pool = ReadbackPool::new("test readback");
        let first = pool.get(&device, 100);
        let id = (*first).clone();
        drop(first);
        assert_eq!(*pool.get(&device, 120), id, "the same size class is reused");
        assert_ne!(*pool.get(&device, 1000), id);
        // Mapped, read, dropped with or without unmapping: reusable.
        for unmap in [true, false] {
            let buffer = pool.get(&device, 100);
            queue.submit([]);
            let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
            buffer.map_read(ready.clone());
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            assert!(ready.load(std::sync::atomic::Ordering::Acquire));
            drop(buffer.slice(..).get_mapped_range());
            if unmap {
                buffer.unmap();
            }
            drop(buffer);
            let again = pool.get(&device, 100);
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&again, 0, None);
            queue.submit([encoder.finish()]);
        }
    }
}
