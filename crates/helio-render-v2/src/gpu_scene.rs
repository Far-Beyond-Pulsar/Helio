//! GPU Scene Buffer — Unreal Engine FGPUScene equivalent.
//!
//! All per-object instance data (transforms, bounds, material IDs) lives in a
//! single large GPU storage buffer.  Objects are assigned stable *slots* via a
//! free-list allocator.  At render time the vertex shader fetches its transform
//! from the storage buffer using `instance_index` — no per-object vertex buffers,
//! no per-object `write_buffer` calls.
//!
//! # Delta Update Protocol
//!
//! Each frame, only the slots whose data actually changed are written to a
//! staging region and copied to the GPU buffer in a single `write_buffer` call
//! over the dirty byte range.  Static geometry at steady state has **zero**
//! CPU→GPU transfer cost.
//!
//! This mirrors Unreal's `FGPUScenePrimitiveUpdater` which batches all dirty
//! primitive transforms into one structured-buffer upload per frame.

use std::collections::HashMap;
use std::sync::Arc;
use bytemuck::Zeroable;

use crate::culling::Aabb;
use crate::material::GpuMaterial;
use crate::mesh::{DrawCall, GpuDrawCall, GpuMesh};
use crate::scene::ObjectId;
use crate::gpu_transfer;

// ── GPU-side per-instance data (must match WGSL `GpuInstance`) ───────────────

/// Per-instance data stored in the GPU scene buffer.
/// 128 bytes per slot — cache-line aligned on most GPUs.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstanceData {
    /// Model-to-world transform (column-major 4×4).
    pub transform: [f32; 16],     // 64 bytes
    /// Inverse-transpose of upper-left 3×3 for normal transform, padded to 48 bytes.
    pub normal_mat: [f32; 12],    // 48 bytes  (3 × vec4 rows)
    /// World-space bounding sphere center.
    pub bounds_center: [f32; 3],  //  12 bytes
    /// Bounding sphere radius.
    pub bounds_radius: f32,       //   4 bytes
    // total: 128 bytes
}

impl GpuInstanceData {
    pub fn from_transform(transform: glam::Mat4, bounds_center: [f32; 3], bounds_radius: f32) -> Self {        let cols = transform.to_cols_array();
        // Compute inverse-transpose of upper-3×3 for correct normal transformation.
        let inv_t = transform.inverse().transpose();
        let it = inv_t.to_cols_array();
        // Pack as 3 × vec4 rows (WGSL array<vec4<f32>, 3>)
        let normal_mat = [
            it[0], it[1], it[2], 0.0,
            it[4], it[5], it[6], 0.0,
            it[8], it[9], it[10], 0.0,
        ];
        // Transform bounds center to world space
        let c = glam::Vec3::from(bounds_center);
        let world_center = transform.transform_point3(c);
        // Scale radius by max scale factor of the transform
        let sx = glam::Vec3::new(transform.x_axis.x, transform.x_axis.y, transform.x_axis.z).length();
        let sy = glam::Vec3::new(transform.y_axis.x, transform.y_axis.y, transform.y_axis.z).length();
        let sz = glam::Vec3::new(transform.z_axis.x, transform.z_axis.y, transform.z_axis.z).length();
        let max_scale = sx.max(sy).max(sz);
        Self {
            transform: cols,
            normal_mat,
            bounds_center: world_center.into(),
            bounds_radius: bounds_radius * max_scale,
        }
    }
}

const INSTANCE_SIZE: usize = std::mem::size_of::<GpuInstanceData>(); // 128 bytes
const INITIAL_CAPACITY: u32 = 16 * 1024; // 16K objects by default

/// Per-material draw range used by GBufferPass for multi_draw_indexed_indirect batching.
/// `start` and `count` index into the GPU draw-call buffer (opaque draws only).
#[derive(Clone)]
pub struct MaterialRange {
    pub bind_group: Arc<wgpu::BindGroup>,
    pub start: u32,
    pub count: u32,
}

// ── Per-instance AABB for Hi-Z occlusion culling ─────────────────────────────

/// World-space AABB per instance, stored in a dedicated GPU buffer for
/// the occlusion culling compute pass.  Updated whenever the instance
/// transform changes.  32 bytes per slot (vec4 min + vec4 max with padding).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstanceAabb {
    /// World-space AABB minimum corner.
    pub aabb_min: [f32; 3],
    pub _pad0: f32,
    /// World-space AABB maximum corner.
    pub aabb_max: [f32; 3],
    pub _pad1: f32,
}

const AABB_SIZE: usize = std::mem::size_of::<GpuInstanceAabb>(); // 32 bytes

/// Free-list slot allocator.  O(1) alloc and free.
struct SlotAllocator {
    capacity: u32,
    free_list: Vec<u32>,
    high_water: u32,
}

impl SlotAllocator {
    fn new(capacity: u32) -> Self {
        Self {
            capacity,
            free_list: Vec::with_capacity(256),
            high_water: 0,
        }
    }

    /// Allocate a slot.  Returns `None` if the buffer is full and needs to grow.
    fn alloc(&mut self) -> Option<u32> {
        if let Some(slot) = self.free_list.pop() {
            Some(slot)
        } else if self.high_water < self.capacity {
            let slot = self.high_water;
            self.high_water += 1;
            Some(slot)
        } else {
            None // need to grow
        }
    }

    fn free(&mut self, slot: u32) {
        self.free_list.push(slot);
    }

    fn grow(&mut self, new_capacity: u32) {
        self.capacity = new_capacity;
    }

    fn active_count(&self) -> u32 {
        self.high_water - self.free_list.len() as u32
    }

    fn high_water_mark(&self) -> u32 {
        self.high_water
    }
}

// ── CPU-side proxy (minimal per-object bookkeeping) ──────────────────────────

/// CPU-side record for a registered scene primitive.
/// Mirrors Unreal's `FPrimitiveSceneProxy` — stores only what the CPU needs
/// to decide whether to update the GPU buffer.
pub(crate) struct SceneProxy {
    /// Slot index into the GPU instance buffer.
    pub slot: u32,
    /// FNV-1a hash of the last-uploaded transform columns.
    pub transform_hash: u64,
    /// Cached mesh reference for draw-call emission.
    pub mesh: GpuMesh,
    /// Material bind group (shared via Arc).
    pub material_bind_group: Arc<wgpu::BindGroup>,
    /// True if material uses alpha blending.
    pub transparent: bool,
    /// Local-space bounding sphere center (from mesh).
    pub local_bounds_center: [f32; 3],
    /// Local-space bounding sphere radius (from mesh).
    pub local_bounds_radius: f32,
    /// Local-space AABB minimum corner (from mesh).
    pub local_aabb_min: [f32; 3],
    /// Local-space AABB maximum corner (from mesh).
    pub local_aabb_max: [f32; 3],
    /// When false the object is excluded from all draw lists.
    pub enabled: bool,
}

// ── GpuScene ─────────────────────────────────────────────────────────────────

/// GPU-resident scene buffer.  The single source of truth for all per-instance
/// data on the GPU — transforms, normal matrices, bounds, material IDs.
///
/// This is the Helio equivalent of Unreal Engine's `FGPUScene` / `FGPUScenePrimitiveUpdater`.
pub struct GpuScene {
    // ── GPU resources ────────────────────────────────────────────────────
    /// The main storage buffer holding `GpuInstanceData` for every active slot.
    instance_buffer: wgpu::Buffer,
    /// Bind group exposing the instance buffer to shaders.
    pub(crate) bind_group: wgpu::BindGroup,
    /// Bind group layout (needed for pipeline creation).
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,

    /// Per-slot world-space AABB buffer used by the Hi-Z occlusion culling pass.
    /// Updated alongside `instance_buffer` whenever a transform changes.
    /// 32 bytes per slot (`GpuInstanceAabb`).
    pub(crate) aabb_buffer: Arc<wgpu::Buffer>,

    /// Visibility bitmask written by the occlusion culling compute pass.
    /// 1 bit per slot; bit N = 1 means slot N passed occlusion test.
    /// Sized for `ceil(capacity / 32)` u32 words.
    pub(crate) visibility_buffer: Arc<wgpu::Buffer>,

    // ── GPU draw call buffer (non-compacting indirect) ────────────────────
    /// One `GpuDrawCall` per enabled proxy (opaque sorted by material, transparent last).
    /// Uploaded to `draw_call_buffer` when `draw_call_dirty` is set.
    draw_call_cpu: Vec<GpuDrawCall>,
    /// GPU buffer holding `draw_call_cpu` data.
    pub(crate) draw_call_buffer: Arc<wgpu::Buffer>,
    /// Capacity of `draw_call_buffer` in draw calls.
    draw_call_buffer_capacity: u32,
    /// Number of draw calls currently in `draw_call_cpu`.
    pub(crate) draw_call_count: u32,
    /// Set when `draw_call_cpu` needs flushing to `draw_call_buffer`.
    draw_call_dirty: bool,
    /// Per-material ranges into `draw_call_cpu` (opaque only).
    /// Index of first transparent draw = `transparent_start` (same as cached list).
    pub(crate) material_ranges: Vec<MaterialRange>,

    // ── CPU mirror + dirty tracking ──────────────────────────────────────
    /// CPU-side copy of the instance data (indexed by slot).
    cpu_data: Vec<GpuInstanceData>,
    /// CPU-side copy of world-space AABBs (indexed by slot).
    cpu_aabbs: Vec<GpuInstanceAabb>,
    /// Bitset: one bit per slot.  Set when slot data changes, cleared after upload.
    dirty_bits: Vec<u64>,
    /// Lowest and highest dirty slot indices (inclusive).  Avoids scanning the
    /// full bitset when only a few objects moved.
    dirty_range: Option<(u32, u32)>,

    // ── Slot management ──────────────────────────────────────────────────
    allocator: SlotAllocator,

    // ── Object registry ──────────────────────────────────────────────────
    /// ObjectId → SceneProxy mapping.
    pub(crate) proxies: HashMap<u64, SceneProxy>,
    next_object_id: u64,

    /// Structural generation counter (bumped on add/remove, NOT on transform update).
    pub(crate) generation: u64,

    // ── Cached draw lists (rebuilt only on generation change) ─────────
    /// Cached opaque+transparent draw calls, rebuilt when `generation` changes.
    cached_draw_list: Vec<DrawCall>,
    /// Cached shadow draw calls (same content, separate vec for independent locking).
    cached_shadow_draw_list: Vec<DrawCall>,
    /// Generation at which the caches were last rebuilt.
    cached_generation: u64,

    // ── Cached draw list partition ────────────────────────────────────────
    /// Index of the first transparent draw in `cached_draw_list`.
    pub(crate) transparent_start: usize,

    // ── Stats ────────────────────────────────────────────────────────────
    /// Number of slots written to GPU this frame (for profiler).
    pub(crate) last_upload_slot_count: u32,
    /// Bytes written to GPU this frame.
    pub(crate) last_upload_bytes: u64,
}

impl GpuScene {
    pub fn new(device: &wgpu::Device) -> Self {
        let capacity = INITIAL_CAPACITY;
        let buffer_size = (capacity as usize * INSTANCE_SIZE) as u64;
        let aabb_buffer_size = capacity as u64 * AABB_SIZE as u64;
        let visibility_words = ((capacity as usize) + 31) / 32;
        let visibility_buffer_size = (visibility_words * 4) as u64;
        let draw_call_capacity = capacity;
        let draw_call_buffer_size = draw_call_capacity as u64 * std::mem::size_of::<GpuDrawCall>() as u64;

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene Instance Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let aabb_buffer_arc = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene AABB Buffer"),
            size: aabb_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let visibility_buffer_arc = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene Visibility Buffer"),
            size: visibility_buffer_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let draw_call_buffer_arc = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene Draw Call Buffer"),
            size: draw_call_buffer_size.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Scene BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Scene BG"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.as_entire_binding(),
            }],
        });

        let words = ((capacity as usize) + 63) / 64;

        let s = Self {
            instance_buffer,
            bind_group,
            bind_group_layout,
            aabb_buffer: aabb_buffer_arc,
            visibility_buffer: visibility_buffer_arc,
            draw_call_cpu: Vec::new(),
            draw_call_buffer: draw_call_buffer_arc,
            draw_call_buffer_capacity: draw_call_capacity,
            draw_call_count: 0,
            draw_call_dirty: false,
            material_ranges: Vec::new(),
            cpu_data: vec![GpuInstanceData::zeroed(); capacity as usize],
            cpu_aabbs: vec![GpuInstanceAabb::zeroed(); capacity as usize],
            dirty_bits: vec![0u64; words],
            dirty_range: None,
            allocator: SlotAllocator::new(capacity),
            proxies: HashMap::with_capacity(1024),
            next_object_id: 1,
            generation: 0,
            cached_draw_list: Vec::new(),
            cached_shadow_draw_list: Vec::new(),
            cached_generation: u64::MAX,
            transparent_start: 0,
            last_upload_slot_count: 0,
            last_upload_bytes: 0,
        };

        gpu_transfer::track_alloc(buffer_size + aabb_buffer_size + visibility_buffer_size + draw_call_buffer_size);
        s
    }

    /// Register a new object.  Returns a stable `ObjectId`.
    ///
    /// Equivalent to Unreal's `FScene::AddPrimitive()`.
    pub fn add_object(
        &mut self,
        device: &wgpu::Device,
        mesh: &GpuMesh,
        material: Option<&GpuMaterial>,
        default_material_bg: &Arc<wgpu::BindGroup>,
        transform: glam::Mat4,
    ) -> ObjectId {
        let id = ObjectId(self.next_object_id);
        self.next_object_id += 1;

        // Allocate a slot, growing if necessary
        let slot = loop {
            if let Some(s) = self.allocator.alloc() {
                break s;
            }
            self.grow(device);
        };

        // Build instance data
        let instance = GpuInstanceData::from_transform(
            transform,
            mesh.bounds_center,
            mesh.bounds_radius,
        );

        // World-space AABB via Arvo transform of local AABB.
        let local_aabb = Aabb::new(mesh.aabb_min.into(), mesh.aabb_max.into());
        let world_aabb = local_aabb.transform(&transform);

        // Write to CPU mirrors
        let mat = transform.to_cols_array();
        self.cpu_data[slot as usize] = instance;
        self.cpu_aabbs[slot as usize] = GpuInstanceAabb {
            aabb_min: world_aabb.min.into(),
            _pad0: 0.0,
            aabb_max: world_aabb.max.into(),
            _pad1: 0.0,
        };
        self.mark_dirty(slot);

        let transform_hash = fnv1a_mat(&mat);

        let (bind_group, transparent) = match material {
            Some(m) => (Arc::clone(&m.bind_group), m.transparent_blend),
            None    => (Arc::clone(default_material_bg), false),
        };

        self.proxies.insert(id.0, SceneProxy {
            slot,
            transform_hash,
            mesh: mesh.clone(),
            material_bind_group: bind_group,
            transparent,
            local_bounds_center: mesh.bounds_center,
            local_bounds_radius: mesh.bounds_radius,
            local_aabb_min: mesh.aabb_min,
            local_aabb_max: mesh.aabb_max,
            enabled: true,
        });

        self.generation = self.generation.wrapping_add(1);
        id
    }

    /// Remove an object.
    ///
    /// Equivalent to Unreal's `FScene::RemovePrimitive()`.
    pub fn remove_object(&mut self, id: ObjectId) {
        if let Some(proxy) = self.proxies.remove(&id.0) {
            // Zero out the slot so shaders don't read stale data
            self.cpu_data[proxy.slot as usize] = GpuInstanceData::zeroed();
            self.cpu_aabbs[proxy.slot as usize] = GpuInstanceAabb::zeroed();
            self.mark_dirty(proxy.slot);
            self.allocator.free(proxy.slot);
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Update the transform of a registered object.
    /// Only writes to the CPU mirror if the hash changed — zero cost for static objects.
    ///
    /// Equivalent to Unreal's `FScene::UpdatePrimitiveTransform()`.
    pub fn update_transform(&mut self, id: ObjectId, transform: glam::Mat4) {
        if let Some(proxy) = self.proxies.get_mut(&id.0) {
            let mat = transform.to_cols_array();
            let hash = fnv1a_mat(&mat);
            if hash != proxy.transform_hash {
                let slot = proxy.slot;
                let instance = GpuInstanceData::from_transform(
                    transform,
                    proxy.local_bounds_center,
                    proxy.local_bounds_radius,
                );
                let local_aabb = Aabb::new(proxy.local_aabb_min.into(), proxy.local_aabb_max.into());
                let world_aabb = local_aabb.transform(&transform);
                self.cpu_data[slot as usize] = instance;
                self.cpu_aabbs[slot as usize] = GpuInstanceAabb {
                    aabb_min: world_aabb.min.into(),
                    _pad0: 0.0,
                    aabb_max: world_aabb.max.into(),
                    _pad1: 0.0,
                };
                proxy.transform_hash = hash;
                self.mark_dirty(slot);
            }
        }
    }

    /// Batch-update many transforms.
    pub fn update_transforms(&mut self, updates: &[(ObjectId, glam::Mat4)]) {
        for &(id, transform) in updates {
            self.update_transform(id, transform);
        }
    }

    /// Enable or disable an object.  Disabled objects are excluded from all
    /// draw lists but keep their GPU slot — re-enabling is instant.
    pub fn set_object_enabled(&mut self, id: ObjectId, enabled: bool) {
        if let Some(proxy) = self.proxies.get_mut(&id.0) {
            if proxy.enabled != enabled {
                proxy.enabled = enabled;
                self.generation = self.generation.wrapping_add(1);
            }
        }
    }

    /// Returns `true` if the object exists and is currently enabled.
    pub fn is_object_enabled(&self, id: ObjectId) -> bool {
        self.proxies.get(&id.0).map(|p| p.enabled).unwrap_or(false)
    }

    /// Override the bounding sphere radius used for culling.
    /// The center is kept at its current world-space position.
    pub fn set_object_bounds(&mut self, id: ObjectId, radius: f32) {
        if let Some(proxy) = self.proxies.get_mut(&id.0) {
            proxy.local_bounds_radius = radius;
            let slot = proxy.slot;
            let center = proxy.local_bounds_center;
            let mat = glam::Mat4::from_cols_array(&self.cpu_data[slot as usize].transform);
            let instance = GpuInstanceData::from_transform(mat, center, radius);
            self.cpu_data[slot as usize] = instance;
            self.mark_dirty(slot);
        }
    }

    /// Update the material of a registered object.
    pub fn update_material(&mut self, id: ObjectId, material: &GpuMaterial) {
        if let Some(proxy) = self.proxies.get_mut(&id.0) {
            proxy.material_bind_group = Arc::clone(&material.bind_group);
            proxy.transparent = material.transparent_blend;
            // Material change doesn't affect the GPU scene buffer (only bind group selection),
            // but it does require draw list rebuild.
            self.generation = self.generation.wrapping_add(1);
        }
    }

    // ── Persistent draw list cache ─────────────────────────────────────────

    /// Rebuild the cached draw lists from the current proxy registry.
    /// Only called when `generation` differs from `cached_generation`.
    ///
    /// After collecting one `DrawCall` per enabled proxy the opaque list is
    /// sorted by `(vertex_buffer_ptr, index_buffer_ptr, material_ptr, instance_buffer_offset)`
    /// and consecutive entries that share the same mesh + material **and** have
    /// contiguous instance-buffer slots are merged into a single `DrawCall`
    /// with `instance_count > 1`.  This turns N instanced draws of the same
    /// repeated object into one GPU draw call, cutting both command-recording
    /// time and `encoder.finish()` translation time proportionally.
    ///
    /// Transparent draws and shadow draws keep their original per-proxy form
    /// (transparent ordering must be done per-frame; shadow pass already uses
    /// its own culling + bundle compilation).
    fn rebuild_draw_lists(&mut self) {
        self.cached_shadow_draw_list.clear();
        self.draw_call_cpu.clear();
        self.material_ranges.clear();

        let proxy_count = self.proxies.len();
        self.cached_shadow_draw_list.reserve(proxy_count);

        // ── Collect per-proxy draw calls ───────────────────────────────────
        let mut raw: Vec<DrawCall> = Vec::with_capacity(proxy_count);
        for proxy in self.proxies.values() {
            if !proxy.enabled { continue; }
            let inst = &self.cpu_data[proxy.slot as usize];
            let dc = DrawCall {
                vertex_buffer:    proxy.mesh.vertex_buffer.clone(),
                index_buffer:     proxy.mesh.index_buffer.clone(),
                index_count:      proxy.mesh.index_count,
                vertex_count:     proxy.mesh.vertex_count,
                material_bind_group: Arc::clone(&proxy.material_bind_group),
                transparent_blend:   proxy.transparent,
                bounds_center:    inst.bounds_center,
                bounds_radius:    inst.bounds_radius,
                slot:             proxy.slot,
                pool_base_vertex: proxy.mesh.pool_base_vertex as i32,
                pool_first_index: proxy.mesh.pool_first_index,
                pool_allocated:   proxy.mesh.pool_allocated,
            };
            self.cached_shadow_draw_list.push(dc.clone());
            raw.push(dc);
        }

        // ── Sort opaque by material, then slot ────────────────────────────
        let (mut opaque, transparent): (Vec<DrawCall>, Vec<DrawCall>) =
            raw.into_iter().partition(|dc| !dc.transparent_blend);

        opaque.sort_unstable_by(|a, b| {
            (Arc::as_ptr(&a.material_bind_group) as usize)
                .cmp(&(Arc::as_ptr(&b.material_bind_group) as usize))
                .then_with(|| a.slot.cmp(&b.slot))
        });

        // ── Build cached_draw_list (CPU, transparent + shadow passes) ─────
        self.cached_draw_list.clear();
        self.cached_draw_list.reserve(opaque.len() + transparent.len());
        self.cached_draw_list.extend_from_slice(&opaque);
        self.transparent_start = self.cached_draw_list.len();
        self.cached_draw_list.extend(transparent);

        // ── Build draw_call_cpu (GPU indirect, opaque only for now) ───────
        // One GpuDrawCall per pool-allocated opaque proxy in the sorted order.
        let mut current_range_bg: Option<Arc<wgpu::BindGroup>> = None;
        let mut current_mat_ptr: usize = 0;
        let mut range_start: u32 = 0;

        for dc in opaque.iter() {
            if !dc.pool_allocated {
                log::warn!("GpuScene: non-pool mesh in slot {} skipped — use renderer.create_mesh_* to upload geometry to the pool", dc.slot);
                continue;
            }
            let mat_ptr  = Arc::as_ptr(&dc.material_bind_group) as usize;
            let pool_idx = self.draw_call_cpu.len() as u32;

            self.draw_call_cpu.push(GpuDrawCall {
                slot:          dc.slot,
                first_index:   dc.pool_first_index,
                base_vertex:   dc.pool_base_vertex,
                index_count:   dc.index_count,
                bounds_center: dc.bounds_center,
                bounds_radius: dc.bounds_radius,
            });

            if pool_idx == 0 {
                // First pool draw: open the first material range.
                current_mat_ptr  = mat_ptr;
                range_start      = 0;
                current_range_bg = Some(Arc::clone(&dc.material_bind_group));
            } else if mat_ptr != current_mat_ptr {
                // Material changed: close previous range and open a new one.
                self.material_ranges.push(MaterialRange {
                    bind_group: current_range_bg.take().unwrap(),
                    start: range_start,
                    count: pool_idx - range_start,
                });
                current_mat_ptr  = mat_ptr;
                range_start      = pool_idx;
                current_range_bg = Some(Arc::clone(&dc.material_bind_group));
            }
        }
        // Flush the last open material range.
        if !self.draw_call_cpu.is_empty() {
            self.material_ranges.push(MaterialRange {
                bind_group: current_range_bg.unwrap(),
                start: range_start,
                count: self.draw_call_cpu.len() as u32 - range_start,
            });
        }

        self.draw_call_count = self.draw_call_cpu.len() as u32;
        self.draw_call_dirty = true;
        self.cached_generation = self.generation;
    }

    /// Ensure cached draw lists are up-to-date and return references to them.
    /// At steady state (no add/remove) this is a no-op — zero cost.
    pub fn draw_lists(&mut self) -> (&[DrawCall], &[DrawCall]) {
        if self.cached_generation != self.generation {
            self.rebuild_draw_lists();
        }
        (&self.cached_draw_list, &self.cached_shadow_draw_list)
    }

    /// Number of persistent draw calls in the cached lists.
    pub fn persistent_draw_count(&self) -> usize {
        self.cached_draw_list.len()
    }

    // ── Per-frame GPU upload ─────────────────────────────────────────────────

    /// Flush all dirty slots to the GPU.  Called once per frame.
    ///
    /// Uses a single contiguous `write_buffer` over the dirty range.
    /// At steady state with no moved objects, this is a no-op.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        let Some((lo, hi)) = self.dirty_range.take() else {
            self.last_upload_slot_count = 0;
            self.last_upload_bytes = 0;
            return;
        };

        let slot_count = (hi - lo + 1) as usize;

        // ── Storage buffer upload (128 bytes per slot) ───────────────────
        let byte_offset = lo as u64 * INSTANCE_SIZE as u64;
        let data_slice = &self.cpu_data[lo as usize..=(hi as usize)];
        queue.write_buffer(&self.instance_buffer, byte_offset, bytemuck::cast_slice(data_slice));

        // ── AABB buffer upload (32 bytes per slot) ────────────────────────
        let aabb_offset = lo as u64 * AABB_SIZE as u64;
        let aabb_slice = &self.cpu_aabbs[lo as usize..=(hi as usize)];
        queue.write_buffer(&self.aabb_buffer, aabb_offset, bytemuck::cast_slice(aabb_slice));

        // Count actual dirty slots for stats
        let mut dirty_count = 0u32;
        for word_idx in (lo as usize / 64)..=((hi as usize) / 64).min(self.dirty_bits.len() - 1) {
            dirty_count += self.dirty_bits[word_idx].count_ones();
        }

        self.last_upload_slot_count = dirty_count;
        self.last_upload_bytes = slot_count as u64 * INSTANCE_SIZE as u64;
        gpu_transfer::track_upload(self.last_upload_bytes);

        for word in &mut self.dirty_bits { *word = 0; }
    }

    /// Flush the GPU draw-call buffer when the draw list was rebuilt.
    /// Call once per frame after `draw_lists()`.
    pub fn flush_draw_calls(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.draw_call_dirty { return; }
        self.draw_call_dirty = false;

        if self.draw_call_cpu.is_empty() { return; }

        // Grow draw_call_buffer if needed
        let needed = self.draw_call_cpu.len() as u32;
        if needed > self.draw_call_buffer_capacity {
            let new_cap = needed.next_power_of_two();
            let new_size = new_cap as u64 * std::mem::size_of::<GpuDrawCall>() as u64;
            self.draw_call_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Scene Draw Call Buffer"),
                size: new_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.draw_call_buffer_capacity = new_cap;
        }

        queue.write_buffer(
            &self.draw_call_buffer,
            0,
            bytemuck::cast_slice(&self.draw_call_cpu),
        );
        gpu_transfer::track_upload(self.draw_call_cpu.len() as u64 * std::mem::size_of::<GpuDrawCall>() as u64);
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    pub fn object_count(&self) -> u32 {
        self.allocator.active_count()
    }

    pub fn capacity(&self) -> u32 {
        self.allocator.capacity
    }

    pub fn high_water_mark(&self) -> u32 {
        self.allocator.high_water_mark()
    }

    pub fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }

    /// The GPU draw-call buffer (one `GpuDrawCall` per enabled opaque proxy).
    pub fn draw_call_buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.draw_call_buffer
    }

    /// The AABB buffer (world-space, 32 bytes per slot) used by the occlusion culling pass.
    pub fn aabb_buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.aabb_buffer
    }

    /// The visibility bitmask buffer written by the occlusion culling pass.
    pub fn visibility_buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.visibility_buffer
    }

    /// Total GPU memory used by all scene buffers.
    pub fn gpu_memory_bytes(&self) -> u64 {
        let cap = self.allocator.capacity as u64;
        cap * INSTANCE_SIZE as u64
            + cap * AABB_SIZE as u64
            + ((cap + 31) / 32) * 4
            + self.draw_call_buffer_capacity as u64 * std::mem::size_of::<GpuDrawCall>() as u64
    }

    /// Get GPU instance data for a slot (for CPU-side culling if needed).
    pub fn get_instance(&self, slot: u32) -> &GpuInstanceData {
        &self.cpu_data[slot as usize]
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn mark_dirty(&mut self, slot: u32) {
        let word = slot as usize / 64;
        let bit = slot as usize % 64;
        self.dirty_bits[word] |= 1u64 << bit;

        match &mut self.dirty_range {
            Some((lo, hi)) => {
                *lo = (*lo).min(slot);
                *hi = (*hi).max(slot);
            }
            None => {
                self.dirty_range = Some((slot, slot));
            }
        }
    }

    fn grow(&mut self, device: &wgpu::Device) {
        let old_storage = self.allocator.capacity as u64 * INSTANCE_SIZE as u64;
        let old_aabb = self.allocator.capacity as u64 * AABB_SIZE as u64;
        gpu_transfer::track_dealloc(old_storage + old_aabb);

        let new_capacity = (self.allocator.capacity * 2).max(INITIAL_CAPACITY);
        log::info!(
            "GpuScene: growing {} → {} slots ({:.1} MB storage + {:.1} MB aabb)",
            self.allocator.capacity,
            new_capacity,
            (new_capacity as f64 * INSTANCE_SIZE as f64) / (1024.0 * 1024.0),
            (new_capacity as f64 * AABB_SIZE as f64) / (1024.0 * 1024.0),
        );

        let buffer_size = new_capacity as u64 * INSTANCE_SIZE as u64;
        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene Instance Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Scene BG"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.instance_buffer.as_entire_binding(),
            }],
        });

        let aabb_size = new_capacity as u64 * AABB_SIZE as u64;
        self.aabb_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene AABB Buffer"),
            size: aabb_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let vis_words = ((new_capacity as usize) + 31) / 32;
        self.visibility_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene Visibility Buffer"),
            size: (vis_words * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        gpu_transfer::track_alloc(buffer_size + aabb_size);

        self.cpu_data.resize(new_capacity as usize, GpuInstanceData::zeroed());
        self.cpu_aabbs.resize(new_capacity as usize, GpuInstanceAabb::zeroed());

        let words = (new_capacity as usize + 63) / 64;
        self.dirty_bits.resize(words, 0u64);

        let slots: Vec<u32> = self.proxies.values().map(|p| p.slot).collect();
        for slot in slots { self.mark_dirty(slot); }

        self.allocator.grow(new_capacity);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn fnv1a_mat(mat: &[f32; 16]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &f in mat {
        h ^= f.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
