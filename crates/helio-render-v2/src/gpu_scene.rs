//! GPU Scene — persistent per-object GPU data buffer.
//!
//! Equivalent to Unreal Engine's `FGPUScene`.  Every registered object has one
//! 144-byte slot in a persistent GPU STORAGE buffer on the device.  The CPU
//! only writes to slots that have changed since the last `flush_dirty()` call,
//! so a completely static scene pays zero GPU upload bandwidth after the first
//! frame.
//!
//! # Shader access
//!
//! The primitive buffer is bound at **group 3, binding 0** across all geometry
//! passes.  The vertex shader reads the model transform with:
//!
//! ```wgsl
//! let prim = gpu_primitives[instance_index];
//! let model = prim.transform;
//! ```
//!
//! where `instance_index` is the `primitive_id` stored in the 4-byte per-
//! instance vertex stream (replacing the old 64-byte mat4 stream).

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};

// ── GpuPrimitive layout (144 bytes, 16-byte aligned) ─────────────────────────
//
//  Offset  Field          Type          Bytes
//  ─────────────────────────────────────────
//    0     transform      [f32; 16]     64   (col-major mat4x4)
//   64     inv_transpose  [f32; 12]     48   (3 cols of vec4, xyz = inv-transp mat3)
//  112     bounds_center  [f32; 3]      12   (world-space sphere centre)
//  124     bounds_radius  f32            4   (world-space sphere radius)
//  128     material_id    u32            4
//  132     flags          u32            4   (PRIM_CAST_SHADOW | PRIM_TRANSPARENT)
//  136     _pad           [u32; 2]       8
//  ───────────────────────────────────────── total 144 bytes

/// GPU-resident primitive data.  Must match the `GpuPrimitive` WGSL struct in
/// every geometry shader that references group 3 binding 0.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuPrimitive {
    /// Column-major model matrix.
    pub transform:     [f32; 16],
    /// Inverse-transpose of the model matrix upper-left 3×3, stored as 3 columns
    /// of vec4 (12 floats) so shaders can build a `mat3x3` for normal transform.
    pub inv_transpose: [f32; 12],
    /// World-space bounding sphere centre.  Used by the GPU culling compute pass.
    pub bounds_center: [f32; 3],
    /// World-space bounding sphere radius.  `f32::MAX` skips culling for this object.
    pub bounds_radius: f32,
    /// Index into the material table (used so the culling pass can group draws by material).
    pub material_id:   u32,
    /// Object flags — combine `PRIM_CAST_SHADOW` / `PRIM_TRANSPARENT`.
    pub flags:         u32,
    pub _pad:          [u32; 2],
}

pub const PRIM_CAST_SHADOW: u32 = 1 << 0;
pub const PRIM_TRANSPARENT: u32 = 1 << 1;

/// Byte size of one `GpuPrimitive`.
pub const GPU_PRIMITIVE_STRIDE: u64 = std::mem::size_of::<GpuPrimitive>() as u64;

const _SIZE_CHECK: () = assert!(std::mem::size_of::<GpuPrimitive>() == 144);

// ── GpuPrimitive construction helpers ────────────────────────────────────────

impl GpuPrimitive {
    /// Build from a glam `Mat4` model matrix, bounds, material/flags.
    pub fn from_transform(
        transform: glam::Mat4,
        bounds_center: [f32; 3],
        bounds_radius: f32,
        material_id:   u32,
        flags:         u32,
    ) -> Self {
        let t = transform.to_cols_array();
        let inv = compute_inv_transpose(&t);
        Self {
            transform:     t,
            inv_transpose: inv,
            bounds_center,
            bounds_radius,
            material_id,
            flags,
            _pad: [0; 2],
        }
    }

    /// Re-use an existing primitive, replacing only the transform.
    pub fn with_transform(mut self, transform: glam::Mat4) -> Self {
        self.transform     = transform.to_cols_array();
        self.inv_transpose = compute_inv_transpose(&self.transform);
        self
    }
}

/// Compute the inverse-transpose of the upper-left 3×3 of a column-major mat4,
/// returns 12 floats (3 columns of vec4, w component zeroed).
fn compute_inv_transpose(m: &[f32; 16]) -> [f32; 12] {
    // Extract upper-left 3×3 (column-major)
    let c0 = glam::Vec3::new(m[0], m[1], m[2]);
    let c1 = glam::Vec3::new(m[4], m[5], m[6]);
    let c2 = glam::Vec3::new(m[8], m[9], m[10]);

    let mat3 = glam::Mat3::from_cols(c0, c1, c2);
    let inv_t = mat3.inverse().transpose();

    [
        inv_t.x_axis.x, inv_t.x_axis.y, inv_t.x_axis.z, 0.0,
        inv_t.y_axis.x, inv_t.y_axis.y, inv_t.y_axis.z, 0.0,
        inv_t.z_axis.x, inv_t.z_axis.y, inv_t.z_axis.z, 0.0,
    ]
}

// ── PrimitiveSlot ──────────────────────────────────────────────────────────────

/// Typed slot index into the GPU Scene primitive array.
///
/// Returned by [`GpuScene::alloc`], used by [`GpuScene::update`] and
/// [`GpuScene::free`].  The contained `u32` equals `first_instance` in all
/// indirect draw commands, so the vertex shader can use
/// `@builtin(instance_index)` to look up the primitive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PrimitiveSlot(pub u32);

// ── GpuScene ──────────────────────────────────────────────────────────────────

/// Manages the persistent GPU Scene storage buffer.
///
/// # Lifecycle
/// * `new` — allocate the initial GPU buffer and create the bind group
/// * `alloc` — add an object, returns its slot  
/// * `update` — modify an object's data (marks slot dirty)
/// * `free` — remove an object (zeroes the slot, recycles it)
/// * `flush_dirty` — called once per frame; uploads only changed slots
pub struct GpuScene {
    /// `STORAGE | COPY_DST` buffer holding all `GpuPrimitive` data.
    /// Bound at **group 3, binding 0** for all geometry passes.
    pub primitive_buf:      Arc<wgpu::Buffer>,
    /// Bind group wrapping `primitive_buf` — bind at set 3 before every draw.
    pub bind_group:         Arc<wgpu::BindGroup>,
    /// Layout for the bind group above — stored so `grow()` can rebuild.
    pub bind_group_layout:  Arc<wgpu::BindGroupLayout>,
    /// Current buffer capacity in slots.
    pub capacity:           u32,

    primitives:  Vec<GpuPrimitive>,
    freelist:    Vec<u32>,
    next_slot:   u32,
    dirty:       Vec<u32>,
}

impl GpuScene {
    /// Create a GPU Scene with at least `initial_capacity` slots.
    /// The buffer will grow automatically as more objects are registered.
    pub fn new(device: &wgpu::Device, initial_capacity: u32) -> Self {
        let capacity = initial_capacity.max(64).next_power_of_two();
        let (primitive_buf, bind_group_layout, bind_group) =
            Self::create_buffer_and_bg(device, capacity);

        Self {
            primitive_buf,
            bind_group,
            bind_group_layout,
            capacity,
            primitives: vec![GpuPrimitive::zeroed(); capacity as usize],
            freelist:   Vec::new(),
            next_slot:  0,
            dirty:      Vec::new(),
        }
    }

    /// Allocate a new slot, initialise it with `primitive`, and mark it dirty.
    ///
    /// Returns the [`PrimitiveSlot`] that uniquely identifies this object.
    /// The slot is valid until [`GpuScene::free`] is called.
    pub fn alloc(&mut self, primitive: GpuPrimitive, device: &wgpu::Device) -> PrimitiveSlot {
        let slot = if let Some(s) = self.freelist.pop() {
            s
        } else {
            let s = self.next_slot;
            self.next_slot += 1;
            s
        };

        if slot >= self.capacity {
            self.grow(device);
        }

        self.primitives[slot as usize] = primitive;
        self.dirty.push(slot);
        PrimitiveSlot(slot)
    }

    /// Release a slot back to the freelist.  The slot is zeroed and marked dirty
    /// so the GPU doesn't render garbage if it's inadvertently referenced.
    pub fn free(&mut self, slot: PrimitiveSlot, device: &wgpu::Device) {
        self.primitives[slot.0 as usize] = GpuPrimitive::zeroed();
        self.dirty.push(slot.0);
        self.freelist.push(slot.0);
        let _ = device; // kept for API symmetry / future use
    }

    /// Replace the primitive data for `slot` and mark the slot dirty.
    pub fn update(&mut self, slot: PrimitiveSlot, primitive: GpuPrimitive) {
        self.primitives[slot.0 as usize] = primitive;
        self.dirty.push(slot.0);
    }

    /// Upload all dirty slots to the GPU in minimal write operations.
    ///
    /// Called once per frame at the very start of `render()`, before any passes
    /// read the primitive buffer.  For a completely static scene this is a no-op
    /// after the first frame.
    pub fn flush_dirty(&mut self, queue: &wgpu::Queue) {
        if self.dirty.is_empty() { return; }

        // Sort + dedup so contiguous slots can be merged into a single write.
        self.dirty.sort_unstable();
        self.dirty.dedup();

        let mut i = 0;
        while i < self.dirty.len() {
            let range_start = self.dirty[i] as usize;
            let mut range_end = range_start;

            // Extend to the next contiguous slot
            while i + 1 < self.dirty.len()
                && self.dirty[i + 1] as usize == range_end + 1
            {
                i += 1;
                range_end += 1;
            }

            let byte_offset = range_start as u64 * GPU_PRIMITIVE_STRIDE;
            let data = bytemuck::cast_slice(
                &self.primitives[range_start..=range_end],
            );
            queue.write_buffer(&self.primitive_buf, byte_offset, data);
            i += 1;
        }

        self.dirty.clear();
    }

    /// Number of currently allocated (live) slots.
    pub fn live_count(&self) -> u32 {
        self.next_slot.saturating_sub(self.freelist.len() as u32)
    }

    /// Total capacity of the GPU buffer in slots.
    pub fn capacity(&self) -> u32 { self.capacity }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn create_buffer_and_bg(
        device: &wgpu::Device,
        capacity: u32,
    ) -> (Arc<wgpu::Buffer>, Arc<wgpu::BindGroupLayout>, Arc<wgpu::BindGroup>) {
        let buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scene Primitive Buffer"),
            size:  capacity as u64 * GPU_PRIMITIVE_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let bgl = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GPU Scene BGL (group 3)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                }],
            },
        ));

        let bg = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("GPU Scene BG (group 3)"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: buf.as_entire_binding(),
            }],
        }));

        (buf, bgl, bg)
    }

    /// Double the buffer capacity, re-upload all live slots.
    fn grow(&mut self, device: &wgpu::Device) {
        let new_cap = (self.capacity * 2).max(64);
        let (new_buf, _, new_bg) = Self::create_buffer_and_bg(device, new_cap);

        self.primitives.resize(new_cap as usize, GpuPrimitive::zeroed());

        // Mark all live slots dirty so they get re-uploaded into the new buffer.
        self.dirty.clear();
        for i in 0..self.next_slot {
            if !self.freelist.contains(&i) {
                self.dirty.push(i);
            }
        }

        self.primitive_buf = new_buf;
        self.bind_group    = new_bg;
        self.capacity      = new_cap;

        log::debug!(
            "GPU Scene grew to {} slots ({} KB)",
            new_cap,
            new_cap as u64 * GPU_PRIMITIVE_STRIDE / 1024,
        );
    }
}
