//! Scene derivation building blocks no kind of object owns.
//!
//! - The layouts of the rows every join reads: owners, generations, hidden
//!   flags, transforms, mesh bounds ([`OWNER_ROW_BYTES`], …), the shared
//!   WGSL ([`join_source`]) and the frame-origin helpers.
//! - [`RowJoin`]: places a kind's component rows (one source row per
//!   component instance, in the owner's space) into the rows its passes read
//!   (in the frame's space), on the GPU. A kind describes its tables
//!   ([`RowTable`]) and builds its own `RowJoin`: fog, post-process volumes,
//!   the lens, water, foliage, the sky, decals, particle emitters and sprites
//!   each have one in their crate. The hierarchy resolve and the scene join
//!   (meshes, lights) stay in `helio-scene-derivations`.

mod row_join;

pub use row_join::{placement, placement_source, Placement, RowJoin, RowJoinKeys, RowTable};

pub use helio_core::{
    BufferHandle, BufferKey, SceneBufferProjection, SceneDerivation, SceneDerivationContext,
    SceneDerivationOutput,
};

/// An owner row: `entity index`, `generation`, `owner index` (`u32` each).
pub const OWNER_ROW_BYTES: u64 = 12;
/// One `u32` generation per entity index.
pub const GENERATION_ROW_BYTES: u64 = 4;
/// An object's hidden flag (`u32`, zero = visible).
pub const HIDDEN_ROW_BYTES: u64 = 4;
/// An object's transform (the frontend's `TransformRow`): its f64 position
/// as an f32 pair (`hi`, the nearest f32, then `lo`, what that drops), YXZ
/// Euler rotation in degrees, scale (four `[f32; 3]`).
pub const TRANSFORM_ROW_BYTES: u64 = 48;
/// A mesh instance's local bounding sphere `[center.xyz, radius]`.
pub const MESH_BOUNDS_ROW_BYTES: u64 = 16;

/// A join shader with the shared join WGSL (row layouts, transforms) before
/// it.
pub fn join_source(source: &str) -> String {
    format!(
        "{}\n{source}",
        include_str!("../shaders/scene_join_common.wgsl")
    )
}

/// The row join's shader as compiled with no kind's placement (every hook's
/// default; shader validation without a device). A kind's is
/// [`placement_source`].
pub fn row_join_source() -> String {
    placement_source(None)
}

/// The row join's shader without any placement hook.
fn row_join_base() -> String {
    join_source(include_str!("../shaders/row_join.wgsl"))
}

/// Several scene derivations run as one, in order, each seeing the buffers
/// those before it published: how a host installs the derivations of the
/// kinds it renders where it takes one derivation (a renderer builder slot,
/// a test).
pub struct DerivationGroup {
    name: &'static str,
    members: Vec<Box<dyn SceneDerivation>>,
}

impl DerivationGroup {
    pub fn new(name: &'static str, members: Vec<Box<dyn SceneDerivation>>) -> Self {
        Self { name, members }
    }
}

impl SceneDerivation for DerivationGroup {
    fn name(&self) -> &'static str {
        self.name
    }

    fn derive(
        &mut self,
        ctx: &SceneDerivationContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> SceneDerivationOutput {
        let mut inputs = ctx.inputs.clone();
        let mut output = SceneDerivationOutput {
            buffers: Vec::new(),
            flushed: Vec::new(),
            recorded: false,
        };
        for member in &mut self.members {
            let out = member.derive(
                &SceneDerivationContext {
                    device: ctx.device,
                    queue: ctx.queue,
                    inputs: &inputs,
                    world_origin: ctx.world_origin,
                },
                encoder,
            );
            for (key, handle) in &out.buffers {
                inputs.insert(*key, handle.clone());
            }
            output.buffers.extend(out.buffers);
            output.flushed.extend(out.flushed);
            output.recorded |= out.recorded;
        }
        output
    }
}

/// The frame's world origin as an f32 pair (`hi`, `lo`): what joins subtract
/// from placed positions so every output is in the frame's space.
pub fn origin_pair(origin: glam::DVec3) -> ([f32; 4], [f32; 4]) {
    let hi = origin.as_vec3();
    let lo = (origin - hi.as_dvec3()).as_vec3();
    (hi.extend(1.0).to_array(), lo.extend(0.0).to_array())
}

/// The bits of `origin`, for an input signature: a moved origin moves
/// every output position.
pub fn origin_bits(origin: glam::DVec3) -> [u64; 3] {
    [origin.x.to_bits(), origin.y.to_bits(), origin.z.to_bits()]
}

/// Each object composed only up to its nearest rigid frame (48-byte transform
/// rows, like the world rows), as the hierarchy resolve publishes it. Equal to
/// the world rows for objects in no frame.
pub const LOCAL_PLACEMENT_KEY: BufferKey = BufferKey::of("Transform::placement");
/// Each object's space: the slot of its nearest rigid frame ancestor, `0` for
/// the world (`u32` rows). Its `content_generation` moves only when a parent
/// link or frame changed (or after a full resolve), never for a plain move.
pub const OBJECT_SPACE_KEY: BufferKey = BufferKey::of("object_space");
/// Published with the hierarchy resolve's moved list: `[x, y, z]` workgroup
/// counts sized for it. Its `content_generation` equals the placement
/// buffer's, so a join can tell the list belongs to the rows it is reading.
pub const HIERARCHY_MOVED_ARGS_KEY: BufferKey = BufferKey::of("hierarchy_moved_args");

/// A buffer a join owns and publishes, grown by doubling.
pub struct JoinOutput {
    label: &'static str,
    pub buffer: wgpu::Buffer,
    pub rows: u32,
    row_bytes: u64,
    pub epoch: u64,
    pub content_generation: u64,
}

impl JoinOutput {
    pub fn new(device: &wgpu::Device, label: &'static str, row_bytes: u64) -> Self {
        Self {
            label,
            buffer: Self::allocate(device, label, row_bytes, 1),
            rows: 1,
            row_bytes,
            epoch: 0,
            content_generation: 0,
        }
    }

    fn allocate(device: &wgpu::Device, label: &str, row_bytes: u64, rows: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: row_bytes * u64::from(rows.max(1)),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Hold at least `rows` rows (doubling), returning whether it grew.
    pub fn ensure(&mut self, device: &wgpu::Device, rows: u32) -> bool {
        if rows <= self.rows {
            return false;
        }
        let rows = rows.next_power_of_two();
        self.buffer = Self::allocate(device, self.label, self.row_bytes, rows);
        self.rows = rows;
        self.epoch += 1;
        true
    }

    pub fn handle(&self) -> BufferHandle {
        BufferHandle {
            buffer: self.buffer.clone(),
            epoch: self.epoch,
            row_bytes: self.row_bytes,
            content_generation: self.content_generation,
        }
    }
}

/// Rows of `row_bytes` bytes `handle` holds.
pub fn rows_of(handle: &BufferHandle, row_bytes: u64) -> u32 {
    (handle.buffer.size() / row_bytes).min(u64::from(u32::MAX)) as u32
}

/// A value that moves whenever any of `handles` was reallocated, rewritten
/// or resized, or `extra` changed.
pub fn signature(handles: &[Option<&BufferHandle>], extra: &[u64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    };
    for handle in handles {
        match handle {
            Some(handle) => {
                mix(handle.epoch);
                mix(handle.content_generation);
                mix(handle.buffer.size());
            }
            None => mix(u64::MAX),
        }
    }
    for value in extra {
        mix(*value);
    }
    hash
}

/// A whole-buffer bind group entry.
pub fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
