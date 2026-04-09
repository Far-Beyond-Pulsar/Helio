//! GPU-side data types for the Terra Forge voxel pass (chunked brick-map).

use bytemuck::{Pod, Zeroable};

/// Default brick dimension (8³ voxels per brick).
pub const BRICK_DIM: u32 = 8;

/// Bricks per chunk axis.
pub const CHUNK_DIM_BRICKS: u32 = 32;

/// Number of u32 words per brick (8³ voxels / 4 voxels per u32).
pub const WORDS_PER_BRICK: u32 = (BRICK_DIM * BRICK_DIM * BRICK_DIM) / 4;

/// Sentinel: brick has no allocated voxel data.
pub const BRICK_EMPTY: u32 = 0xFFFF_FFFF;

/// Sentinel: brick is fully solid (all voxels are material 1, no pool data needed).
pub const BRICK_SOLID: u32 = 0xFFFF_FFFE;

/// Max simultaneously loaded chunks.
pub const MAX_LOADED_CHUNKS: u32 = 64;

/// Max mixed (surface) bricks per chunk that need voxel pool data.
pub const MAX_MIXED_BRICKS_PER_CHUNK: u32 = 4096;

/// Indirection grid dimension (cubed = total slots for O(1) chunk lookup).
pub const INDIR_GRID_DIM: u32 = 16;

/// Indirection grid sentinel: no chunk loaded at this position.
pub const INDIR_EMPTY: u32 = 0xFFFF_FFFF;

/// Bricks per chunk (32³).
pub const BRICKS_PER_CHUNK: u32 = CHUNK_DIM_BRICKS * CHUNK_DIM_BRICKS * CHUNK_DIM_BRICKS;

/// Max SDF edit operations.
pub const MAX_EDITS: u32 = 256;

// ── Shared structs ──────────────────────────────────────────────────────────

/// Brick metadata (8 bytes). One per brick.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BrickMeta {
    /// Index into the voxel pool (brick slot index, not byte offset).
    /// `BRICK_EMPTY` = unallocated / fully air.
    /// `BRICK_SOLID` = fully solid, no pool data.
    pub data_offset: u32,
    /// Count of solid voxels (0 = empty).
    pub occupancy: u32,
}

/// Material palette entry (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuMaterial {
    pub color: [f32; 3],
    pub roughness: f32,
}

/// Per-chunk info stored in the chunk table (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ChunkInfo {
    /// Chunk position in chunk-space (integer coordinates).
    pub pos: [i32; 3],
    /// 0 = empty slot, 1 = loaded.
    pub status: u32,
    /// Starting index in brick_pool (in BrickMeta units).
    pub brick_pool_offset: u32,
    /// Starting brick slot in voxel_pool for this chunk's data.
    pub voxel_pool_offset: u32,
    pub _pad: [u32; 2],
}

/// SDF edit operation (48 bytes). Applied during chunk gen and far-field tracing.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct EditOp {
    /// Shape: 0 = sphere, 1 = box.
    pub shape_type: u32,
    /// Operation: 0 = add (smooth union), 1 = subtract (smooth subtraction).
    pub op_type: u32,
    /// Material index to apply (for add ops).
    pub material: u32,
    /// Smoothness parameter for Quilez smooth min/max.
    pub blend_k: f32,
    /// World-space center of the edit.
    pub position: [f32; 3],
    pub _pad0: f32,
    /// Radius (sphere) or half-extents (box).
    pub size: [f32; 3],
    pub _pad1: f32,
}

// ── Per-frame uniforms for the ray marcher (48 bytes) ────────────────────────

/// Ray march uniforms (80 bytes, matches WGSL alignment).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuUniforms {
    pub width: u32,
    pub height: u32,
    pub brick_dim: u32,
    pub chunk_dim_bricks: u32,
    pub voxel_size: f32,
    pub planet_radius: f32,
    pub indir_grid_dim: u32,
    /// Number of active SDF edits.
    pub edit_count: u32,
    /// Chunk-space origin of the indirection grid (bottom-left-front corner).
    pub indir_origin: [i32; 3],
    /// Far-field cell size (computed once per frame from camera-planet distance).
    pub ff_cell_size: f32,
    /// Camera world-space position, subtracted from all coordinates for precision.
    pub camera_offset: [f32; 3],
    pub _pad_cam: f32,
    /// TAA Halton subpixel jitter in pixel space [-0.5, 0.5); matches TaaPass sequence.
    pub jitter: [f32; 2],
    pub _jitter_pad: [f32; 2],
}

// ── Gen uniforms for per-chunk SDF generation (48 bytes) ─────────────────────

/// Uniforms for the GPU generation compute shader (48 bytes).
///
/// `chunk_world_origin` (vec3<f32>) at offset 16 is naturally 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GenUniforms {
    pub chunk_dim_bricks: u32,
    pub brick_dim: u32,
    pub voxel_size: f32,
    pub planet_radius: f32,
    /// World-space position of this chunk's corner (min x,y,z).
    pub chunk_world_origin: [f32; 3],
    /// Max mixed brick slots for this chunk.
    pub max_mixed_bricks: u32,
    /// Starting index in brick_pool for this chunk's BrickMeta data.
    pub brick_pool_offset: u32,
    /// Starting brick slot in voxel_pool for this chunk's voxel data.
    pub voxel_pool_offset: u32,
    /// Number of active SDF edits.
    pub edit_count: u32,
    pub _pad1: u32,
}

// ── Static assertions ────────────────────────────────────────────────────────

const _: () = assert!(std::mem::size_of::<GpuUniforms>() == 80);
const _: () = assert!(std::mem::size_of::<BrickMeta>() == 8);
const _: () = assert!(std::mem::size_of::<GpuMaterial>() == 16);
const _: () = assert!(std::mem::size_of::<GenUniforms>() == 48);
const _: () = assert!(std::mem::size_of::<ChunkInfo>() == 32);
const _: () = assert!(std::mem::size_of::<EditOp>() == 48);
