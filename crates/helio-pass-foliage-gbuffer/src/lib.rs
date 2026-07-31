//! Foliage G-buffer pass — grass blades and cards rasterised into the shared 8-target
//! G-buffer.
//!
//! Phase 2 of the foliage plan (`docs/foliage_rendering_plan.md`). This is the consumer
//! half of the grass pipeline: `FoliagePlacePass` decides *which* blades are visible and
//! at what LOD, and this pass turns those four compacted index lists into four
//! `draw_indirect` calls that fill albedo, normal, ORM, emissive and velocity.
//!
//! # Why it draws into the G-buffer and not a composite
//!
//! `BillboardPass` exists and would have been the cheap way to get vegetation on screen.
//! It composites into `pre_aa` *after* deferred lighting, so anything drawn through it
//! receives no shadows, no SSAO, no GI and no correct TAA. Foliage that does not go
//! through the G-buffer cannot be lit like the rest of the scene, and that is not a
//! quality knob — it is the difference between grass that sits in the world and grass
//! that floats on top of a screenshot of it.
//!
//! # Four draws, no vertex buffer, no index buffer
//!
//! `PrimitiveTopology::TriangleStrip`, `strip_index_format: None`, `cull_mode: None`,
//! and every vertex derived procedurally from `@builtin(vertex_index)`. Per-instance
//! strip restart is guaranteed by the WebGPU spec's primitive-assembly algorithm —
//! assembly runs per instance and a strip is split on the restart value only for
//! *indexed* draws — so a blade is one primitive with no degenerate stitch triangles.
//! `strip_index_format: None` has no effect on a non-indexed draw, but on Vulkan a
//! `Some` value sets `primitiveRestartEnable`, and `None` is the correct state here.
//!
//! Exactly four `draw_indirect` calls covers all the grass in the world on every
//! backend, including WebGPU, which has no `MULTI_DRAW_INDIRECT_COUNT`.
//!
//! # The 8-target pipeline with masked writes
//!
//! Grass writes neither `lightmap_uv`, `sss` nor `extra`, and the obvious optimisation —
//! a 5-target pipeline — is invalid. A pipeline's fragment targets must match the render
//! pass's colour attachments in count and format (`RenderPassContext::check_compatible`
//! compares the lists element-wise), so a 5-target pipeline needs its own render pass,
//! which forfeits subpass fusion. Breaking the chain forces a store and reload of every
//! touched attachment; at 1080p and 48 bytes/sample that is ~100 MiB each way on a
//! tile-based GPU, far more than the lever was ever going to save.
//!
//! So: all eight targets, identical formats to `GBufferPass`'s, and
//! [`wgpu::ColorWrites::empty()`] on the three grass does not write, with those
//! `@location`s omitted from the fragment shader. **The empty write mask is mandatory,
//! not cosmetic** — a declared fragment target with no shader output has an undefined
//! value, so without the mask those three G-buffer channels are filled with garbage
//! wherever grass covers a pixel, and the deferred pass reads them everywhere.
//!
//! # Zero overhead when absent
//!
//! When `frame.foliage` is unwritten, [`FoliageGBufferPass::prepare`] early-returns
//! before any upload and [`FoliageGBufferPass::execute`] records nothing. It still
//! returns `Some(descriptor)` from [`FoliageGBufferPass::render_pass_descriptor`]
//! whenever the G-buffer views exist. That is not an oversight: a pass that
//! conditionally returns `None` based on per-frame state can never participate in
//! subpass fusion, because the executor probes chain membership *once*, at lock time.
//! `BillboardPass` documents the same trap around `helio-pass-billboard/src/lib.rs:494`.
//!
//! # Pass order
//!
//! This pass must sit **immediately after `GBufferPass` and before
//! `VirtualGeometryPass`**, returning the byte-identical eight views in the same order.
//! Chain formation requires an exact attachment-view match
//! (`helio-core/src/graph/scheduling.rs:82-88`), and `VirtualGeometryPass` binds only
//! seven attachments — it omits `gbuffer_velocity` — so it can neither fuse with this
//! pass nor be skipped over by the chain scan (it is a raster pass, not
//! `chain_transparent`). Placing foliage after VG makes the fusion impossible.

mod geometry;

pub use geometry::*;

use std::num::NonZeroU64;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_foliage_core::{FoliageQuality, GpuFoliageType, DEFAULT_SCALE_IN_BAND};

// ═══════════════════════════════════════════════════════════════════════════════
// Interface contract with `FoliagePlacePass`
// ═══════════════════════════════════════════════════════════════════════════════

/// Stride of one `wgpu::util::DrawIndirectArgs` in the `foliage_indirect` buffer.
///
/// `vertex_count`, `instance_count`, `first_vertex`, `first_instance` — four `u32`.
/// The four LOD commands live at byte offsets 0, 16, 32 and 48.
pub const DRAW_INDIRECT_STRIDE: u64 = 16;

/// Recover the tile slot that owns a blade, from its flat arena index.
///
/// # Why the consumer has to do this at all
///
/// A `visible_blades[]` entry is a flat index into `blade_arena`. A blade's packed
/// position, though, is **tile-local** — that is what makes placement a pure function of
/// `(tile_coord, lane, generation)` and therefore reproducible across GPUs — so
/// reconstructing a world position needs the blade's *tile*, which the index alone does
/// not name.
///
/// `FoliagePlacePass` makes that recoverable by partitioning the arena into equal
/// fixed-size slabs, one per tile ring slot, and publishing the slab size as
/// `FoliagePlacePass::blades_per_tile`. The division is then exact and O(1) — the
/// alternative, searching a 4096-entry tile table for the slab containing an index, is
/// not something a vertex shader can do once per vertex.
///
/// The `max(1)` is not decoration: a zero slab size would be a division by zero in the
/// vertex stage, whose result is implementation-defined, and the symptom would be every
/// blade in the world reading tile slot garbage. One blade drawn at the wrong tile is
/// recoverable; an undefined index into the tile table is not.
#[inline]
pub const fn tile_slot_of_blade(blade_index: u32, blades_per_tile: u32) -> u32 {
    blade_index / if blades_per_tile == 0 { 1 } else { blades_per_tile }
}

// ── Packed `visible_blades[]` entry ─────────────────────────────────────────────
//
// The live encoding. A `visible_blades[]` entry is **not** a flat arena index: it is the
// owning tile's ring slot in the high 16 bits and the blade's tile-local index in the low
// 16. `foliage_cull.wgsl` writes it, `foliage_gbuffer.wgsl` reads it back with the same
// shift and mask, and both must change together.
//
// Packing rather than a flat index because the consumer needs the *tile* regardless — a
// blade's stored position is tile-local, so reconstructing a world position requires the
// tile's origin and `bounds_center_y`. Handing the slot over directly saves a division
// per vertex in the hottest shader in the foliage path. `tile_slot_of_blade` above is the
// flat-index alternative, kept only for callers holding a raw arena index.
//
// Both halves have room: the ring is 4096 slots and a tile's slab is a few hundred
// blades, against a 65 536 ceiling each.

/// Bits the owning tile slot is shifted left by in a `visible_blades[]` entry.
pub const VISIBLE_TILE_SHIFT: u32 = 16;

/// Mask selecting the tile-local blade index from a `visible_blades[]` entry.
pub const VISIBLE_LOCAL_MASK: u32 = 0xffff;

/// Pack a `(tile_slot, tile_local_index)` pair into a `visible_blades[]` entry.
#[inline]
pub const fn pack_visible_blade(tile_slot: u32, local_index: u32) -> u32 {
    (tile_slot << VISIBLE_TILE_SHIFT) | (local_index & VISIBLE_LOCAL_MASK)
}

/// Inverse of [`pack_visible_blade`].
#[inline]
pub const fn unpack_visible_blade(packed: u32) -> (u32, u32) {
    (packed >> VISIBLE_TILE_SHIFT, packed & VISIBLE_LOCAL_MASK)
}

/// Element offset of LOD `lod`'s region in `visible_blades[]`, given the per-region
/// stride.
///
/// # The region layout this pass assumes
///
/// `visible_blades` is one buffer holding four **equal-size, contiguous** regions, in
/// LOD order, and region `n` begins at element `n * stride`. The stride is derived from
/// the buffer's own size in [`FoliageGBufferPass::new`] as
/// `buffer.size() / 4 / size_of::<u32>()`, so the producer sizes the buffer and the
/// consumer follows — there is no third place for the two to disagree. For
/// `FoliagePlacePass`'s 16 MiB buffer that comes out at `1 << 20` elements per region,
/// which is what its `FOLIAGE_VISIBLE_PER_LOD_CAPACITY` states.
///
/// Equal-size regions waste a lot of space, and that is deliberate: per-LOD occupancy is
/// bimodal and camera-dependent — a camera looking straight down puts nearly everything
/// in L0, one on the horizon puts nearly everything in L3 — so any unequal split has an
/// ordinary camera angle that overflows it. A single stride also means the region base is
/// one multiply in the shader instead of a table lookup.
///
/// The producer must additionally write `first_instance = 0` and `first_vertex = 0` into
/// every `DrawIndirectArgs`: the region base is applied by this pass from its per-LOD
/// uniform, and WebGPU starts `@builtin(instance_index)` at `firstInstance`, so a
/// non-zero value there would offset the region twice.
#[inline]
pub const fn visible_region_offset(lod: u32, stride: u32) -> u32 {
    lod * stride
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pipeline contract
// ═══════════════════════════════════════════════════════════════════════════════

/// The eight G-buffer colour formats, in the exact order and format `GBufferPass`
/// declares them.
///
/// Duplicated here rather than imported because importing them would make this crate
/// depend on `helio-pass-gbuffer` for a constant. The list is pinned against the real
/// one in `tests/pipeline_contract.rs`; if that ever fails, the fix is here, not there.
pub const GBUFFER_TARGET_FORMATS: [wgpu::TextureFormat; 8] = [
    wgpu::TextureFormat::Rgba8Unorm,  // 0 albedo
    wgpu::TextureFormat::Rgba16Float, // 1 normal
    wgpu::TextureFormat::Rgba8Unorm,  // 2 orm
    wgpu::TextureFormat::Rgba16Float, // 3 emissive
    wgpu::TextureFormat::Rg16Float,   // 4 lightmap_uv
    wgpu::TextureFormat::Rgba16Float, // 5 sss
    wgpu::TextureFormat::Rgba16Float, // 6 extra
    wgpu::TextureFormat::Rg16Float,   // 7 velocity
];

/// The three targets grass does not write, which therefore carry
/// [`wgpu::ColorWrites::empty()`].
///
/// See the crate docs: this is the corrected §12 perf lever, and the mask is what makes
/// it *correct* rather than merely cheap.
pub const UNWRITTEN_TARGET_INDICES: [usize; 3] = [4, 5, 6];

/// The pipeline's fragment target list.
///
/// A function rather than a `const` because [`wgpu::ColorTargetState`] is not
/// const-constructible; the test in `tests/pipeline_contract.rs` reads it, so the thing
/// the pipeline is built from is the thing that is asserted.
pub fn color_target_states() -> [Option<wgpu::ColorTargetState>; 8] {
    std::array::from_fn(|index| {
        Some(wgpu::ColorTargetState {
            format: GBUFFER_TARGET_FORMATS[index],
            // No blending. The cross-fade is a stochastic alpha *test* resolved by TAA,
            // not a blend: blending grass into the G-buffer would write a half-weight
            // normal and a half-weight depth, which is not a surface any deferred
            // lighting model can interpret.
            blend: None,
            write_mask: if UNWRITTEN_TARGET_INDICES.contains(&index) {
                wgpu::ColorWrites::empty()
            } else {
                wgpu::ColorWrites::ALL
            },
        })
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Uniforms
// ═══════════════════════════════════════════════════════════════════════════════

/// `globals.flags` bit: the bound interaction view is the real field, not the 1×1
/// placeholder this pass creates in its constructor.
pub const FLAG_INTERACTION_VALID: u32 = 1;

/// Tint each blade by its LOD instead of shading it, so LOD band boundaries are visible.
///
/// Enabled with `HELIO_FOLIAGE_DEBUG=lod`. Density and coverage artefacts in foliage all
/// look alike from a screenshot — a ring, a step, a gap — and which LOD boundary they sit
/// on is the single most useful fact for diagnosing them. Guessing that from a
/// description costs a build/run cycle each time; colouring it costs one.
///
/// L0 red, L1 green, L2 blue, L3 yellow.
pub const FLAG_DEBUG_LOD: u32 = 2;

/// Whether `HELIO_FOLIAGE_DEBUG=lod` is set.
///
/// Read once and cached: `execute` runs every frame, and an environment lookup per frame
/// would be a syscall on the hot path for a value that cannot change.
fn debug_lod_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HELIO_FOLIAGE_DEBUG")
            .map(|value| value.eq_ignore_ascii_case("lod"))
            .unwrap_or(false)
    })
}

/// Per-frame foliage globals. Mirrors `FoliageGlobals` in `foliage_gbuffer.wgsl`
/// (64 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct FoliageGlobals {
    /// Render target size in pixels. The velocity target is in pixels/frame, so this is
    /// load-bearing rather than informational — a stale size scales every foliage motion
    /// vector and TAA rejects them as outliers.
    pub screen_size: [f32; 2],
    /// Frame index; the temporal axis of the cross-fade dither.
    pub frame: u32,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// xyz unused (the camera position comes from the camera uniform), **w = resident
    /// ring radius in metres** — the input to the scale-in factor.
    ///
    /// A `vec4` rather than a bare `f32`, because that is what the WGSL side declares and
    /// WGSL gives a `vec4<f32>` 16-byte alignment. This field is where an earlier version
    /// of this struct went wrong: it had four consecutive scalars here
    /// (`ring_radius`, `blades_per_tile`, `lod_quality_scale`, `scale_in_band`) against
    /// the shader's single `vec4`, so `camera_ring.w` read back whatever was in
    /// `scale_in_band` — a couple of metres — and every blade's `scale_in` evaluated to
    /// zero. Zero scale-in means zero height, so all 236 000 blades rendered as
    /// degenerate strips: no error, no warning, just bare ground.
    pub camera_ring: [f32; 4],
    /// xy = interaction field world-XZ origin, z = extent in metres, w = `1/extent`.
    pub interaction_field: [f32; 4],
    /// `FoliageQuality::lod_distance_scale`, already sanitised here so the shader does
    /// not have to repeat `select_blade_lod`'s defensive branch on it.
    pub lod_quality_scale: f32,
    /// Ring-entry scale-in band width in metres.
    pub scale_in_band: f32,
    /// Width of the stochastic LOD cross-fade band in metres.
    pub lod_fade_band: f32,
    /// Global multiplier on the interaction bend.
    pub interaction_strength: f32,
}

const _: () = {
    assert!(
        std::mem::size_of::<FoliageGlobals>() == 64,
        "FoliageGlobals must be 64 bytes and match `struct FoliageGlobals` in \
         foliage_gbuffer.wgsl field for field; a mismatch here does not fail to compile, \
         it silently feeds the vertex shader the wrong numbers"
    );
};

/// Per-LOD constants, bound with a dynamic offset so all four draws share one pipeline
/// and one buffer. Mirrors `FoliageLod` in `foliage_gbuffer.wgsl` (32 bytes).
///
/// A dynamic-offset uniform rather than immediate/push constants because immediate data
/// availability varies by backend, and rather than `first_instance` because that would
/// put the region base under the producer's control on one side of the contract and this
/// pass's on the other.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct FoliageLodUniform {
    pub lod: u32,
    pub segments: u32,
    pub vertex_count: u32,
    pub region_base: u32,
    pub width_scale: f32,
    pub height_scale: f32,
    /// 1 when this LOD draws a flat card rather than a tapered blade strip.
    pub is_card: u32,
    pub _pad: u32,
}

/// Stride between per-LOD uniform slots.
///
/// 256 rather than `size_of::<FoliageLodUniform>()` because a dynamic uniform offset
/// must be a multiple of the adapter's `min_uniform_buffer_offset_alignment`, which
/// wgpu caps at 256. Using the maximum is valid on every adapter (256 is a multiple of
/// any smaller power-of-two alignment) and costs 896 wasted bytes, once.
pub const LOD_UNIFORM_STRIDE: u32 = 256;

/// Maximum foliage types, fixed by `GpuBladeInstance`'s 8-bit type id.
pub const MAX_FOLIAGE_TYPES: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// Per-frame decision
// ═══════════════════════════════════════════════════════════════════════════════

/// What the publisher gave us this frame, reduced to the parts that decide whether the
/// pass does anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoliageTables {
    /// Number of entries in `FoliageFrameData::types`.
    pub type_count: u32,
    /// `FoliageFrameData::generation`. Wind must not advance it — see the field's docs
    /// in `libhelio::frame`.
    pub generation: u64,
}

/// The outcome of [`decide_frame`]: everything `prepare` and `execute` need to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoliageFrameDecision {
    /// Whether `execute` records anything at all.
    pub enabled: bool,
    /// Whether the type table needs re-uploading (first frame, or the tables changed).
    pub upload_types: bool,
    /// Whether the wind uniform needs uploading. Every enabled frame: wind carries two
    /// timestamps and both move every frame.
    pub upload_wind: bool,
    /// Number of `draw_indirect` calls `execute` records.
    pub draw_count: u32,
}

/// The zero-overhead guarantee, as a pure function.
///
/// Extracted from `prepare` so the plan's §10 guarantee 2 is testable without a GPU,
/// which is the only way it stays true: "the pass does nothing when there is no foliage"
/// is exactly the kind of property that survives a refactor in spirit and dies in
/// practice, because the cost of breaking it is a constant-size buffer write nobody
/// notices in a profile.
///
/// Three cases:
///
/// - `frame.foliage` unwritten ⇒ nothing at all. Not "upload an empty table", not
///   "record four empty draws" — the publisher deliberately leaves the slot empty rather
///   than writing an empty [`FoliageFrameData`](libhelio::frame::FoliageFrameData) for
///   this reason.
/// - Published but no types registered ⇒ still nothing: a type id indexes the descriptor
///   table directly, so an empty table means no blade can resolve.
/// - Types registered ⇒ four draws, whatever the ring holds. An empty ring is four
///   `draw_indirect` calls with `instance_count == 0`, which is the plan's §10
///   guarantee 3 and costs under 0.02 ms.
pub fn decide_frame(
    tables: Option<FoliageTables>,
    uploaded_generation: Option<u64>,
) -> FoliageFrameDecision {
    match tables {
        Some(tables) if tables.type_count > 0 => FoliageFrameDecision {
            enabled: true,
            upload_types: uploaded_generation != Some(tables.generation),
            upload_wind: true,
            draw_count: LOD_COUNT as u32,
        },
        _ => FoliageFrameDecision {
            enabled: false,
            upload_types: false,
            upload_wind: false,
            draw_count: 0,
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOD cross-fade — CPU mirror
// ═══════════════════════════════════════════════════════════════════════════════

/// Upper distance bound of LOD `level`, with the same non-decreasing repair
/// [`helio_foliage_core::select_blade_lod`] applies.
///
/// The repair matters: a mis-authored ladder like `[8, 45, 20, 120]` must degrade to an
/// empty L2 rather than let a band be skipped, or the scene pops straight from a
/// 3-segment blade to a clump card with no fade between them.
pub fn lod_threshold(lod_distances: [f32; 4], level: u32, quality_scale: f32) -> f32 {
    let scale = if quality_scale.is_finite() && quality_scale > 0.0 {
        quality_scale
    } else {
        1.0
    };
    let mut threshold = 0.0f32;
    for entry in lod_distances.iter().take(level.min(3) as usize + 1) {
        let scaled = entry * scale;
        if scaled > threshold {
            threshold = scaled;
        }
    }
    threshold
}

/// Fraction of blades at `level` that survive the stochastic dither at `distance`.
///
/// CPU mirror of `foliage_cross_fade` in the shader. Symmetric by construction: at any
/// LOD threshold the near level's weight is `f` and the far level's is `1 - f`, so the
/// two sum to exactly one blade's worth of coverage everywhere in the band. That is what
/// makes the dither resolve to constant density instead of a thinning or doubling ring,
/// and it is what `tests/lod.rs` asserts.
pub fn cross_fade_alpha(
    level: u32,
    distance: f32,
    lod_distances: [f32; 4],
    quality_scale: f32,
    band: f32,
) -> f32 {
    let upper = lod_threshold(lod_distances, level, quality_scale);
    let mut alpha = helio_foliage_core::lod_fade_alpha(distance, upper - band, upper);
    if level > 0 {
        let lower = lod_threshold(lod_distances, level - 1, quality_scale);
        alpha = alpha.min(1.0 - helio_foliage_core::lod_fade_alpha(distance, lower - band, lower));
    }
    alpha
}

// ═══════════════════════════════════════════════════════════════════════════════
// The pass
// ═══════════════════════════════════════════════════════════════════════════════

/// Rasterises grass blades and cards into the shared G-buffer.
///
/// Construct it with the four buffers `FoliagePlacePass` owns, following the
/// `ShadowDirtyPass` → `ShadowCullPass` handoff in `helio-default-graphs`: the producer
/// pass is built first and its `Arc<wgpu::Buffer>` fields are cloned into the consumer.
pub struct FoliageGBufferPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,

    // ── Producer-owned buffers (the pinned contract) ─────────────────────────
    blade_arena: Arc<wgpu::Buffer>,
    tile_table: Arc<wgpu::Buffer>,
    visible_blades: Arc<wgpu::Buffer>,
    foliage_indirect: Arc<wgpu::Buffer>,

    // ── Pass-owned buffers ───────────────────────────────────────────────────
    globals_buf: wgpu::Buffer,
    wind_buf: wgpu::Buffer,
    /// This pass's own copy of the foliage type descriptor table.
    ///
    /// A copy rather than a handle to the placement pass's copy because the pinned
    /// constructor contract carries four buffers and this is not one of them. It costs
    /// 24 KiB and one upload per authoring edit — `generation` gates it, and wind
    /// deliberately does not advance `generation`, so a steady-state frame uploads
    /// nothing here.
    type_buf: wgpu::Buffer,
    lod_buf: wgpu::Buffer,
    lod_uniforms_written: bool,

    // ── Interaction placeholder ──────────────────────────────────────────────
    /// TEMPORARY: `FoliageInteractionPass` is phase 3 and does not exist yet, so
    /// `frame.foliage_interaction` is normally unwritten. The bind group layout still
    /// needs a texture and a sampler, so the pass owns a 1×1 stand-in. It is never
    /// sampled: `FLAG_INTERACTION_VALID` stays clear and the shader returns a zero bend
    /// before touching it. Delete this together with the flag once phase 3 lands.
    placeholder_view: wgpu::TextureView,
    placeholder_sampler: wgpu::Sampler,

    bind_group_0: Option<wgpu::BindGroup>,
    bind_group_0_key: Option<(usize, usize, usize)>,
    bind_group_1: wgpu::BindGroup,

    /// Elements per `visible_blades` region. See [`visible_region_offset`].
    region_stride: u32,
    /// `FoliagePlacePass::blades_per_tile` — the fixed arena slab size that makes
    /// [`tile_slot_of_blade`] an exact division.
    blades_per_tile: u32,

    decision: FoliageFrameDecision,
    uploaded_generation: Option<u64>,

    /// Quality preset. Drives the ring radius and the LOD ladder scale, and must match
    /// what `FoliagePlacePass` was configured with — the two independently apply the
    /// same scale, and a mismatch shears the cross-fade.
    pub quality: FoliageQuality,
    /// Width of the stochastic cross-fade band in metres.
    ///
    /// Four metres against the default `[8, 20, 45, 120]` ladder: wide enough that the
    /// dissolve is below the perceptual threshold at walking speed, narrow enough that
    /// the doubled draw cost inside the band stays a few percent of the ring.
    pub lod_fade_band: f32,
    /// Global multiplier on the interaction bend.
    pub interaction_strength: f32,
    /// TEMPORARY: world-XZ origin and extent of the interaction field, `[x, z, extent]`.
    /// Set by phase 3's `FoliageInteractionPass` once it publishes a field; ignored
    /// while `FLAG_INTERACTION_VALID` is clear.
    interaction_field: [f32; 3],
    /// Whether [`FoliageGBufferPass::set_interaction_field`] has ever been called.
    /// Gates `FLAG_INTERACTION_VALID` alongside the published view — see `prepare`.
    interaction_field_published: bool,
}

impl FoliageGBufferPass {
    /// Create the pass against the four buffers `FoliagePlacePass` publishes.
    ///
    /// * `blade_arena` — `GpuBladeInstance[]`, partitioned into equal fixed slabs, one
    ///   per tile ring slot.
    /// * `tile_table` — `GpuFoliageTile[]`, the resident tile ring.
    /// * `visible_blades` — flat `blade_arena` indices in four equal contiguous regions;
    ///   see [`visible_region_offset`] for the region layout.
    /// * `foliage_indirect` — four `DrawIndirectArgs` at byte offsets 0/16/32/48.
    /// * `blades_per_tile` — `FoliagePlacePass::blades_per_tile`, the arena slab size.
    ///   Not derivable from the buffers: it is the one piece of the producer's layout a
    ///   consumer cannot see, and without it a blade index cannot be resolved to the
    ///   tile whose origin its position is relative to. See [`tile_slot_of_blade`].
    pub fn new(
        device: &wgpu::Device,
        blade_arena: Arc<wgpu::Buffer>,
        tile_table: Arc<wgpu::Buffer>,
        visible_blades: Arc<wgpu::Buffer>,
        foliage_indirect: Arc<wgpu::Buffer>,
        blades_per_tile: u32,
    ) -> Self {
        // ── Region stride ─────────────────────────────────────────────────────
        // Derived from the buffer the producer sized, so there is no third place for
        // the two sides to disagree. `max(1)` keeps a degenerate (test-sized) buffer
        // from producing a zero stride that silently aliases all four regions onto
        // region 0.
        let region_stride =
            ((visible_blades.size() / (LOD_COUNT as u64 * 4)) as u32).max(1);

        // ── Buffers ───────────────────────────────────────────────────────────
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FoliageGBuffer/Globals"),
            size: std::mem::size_of::<FoliageGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wind_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FoliageGBuffer/Wind"),
            size: std::mem::size_of::<libhelio::wind::GpuWind>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Fixed size: the type id is 8 bits, so 256 descriptors is the ceiling and the
        // table can never need to grow. 24 KiB, allocated once.
        let type_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FoliageGBuffer/Types"),
            size: (MAX_FOLIAGE_TYPES * std::mem::size_of::<GpuFoliageType>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lod_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FoliageGBuffer/LodConstants"),
            size: (LOD_COUNT as u32 * LOD_UNIFORM_STRIDE) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Interaction placeholder (temporary; see the field's docs) ─────────
        let placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FoliageGBuffer/InteractionPlaceholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            // COPY_DST alongside TEXTURE_BINDING so wgpu can zero-initialise it. It is
            // never sampled — the flag gates that — but an unclearable texture in a bind
            // group is a validation problem waiting for the one build that does sample it.
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let placeholder_view = placeholder.create_view(&wgpu::TextureViewDescriptor::default());
        let placeholder_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FoliageGBuffer/InteractionSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // ── Bind group layouts ────────────────────────────────────────────────
        //
        // Everything here is VERTEX-visible except `globals`, which the fragment stage
        // also reads for the screen size and the dither's frame index. Blade geometry,
        // wind, LOD selection and the interaction bend are all vertex-stage work; the
        // fragment stage only shades and dithers.
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FoliageGBuffer BGL 0"),
                entries: &[
                    // Camera is a read-only storage array; `view_index` selects the
                    // eye in a future single-pass stereo path (mono reads index 0).
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    storage(3), // foliage_types
                    storage(4), // blade_arena
                    storage(5), // tile_table
                    storage(6), // visible_blades
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FoliageGBuffer BGL 1"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // One buffer, four slots, one dynamic offset per draw.
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<FoliageLodUniform>() as u64
                        ),
                    },
                    count: None,
                }],
            });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FoliageGBuffer BG 1"),
            layout: &bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &lod_buf,
                    offset: 0,
                    // One slot's worth, so `offset + size` stays inside the buffer at
                    // every dynamic offset including the last.
                    size: NonZeroU64::new(LOD_UNIFORM_STRIDE as u64),
                }),
            }],
        });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let shader = helio_core::shader::module(
            device,
            "FoliageGBuffer Shader",
            include_str!("../shaders/foliage_gbuffer.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FoliageGBuffer PL"),
            bind_group_layouts: &[Some(&bind_group_layout_0), Some(&bind_group_layout_1)],
            immediate_size: 0,
        });
        let targets = color_target_states();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FoliageGBuffer Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // No vertex buffer. Every attribute is derived from `vertex_index` and
                // the packed instance record — see `geometry`.
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                // No effect on a non-indexed draw, but on Vulkan a `Some` value sets
                // `primitiveRestartEnable`, and `None` is the correct state here.
                strip_index_format: None,
                // Blades are two-sided: a grass blade is a surface with no back. The
                // fragment shader flips the normal by `@builtin(front_facing)`.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Grass is opaque geometry and must occlude what is behind it. The
                // stochastic cross-fade is a discard, not a blend, so a surviving
                // fragment is fully opaque and writing depth is correct.
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            // Single-sampled, like every other Helio raster pass. This is also why the
            // cross-fade cannot use alpha-to-coverage: WebGPU rejects
            // `alphaToCoverageEnabled` when `count == 1`.
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout_0,
            blade_arena,
            tile_table,
            visible_blades,
            foliage_indirect,
            globals_buf,
            wind_buf,
            type_buf,
            lod_buf,
            lod_uniforms_written: false,
            placeholder_view,
            placeholder_sampler,
            bind_group_0: None,
            bind_group_0_key: None,
            bind_group_1,
            region_stride,
            // Clamped once here rather than defended at every use: a zero slab size is a
            // wiring mistake, and one blade per tile is a bounded, visible failure —
            // whereas a division by zero in the vertex stage is implementation-defined.
            blades_per_tile: blades_per_tile.max(1),
            decision: decide_frame(None, None),
            uploaded_generation: None,
            quality: FoliageQuality::default(),
            lod_fade_band: 4.0,
            interaction_strength: 1.0,
            interaction_field: [0.0, 0.0, 64.0],
            interaction_field_published: false,
        }
    }

    /// Elements per `visible_blades` region, derived from the buffer's size.
    /// See [`visible_region_offset`].
    #[inline]
    pub fn region_stride(&self) -> u32 {
        self.region_stride
    }

    /// What the pass decided to do this frame. See [`decide_frame`].
    #[inline]
    pub fn decision(&self) -> FoliageFrameDecision {
        self.decision
    }

    /// TEMPORARY: publish the interaction field's world placement.
    ///
    /// Phase 3's `FoliageInteractionPass` owns the snapped, camera-relative field and is
    /// the only thing that can know this. Until it exists nothing calls this and the
    /// bend is skipped entirely.
    pub fn set_interaction_field(&mut self, origin_xz: [f32; 2], extent_meters: f32) {
        self.interaction_field = [origin_xz[0], origin_xz[1], extent_meters.max(1.0e-3)];
        self.interaction_field_published = true;
    }

    /// The per-LOD constants, in the order they are written into `lod_buf`.
    ///
    /// Public so the LOD ladder's vertex counts can be asserted against
    /// [`LOD_VERTEX_COUNTS`] without a GPU.
    pub fn lod_uniforms(region_stride: u32, cluster_granularity: u32) -> [FoliageLodUniform; LOD_COUNT] {
        // The clump card's width is a *coverage* quantity, not a style choice, so it is
        // derived rather than hardcoded. The producer emits one L3 card per cluster, so
        // that card stands in for `cluster_granularity` blades and must cover their
        // footprint: area scales with the square of width, hence `sqrt(N)`.
        //
        // A fixed constant cannot be right here, because the cluster size is a quality
        // setting — 16 blades on Medium and above, 64 on Low, needing 4x and 8x
        // respectively. The previous hardcoded 2.5 covered about six blades' worth in
        // both cases, which is what produced a hard step down in density at the L2→L3
        // boundary and a far ring visibly thinner than the band in front of it.
        let clump_width = (cluster_granularity.max(1) as f32).sqrt();
        std::array::from_fn(|lod| FoliageLodUniform {
            lod: lod as u32,
            segments: LOD_SEGMENTS[lod],
            vertex_count: LOD_VERTEX_COUNTS[lod],
            region_base: visible_region_offset(lod as u32, region_stride),
            width_scale: if lod == CLUMP_LOD {
                clump_width
            } else {
                LOD_WIDTH_SCALE[lod]
            },
            height_scale: LOD_HEIGHT_SCALE[lod],
            is_card: LOD_IS_CARD[lod] as u32,
            _pad: 0,
        })
    }
}

impl RenderPass for FoliageGBufferPass {
    fn name(&self) -> &'static str {
        "FoliageGBuffer"
    }

    fn reads(&self) -> &'static [&'static str] {
        // The plan's §10 also lists `foliage_interaction`. It is deliberately absent
        // until phase 3: `RenderGraph::validate_dependencies` rejects a read that no
        // prior pass writes, and `FoliageInteractionPass` does not exist yet, so
        // declaring it now would make any graph containing this pass fail to build.
        // Add it in the same commit that adds the pass that writes it.
        &["gbuffer", "foliage_draws"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["gbuffer", "gbuffer_velocity"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // Read-only. The eight G-buffer targets are declared and owned by `GBufferPass`;
        // this pass loads them and draws into them, exactly as `VirtualGeometryPass`
        // does. Declaring a second writer would allocate a second set of textures and
        // the fusion this pass is positioned for would become impossible.
        builder.read("gbuffer");
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // Returns `Some` whenever the G-buffer exists, regardless of whether there is
        // any foliage this frame. A pass that returns `None` on per-frame state can
        // never participate in subpass fusion, because the executor probes chain
        // membership once at lock time — see the crate docs and
        // `helio-pass-billboard/src/lib.rs:494`.
        //
        // The views must be byte-identical to `GBufferPass`'s, in the same order:
        // `compute_chains` requires an exact attachment match before it will fuse.
        let gbuffer = resources.gbuffer.read("FoliageGBuffer")?;
        let lightmap_uv = resources.gbuffer_lightmap_uv.read("FoliageGBuffer")?;
        let sss_target = resources.gbuffer_sss.read("FoliageGBuffer")?;
        let extra_target = resources.gbuffer_extra.read("FoliageGBuffer")?;
        let velocity_target = resources.gbuffer_velocity.read("FoliageGBuffer")?;

        // `LoadOp::Load` on all eight: this pass adds geometry to a G-buffer that
        // `GBufferPass` has already filled. A clear here would erase the scene.
        //
        // Written out longhand rather than through a helper because the *order* is the
        // contract — it has to be readable next to `GBufferPass`'s list and diffable
        // against it.
        const LOAD: wgpu::Operations<wgpu::Color> = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([
                Some(wgpu::RenderPassColorAttachment {
                    view: gbuffer.albedo,
                    resolve_target: None,
                    depth_slice: None,
                    ops: LOAD,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: gbuffer.normal,
                    resolve_target: None,
                    depth_slice: None,
                    ops: LOAD,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: gbuffer.orm,
                    resolve_target: None,
                    depth_slice: None,
                    ops: LOAD,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: gbuffer.emissive,
                    resolve_target: None,
                    depth_slice: None,
                    ops: LOAD,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: lightmap_uv,
                    resolve_target: None,
                    depth_slice: None,
                    ops: LOAD,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: sss_target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: LOAD,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: extra_target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: LOAD,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: velocity_target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: LOAD,
                }),
            ]));

        Some(wgpu::RenderPassDescriptor {
            label: Some("FoliageGBuffer"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    // Load, not clear: blades depth-test against the opaque scene
                    // `GBufferPass` just drew.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let foliage = ctx.frame_resources.foliage.get();
        self.decision = decide_frame(
            foliage.map(|f| FoliageTables {
                type_count: f.type_count,
                generation: f.generation,
            }),
            self.uploaded_generation,
        );
        if !self.decision.enabled {
            // The zero-overhead path: not one buffer write. See `decide_frame`.
            return Ok(());
        }
        // Unreachable by construction — `decide_frame` only enables on `Some` tables —
        // but written as a fallthrough rather than an `expect`, because a panic inside
        // `prepare` takes the whole frame down and there is nothing here worth that.
        let Some(foliage) = foliage else {
            return Ok(());
        };

        // ── Type table (authoring edits only) ─────────────────────────────────
        if self.decision.upload_types {
            let entry = std::mem::size_of::<GpuFoliageType>();
            let wanted = (foliage.type_count as usize).min(MAX_FOLIAGE_TYPES) * entry;
            // Truncate rather than trust: `FoliageFrameData` carries raw bytes and a
            // separate count, so a publisher that got the cast wrong hands us a slice
            // shorter than the count claims. Reading past it would be a CPU-side
            // out-of-bounds; drawing the types that did arrive is recoverable.
            let bytes = &foliage.types[..wanted.min(foliage.types.len())];
            if !bytes.is_empty() {
                ctx.write_buffer(&self.type_buf, 0, bytes);
            }
            self.uploaded_generation = Some(foliage.generation);
        }

        // ── Per-LOD constants (once) ──────────────────────────────────────────
        if !self.lod_uniforms_written {
            for (lod, uniform) in Self::lod_uniforms(self.region_stride, self.quality.cluster_granularity()).iter().enumerate() {
                ctx.write_buffer(
                    &self.lod_buf,
                    lod as u64 * LOD_UNIFORM_STRIDE as u64,
                    bytemuck::bytes_of(uniform),
                );
            }
            self.lod_uniforms_written = true;
        }

        // ── Wind (every frame — both timestamps move) ─────────────────────────
        ctx.write_buffer(&self.wind_buf, 0, bytemuck::bytes_of(&foliage.wind));

        // ── Globals ───────────────────────────────────────────────────────────
        // Three conditions, all required. The field placement is the one that is easy to
        // forget: a published view with no published origin would bend every blade in
        // the world against a 64 m field sitting at the world origin, which looks like a
        // wind bug rather than a wiring bug.
        let interaction_valid = ctx.frame_resources.foliage_interaction.is_some()
            && ctx.frame_resources.foliage_interaction_sampler.is_some()
            && self.interaction_field_published;
        let extent = self.interaction_field[2].max(1.0e-3);
        let globals = FoliageGlobals {
            screen_size: [ctx.width as f32, ctx.height as f32],
            frame: ctx.frame_num as u32,
            flags: if interaction_valid { FLAG_INTERACTION_VALID } else { 0 }
                | if debug_lod_enabled() { FLAG_DEBUG_LOD } else { 0 },
            camera_ring: [0.0, 0.0, 0.0, self.quality.ring_radius()],
            // Sanitised here so the shader does not have to repeat
            // `select_blade_lod`'s defensive branch on every vertex.
            lod_quality_scale: {
                let scale = self.quality.lod_distance_scale();
                if scale.is_finite() && scale > 0.0 {
                    scale
                } else {
                    1.0
                }
            },
            scale_in_band: DEFAULT_SCALE_IN_BAND,
            lod_fade_band: self.lod_fade_band.max(0.0),
            interaction_strength: self.interaction_strength,
            interaction_field: [
                self.interaction_field[0],
                self.interaction_field[1],
                extent,
                1.0 / extent,
            ],
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if ctx.frame_num < 3 || ctx.frame_num % 120 == 0 {
            log::debug!(
                "[foliage][raster] frame={} enabled={} render_pass_open={}",
                ctx.frame_num,
                self.decision.enabled,
                ctx.active_render_pass_ptr().is_some(),
            );
        }
        if !self.decision.enabled {
            // Zero recorded commands. Not four empty draws — nothing.
            return Ok(());
        }
        let Some(pass_ptr) = ctx.active_render_pass_ptr() else {
            // The executor did not open our render pass, which means
            // `render_pass_descriptor` returned `None` (no G-buffer). Nothing to draw
            // into; not an error.
            return Ok(());
        };

        // ── Bind group 0 ──────────────────────────────────────────────────────
        //
        // Rebuilt when the camera buffer reallocates or the interaction view changes.
        // The producer's four buffers are `Arc`s owned by this pass, so they cannot move
        // underneath it and are not part of the key.
        //
        // Built inside a block that ends before the assignment: the interaction view is
        // borrowed either out of `ctx.resources` or out of `self`, and holding that
        // borrow across `self.bind_group_0 = ...` would not compile. The created bind
        // group owns its references, so nothing escapes the block.
        let (key, rebuilt) = {
            let interaction_view = ctx
                .resources
                .foliage_interaction
                .get()
                .unwrap_or(&self.placeholder_view);
            let interaction_sampler = ctx
                .resources
                .foliage_interaction_sampler
                .get()
                .unwrap_or(&self.placeholder_sampler);
            let key = (
                ctx.scene.camera as *const _ as usize,
                interaction_view as *const wgpu::TextureView as usize,
                interaction_sampler as *const wgpu::Sampler as usize,
            );
            let rebuilt = if self.bind_group_0_key != Some(key) {
                log::debug!("FoliageGBuffer: rebuilding bind group 0");
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("FoliageGBuffer BG 0"),
                    layout: &self.bind_group_layout_0,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ctx.scene.camera.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.globals_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.wind_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.type_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.blade_arena.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.tile_table.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.visible_blades.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(interaction_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::Sampler(interaction_sampler),
                        },
                    ],
                }))
            } else {
                None
            };
            (key, rebuilt)
        };
        if let Some(bind_group) = rebuilt {
            self.bind_group_0 = Some(bind_group);
            self.bind_group_0_key = Some(key);
        }

        // ── Four draws, one per LOD ───────────────────────────────────────────
        let pass = unsafe { &mut *pass_ptr };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
        for lod in 0..self.decision.draw_count {
            pass.set_bind_group(1, &self.bind_group_1, &[lod * LOD_UNIFORM_STRIDE]);
            pass.draw_indirect(&self.foliage_indirect, lod as u64 * DRAW_INDIRECT_STRIDE);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_blade_reference_round_trips() {
        for tile_slot in [0u32, 1, 4095, 65535] {
            for local in [0u32, 1, 2559, 65535] {
                let packed = pack_visible_blade(tile_slot, local);
                assert_eq!(unpack_visible_blade(packed), (tile_slot, local));
            }
        }
        // The halves must not bleed into each other, which is the failure that draws
        // grass from a neighbouring tile at a plausible-looking position.
        assert_eq!(pack_visible_blade(0xffff, 0), 0xffff_0000);
        assert_eq!(pack_visible_blade(0, 0xffff), 0x0000_ffff);
    }

    #[test]
    fn region_offsets_are_contiguous_and_in_lod_order() {
        let stride = 1024;
        let offsets: Vec<u32> = (0..4).map(|l| visible_region_offset(l, stride)).collect();
        assert_eq!(offsets, vec![0, 1024, 2048, 3072]);
    }
}
