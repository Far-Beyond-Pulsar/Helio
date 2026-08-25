//! Virtual-texturing shared contract (Helio#238, texel-streaming S2).
//!
//! This module is the single CPU-side source of truth for everything the
//! `//!use helio_vt` WGSL module (helio-core/src/shader/vt_sample.wgsl) and
//! the density-feedback pipeline agree on:
//!
//! - [`GpuVtMetaRow`] — the per-slot meta row the frame builds transiently
//!   every frame (dims / format / mip geometry / residency floor). The WGSL
//!   `VtMetaRow` struct mirrors it field-for-field; a rename on either side
//!   surfaces as a garbage row (or a validation error), never silently.
//! - The feedback pack scheme — `(slot << 8) | (wanted_mip + 1)` — shared by
//!   the raster shaders' writes, the compaction compute, and
//!   [`unpack_feedback_cell`].
//! - [`mip_first_rank_for`] — the mip→rank LUT that lets the GPU answer "is
//!   this mip's whole page span resident?" with one compare. It is the
//!   arithmetic twin of `helio_component::texture_cache::page_table`
//!   (coarse-first canonical order); the authority test in helio-component
//!   cross-checks the two so they cannot drift.
//!
//! # Layering
//!
//! Lives in libhelio because every consumer already links it: the renderer
//! crates build and bind the rows, the pass crates declare them, and
//! helio-component (which depends on libhelio, not the reverse) consumes the
//! same type from its tier policy without a dependency cycle.

use bytemuck::{Pod, Zeroable};

/// Bindless-table slot count every VT structure is sized for. Matches the
/// 256-wide `scene_textures`/`scene_samplers` binding arrays.
pub const VT_SLOT_COUNT: usize = 256;

/// Bytes per `.ptex` payload page — the rank granularity of SceneDB's static
/// texture-payload layout (rank *r* covers bytes `[r·PAGE, (r+1)·PAGE)`).
///
/// Mirrors `helio_component::texture_cache::PAGE_SIZE`. Duplicated as a plain
/// constant because libhelio cannot see helio-component; the authority test
/// pins the two together.
pub const VT_PAGE_BYTES: u64 = 16 * 1024;

/// Sentinel [`GpuVtMetaRow::floor_flags`] mode meaning "no streaming state is
/// published for this slot" — sampling must behave exactly like an unmanaged
/// fully-resident mip chain, and feedback writes are still recorded (they are
/// observations, not commitments).
pub const VT_MODE_UNMANAGED: u32 = 0;
/// Mode meaning "floor/mip_first_rank columns are authoritative": rule 1
/// clamps LOD to `floor_mip`, rule 2 consults the rank table.
pub const VT_MODE_STREAMED: u32 = 1;

/// Engine-side description of one uploaded texture's `.ptex` geometry — what
/// `TextureUpload` carries so [`GpuVtMetaRow::for_asset`] can build the slot's
/// GPU row at insert time. All six facts come straight off the hydrated
/// component's decoded header; `None` on the upload means "not a streamed
/// asset" and the slot binds as [`GpuVtMetaRow::UNMANAGED`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtTextureMeta {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    /// `BcnFormat`'s stable on-disk discriminant (1/3/4/5/7).
    pub format_discriminant: u32,
    pub srgb: bool,
    /// Compressed bytes per 4×4 block for the format above.
    pub block_bytes: u32,
}

/// One slot's row of the frame-transient VT meta buffer (64 bytes).
///
/// Field groups (also the WGSL mirror order):
/// - `dims_xy_mips` — `[width, height, mip_count, format_discriminant]`.
///   Width/height/mip_count use the same conventions as
///   `texture_cache::{mip_dims, mip_count_for}` (mip 0 = finest, NPOT ceil
///   halving). Format discriminants are `BcnFormat`'s stable on-disk values
///   (1/3/4/5/7); `u32::MAX` = not a compressed asset (no .ptex metadata).
/// - `floor_flags` — `[resident_through_rank, floor_mip, flags, total_ranks]`.
///   `resident_through_rank` is SceneDB's inclusive rank-prefix watermark
///   (`TierSpan::ThroughRank(r)` vocabulary); `u32::MAX` is the unrestricted
///   sentinel ("everything resident", also the default). `floor_mip` is the
///   FINEST mip index whose entire page span lies inside that prefix — the
///   scalar rule 1 clamps LOD against and rule 2 falls back to on a miss.
///   Flags word: bit 0 = sRGB, bits 1.. = mode (`VT_MODE_*`). `total_ranks`
///   lets the GPU close the base mip's span without re-deriving it.
/// - `mip_first_rank` — for the eight FINEST mips (`mip == index`), the first
///   payload rank covering that mip. Mips ≥ 8 are the coarse tail: coarse-first
///   canonical order starts each of them strictly before `mip_first_rank[7]`,
///   so one compare against that entry proves them resident.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuVtMetaRow {
    pub dims_xy_mips: [u32; 4],
    pub floor_flags: [u32; 4],
    pub mip_first_rank: [u32; 8],
}

impl GpuVtMetaRow {
    /// The row bound for a slot with no streaming metadata at all: sampling
    /// degenerates to today's behavior (whole chain treated as resident).
    pub const UNMANAGED: Self = Self {
        dims_xy_mips: [0, 0, 1, u32::MAX],
        floor_flags: [u32::MAX, 0, VT_MODE_UNMANAGED << 1, 0],
        mip_first_rank: [0; 8],
    };

    /// Builds the row for one asset from its `.ptex` header facts.
    ///
    /// Residency starts at the unrestricted sentinel — an unmanaged asset is
    /// fully sampleable until the engine publishes real residency for its
    /// slot (mode stays [`VT_MODE_UNMANAGED`] until then either way).
    pub fn for_asset(
        width: u32,
        height: u32,
        mip_count: u32,
        format_discriminant: u32,
        srgb: bool,
        block_bytes: u32,
    ) -> Self {
        let mip_count = mip_count.max(1);
        let mut row = Self {
            dims_xy_mips: [width.max(1), height.max(1), mip_count, format_discriminant],
            floor_flags: [
                0,
                0,
                vt_mode_flags(VT_MODE_UNMANAGED, srgb),
                mip_first_rank_for(width, height, mip_count, block_bytes).total_ranks,
            ],
            mip_first_rank: mip_first_rank_for(width, height, mip_count, block_bytes).ranks,
        };
        // Unpublished residency = everything resident; floor follows.
        row.set_resident_through(u32::MAX);
        // Mode stays UNMANAGED until the engine publishes streaming state
        // (set_resident_through flips it on; flip back here).
        row.floor_flags[2] = vt_mode_flags(VT_MODE_UNMANAGED, srgb);
        row
    }

    /// Publishes engine-side residency: clamps the floor to what the rank
    /// prefix actually proves (a partially-covered mip is not sampleable).
    /// The unrestricted sentinel ([`u32::MAX`]) means "everything".
    pub fn set_resident_through(&mut self, resident_through_rank: u32) {
        self.floor_flags[0] = resident_through_rank;
        self.floor_flags[1] = finest_fully_resident_mip(self, resident_through_rank);
        let srgb = self.srgb();
        self.floor_flags[2] = vt_mode_flags(VT_MODE_STREAMED, srgb);
    }

    /// sRGB flag carried beside the mode bits (flags word bit 0).
    pub fn srgb(&self) -> bool {
        self.floor_flags[2] & 1 == 1
    }

    /// Streaming mode bits (`VT_MODE_*`, stored shifted into bits 1..).
    pub fn mode(&self) -> u32 {
        self.floor_flags[2] >> 1
    }
}

impl Default for GpuVtMetaRow {
    fn default() -> Self {
        Self::UNMANAGED
    }
}

fn vt_mode_flags(mode: u32, srgb: bool) -> u32 {
    // Flags-word layout: bit 0 = sRGB, bits 1.. = mode value. Mode must not
    // occupy bit 0 or it would collide with the color-space flag.
    (mode << 1) | u32::from(srgb)
}

/// Total ranks implied by a row's own LUT + dims (used by the helpers above;
/// recomputed rather than stored to keep the row exactly 64 bytes).
pub fn total_ranks_for_row(row: &GpuVtMetaRow) -> u32 {
    mip_first_rank_for(
        row.dims_xy_mips[0],
        row.dims_xy_mips[1],
        row.dims_xy_mips[2],
        block_bytes_from_discriminant(row.dims_xy_mips[3]).unwrap_or(16),
    )
    .total_ranks
}

/// `BcnFormat::block_bytes` for an on-disk discriminant; `None` when the row
/// does not describe a compressed asset (`u32::MAX`) or the value is unknown.
pub fn block_bytes_from_discriminant(discriminant: u32) -> Option<u32> {
    match discriminant {
        1 | 4 => Some(8),  // BC1 / BC4
        3 | 5 | 7 => Some(16), // BC3 / BC5 / BC7
        _ => None,
    }
}

/// Result of [`mip_first_rank_for`]: the per-mip first-rank LUT plus the
/// total rank count of the whole chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipFirstRankTable {
    /// First rank covering mip `index`, for the eight FINEST mips only
    /// (index 0 = base mip). Entries past the actual mip count saturate at
    /// the last valid entry.
    pub ranks: [u32; 8],
    /// Total number of page ranks in the whole chain.
    pub total_ranks: u32,
}

/// Arithmetic twin of `texture_cache::page_table`: walks mips COARSE-FIRST
/// accumulating page-rounded spans, then inverts into a finest-eight lookup.
///
/// Must stay formula-identical to the authoritative implementation; pinned by
/// helio-component's `vt_meta_matches_page_table` test.
pub fn mip_first_rank_for(
    width: u32,
    height: u32,
    mip_count: u32,
    block_bytes: u32,
) -> MipFirstRankTable {
    let mip_count = mip_count.max(1);
    // Coarse→fine walk, exactly like texture_cache::mip_segment_table.
    let mut first_rank_by_mip = [0u64; 32];
    let mut offset_pages = 0u64;
    for mip in (0..mip_count).rev() {
        first_rank_by_mip[mip as usize] = offset_pages;
        let bytes = encoded_bytes(width.max(1), height.max(1), mip, block_bytes);
        offset_pages += bytes.div_ceil(VT_PAGE_BYTES);
    }
    let total_ranks = u32::try_from(offset_pages).unwrap_or(u32::MAX);
    let mut ranks = [total_ranks; 8];
    for i in 0..8usize {
        if i < mip_count as usize {
            ranks[i] = u32::try_from(first_rank_by_mip[i]).unwrap_or(u32::MAX);
        }
        // Beyond the chain the entry keeps `total_ranks`: never falsely
        // "resident", and the WGSL side clamps its index into the real chain
        // before looking here anyway.
    }
    MipFirstRankTable { ranks, total_ranks }
}

/// Encoded byte length of one mip — same ceil-to-block geometry as
/// `texture_cache::mip_encoded_bytes`.
fn encoded_bytes(width: u32, height: u32, mip: u32, block_bytes: u32) -> u64 {
    let shift = |v: u32| {
        if mip >= 32 {
            1
        } else {
            (v >> mip).max(1)
        }
    };
    let bx = shift(width).div_ceil(4).max(1) as u64;
    let by = shift(height).div_ceil(4).max(1) as u64;
    bx * by * u64::from(block_bytes)
}

/// The finest mip index whose ENTIRE page span lies within
/// `[0, resident_through_rank]` — the scalar the sampling contract calls the
/// floor. A mip only PARTIALLY covered by the prefix is not sampleable
/// (prefix-complete reads would hit un-promised pages mid-mip), so its span
/// END — the first rank of the next-finer mip, minus one — decides, not its
/// start.
///
/// Coarse-first body order also gives the tabulated range a clean closure:
/// every mip coarser than index 7 starts strictly before `mip_first_rank[7]`,
/// so a prefix reaching `ranks[7]` proves the entire coarse tail at once.
pub fn finest_fully_resident_mip(row: &GpuVtMetaRow, resident_through_rank: u32) -> u32 {
    if resident_through_rank == u32::MAX {
        return 0; // unrestricted sentinel: the base mip is resident
    }
    let mip_count = row.dims_xy_mips[2].max(1);
    let block_bytes = block_bytes_from_discriminant(row.dims_xy_mips[3]).unwrap_or(16);
    let table = mip_first_rank_for(
        row.dims_xy_mips[0],
        row.dims_xy_mips[1],
        mip_count,
        block_bytes,
    );
    let tabulated = mip_count.min(8) as usize;
    // Walk finest→coarsest; the first fully-covered mip wins. Span end for
    // mip m is `ranks[m-1] - 1` (next finer mip starts one past it), and the
    // base mip's end is the last rank of the chain.
    for mip in 0..tabulated {
        let end_exclusive = if mip == 0 {
            table.total_ranks.max(1)
        } else {
            table.ranks[mip - 1].max(1)
        };
        if end_exclusive == 0 || end_exclusive - 1 <= resident_through_rank {
            return mip as u32;
        }
    }
    // Nothing tabulated is fully covered. The coarse tail beyond the table
    // still exists physically (its ranks precede ranks[7]), but the LUT cannot
    // name its exact finest index; fall back to the coarsest tabulated mip,
    // which is where the shader's own miss-path lands anyway.
    (tabulated - 1) as u32
}

// ── Feedback cell packing ─────────────────────────────────────────────────

/// Packs one quarter-res feedback cell write: `(slot << 8) | (wanted_mip + 1)`.
///
/// Bit budget (deliberate): slots get the high 24 bits (the bindless table is
/// 256 wide today; 2²⁴ leaves headroom nobody can outgrow soon), wanted mip +
/// one bias gets the low 8. The +1 bias makes zero the unambiguous "cell never
/// sampled this frame" sentinel — slot 0 demanding mip 0 packs to 1, not 0.
/// 255 mips is far beyond the 32-max NPOT chain, so the budget is safe.
#[inline]
pub fn pack_feedback(slot: u32, wanted_mip: u32) -> u32 {
    debug_assert!(slot < (1 << 24), "slot exceeds the 24-bit feedback budget");
    debug_assert!(wanted_mip < 255, "wanted_mip exceeds the 8-bit feedback budget");
    (slot << 8) | (wanted_mip.min(254) + 1)
}

/// Inverse of [`pack_feedback`]; `None` for the untouched sentinel.
#[inline]
pub fn unpack_feedback(cell: u32) -> Option<(u32, u32)> {
    if cell == 0 {
        None
    } else {
        Some((cell >> 8, (cell & 0xFF) - 1))
    }
}

/// Words in the compaction output buffer: 256 per-slot packed maxima plus the
/// touched-store counter (see helio-pass-vt-density).
pub const VT_FEEDBACK_WORDS: usize = 264;

/// Parsed per-frame feedback snapshot — what `Renderer::take_vt_feedback`
/// hands to whoever runs the policy loop. Consuming is TAKING: the renderer
/// retains nothing after the take, keeping its frame-side stateless.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VtFeedbackSnapshot {
    /// One `(slot, max_wanted_mip)` pair per touched slot, ascending by slot.
    pub wanted_mips: Vec<(u32, u32)>,
    /// Raw feedback stores observed this frame (compaction's atomic counter).
    pub touched_stores: u32,
}

impl VtFeedbackSnapshot {
    /// Decodes one mapped staging copy (`VT_FEEDBACK_WORDS` little-endian
    /// words). Zero cells decode to nothing — the sentinel survives the whole
    /// pipeline.
    pub fn parse(words: &[u32]) -> Self {
        let mut out = Self { touched_stores: words.get(256).copied().unwrap_or(0), wanted_mips: Vec::new() };
        for (slot, &word) in words.iter().take(256).enumerate() {
            if let Some((s, mip)) = unpack_feedback(word) {
                let _ = s; // slot index IS the array index; packed slot kept for symmetry
                out.wanted_mips.push((slot as u32, mip));
            }
        }
        out
    }
}

/// CPU mirror of the WGSL derivative-based wanted-mip estimate: the mip whose
/// texel footprint best matches the fragment's screen-space UV footprint.
///
/// `duv_dx`/`duv_dy` are `dpdx(uv)`/`dpdy(uv)`; `tex_w`/`tex_h` the slot's
/// base dimensions; `mip_count` the slot's chain length (the WGSL twin clamps
/// against the meta row's `dims_xy_mips.z` — passing that value here keeps the
/// two bit-identical). Used by the golden tests as the reference the shader
/// math must reproduce, and by host-side tooling that replays feedback
/// captures.
pub fn wanted_mip_from_derivatives(
    duv_dx: [f32; 2],
    duv_dy: [f32; 2],
    tex_w: f32,
    tex_h: f32,
    mip_count: u32,
) -> u32 {
    let fx = duv_dx[0].abs() * tex_w;
    let fy = duv_dx[1].abs() * tex_h;
    let gx = duv_dy[0].abs() * tex_w;
    let gy = duv_dy[1].abs() * tex_h;
    let footprint = fx.max(fy).max(gx).max(gy);
    let raw = footprint.max(1.0).log2();
    // Round-to-nearest matches textureSampleLevel's trilinear intent closely
    // enough for demand measurement, while never reporting finer than exists —
    // or coarser than the chain's top (same clamp as vt_sample.wgsl).
    let chain_top = mip_count.max(1).saturating_sub(1);
    let mip = raw.round() as i32;
    mip.clamp(0, chain_top.min(i32::MAX as u32) as i32) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_and_sentinel() {
        assert_eq!(pack_feedback(0, 0), 1, "bias keeps zero the untouched sentinel");
        assert_eq!(pack_feedback(7, 3), (7 << 8) | 4);
        assert_eq!(unpack_feedback(0), None);
        assert_eq!(unpack_feedback((7 << 8) | 4), Some((7, 3)));
        assert_eq!(unpack_feedback(pack_feedback(255, 254)), Some((255, 254)));
    }

    #[test]
    fn mip_first_rank_matches_hand_computed_chain() {
        // 1024×1024 BC7 (16 B/block): mip k dims = 1024>>k, blocks = ((d+3)/4)²,
        // bytes = blocks·16. Pages per mip (fine k=0→coarse k=10):
        //   k=0: 65536 blk = 1 MiB  → 64 pages
        //   k=1: 16384 blk = 256 KiB → 16 pages
        //   k=2:  4096 blk = 64 KiB  → 4 pages
        //   k=3:  1024 blk = 16 KiB  → 1 page (exactly one page)
        //   k=4..10: 256 blk…1 blk   → 1 page each (7 mips)
        // Coarse-first cumulative starts: mips 10..0 begin at
        // [0,1,2,3,4,5,6,7,8,12,28]; total = 28 + 64 = 92.
        let t = mip_first_rank_for(1024, 1024, 11, 16);
        assert_eq!(t.ranks[0], 28);
        assert_eq!(t.ranks[1], 12);
        assert_eq!(t.ranks[2], 8);
        assert_eq!(t.ranks[3], 7);
        assert_eq!(t.ranks[4], 6);
        assert_eq!(t.ranks[5], 5);
        assert_eq!(t.ranks[6], 4);
        assert_eq!(t.ranks[7], 3);
        assert_eq!(t.total_ranks, 92);

        let mut row = GpuVtMetaRow::for_asset(1024, 1024, 11, 7, false, 16);
        // Unpublished residency reads as fully resident…
        assert_eq!(row.floor_flags[1], 0);
        // …prefix through rank 43 covers all of mip 1's span (ends at
        // ranks[0]-1 = 27) but none of the base mip → floor stays mip 1.
        row.set_resident_through(43);
        assert_eq!(row.floor_flags[1], 1);
        // Full chain → base mip.
        row.set_resident_through(u32::MAX);
        assert_eq!(row.floor_flags[1], 0);
        // Partial coverage of mip 1 (its span ends at rank 27; prefix stops at
        // 20) must NOT count it — finest fully-resident is mip 2 (ends at
        // ranks[1]-1 = 11 ≤ 20).
        row.set_resident_through(20);
        assert_eq!(row.floor_flags[1], 2);
    }

    #[test]
    fn npot_geometry_halves_with_ceil() {
        // 100×63 BC1 (8 B/block): base blocks = 25×16 = 400 → 3200 B → 1 page.
        // Whole chain stays single-page until a mip exceeds 16 KiB (never here),
        // so every mip's first rank equals its coarse-first ordinal.
        let mip_count = 7; // max dim 100 → 2^⌈log2(100)⌉=128 → 7 mips
        let t = mip_first_rank_for(100, 63, mip_count, 8);
        assert_eq!(t.total_ranks, 7);
        for (i, r) in t.ranks.iter().enumerate().take(mip_count as usize) {
            assert_eq!(*r, (mip_count as u32 - 1 - i as u32), "mip {i}");
        }
    }

    #[test]
    fn wanted_mip_reference_math() {
        // One texel per pixel at mip 0 scale → wants mip 0.
        assert_eq!(
            wanted_mip_from_derivatives([1.0 / 64.0, 0.0], [0.0, 1.0 / 64.0], 64.0, 64.0, 7),
            0
        );
        // Four texels per pixel horizontally → footprint 4 → mip 2.
        assert_eq!(
            wanted_mip_from_derivatives([4.0 / 64.0, 0.0], [0.0, 1.0 / 64.0], 64.0, 64.0, 7),
            2
        );
    }
}
