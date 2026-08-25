//! Engine-native baked texture assets (Helio#237) — the texture twin of
//! [`crate::mesh_cache`] (issues #391/#409/Pulsar-Native#632 precedent).
//!
//! Model import happens **at copy time**: a dropped source image
//! (png/jpg/bmp/tga/gif/webp) is converted — decoded to rgba8, mip-mapped,
//! block-compressed — into an engine-native `.ptex` asset written into the
//! project. **The source file itself is not brought into the project** — only
//! the native asset. Components reference the `.ptex` asset (or, identically,
//! the raw source image, converted in memory at hydrate time without writing
//! anything) through [`crate::components::TextureAssetPath`] wrapper fields;
//! `hydrate_static_mesh_component` populates the component's `#[gpu]` texel
//! fields exactly like it does `vertices`/`indices` for meshes.
//!
//! # Format (`PTEX`)
//!
//! A fixed 48-byte header (magic + version + `content_id: u128` + dims +
//! format + sRGB flag + mip/page counts, all little-endian), then a
//! `page_count`-entry directory (`(mip, valid_len)` u32 pairs), then the body:
//! `page_count` contiguous [`PAGE_SIZE`]-byte pages.
//!
//! The body's canonical order is **coarse-first**: the mip chain is stored
//! starting at the COARSEST mip (the permanent floor) and ending at the finest
//! (base) mip, each mip's byte span padded up to a whole number of pages so
//! every mip starts on a page boundary (only a mip's LAST page may be
//! partially valid — `valid_len` in the directory; waste ≤ one page per mip).
//! This ordering is what makes the SceneDB tier layout below a single static,
//! per-TYPE declaration rather than a per-instance one: rank *r* of the
//! registered layout IS page *r*, and page *r* is always coarser than page
//! *r+1* for every asset this module produces.
//!
//! # Segments (SceneDB#61 / Helio#237 texel-streaming S1)
//!
//! The GPU payload type is [`TexturePayload`] — a transparent `u8` newtype —
//! so every texture chain's `#[gpu] Vec<TexturePayload>` field lands in an
//! interned var-len pool typed by ONE `TypeId`, and ONE static segment layout
//! registered for that type covers every instance:
//!
//! - rank *r* = byte range `[r·PAGE_SIZE, (r+1)·PAGE_SIZE)` for
//!   *r* in `0..MAX_PAYLOAD_PAGES`. Because payloads are coarse-first, rank
//!   order IS the eviction preference order reversed: rank 0 is the coarse
//!   floor (last-to-leave), the highest ranks are the finest mips (evict
//!   first) — exactly `.ptex`'s "fine mips evict first" contract, expressed in
//!   S0's ascending-promotion / reverse-rank-eviction vocabulary.
//! - Every promotable unit is one `(offset,len)` pair (one page), satisfying
//!   the VT-ready clause; bodies beyond the declared bands clamp (S0 skips
//!   empty-clamped units), so gigantic textures degrade gracefully instead of
//!   breaking the prefix invariant.
//!
//! Registration happens ONCE per process ([`ensure_payload_layout_registered`],
//! `std::sync::Once`-guarded; S0's registry additionally treats identical
//! re-registration as an idempotent no-op).
//!
//! # Content identity (PMSH-v2 convention, Pulsar-Native#632)
//!
//! `content_id` is the xxh3-128 hash of the canonical BODY (the concatenated
//! valid page bytes — precisely the bytes a `Vec<TexturePayload>` GPU payload
//! carries, which is what SceneDB interns by). Fresh imports compute it once
//! via [`build_and_encode`]; [`encode`] stores it in the header so loads read
//! it directly. A header id of `0` means "absent" (hand-built/legacy file):
//! [`decode`] backfills by hashing the body once, and the hydrate path primes
//! the memoized path cache from the result so the derive-driven resolve is a
//! warm hit — same shape as the mesh v1 backfill story.
//!
//! # BCn encoding
//!
//! All five encoder paths are implemented here (safe Rust, no external
//! codec): BC1 (opaque color), BC3 (BC4 alpha half + BC1 color half), BC4
//! (single-channel masks), BC5 (two independent BC4 halves — normals/MR), and
//! BC7 restricted to **mode 6** (one subset, RGBAP 7.7.7.7 + unique P-bits,
//! 4-bit indices — the highest-quality single-subset mode per Microsoft's
//! BC7 mode reference; mode-6-only keeps the bitstream writer auditable:
//! 7 bits of mode code + 8×7-bit endpoints + 2 P-bits + 63 index bits =
//! exactly 128). Matching decoders live beside the encoders and are exercised
//! against them (plus constant-color known-answer vectors) by
//! `tests/texture_container.rs`.
//!
//! Semantic → format map ([`semantic_target_format`]): BaseColor/Emissive/
//! SpecularColor → BC7; Normal and MetallicRoughness → BC5 (two channels);
//! Occlusion and SpecularWeight → BC4. BC1/BC3 remain supported paths of the
//! codec itself (and of [`BcnFormat`]); they are exercised directly by the
//! container tests rather than chosen by the default semantic map.
//!
//! # What this module does NOT do
//!
//! No shader/sampling changes (that's S2/Helio#238): nothing here touches how
//! GPUs sample these bytes — the container and its segments are storage-only.
//! Renderer crates still store nothing; scene JSON still stores plain paths;
//! the user-facing placing-textures UX stays "assign a path".

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Once, OnceLock};

use pulsar_scenedb::gpu::{register_segment_layout, Segment};

// ---------------------------------------------------------------------------
// Container constants
// ---------------------------------------------------------------------------

/// Magic for native `.ptex` assets (sibling of `mesh_cache`'s `b"PMSH"`).
const MAGIC: &[u8; 4] = b"PTEX";
/// Current WRITE version — every fresh [`encode`] produces this.
const VERSION: u32 = 1;
/// Header size: magic(4) + version(4) + content_id(16) + width(4) + height(4)
/// + format(4) + srgb(4) + mip_count(4) + page_count(4), all little-endian.
const HEADER: usize = 48;
/// One directory entry: `(mip: u32, valid_len: u32)` little-endian.
const DIR_ENTRY: usize = 8;

/// Canonical page size in bytes: one 128×128-texel page for the 16-byte-block
/// BCn formats (BC3/BC5/BC7), exactly two such native pages for the
/// 8-byte-block formats (BC1/BC4). This is the ".ptex default 128px page" of
/// the issue text, expressed at payload-byte granularity so ONE static SceneDB
/// layout serves every format.
pub const PAGE_SIZE: usize = 16 * 1024;
const PAGE_SIZE_U64: u64 = PAGE_SIZE as u64;

/// How many rank bands the STATIC SceneDB layout declares — the coverage
/// ceiling for one texture payload (4096 × 16 KiB = 64 MiB). Bodies beyond
/// this clamp: trailing finest-page ranks simply never exist for those assets
/// (S0 skips empty-clamped units; the coarse prefix stays complete).
pub const MAX_PAYLOAD_PAGES: u32 = 4096;

// ---------------------------------------------------------------------------
// Block-compressed formats
// ---------------------------------------------------------------------------

/// The BCn family `.ptex` bodies may carry. Discriminants are stable on-disk
/// values (little-endian u32 in the header).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BcnFormat {
    /// 4×4 blocks, 8 bytes/block. Opaque RGB (alpha decoded as 255).
    Bc1,
    /// 4×4 blocks, 16 bytes/block: BC4 alpha half + BC1 color half.
    Bc3,
    /// 4×4 blocks, 8 bytes/block. Single channel.
    Bc4,
    /// 4×4 blocks, 16 bytes/block. Two channels (normals: X,Y — Z is
    /// reconstructed shader-side, which is S2 territory and deliberately not
    /// this module's concern).
    Bc5,
    /// 4×4 blocks, 16 bytes/block. High-quality RGBA. This codec emits
    /// mode-6-only blocks (see the module doc).
    Bc7,
}

impl BcnFormat {
    const BC1_DISCRIMINANT: u32 = 1;
    const BC3_DISCRIMINANT: u32 = 3;
    const BC4_DISCRIMINANT: u32 = 4;
    const BC5_DISCRIMINANT: u32 = 5;
    const BC7_DISCRIMINANT: u32 = 7;

    /// On-disk header discriminant.
    pub const fn discriminant(self) -> u32 {
        match self {
            BcnFormat::Bc1 => Self::BC1_DISCRIMINANT,
            BcnFormat::Bc3 => Self::BC3_DISCRIMINANT,
            BcnFormat::Bc4 => Self::BC4_DISCRIMINANT,
            BcnFormat::Bc5 => Self::BC5_DISCRIMINANT,
            BcnFormat::Bc7 => Self::BC7_DISCRIMINANT,
        }
    }

    /// Inverse of [`Self::discriminant`]; `None` for anything else (unknown
    /// on-disk values reject loudly rather than misreading bytes).
    pub const fn from_discriminant(d: u32) -> Option<Self> {
        match d {
            Self::BC1_DISCRIMINANT => Some(BcnFormat::Bc1),
            Self::BC3_DISCRIMINANT => Some(BcnFormat::Bc3),
            Self::BC4_DISCRIMINANT => Some(BcnFormat::Bc4),
            Self::BC5_DISCRIMINANT => Some(BcnFormat::Bc5),
            Self::BC7_DISCRIMINANT => Some(BcnFormat::Bc7),
            _ => None,
        }
    }

    /// Compressed bytes per 4×4 block.
    pub const fn block_bytes(self) -> usize {
        match self {
            // 16-byte-block formats (BC3/BC5/BC7): one canonical page ==
            // one native 128×128 page. 8-byte formats (BC1/BC4): two.
            BcnFormat::Bc1 | BcnFormat::Bc4 => 8,
            BcnFormat::Bc3 | BcnFormat::Bc5 | BcnFormat::Bc7 => 16,
        }
    }

    /// The `wgpu::TextureFormat` these bytes upload as (storage-side mapping
    /// only — binding/sampling decisions stay with the renderer, S2).
    pub const fn to_wgpu(self, srgb: bool) -> wgpu::TextureFormat {
        match (self, srgb) {
            (BcnFormat::Bc1, false) => wgpu::TextureFormat::Bc1RgbaUnorm,
            (BcnFormat::Bc1, true) => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
            (BcnFormat::Bc3, false) => wgpu::TextureFormat::Bc3RgbaUnorm,
            (BcnFormat::Bc3, true) => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
            (BcnFormat::Bc4, _) => wgpu::TextureFormat::Bc4RUnorm,
            (BcnFormat::Bc5, _) => wgpu::TextureFormat::Bc5RgUnorm,
            (BcnFormat::Bc7, false) => wgpu::TextureFormat::Bc7RgbaUnorm,
            (BcnFormat::Bc7, true) => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
        }
    }
}

impl fmt::Display for BcnFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            BcnFormat::Bc1 => "BC1",
            BcnFormat::Bc3 => "BC3",
            BcnFormat::Bc4 => "BC4",
            BcnFormat::Bc5 => "BC5",
            BcnFormat::Bc7 => "BC7",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// Mip geometry (NPOT + 4×4 block alignment)
// ---------------------------------------------------------------------------

/// Dimensions of mip level `mip` (0 = finest/base): each level halves with
/// ceil, clamping at 1 — the standard NPOT chain convention.
pub fn mip_dims(width: u32, height: u32, mip: u32) -> (u32, u32) {
    let shift = |v: u32| {
        if mip >= 32 {
            1
        } else {
            (v >> mip).max(1)
        }
    };
    (shift(width), shift(height))
}

/// Number of mips in the chain down to 1×1 (inclusive).
pub fn mip_count_for(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// Encoded byte length of one mip: dimensions ceil-div'd UP to 4×4 block
/// granularity (a 2×1 tail occupies a full padded 4×4 block — the documented
/// padding rule; sub-block tails never truncate mid-block).
pub fn mip_encoded_bytes(width: u32, height: u32, mip: u32, format: BcnFormat) -> u64 {
    let (w, h) = mip_dims(width, height, mip);
    let bx = w.div_ceil(4).max(1) as u64;
    let by = h.div_ceil(4).max(1) as u64;
    bx * by * format.block_bytes() as u64
}

/// One entry of the per-MIP segment table (CPU-facing, exact): the mip's
/// `(offset, len)` within the canonical coarse-first body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipSegment {
    /// Mip level (0 = base/finest).
    pub mip: u32,
    /// Byte offset of this mip's start within the body (page-aligned by
    /// construction).
    pub offset: u64,
    /// Exact encoded byte length (NOT page-rounded).
    pub len: u64,
}

/// The canonical coarse-first per-MIP table: mips ordered coarsest-first, each
/// starting where the previous one's PAGE-rounded span ends. Hand-checkable
/// (see `tests/texture_container.rs`'s NPOT case: 100×63 BC1).
pub fn mip_segment_table(width: u32, height: u32, mip_count: u32, format: BcnFormat) -> Vec<MipSegment> {
    let mut coarse_to_fine = Vec::with_capacity(mip_count as usize);
    let mut offset = 0u64;
    for mip in (0..mip_count).rev() {
        let len = mip_encoded_bytes(width, height, mip, format);
        coarse_to_fine.push(MipSegment { mip, offset, len });
        // Every mip starts on a page boundary: round the SPAN (not the len)
        // up to PAGE_SIZE so the next mip begins page-aligned.
        offset += len.div_ceil(PAGE_SIZE_U64) * PAGE_SIZE_U64;
    }
    coarse_to_fine
}

/// One canonical page: which mip it belongs to and how many of its
/// [`PAGE_SIZE`] bytes are valid (only a mip's last page may be partial).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageEntry {
    pub mip: u32,
    pub valid_len: u32,
}

/// The per-PAGE table (payload/GPU-facing): page *i* lives at body offset
/// `i·PAGE_SIZE`. This is exactly the shape the static SceneDB layout ranks —
/// entry *i*'s rank is *i* (coarse-first), so promotion walks coarse→fine and
/// eviction walks fine→coarse for every instance.
pub fn page_table(width: u32, height: u32, mip_count: u32, format: BcnFormat) -> Vec<PageEntry> {
    let mut pages = Vec::new();
    for seg in mip_segment_table(width, height, mip_count, format) {
        let full_pages = (seg.len / PAGE_SIZE_U64) as usize;
        let tail = seg.len % PAGE_SIZE_U64;
        for _ in 0..full_pages {
            pages.push(PageEntry { mip: seg.mip, valid_len: PAGE_SIZE as u32 });
        }
        if tail > 0 {
            pages.push(PageEntry { mip: seg.mip, valid_len: tail as u32 });
        }
    }
    pages
}

// ---------------------------------------------------------------------------
// Image + container encode/decode
// ---------------------------------------------------------------------------

/// An in-memory `.ptex` image: metadata plus the canonical coarse-first page
/// sequence (`pages[i].1.len()` in `1..=PAGE_SIZE`; only a mip's last page may
/// be shorter).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PTexImage {
    pub width: u32,
    pub height: u32,
    pub format: BcnFormat,
    pub srgb: bool,
    pub mip_count: u32,
    /// Coarse-first canonical pages as `(mip, valid bytes)`.
    pub pages: Vec<(u32, Vec<u8>)>,
}

impl PTexImage {
    /// The canonical BODY: valid page bytes concatenated, coarse-first — the
    /// exact byte stream a `Vec<TexturePayload>` GPU payload carries and the
    /// exact bytes [`content_id_for_body`] hashes.
    pub fn body(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pages.iter().map(|(_, d)| d.len()).sum());
        for (_, data) in &self.pages {
            out.extend_from_slice(data);
        }
        out
    }

    /// Valid bytes belonging to one mip, in body order (convenience for
    /// tests/decoders).
    pub fn mip_bytes(&self, mip: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for (m, data) in &self.pages {
            if *m == mip {
                out.extend_from_slice(data);
            }
        }
        out
    }
}

/// xxh3-128 over the canonical BODY — the content identity SceneDB interns by
/// (see the module doc). Single source for fresh imports and header backfill.
pub fn content_id_for_body(body: &[u8]) -> u128 {
    twox_hash::XxHash3_128::oneshot(body)
}

/// Serialise a [`PTexImage`] into native `.ptex` bytes (always current
/// version). Prefer [`build_and_encode`] for fresh images: it derives the
/// header id from the body itself. `content_id == 0` is reserved for
/// "absent" (decode-side backfill) and is rejected here so an import can
/// never write an id-less file by accident.
pub fn encode(image: &PTexImage, content_id: u128) -> Vec<u8> {
    assert!(content_id != 0, "content_id 0 is the reserved 'absent' sentinel");
    debug_assert!(image.pages.iter().all(|(_, d)| !d.is_empty() && d.len() <= PAGE_SIZE));
    let mut out = Vec::with_capacity(HEADER + image.pages.len() * (DIR_ENTRY + PAGE_SIZE));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&content_id.to_le_bytes());
    out.extend_from_slice(&image.width.to_le_bytes());
    out.extend_from_slice(&image.height.to_le_bytes());
    out.extend_from_slice(&image.format.discriminant().to_le_bytes());
    out.extend_from_slice(&(u32::from(image.srgb)).to_le_bytes());
    out.extend_from_slice(&image.mip_count.to_le_bytes());
    out.extend_from_slice(&(image.pages.len() as u32).to_le_bytes());

    for (mip, data) in &image.pages {
        out.extend_from_slice(&mip.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    }
    for (_, data) in &image.pages {
        out.extend_from_slice(data);
        out.resize(out.len() + (PAGE_SIZE - data.len()), 0); // zero-pad the tail
    }
    out
}

/// Build a fresh image and serialise it with its own body-derived content id
/// (the common import path — one call, one hash, header always populated).
pub fn build_and_encode(
    width: u32,
    height: u32,
    format: BcnFormat,
    srgb: bool,
    mip_count: u32,
    pages: Vec<(u32, Vec<u8>)>,
) -> (PTexImage, Vec<u8>, u128) {
    let image = PTexImage { width, height, format, srgb, mip_count, pages };
    let id = content_id_for_body(&image.body());
    let bytes = encode(&image, id);
    (image, bytes, id)
}

/// Parse native `.ptex` bytes, or `None` if invalid / truncated / an unknown
/// format. A header id of `0` ("absent") is backfilled by hashing the valid
/// body once ([`content_id_for_body`]).
pub fn decode(bytes: &[u8]) -> Option<(PTexImage, u128)> {
    if bytes.len() < HEADER || &bytes[0..4] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    if version != VERSION {
        return None;
    }
    let stored_id = u128::from_le_bytes(bytes[8..24].try_into().ok()?);
    let width = u32::from_le_bytes(bytes[24..28].try_into().ok()?);
    let height = u32::from_le_bytes(bytes[28..32].try_into().ok()?);
    let format = BcnFormat::from_discriminant(u32::from_le_bytes(bytes[32..36].try_into().ok()?))?;
    let srgb = match u32::from_le_bytes(bytes[36..40].try_into().ok()?) {
        0 => false,
        1 => true,
        _ => return None,
    };
    let mip_count = u32::from_le_bytes(bytes[40..44].try_into().ok()?);
    let page_count = u32::from_le_bytes(bytes[44..48].try_into().ok()?) as usize;

    if bytes.len() < HEADER + page_count * DIR_ENTRY {
        return None;
    }
    let mut directory = Vec::with_capacity(page_count);
    for i in 0..page_count {
        let base = HEADER + i * DIR_ENTRY;
        let mip = u32::from_le_bytes(bytes[base..base + 4].try_into().ok()?);
        let valid = u32::from_le_bytes(bytes[base + 4..base + 8].try_into().ok()?) as usize;
        if valid == 0 || valid > PAGE_SIZE || mip >= mip_count.max(1) {
            return None;
        }
        directory.push((mip, valid));
    }

    let mut image = PTexImage {
        width,
        height,
        format,
        srgb,
        mip_count,
        pages: Vec::with_capacity(page_count),
    };
    // The body is exactly `page_count` PAGE_SIZE blocks after the directory
    // ([encode] pads every page); `valid_len` is the meaningful prefix.
    let body_start = HEADER + page_count * DIR_ENTRY;
    if bytes.len() < body_start + page_count * PAGE_SIZE {
        return None;
    }
    let mut body = Vec::with_capacity(directory.iter().map(|&(_, v)| v).sum());
    for (i, (mip, valid)) in directory.into_iter().enumerate() {
        let start = body_start + i * PAGE_SIZE;
        image.pages.push((mip, bytes[start..start + valid].to_vec()));
        body.extend_from_slice(&bytes[start..start + valid]);
    }

    let id = if stored_id == 0 { content_id_for_body(&body) } else { stored_id };
    Some((image, id))
}

// ---------------------------------------------------------------------------
// SceneDB segment registration (SceneDB#61 §3 — type-author duty, once)
// ---------------------------------------------------------------------------

/// The GPU payload element type for every texture slot's `#[gpu]`
/// `Vec<TexturePayload>` field. Transparent over `u8` so the pool's byte span
/// IS the canonical `.ptex` body byte-for-byte; registered once with the
/// static rank-band layout below.
///
/// # Safety (SceneDB `Pod`)
/// All-zero bytes are trivially valid (`u8`) and there is no `Drop` glue —
/// same one-line contract as `helio::mesh::PackedVertex`'s impl.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TexturePayload(pub u8);

unsafe impl pulsar_scenedb::Pod for TexturePayload {}

static LAYOUT_ONCE: Once = Once::new();

/// Register [`TexturePayload`]'s segment layout with SceneDB exactly once per
/// process (type-author duty per S0 §3). Rank *r* covers payload bytes
/// `[r·PAGE_SIZE, (r+1)·PAGE_SIZE)` — see the module doc for why a single
/// static table is EXACT for every instance (coarse-first canonical order)
/// and how oversize bodies behave (clamp). Idempotent: the `Once` guards the
/// call site AND S0's registry treats identical re-registration as a no-op,
/// so repeated hydrates never trip duplicate-registration errors.
pub fn ensure_payload_layout_registered() {
    LAYOUT_ONCE.call_once(|| {
        let segments: Vec<Segment> = (0..MAX_PAYLOAD_PAGES as u64)
            .map(|rank| Segment::new(rank * PAGE_SIZE_U64, PAGE_SIZE_U64, rank as u32))
            .collect();
        register_segment_layout::<TexturePayload>(&segments)
            .expect("texture payload segment layout must register cleanly");
    });
}

// ---------------------------------------------------------------------------
// Texture semantics (mirrored vocabulary)
// ---------------------------------------------------------------------------

/// The seven authored texture slots. Mirrored — variant-for-variant and in the
/// same order — from `helio_asset_compat`'s crate-private
/// `texture_loader::TextureSemantic`: that crate resolves from a git pin in
/// the Pulsar-Native graph (see `helio-component/Cargo.toml`'s workspace doc),
/// so its local tree is not extensible from here; the vocabulary is stable
/// engine surface either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSemantic {
    BaseColor,
    MetallicRoughness,
    Normal,
    Occlusion,
    Emissive,
    SpecularColor,
    SpecularWeight,
}

impl TextureSemantic {
    /// sRGB-encoded channels (gamma-correct mip filtering applies): color
    /// semantics only — data/normal/mask semantics are linear.
    pub fn is_srgb(self) -> bool {
        matches!(
            self,
            TextureSemantic::BaseColor | TextureSemantic::Emissive | TextureSemantic::SpecularColor
        )
    }

    /// Stable lowercase identifier (import options persistence).
    pub fn suffix(self) -> &'static str {
        match self {
            TextureSemantic::BaseColor => "base-color",
            TextureSemantic::MetallicRoughness => "metallic-roughness",
            TextureSemantic::Normal => "normal",
            TextureSemantic::Occlusion => "occlusion",
            TextureSemantic::Emissive => "emissive",
            TextureSemantic::SpecularColor => "specular-color",
            TextureSemantic::SpecularWeight => "specular-weight",
        }
    }

    /// All seven, in declaration order (options-schema enumeration).
    pub const ALL: [TextureSemantic; 7] = [
        TextureSemantic::BaseColor,
        TextureSemantic::MetallicRoughness,
        TextureSemantic::Normal,
        TextureSemantic::Occlusion,
        TextureSemantic::Emissive,
        TextureSemantic::SpecularColor,
        TextureSemantic::SpecularWeight,
    ];

    /// Parse a [`Self::suffix`] back (options persistence round-trip).
    pub fn from_suffix(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|&v| v.suffix() == s)
    }
}

/// The explicit semantic → BCn map (issue item 3: every variant mapped):
///
/// | Semantic           | Format | Why                                   |
/// |--------------------|--------|---------------------------------------|
/// | BaseColor          | BC7    | high-quality RGBA color               |
/// | Emissive           | BC7    | high-quality RGBA color               |
/// | SpecularColor      | BC7    | high-quality RGBA color               |
/// | Normal             | BC5    | two-channel X,Y; Z reconstructed (S2) |
/// | MetallicRoughness  | BC5    | two-channel R=roughness, G=metallic   |
/// | Occlusion          | BC4    | single-channel mask                   |
/// | SpecularWeight     | BC4    | single-channel mask                   |
pub fn semantic_target_format(semantic: TextureSemantic) -> BcnFormat {
    match semantic {
        TextureSemantic::BaseColor
        | TextureSemantic::Emissive
        | TextureSemantic::SpecularColor => BcnFormat::Bc7,
        TextureSemantic::Normal | TextureSemantic::MetallicRoughness => BcnFormat::Bc5,
        TextureSemantic::Occlusion | TextureSemantic::SpecularWeight => BcnFormat::Bc4,
    }
}

// ---------------------------------------------------------------------------
// BCn codecs (encoders + matching decoders; safe Rust, no external codec)
// ---------------------------------------------------------------------------

/// BC interpolation weights (from Microsoft's BC7 decode spec): mode 6's
/// 4-bit ladder is the only one this module's bitstreams use.
const WEIGHTS4: [u16; 16] = [
    0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64,
];

/// `((64-w)·e0 + w·e1 + 32) >> 6` — the spec's own interpolate().
#[inline]
fn lerp_weighted(e0: u8, e1: u8, weight: u16) -> u8 {
    (((64 - weight) * u16::from(e0) + weight * u16::from(e1) + 32) >> 6) as u8
}

/// Squared RGBA distance (encoder-side nearest-palette fit metric).
#[inline]
fn dist2(a: &[u8; 4], b: &[u8; 4]) -> u32 {
    let d = |x: u8, y: u8| {
        let d = i32::from(x) - i32::from(y);
        (d * d) as u32
    };
    d(a[0], b[0]) + d(a[1], b[1]) + d(a[2], b[2]) + d(a[3], b[3])
}

/// Expand a 5-bit/6-bit channel to 8 bits by bit replication.
#[inline]
fn expand_channel(c: u32, bits: u32) -> u8 {
    let v = (c & ((1 << bits) - 1)) as u8;
    if bits == 5 {
        (v << 3) | (v >> 2)
    } else {
        (v << 2) | (v >> 4)
    }
}

/// Round an 8-bit channel down to `bits` (encoder endpoint quantization):
/// `(c·max + 127) / 255`, i.e. true nearest-of-(max+1) levels.
#[inline]
fn quantize_channel(c: u8, bits: u32) -> u16 {
    if bits == 5 {
        (u16::from(c) * 31 + 127) / 255
    } else {
        (u16::from(c) * 63 + 127) / 255
    }
}

/// One 4×4 texel block of RGBA8, row-major, edge-replicated out of a full mip.
fn block_rgba(rgba: &[u8], width: u32, height: u32, bx: u32, by: u32) -> [[u8; 4]; 16] {
    let mut out = [[0u8; 4]; 16];
    for ty in 0..4u32 {
        for tx in 0..4u32 {
            let px = (bx * 4 + tx).min(width - 1) as usize;
            let py = (by * 4 + ty).min(height - 1) as usize;
            let src = (py * width as usize + px) * 4;
            out[(ty * 4 + tx) as usize] =
                [rgba[src], rgba[src + 1], rgba[src + 2], rgba[src + 3]];
        }
    }
    out
}

/// Encode a whole mip (`rgba`, `width`×`height`) into `format` bytes. Dims are
/// ceil-div'd to 4×4 blocks with edge-replicated padding — the exact inverse
/// of [`mip_encoded_bytes`]'s geometry, so lengths always agree.
pub fn encode_bcn_mip(format: BcnFormat, rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    assert_eq!(rgba.len(), (width * height * 4) as usize, "mip byte length");
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut out = Vec::with_capacity((blocks_x * blocks_y) as usize * format.block_bytes());
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block = block_rgba(rgba, width, height, bx, by);
            match format {
                BcnFormat::Bc1 => out.extend_from_slice(&encode_bc1_block(&block)),
                BcnFormat::Bc3 => out.extend_from_slice(&encode_bc3_block(&block)),
                BcnFormat::Bc4 => out.extend_from_slice(&encode_bc4_block(&block, 0)),
                BcnFormat::Bc5 => out.extend_from_slice(&encode_bc5_block(&block)),
                BcnFormat::Bc7 => out.extend_from_slice(&encode_bc7_block(&block)),
            }
        }
    }
    out
}

/// Decode a whole compressed mip back to RGBA (B/A zeroed where the format
/// doesn't carry them: BC4 → R only; BC5 → R,G only; BC1 → A=255).
pub fn decode_bcn_mip(format: BcnFormat, bytes: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let expected = mip_encoded_bytes(width, height, 0, format) as usize;
    if bytes.len() < expected || width == 0 || height == 0 {
        return None;
    }
    let blocks_x = width.div_ceil(4).max(1) as usize;
    let mut out = vec![0u8; (width * height * 4) as usize];
    let bb = format.block_bytes();
    let mut block_bytes = [0u8; 16];
    for by in 0..height.div_ceil(4).max(1) as usize {
        for bx in 0..blocks_x {
            let src = (by * blocks_x + bx) * bb;
            block_bytes[..bb].copy_from_slice(&bytes[src..src + bb]);
            let decoded = match format {
                BcnFormat::Bc1 => decode_bc1_block(&block_bytes),
                BcnFormat::Bc3 => decode_bc3_block(&block_bytes),
                BcnFormat::Bc4 => decode_bc4_block(&block_bytes, 0),
                BcnFormat::Bc5 => decode_bc5_block(&block_bytes),
                BcnFormat::Bc7 => decode_bc7_block(&block_bytes)?,
            };
            for ty in 0..4usize {
                let py = by * 4 + ty;
                if py >= height as usize {
                    break;
                }
                for tx in 0..4usize {
                    let px = bx * 4 + tx;
                    if px >= width as usize {
                        break;
                    }
                    let dst = (py * width as usize + px) * 4;
                    out[dst..dst + 4].copy_from_slice(&decoded[ty * 4 + tx]);
                }
            }
        }
    }
    Some(out)
}

// ── BC1 ───────────────────────────────────────────────────────────────────

/// Encode ONE 4×4 RGBA block as BC1 (opaque color; alpha ignored, decodes as
/// 255). RGB565 min/max endpoints + full 4-color palette + nearest-index fit.
pub fn encode_bc1_block(block: &[[u8; 4]; 16]) -> [u8; 8] {
    let mut min = [255u8; 3];
    let mut max = [0u8; 3];
    for texel in block {
        for c in 0..3 {
            min[c] = min[c].min(texel[c]);
            max[c] = max[c].max(texel[c]);
        }
    }
    let pack565 = |rgb: &[u8; 3]| -> u16 {
        quantize_channel(rgb[0], 5) << 11
            | quantize_channel(rgb[1], 6) << 5
            | quantize_channel(rgb[2], 5)
    };
    let (c0, c1) = (pack565(&max), pack565(&min));
    let endpoint565 = |c: u16| -> [u8; 4] {
        [
            expand_channel(u32::from(c >> 11), 5),
            expand_channel(u32::from((c >> 5) & 63), 6),
            expand_channel(u32::from(c & 31), 5),
            255,
        ]
    };
    let (e0, e1) = (endpoint565(c0), endpoint565(c1));
    let mix = |num: u16| -> [u8; 4] {
        // (den-num)·e0 + num·e1, rounded; den = 3.
        let f = |a: u8, b: u8| ((3 - num) * u16::from(a) + num * u16::from(b) + 1) / 3;
        [f(e0[0], e1[0]) as u8, f(e0[1], e1[1]) as u8, f(e0[2], e1[2]) as u8, 255]
    };
    let palette = [e0, e1, mix(1), mix(2)];

    let mut bits = BlockBits::new();
    for texel in block {
        let idx = (0..4usize)
            .min_by_key(|&i| dist2(texel, &palette[i]))
            .unwrap();
        bits.write(idx as u32, 2);
    }
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.out[..4]); // 16 × 2-bit indices = 32 bits
    out
}

/// Decode ONE BC1 block to 4×4 RGBA (alpha 255).
pub fn decode_bc1_block(bytes: &[u8]) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([bytes[0], bytes[1]]);
    let c1 = u16::from_le_bytes([bytes[2], bytes[3]]);
    let endpoint565 = |c: u16| -> [u8; 4] {
        [
            expand_channel(u32::from(c >> 11), 5),
            expand_channel(u32::from((c >> 5) & 63), 6),
            expand_channel(u32::from(c & 31), 5),
            255,
        ]
    };
    let (e0, e1) = (endpoint565(c0), endpoint565(c1));
    let mix = |num: u16| -> [u8; 4] {
        let f = |a: u8, b: u8| ((3 - num) * u16::from(a) + num * u16::from(b) + 1) / 3;
        [f(e0[0], e1[0]) as u8, f(e0[1], e1[1]) as u8, f(e0[2], e1[2]) as u8, 255]
    };
    let palette = [e0, e1, mix(1), mix(2)];
    let mut reader = BlockBitsReader::new(&bytes[4..8]);
    let mut out = [[0u8; 4]; 16];
    for texel in &mut out {
        *texel = palette[reader.read(2) as usize];
    }
    out
}

// ── BC4 ───────────────────────────────────────────────────────────────────

/// Encode ONE 4×4 block's channel `channel` (0=R .. 3=A) as BC4, 8-interpolant
/// mode (endpoints ordered max,min — which is what selects the 8-level
/// palette over the 6+2 one).
pub fn encode_bc4_block(block: &[[u8; 4]; 16], channel: usize) -> [u8; 8] {
    let mut min = 255u8;
    let mut max = 0u8;
    for texel in block {
        let v = texel[channel];
        min = min.min(v);
        max = max.max(v);
    }
    let (a0, a1) = (max, min);
    let palette = bc4_palette(a0, a1);
    let mut bits = BlockBits::new();
    for texel in block {
        let v = texel[channel];
        let idx = (0..8usize)
            .min_by_key(|&i| u32::from(palette[i]).abs_diff(u32::from(v)))
            .unwrap();
        bits.write(idx as u32, 3);
    }
    let mut out = [0u8; 8];
    out[0] = a0;
    out[1] = a1;
    out[2..8].copy_from_slice(&bits.out[..6]); // exactly 48 index bits
    out
}

/// The BC4 8-interpolant palette: `{a0, a1, ⌊((8-j)a0+(j-1)a1)/7⌋}` — the
/// decoder-side arithmetic both encoder fit and our decoder share.
fn bc4_palette(a0: u8, a1: u8) -> [u8; 8] {
    let mut p = [0u8; 8];
    p[0] = a0;
    p[1] = a1;
    for j in 2..8usize {
        p[j] = (((8 - j) as u16 * u16::from(a0) + (j - 1) as u16 * u16::from(a1) + 3) / 7) as u8;
    }
    p
}

/// Decode ONE BC4 block's channel into position `channel_out` of RGBA rows.
pub fn decode_bc4_block(bytes: &[u8], channel_out: usize) -> [[u8; 4]; 16] {
    let palette = bc4_palette(bytes[0], bytes[1]);
    let mut reader = BlockBitsReader::new(&bytes[2..8]);
    let mut out = [[0u8; 4]; 16];
    for texel in &mut out {
        texel[channel_out] = palette[reader.read(3) as usize];
    }
    out
}

// ── BC3 / BC5 ─────────────────────────────────────────────────────────────

/// Encode ONE 4×4 RGBA block as BC3: a BC4 alpha half followed by a BC1 color
/// half.
pub fn encode_bc3_block(block: &[[u8; 4]; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&encode_bc4_block(block, 3));
    out[8..16].copy_from_slice(&encode_bc1_block(block));
    out
}

/// Decode ONE BC3 block to 4×4 RGBA.
pub fn decode_bc3_block(bytes: &[u8]) -> [[u8; 4]; 16] {
    let mut out = decode_bc1_block(&bytes[8..16]);
    let alpha = decode_bc4_block(&bytes[0..8], 3);
    for (texel, a) in out.iter_mut().zip(alpha.iter()) {
        texel[3] = a[3];
    }
    out
}

/// Encode ONE 4×4 RGBA block as BC5: two independent BC4 halves (R then G).
/// Blue/Z is intentionally dropped — reconstruction happens shader-side
/// (S2/Helio#238), never here.
pub fn encode_bc5_block(block: &[[u8; 4]; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&encode_bc4_block(block, 0));
    out[8..16].copy_from_slice(&encode_bc4_block(block, 1));
    out
}

/// Decode ONE BC5 block's R,G (B,A left zeroed).
pub fn decode_bc5_block(bytes: &[u8]) -> [[u8; 4]; 16] {
    let r_half = decode_bc4_block(&bytes[0..8], 0);
    let g_half = decode_bc4_block(&bytes[8..16], 1);
    let mut out = [[0u8; 4]; 16];
    for i in 0..16 {
        out[i][0] = r_half[i][0];
        out[i][1] = g_half[i][1];
    }
    out
}

// ── BC7 (mode 6 only — see module doc) ────────────────────────────────────

/// Encode ONE 4×4 RGBA block as BC7 **mode 6**: one subset; each endpoint is
/// RGBA at 7 stored bits plus ONE P-bit shared by all four of its components
/// ("RGBAP 7.7.7.7.1, unique P-bit per endpoint"); 16 4-bit indices with the
/// fix-up trick (texel 0's MSB implied 0). Bitstream, LSB-first:
/// `0000001` (mode code) | R0 R1 G0 G1 B0 B1 A0 A1 (7b each) | P0 P1 |
/// idx0 low 3 bits | idx1..idx15 (4b each) — 7+56+2+3+60 = 128.
pub fn encode_bc7_block(block: &[[u8; 4]; 16]) -> [u8; 16] {
    let mut min = [255u8; 4];
    let mut max = [0u8; 4];
    for texel in block {
        for c in 0..4 {
            min[c] = min[c].min(texel[c]);
            max[c] = max[c].max(texel[c]);
        }
    }
    // Quantize an endpoint's four components to 7 bits each and pick the ONE
    // shared P-bit that reconstructs closest to the intended 8-bit vector.
    let quantize_endpoint = |target: [u8; 4]| -> ([u8; 4], u8) {
        let q = target.map(|c| c >> 1);
        let best_p = {
            let err = |p: u8| -> u32 {
                dist2(
                    &q.map(|c| (c << 1) | p),
                    &[target[0], target[1], target[2], target[3]],
                )
            };
            if err(0) <= err(1) { 0 } else { 1 }
        };
        (q, best_p)
    };
    let (mut q0, mut p0) = quantize_endpoint(max);
    let (mut q1, mut p1) = quantize_endpoint(min);

    let build_palette = |q0: [u8; 4], p0: u8, q1: [u8; 4], p1: u8| -> [[u8; 4]; 16] {
        let e0 = q0.map(|c| (c << 1) | p0);
        let e1 = q1.map(|c| (c << 1) | p1);
        let mut pal = [[0u8; 4]; 16];
        for (i, entry) in pal.iter_mut().enumerate() {
            let w = WEIGHTS4[i];
            for c in 0..4 {
                entry[c] = lerp_weighted(e0[c], e1[c], w);
            }
        }
        pal
    };

    let mut indices = [0usize; 16];
    let fit_indices = |pal: &[[u8; 4]; 16], out: &mut [usize; 16]| {
        for (i, texel) in block.iter().enumerate() {
            out[i] = (0..16usize).min_by_key(|&p| dist2(texel, &pal[p])).unwrap();
        }
    };
    let palette = build_palette(q0, p0, q1, p1);
    fit_indices(&palette, &mut indices);
    // Fix-up contract: texel 0's stored MSB is implied 0 (its index must be
    // < 8). The endpoint swap mirrors the palette symmetrically (index i on
    // swapped endpoints == 15-i), so mirroring the indices is sufficient.
    if indices[0] >= 8 {
        std::mem::swap(&mut q0, &mut q1);
        std::mem::swap(&mut p0, &mut p1);
        for idx in indices.iter_mut() {
            *idx = 15 - *idx;
        }
        debug_assert!(indices[0] < 8);
    }

    let mut bits = BlockBits::new();
    bits.write(0, 6); // mode code 6: six 0s...
    bits.write(1, 1); // ...terminated by a 1
    for c in 0..4 {
        bits.write(u32::from(q0[c]), 7);
        bits.write(u32::from(q1[c]), 7);
    }
    bits.write(u32::from(p0), 1);
    bits.write(u32::from(p1), 1);
    for (i, &idx) in indices.iter().enumerate() {
        if i == 0 {
            bits.write(idx as u32 & 0b111, 3); // MSB implied 0
        } else {
            bits.write(idx as u32, 4);
        }
    }
    debug_assert_eq!(bits.cursor, 128);
    bits.out
}

/// Decode ONE BC7 block back to 4×4 RGBA. Only mode 6 is accepted — the only
/// mode this module writes (documented container subset; conformant hardware
/// decodes mode 6 blocks regardless of what other modes it supports).
pub fn decode_bc7_block(bytes: &[u8]) -> Option<[[u8; 4]; 16]> {
    let mut reader = BlockBitsReader::new(bytes);
    let mut mode = 0u32;
    while mode <= 8 {
        if reader.read(1) == 1 {
            break;
        }
        mode += 1;
    }
    if mode != 6 {
        return None;
    }
    let mut q0 = [0u8; 4];
    let mut q1 = [0u8; 4];
    for c in 0..4 {
        q0[c] = reader.read(7) as u8;
        q1[c] = reader.read(7) as u8;
    }
    let p0 = reader.read(1) as u8;
    let p1 = reader.read(1) as u8;
    let e0 = q0.map(|c| (c << 1) | p0);
    let e1 = q1.map(|c| (c << 1) | p1);
    let mut palette = [[0u8; 4]; 16];
    for (i, entry) in palette.iter_mut().enumerate() {
        let w = WEIGHTS4[i];
        for c in 0..4 {
            entry[c] = lerp_weighted(e0[c], e1[c], w);
        }
    }
    let mut out = [[0u8; 4]; 16];
    for (i, texel) in out.iter_mut().enumerate() {
        let idx = if i == 0 { reader.read(3) as usize } else { reader.read(4) as usize };
        if idx >= 16 {
            return None;
        }
        *texel = palette[idx];
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Bitstream primitives (LSB-first, per 4×4 block)
// ---------------------------------------------------------------------------

struct BlockBits {
    out: [u8; 16],
    cursor: usize,
}

impl BlockBits {
    fn new() -> Self {
        Self { out: [0; 16], cursor: 0 }
    }

    fn write(&mut self, value: u32, bits: usize) {
        debug_assert!(bits <= 24 && self.cursor + bits <= 128);
        for i in 0..bits {
            let bit = ((value >> i) & 1) as u8;
            let idx = self.cursor + i;
            self.out[idx / 8] |= bit << (idx % 8);
        }
        self.cursor += bits;
    }
}

struct BlockBitsReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> BlockBitsReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn read(&mut self, bits: usize) -> u32 {
        debug_assert!(self.cursor + bits <= self.bytes.len() * 8);
        let mut v = 0u32;
        for i in 0..bits {
            let idx = self.cursor + i;
            v |= u32::from((self.bytes[idx / 8] >> (idx % 8)) & 1) << i;
        }
        self.cursor += bits;
        v
    }
}

// ---------------------------------------------------------------------------
// Mip generation (gamma-correct; box default, Kaiser for normals)
// ---------------------------------------------------------------------------

/// Mip filter choice. [`MipFilter::Box`] (area-average) is the default for
/// color/data semantics; [`MipFilter::Kaiser`] is selected for normal maps
/// (sharper high-frequency retention than a plain average before
/// renormalization).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipFilter {
    Box,
    Kaiser,
}

/// sRGB EOTF (exact piecewise transfer, not the 2.2 approximation).
fn srgb_to_linear(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB OETF back to 8-bit storage.
fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    ((s * 255.0).round()) as u8
}

/// Area-average resample of an RGBA mip level to `tw×th`. For sRGB content
/// the filtering runs in LINEAR light (`srgb == true`) — averaging gamma-
/// encoded values darkens mips; this is the "gamma-correct mipgen" clause.
/// For exact halving this degenerates to the plain 2×2 mean.
fn resample_box_rgba(rgba: &[u8], sw: u32, sh: u32, tw: u32, th: u32, srgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for ty in 0..th {
        // Source-space pixel footprint of the destination row.
        let sy0 = (ty as f64 * sh as f64 / th as f64).floor() as u32;
        let sy1 = (((ty + 1) as f64 * sh as f64 / th as f64).ceil() as u32).clamp(sy0 + 1, sh);
        for tx in 0..tw {
            let sx0 = (tx as f64 * sw as f64 / tw as f64).floor() as u32;
            let sx1 = (((tx + 1) as f64 * sw as f64 / tw as f64).ceil() as u32).clamp(sx0 + 1, sw);
            let mut lin = [0f32; 4];
            let mut n = 0u32;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let src = ((sy * sw + sx) * 4) as usize;
                    if srgb {
                        for c in 0..3 {
                            lin[c] += srgb_to_linear(rgba[src + c]);
                        }
                        lin[3] += f32::from(rgba[src + 3]);
                    } else {
                        for c in 0..4 {
                            lin[c] += f32::from(rgba[src + c]);
                        }
                    }
                    n += 1;
                }
            }
            let dst = ((ty * tw + tx) * 4) as usize;
            if srgb {
                for c in 0..3 {
                    out[dst + c] = linear_to_srgb_u8(lin[c] / n as f32);
                }
                out[dst + 3] = (lin[3] / n as f32).round().clamp(0.0, 255.0) as u8;
            } else {
                for c in 0..4 {
                    out[dst + c] = (lin[c] / n as f32).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    out
}

/// Modified Bessel function of the first kind, order 0 (Kaiser window's I₀).
/// Power series — converges quickly for Kaiser's typical β range.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_sq = (x / 2.0) * (x / 2.0);
    for k in 1..32usize {
        term *= half_sq / (k * k) as f64;
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

/// The separable Kaiser-windowed sinc kernel (β = 6, radius 2 output taps):
/// `sinc(t/2)·I₀(6√(1-(t/2)²))/I₀(6)` normalized over its support. Chosen per
/// the issue text as the normal-map mip option ("box default, Kaiser option
/// for normals").
fn kaiser_kernel() -> [f64; 5] {
    const BETA: f64 = 6.0;
    const RADIUS: f64 = 2.0;
    let i0_beta = bessel_i0(BETA);
    let sinc_pi = |x: f64| if x.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
    let mut raw = [0f64; 5];
    for (i, t) in (-2i32..=2).enumerate() {
        let tf = f64::from(t);
        let window_phase = tf / RADIUS;
        let window = bessel_i0(BETA * (1.0 - window_phase * window_phase).max(0.0).sqrt()) / i0_beta;
        raw[i] = sinc_pi(tf / 2.0) * window; // cutoff π/2 → sinc(t/2)
    }
    let sum: f64 = raw.iter().sum();
    raw.map(|v| v / sum)
}

/// One channel of one row/column through the fixed [`kaiser_kernel`], edges
/// clamped. Returns samples at every OTHER source index (the ×2 decimation).
fn kaiser_decimate_line<F>(get: F, len: usize) -> Vec<f32>
where
    F: Fn(usize) -> f32,
{
    let k = kaiser_kernel();
    let sample = |center: i64| -> f32 {
        let mut acc = 0f32;
        for (ki, t) in (-2i64..=2).enumerate() {
            let px = (center + t).clamp(0, len as i64 - 1);
            acc += (k[ki] * f64::from(get(px as usize))) as f32;
        }
        acc.clamp(0.0, 255.0)
    };
    (0..len.div_ceil(2)).map(|o| sample((o * 2) as i64)).collect()
}

/// Reconstruct a tangent-space normal vector from stored X,Y bytes.
fn normal_from_bytes(r: u8, g: u8) -> [f32; 3] {
    let nx = f32::from(r) / 255.0 * 2.0 - 1.0;
    let ny = f32::from(g) / 255.0 * 2.0 - 1.0;
    let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
    [nx, ny, nz]
}

/// Build the full mip chain for a normal map (R,G = X,Y): each level filters
/// X/Y with [`MipFilter::Kaiser`] and RENORMALIZES the reconstructed vectors so
/// decoded normals keep unit length within tolerance (issue test 5). Z is
/// recomputed from filtered X,Y and written to B so the intermediate stays a
/// valid normal map even though BC5 keeps only R,G.
fn build_normal_mip_chain(rgba: &[u8], w: u32, h: u32, mip_count: u32) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = Vec::with_capacity(mip_count as usize);
    levels.push((w, h, rgba.to_vec()));
    for _ in 1..mip_count {
        let &(pw, ph, ref prev) = levels.last().expect("seeded above");
        let tw = pw.div_ceil(2).max(1);
        let th = ph.div_ceil(2).max(1);
        // Kaiser-decimate the X and Y byte planes independently...
        let mut planes = [vec![0u8; (tw * th) as usize], vec![0u8; (tw * th) as usize]];
        for c in 0..2usize {
            let mut tmp_rows = vec![0u8; (tw * ph) as usize]; // horizontal pass first
            for y in 0..ph {
                let line = kaiser_decimate_line(
                    |x| f32::from(prev[(y * pw + x as u32) as usize * 4 + c]),
                    pw as usize,
                );
                for (x, v) in line.iter().enumerate() {
                    tmp_rows[(y * tw + x as u32) as usize] = v.round() as u8;
                }
            }
            for x in 0..tw {
                let line =
                    kaiser_decimate_line(|y| f32::from(tmp_rows[(y as u32 * tw + x) as usize]), th as usize);
                for (y, v) in line.iter().enumerate() {
                    planes[c][(y as u32 * tw + x) as usize] = v.round() as u8;
                }
            }
        }
        // ...then renormalize reconstructed vectors into the new level.
        let mut level = vec![0u8; (tw * th * 4) as usize];
        for i in 0..(tw * th) as usize {
            let [nx, ny, nz] = normal_from_bytes(planes[0][i], planes[1][i]);
            let inv = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt().max(f32::EPSILON);
            level[i * 4] = ((nx * inv * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            level[i * 4 + 1] = ((ny * inv * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            level[i * 4 + 2] = ((nz * inv * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            level[i * 4 + 3] = 255;
        }
        levels.push((tw, th, level));
    }
    levels
}

/// Build the full mip chain for a color/data image: box area-average,
/// gamma-corrected when `srgb` (linear-light filtering).
fn build_color_mip_chain(
    rgba: &[u8],
    w: u32,
    h: u32,
    srgb: bool,
    mip_count: u32,
) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = Vec::with_capacity(mip_count as usize);
    levels.push((w, h, rgba.to_vec()));
    for _ in 1..mip_count {
        let &(sw, sh, ref prev) = levels.last().expect("seeded above");
        let (tw, th) = mip_dims(sw, sh, 1);
        levels.push((tw, th, resample_box_rgba(prev, sw, sh, tw, th, srgb)));
    }
    levels
}

/// Full import pipeline (issue item 2): decode source bytes exactly like the
/// existing texture loader path does (`image` crate → rgba8), build the mip
/// chain (gamma-correct; Kaiser iff Normal), block-compress each level into
/// `format`, and assemble the canonical coarse-first page sequence. Returns
/// the built image, its serialized `.ptex` bytes, and the body-derived
/// content id. Registers the SceneDB segment layout as a side effect (the
/// type-author duty — once per process, idempotent).
pub fn import_texture_bytes(
    source_bytes: &[u8],
    semantic: TextureSemantic,
) -> Result<(PTexImage, Vec<u8>, u128), String> {
    ensure_payload_layout_registered();
    let decoded = image::load_from_memory(source_bytes)
        .map_err(|e| format!("texture decode failed: {e}"))?
        .to_rgba8();
    let (width, height) = (decoded.width(), decoded.height());
    if width == 0 || height == 0 {
        return Err("texture has zero extent".to_string());
    }
    let format = semantic_target_format(semantic);
    let mip_count = mip_count_for(width, height);

    let levels = if semantic == TextureSemantic::Normal {
        build_normal_mip_chain(&decoded, width, height, mip_count)
    } else {
        build_color_mip_chain(&decoded, width, height, semantic.is_srgb(), mip_count)
    };

    // Compress each level, then slice coarse-first into pages using exactly
    // the geometry `page_table` computes.
    let mut compressed = vec![Vec::new(); mip_count as usize];
    for (mip, (lw, lh, bytes)) in levels.iter().enumerate() {
        compressed[mip] = encode_bcn_mip(format, bytes, *lw, *lh);
        debug_assert_eq!(
            compressed[mip].len() as u64,
            mip_encoded_bytes(width, height, mip as u32, format)
        );
    }
    let mut pages = Vec::new();
    for seg in mip_segment_table(width, height, mip_count, format) {
        let bytes = &compressed[seg.mip as usize];
        let mut off = 0usize;
        while off < bytes.len() {
            let n = PAGE_SIZE.min(bytes.len() - off);
            pages.push((seg.mip, bytes[off..off + n].to_vec()));
            off += n;
        }
    }
    debug_assert_eq!(
        pages.len(),
        page_table(width, height, mip_count, format).len(),
        "page assembly must match the declared table"
    );
    Ok(build_and_encode(width, height, format, semantic.is_srgb(), mip_count, pages))
}

// ---------------------------------------------------------------------------
// Path → id resolution + bounded decoded-body cache
// ---------------------------------------------------------------------------

/// Path → content id memoization for TEXTURE assets. A native `.ptex` reads
/// its id straight out of the header (a 24-byte read, not a body hash) when
/// one is stored; anything else defers to the shared canonical-path +
/// mtime/len memoized hasher in `mesh_cache` (`a.png` vs `A/../a.png` vs an
/// identical-bytes copy converge on ONE id; editing the file mints a new one)
/// — the exact convergence contract issue test 2 asserts.
pub fn content_id_for_path(abs_path: &Path) -> Option<u128> {
    ensure_payload_layout_registered();
    if abs_path.extension().and_then(|e| e.to_str()) == Some("ptex") {
        if let Ok(mut f) = std::fs::File::open(abs_path) {
            use std::io::Read as _;
            let mut head = [0u8; 24];
            if f.read_exact(&mut head).is_ok()
                && &head[0..4] == MAGIC
                && u32::from_le_bytes(head[4..8].try_into().unwrap_or([0; 4])) == VERSION
            {
                let id = u128::from_le_bytes(head[8..24].try_into().unwrap_or([0; 16]));
                if id != 0 {
                    return Some(id);
                }
            }
        }
    }
    crate::mesh_cache::memoized_content_id_for_file(abs_path)
}

/// Diagnostics hook: how many REAL decodes (source-image imports or `.ptex`
/// body parses) this process has performed since last reset. The benchmark
/// asserting "hydrate of 100 entities sharing K textures ⇒ exactly K
/// decodes" reads this.
pub fn decode_count() -> u64 {
    DECODE_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Reset [`decode_count`] (benchmarks call this before their measured phase).
pub fn reset_decode_count() {
    DECODE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

static DECODE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Byte budget for the process-wide decoded-body cache. Small on purpose:
/// this exists so N entities hydrating the SAME texture resolve to ONE decode
/// (and one Arc clone each), never to hold every asset's whole mip set
/// resident — VRAM residency is SceneDB's job, not this cache's. FIFO
/// eviction past the budget; keyed by content id so a mutated file (new id)
/// can never serve stale bytes.
const BODY_CACHE_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

struct BodyCache {
    map: HashMap<u128, (std::sync::Arc<Vec<u8>>, u64)>,
    order: std::collections::VecDeque<u128>,
    tracked_bytes: u64,
}

impl BodyCache {
    fn new() -> Self {
        Self { map: HashMap::new(), order: std::collections::VecDeque::new(), tracked_bytes: 0 }
    }

    fn get(&mut self, id: u128) -> Option<std::sync::Arc<Vec<u8>>> {
        self.map.get(&id).map(|(arc, _)| std::sync::Arc::clone(arc))
    }

    fn insert(&mut self, id: u128, body: std::sync::Arc<Vec<u8>>) {
        let bytes = body.len() as u64;
        if bytes > BODY_CACHE_BUDGET_BYTES {
            return; // one oversized body would evict everything; don't cache it
        }
        while self.tracked_bytes + bytes > BODY_CACHE_BUDGET_BYTES {
            let Some(victim) = self.order.pop_front() else { break };
            if let Some((_, victim_bytes)) = self.map.remove(&victim) {
                self.tracked_bytes -= victim_bytes;
            }
        }
        self.tracked_bytes += bytes;
        self.order.push_back(id);
        self.map.insert(id, (body, bytes));
    }
}

static BODY_CACHE: OnceLock<std::sync::Mutex<BodyCache>> = OnceLock::new();

fn body_cache() -> &'static std::sync::Mutex<BodyCache> {
    BODY_CACHE.get_or_init(|| std::sync::Mutex::new(BodyCache::new()))
}

/// Test/bench isolation: drop every cached body and zero counters.
pub fn clear_decoded_payload_cache() {
    *body_cache().lock().expect("body cache mutex poisoned") = BodyCache::new();
    reset_decode_count();
}

/// Resolve the canonical BODY of the texture at `abs_path`, decoding at most
/// ONCE per unique content id per process: native `.ptex` files parse their
/// own container; raw source images run the full import pipeline IN MEMORY
/// (no writes — mirroring `load_mesh_upload`'s convert-on-load behavior for
/// non-native sources). This is what `hydrate_static_mesh_component` calls
/// per slot; the memoization is why 100 entities sharing one asset cost one
/// decode (benchmark 3) — unlike the mesh path, which discloses that gap:
/// the texture issue asserts the count explicitly.
///
/// `semantic` drives the BCn mapping ONLY for raw-source conversion (a
/// `.ptex` already carries its format). `None`: unresolvable/unreadable/
/// undecodable — the hydrate caller treats that as "slot stays empty", never
/// a failure (the mesh hydrate's tolerance contract, verbatim).
pub fn decoded_body_for_path(
    abs_path: &Path,
    semantic: TextureSemantic,
) -> Option<std::sync::Arc<Vec<u8>>> {
    let id = content_id_for_path(abs_path)?;
    if let Some(hit) = body_cache().lock().expect("body cache mutex poisoned").get(id) {
        return Some(hit);
    }
    let bytes = std::fs::read(abs_path).ok()?;
    let body = if abs_path.extension().and_then(|e| e.to_str()) == Some("ptex") {
        let (image, _) = decode(&bytes)?;
        image.body()
    } else {
        import_texture_bytes(&bytes, semantic).ok()?.0.body()
    };
    crate::mesh_cache::prime_content_id_cache(abs_path, id);
    let arc = std::sync::Arc::new(body);
    body_cache()
        .lock()
        .expect("body cache mutex poisoned")
        .insert(id, std::sync::Arc::clone(&arc));
    DECODE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(arc)
}

// ---------------------------------------------------------------------------
// Import entry points (mirroring mesh_cache's model flow)
// ---------------------------------------------------------------------------

/// Native asset path for an imported source texture:
/// `<dest_dir>/<stem>.ptex`.
pub fn native_texture_path(dest_dir: &Path, source: &Path) -> PathBuf {
    let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("texture");
    dest_dir.join(format!("{stem}.ptex"))
}

/// Whether `ext` (without leading dot) is a source-image format we import —
/// exactly the decoders Pulsar-Native's root workspace enables on `image`.
pub const TEXTURE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tga", "gif", "webp"];

pub fn is_importable_texture(ext: &str) -> bool {
    TEXTURE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
}

/// The import-options schema advertised for a texture source extension: one
/// enum field selecting the target semantic (default BaseColor). Drives the
/// same configurator UI models use ([`crate::mesh_cache::OptionsSchema`]).
pub fn options_schema(_ext: &str) -> Option<crate::mesh_cache::OptionsSchema> {
    let choices: Vec<String> =
        TextureSemantic::ALL.iter().map(|s| s.suffix().to_string()).collect();
    Some(crate::mesh_cache::OptionsSchema {
        fields: vec![crate::mesh_cache::ImportField {
            key: "semantic".to_string(),
            label: "Texture semantic".to_string(),
            doc: "Which material slot this texture feeds (drives BCn format and sRGB handling)"
                .to_string(),
            type_info: crate::mesh_cache::build_enum_type_info("Texture semantic", &choices),
            default: Box::new(TextureSemantic::BaseColor.suffix().to_string()),
            constraints: Default::default(),
        }],
    })
}

/// Extract the chosen semantic from an option-value map (defaults BaseColor;
/// accepts both the persisted suffix string and numeric enum encodings).
fn semantic_from_values(values: &HashMap<String, Box<dyn Any + Send>>) -> TextureSemantic {
    values
        .get("semantic")
        .and_then(|v| {
            if let Some(s) = v.downcast_ref::<String>() {
                TextureSemantic::from_suffix(s)
            } else if let Some(i) = v.downcast_ref::<u64>() {
                TextureSemantic::ALL.get(*i as usize).copied()
            } else if let Some(i) = v.downcast_ref::<i64>() {
                TextureSemantic::ALL.get(*i as usize).copied()
            } else {
                None
            }
        })
        .unwrap_or(TextureSemantic::BaseColor)
}

/// Import `source` into an engine-native `.ptex` asset at `native`. Persists
/// the chosen options (keyed by the native path) for reimport. Returns the
/// written native path.
pub fn import_texture_to_native(
    source: &Path,
    native: &Path,
    values: &HashMap<String, Box<dyn Any + Send>>,
) -> Result<PathBuf, String> {
    let semantic = semantic_from_values(values);
    let source_bytes = std::fs::read(source)
        .map_err(|e| format!("failed to read source texture {}: {e}", source.display()))?;
    let (_, ptex_bytes, _) = import_texture_bytes(&source_bytes, semantic)?;
    std::fs::write(native, ptex_bytes)
        .map_err(|e| format!("failed to write native texture {}: {e}", native.display()))?;

    // Persist chosen options for reimport / configurator pre-fill (#409 shape).
    if let Some(root) = engine_state::get_project_path() {
        let root = Path::new(&root);
        let key = engine_fs::import_options::asset_key(root, native);
        let mut json_map = serde_json::Map::new();
        json_map.insert("semantic".to_string(), serde_json::Value::from(semantic.suffix()));
        let _ = engine_fs::import_options::set(root, &key, serde_json::Value::Object(json_map));
    }

    Ok(native.to_path_buf())
}

/// Import `source` into `dest_dir` as a native `.ptex`, resolving options from
/// storage (reimport) or schema defaults. Convenience for the drop flow when
/// no configurator modal supplied explicit options. Returns the native path.
pub fn import_texture_to_native_default(source: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
    ensure_payload_layout_registered();
    let native = native_texture_path(dest_dir, source);
    let ext = source.extension().and_then(|e| e.to_str()).unwrap_or("");
    let values = crate::mesh_cache::resolve_options(native.as_path(), ext);
    import_texture_to_native(source, &native, &values)
}


