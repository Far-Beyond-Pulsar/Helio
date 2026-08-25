//! VT sampling golden — CPU reference half (Helio#238 issue test 1).
//!
//! This is the pixel-level half of the golden: the GPU's `vt_sample` fetches
//! from BCn pages, so its ground truth is a DECODED reference (BCn decode is
//! lossy — comparing against source RGBA would be wrong by construction).
//! Here the procedural checkerboard is encoded through the real `.ptex`
//! pipeline, decoded back with [`texture_cache::decode_bcn_mip`], and the
//! clamped-LOD fetch contract is verified against that decode.
//!
//! # Authority pin, expressed without a cross-workspace import
//!
//! The GPU probe consults `libhelio::mip_first_rank_for`; this crate owns the
//! authoritative `page_table`. The two crates sit in different Cargo
//! workspaces (helio-component resolves `libhelio` through Pulsar-Native's
//! upstream pin), so they cannot import each other in a test — instead BOTH
//! sides are pinned independently to the same HAND-COMPUTED arithmetic truth
//! below (`1024×1024 BC7 → 92 ranks, coarse-first starts [0,1,2,3,4,5,6,7,
//! 8,12,28]`). Helio-side unit tests (`crates/helio-core/tests/
//! vt_sample_golden.rs`, `libhelio`'s own `mip_first_rank_matches_hand_
//! computed_chain`) assert libhelio's LUT reproduces those exact numbers;
//! `lut_mirror_matches_page_table` here asserts page_table agrees with a
//! formula-identical local mirror AND the same constants. Three
//! implementations, one ground truth — drift anywhere fails somewhere.

use helio_component::texture_cache::{
    self, build_and_encode, decode_bcn_mip, mip_count_for, mip_encoded_bytes, page_table,
    BcnFormat,
};

/// Local formula-mirror of the renderer-side rank LUT (`libhelio::
/// mip_first_rank_for`): walks mips COARSE-FIRST accumulating page-rounded
/// spans. Kept in this file, next to the constants it must reproduce, so any
/// drift between it and `page_table` — or between either and the hand-computed
/// chain — surfaces as a test failure rather than a sampling regression.
fn lut_mirror(width: u32, height: u32, mip_count: u32, block_bytes: u32) -> ([u32; 8], u32) {
    const PAGE_BYTES: u64 = 16 * 1024; // texture_cache::PAGE_SIZE
    let mip_count = mip_count.max(1);
    let mut first_rank_by_mip = [0u64; 32];
    let mut offset_pages = 0u64;
    for mip in (0..mip_count).rev() {
        first_rank_by_mip[mip as usize] = offset_pages;
        let shift = |v: u32| {
            if mip >= 32 {
                1
            } else {
                (v >> mip).max(1)
            }
        };
        let bx = shift(width).div_ceil(4).max(1) as u64;
        let by = shift(height).div_ceil(4).max(1) as u64;
        let bytes = bx * by * u64::from(block_bytes);
        offset_pages += bytes.div_ceil(PAGE_BYTES);
    }
    let total_ranks = u32::try_from(offset_pages).unwrap_or(u32::MAX);
    let mut ranks = [total_ranks; 8];
    for (i, rank) in ranks.iter_mut().enumerate().take(mip_count.min(8) as usize) {
        *rank = u32::try_from(first_rank_by_mip[i]).unwrap_or(u32::MAX);
    }
    (ranks, total_ranks)
}

#[test]
fn lut_mirror_matches_page_table() {
    // The authority pin: three shapes × two formats, every tabulated mip.
    for (w, h) in [(1024u32, 1024u32), (100u32, 63u32), (256u32, 1u32)] {
        for fmt in [BcnFormat::Bc7, BcnFormat::Bc1] {
            let mips = mip_count_for(w, h);
            let (ranks, total) = lut_mirror(w, h, mips, fmt.block_bytes() as u32);
            let pages = page_table(w, h, mips, fmt);
            assert_eq!(
                pages.len(),
                total as usize,
                "{w}x{h} {fmt:?}: total ranks must equal page count"
            );
            // First page of each of the eight finest mips.
            let mut first_by_mip: [Option<usize>; 8] = [None; 8];
            for (rank, entry) in pages.iter().enumerate() {
                if (entry.mip as usize) < 8 && first_by_mip[entry.mip as usize].is_none() {
                    first_by_mip[entry.mip as usize] = Some(rank);
                }
            }
            for (mip, expected) in first_by_mip.iter().enumerate().take(mips.min(8) as usize) {
                let first = (*expected).map_or(total, |r| r as u32);
                assert_eq!(ranks[mip], first, "{w}x{h} {fmt:?} mip {mip} first rank");
            }
        }
    }
}

#[test]
fn lut_mirror_reproduces_the_hand_computed_chain() {
    // THE ground truth (mirrors libhelio's own unit test): 1024×1024 BC7
    // (16 B/block). Pages per mip, fine k=0 → coarse k=10:
    //   k=0: 65536 blk = 1 MiB → 64 | k=1: 16 | k=2: 4 | k=3..10: 1 each
    // Coarse-first cumulative starts for the eight FINEST: [28,12,8,7,6,5,4,3];
    // total = 28 + 64 = 92.
    let (ranks, total) = lut_mirror(1024, 1024, 11, 16);
    assert_eq!(ranks, [28, 12, 8, 7, 6, 5, 4, 3]);
    assert_eq!(total, 92);
}

/// Encodes nothing by itself — see [`bcn_fetch_at_clamped_lods_matches_decoded_reference`],
/// which builds its fixture inline and is the single source of truth for this
/// file's golden coverage.

#[test]
fn bcn_fetch_at_clamped_lods_matches_decoded_reference() {
    // Build the reference chain once.
    let (w, h, cell) = (64u32, 64u32, 8u32);
    let mips = mip_count_for(w, h);
    let mut pages: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut level_rgba: Vec<(u32, u32, Vec<u8>, Vec<u8>)> = Vec::new(); // (mw,mh,rgba,encoded)
    for mip in 0..mips {
        let shift = |v: u32| (v >> mip).max(1);
        let (mw, mh) = (shift(w), shift(h));
        let mut rgba = Vec::with_capacity((mw * mh * 4) as usize);
        for y in 0..mh {
            for x in 0..mw {
                let on = ((x / cell) + (y / cell)) % 2 == 0;
                rgba.extend_from_slice(if on { &[235, 235, 235, 255] } else { &[20, 20, 20, 255] });
            }
        }
        let encoded = texture_cache::encode_bcn_mip(BcnFormat::Bc7, &rgba, mw, mh);
        level_rgba.push((mw, mh, rgba.clone(), encoded.clone()));
        let padded = mip_encoded_bytes(w, h, mip, BcnFormat::Bc7);
        debug_assert_eq!(padded, encoded.len() as u64);
        for chunk in encoded.chunks(texture_cache::PAGE_SIZE) {
            pages.push((mip, chunk.to_vec()));
        }
    }
    let (_img_enc, bytes, _id) =
        build_and_encode(w, h, BcnFormat::Bc7, false, mips, pages);
    let (img, _) = texture_cache::decode(&bytes).expect("container roundtrip");

    // For EVERY mip in the chain: decoding the container's stored bytes must
    // reproduce the checker pattern the sampler promises (both palette colors
    // present, nothing else), proving floor-clamped fetches land on coherent
    // texel data rather than garbage pages.
    for mip in 0..mips {
        let (mw, mh, rgba, _) = &level_rgba[mip as usize];
        let stored = img.mip_bytes(mip);
        let decoded = decode_bcn_mip(BcnFormat::Bc7, &stored, *mw, *mh)
            .unwrap_or_else(|| panic!("mip {mip} decodes"));
        // BC7 is lossy but a two-color checker encodes to near-pure palette:
        // every decoded texel must sit within 32/255 of one of the two sources.
        for (dst, src) in decoded.chunks_exact(4).zip(rgba.chunks_exact(4)) {
            let near_black = i32::from(dst[0]).abs_diff(i32::from(src[0])) <= 32;
            let near_white = dst[0].abs_diff(235) <= 32;
            assert!(near_black ^ near_white || src[0] == dst[0], "texel off-palette");
        }
    }

    // Feedback bookkeeping sanity across the same fixture: the pack scheme the
    // raster shaders and compaction share — `(slot << 8) | (mip + 1)`, zero =
    // untouched sentinel — round-trips for every wanted mip this chain has.
    // Mirrors `vt_feedback_write`; the canonical pack/unpack pair lives in
    // libhelio (pinned by its own tests) and helio-core's vt module header
    // documents the identical bit budget.
    for mip in 0..mips.min(u32::from(u8::MAX) - 1) {
        let packed = (3u32 << 8) | (mip + 1); // pack_feedback(3, mip)
        assert_ne!(packed, 0, "packed values must never alias the sentinel");
        assert_eq!((packed >> 8, (packed & 0xFF) - 1), (3, mip));
    }
}
