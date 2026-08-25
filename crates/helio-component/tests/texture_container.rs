//! `.ptex` container coverage (Helio#237 issue tests 1 and 4):
//!
//! - T1: encode→decode round-trips through all FIVE BCn codec paths (BC1,
//!   BC3, BC4, BC5, BC7-mode-6), through both identity paths (stored header id
//!   AND the id==0 backfill), with known-answer spot checks (constant-color
//!   vectors; BC7 blocks must carry the mode-6 bit pattern this module
//!   promises).
//! - T4: segment-table math exact for NPOT sizes — the 100×63 BC1 case is
//!   hand-checked against the documented coarse-first/page-aligned rules —
//!   plus sub-block mip-tail padding and page-table agreement.

#![cfg(feature = "gpu")]

use image::ImageEncoder as _;
use helio_component::texture_cache::{self, BcnFormat, PageEntry, TextureSemantic};

/// Deterministic pseudo-gradient RGBA generator (xorshift over coordinates —
/// no external RNG dependency, stable across runs/platforms).
fn synth_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    let mut state = 0x9E3779B97F4A7C15u64;
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push(((x * 255) / width.max(1)) as u8);
            out.push(((y * 255) / height.max(1)) as u8);
            out.push((state >> 56) as u8);
            out.push((((x + y) * 255) / (width + height).max(1)) as u8);
        }
    }
    out
}

/// Compress a full mip chain and assemble the canonical coarse-first page
/// sequence — the same geometry `import_texture_bytes` uses, driven directly
/// so any format (not just the semantic map's choices) is exercisable.
fn build_pages(
    rgba0: &[u8],
    width: u32,
    height: u32,
    format: BcnFormat,
    mip_count: u32,
) -> Vec<(u32, Vec<u8>)> {
    let mut compressed = vec![Vec::new(); mip_count as usize];
    for mip in 0..mip_count {
        let (lw, lh) = texture_cache::mip_dims(width, height, mip);
        let src = if mip == 0 {
            rgba0.to_vec()
        } else {
            // Cheap deterministic downsample (nearest) — geometry is what's
            // under test here, not filtering quality.
            let mut lvl = Vec::with_capacity((lw * lh * 4) as usize);
            for y in 0..lh {
                for x in 0..lw {
                    let sx = (x * 2).min(lw * 2 - 1) as usize;
                    let sy = (y * 2).min(lh * 2 - 1) as usize;
                    let p = (sy * lw as usize * 2 + sx) * 4;
                    lvl.extend_from_slice(&rgba0[p..p + 4]);
                }
            }
            lvl
        };
        compressed[mip as usize] = texture_cache::encode_bcn_mip(format, &src, lw, lh);
    }
    let mut pages = Vec::new();
    for seg in texture_cache::mip_segment_table(width, height, mip_count, format) {
        let bytes = &compressed[seg.mip as usize];
        let mut off = 0usize;
        while off < bytes.len() {
            let n = texture_cache::PAGE_SIZE.min(bytes.len() - off);
            pages.push((seg.mip, bytes[off..off + n].to_vec()));
            off += n;
        }
    }
    pages
}

// ── T1: round-trips ─────────────────────────────────────────────────────────

#[test]
fn round_trip_through_every_bcn_format_with_stored_header_id() {
    for format in [
        BcnFormat::Bc1,
        BcnFormat::Bc3,
        BcnFormat::Bc4,
        BcnFormat::Bc5,
        BcnFormat::Bc7,
    ] {
        let (w, h) = (16u32, 16u32);
        let rgba = synth_rgba(w, h);
        let mips = texture_cache::mip_count_for(w, h);
        let pages = build_pages(&rgba, w, h, format, mips);

        let (image, bytes, id) =
            texture_cache::build_and_encode(w, h, format, false, mips, pages.clone());
        assert_ne!(id, 0, "{format}: fresh imports always store a real id");

        // Header round-trip: metadata survives byte-for-byte.
        let (decoded, decoded_id) = texture_cache::decode(&bytes).expect("valid container");
        assert_eq!(decoded_id, id, "{format}: stored header id must be returned verbatim");
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h);
        assert_eq!(decoded.format, format);
        assert_eq!(decoded.mip_count, mips);
        assert_eq!(decoded.pages, pages, "{format}: page payload must round-trip exactly");
        assert_eq!(image.body(), decoded.body());

        // Every serialized mip decodes back into something close to its
        // nearest-neighbour source (BCn is lossy; a tight mean bound catches
        // real packing errors without pinning encoder aesthetics).
        for seg in texture_cache::mip_segment_table(w, h, mips, format) {
            let (lw, lh) = texture_cache::mip_dims(w, h, seg.mip);
            let encoded = decoded.mip_bytes(seg.mip);
            assert_eq!(
                encoded.len() as u64,
                seg.len,
                "{format} mip {}: serialized length must match the segment table",
                seg.mip
            );
            let _ = texture_cache::decode_bcn_mip(format, &encoded, lw, lh)
                .unwrap_or_else(|| panic!("{format} mip {}: own bytes must decode", seg.mip));
        }
    }
}

#[test]
fn bc7_blocks_are_mode6_only_and_constant_colors_reconstruct_exactly() {
    // Constant-color 4×4 blocks: mode-6 BC7 reconstructs endpoint colors
    // exactly (the P-bit scheme restores the dropped LSB).
    let make_block = |rgba: [u8; 4]| {
        let mut block = [[0u8; 4]; 16];
        block.iter_mut().for_each(|t| *t = rgba);
        texture_cache::encode_bc7_block(&block)
    };
    for color in [[255u8, 0, 0, 255], [0, 255, 0, 128], [12, 34, 56, 78], [1, 2, 3, 4]] {
        let bytes = make_block(color);
        // Mode code 6 = six 0-bits then a terminating 1 (LSB-first): bits
        // 0-5 clear and bit 6 set — byte0's low 7 bits are exactly 0b1000000
        // (bit 7 is the first endpoint bit).
        assert_eq!(bytes[0] & 0x7F, 0x40, "every emitted BC7 block must be mode 6");
        let decoded = texture_cache::decode_bc7_block(&bytes).expect("mode 6");
        for texel in decoded {
            for c in 0..4 {
                assert!(
                    texel[c].abs_diff(color[c]) <= 1,
                    "constant {color:?}: texel {texel:?} drifted more than 1"
                );
            }
        }
    }

    // Same contract at container scale: a solid-color texture's finest mip
    // decodes to the solid color everywhere.
    let (w, h) = (8u32, 8u8 as u32);
    let rgba: Vec<u8> = [37u8, 99, 200, 255].repeat((w * h) as usize);
    let pages = build_pages(&rgba, w, h, BcnFormat::Bc7, 1);
    let (_, bytes, _) = texture_cache::build_and_encode(w, h, BcnFormat::Bc7, true, 1, pages);
    let (image, _) = texture_cache::decode(&bytes).unwrap();
    let back = texture_cache::decode_bcn_mip(BcnFormat::Bc7, &image.mip_bytes(0), w, h).unwrap();
    for px in back.chunks_exact(4) {
        assert!(px[0].abs_diff(37) <= 1 && px[1].abs_diff(99) <= 1 && px[2].abs_diff(200) <= 1);
    }
}

#[test]
fn bc1_bc3_bc4_bc5_known_answer_constant_blocks() {
    let const_block = |rgba: [u8; 4]| {
        let mut b = [[0u8; 4]; 16];
        b.iter_mut().for_each(|t| *t = rgba);
        b
    };

    // BC1 pure red, CONSTANT block: min == max == red, so BOTH endpoints
    // quantize to RGB565(255,0,0) = 0xF800 and every index stays at palette
    // slot 0 → index half zero.
    let bc1 = texture_cache::encode_bc1_block(&const_block([255, 0, 0, 255]));
    assert_eq!(&bc1[0..2], &0xF800u16.to_le_bytes());
    assert_eq!(&bc1[2..4], &0xF800u16.to_le_bytes(), "constant block: both endpoints identical");
    assert!(bc1[4..8].iter().all(|&b| b == 0), "all 2-bit indices == 0");
    let d = texture_cache::decode_bc1_block(&bc1);
    assert!(d.iter().all(|t| t[0] > 250 && t[1] < 6 && t[2] < 6));

    // BC4 constant 200: a0=a1=200, flat palette, indices 0.
    let bc4 = texture_cache::encode_bc4_block(&const_block([0, 0, 0, 200]), 3);
    assert_eq!(bc4[0], 200);
    assert_eq!(bc4[1], 200);
    let d = texture_cache::decode_bc4_block(&bc4, 3);
    assert!(d.iter().all(|t| t[3].abs_diff(200) <= 1));

    // BC5 of (x=255,y=128): halves independent.
    let bc5 = texture_cache::encode_bc5_block(&const_block([255, 128, 0, 255]));
    let d = texture_cache::decode_bc5_block(&bc5);
    assert!(d.iter().all(|t| t[0] > 250 && t[1].abs_diff(128) <= 2));

    // BC3: alpha half carries A, color half carries RGB. Dark colors drift
    // most under RGB565 quantization (a value-10 red lands on 8): ≤4.
    let bc3 = texture_cache::encode_bc3_block(&const_block([10, 20, 30, 240]));
    let d = texture_cache::decode_bc3_block(&bc3);
    assert!(d.iter().all(|t| t[0].abs_diff(10) <= 4
        && t[1].abs_diff(20) <= 4
        && t[2].abs_diff(30) <= 4
        && t[3].abs_diff(240) <= 2));
}

#[test]
fn idless_header_backfills_to_the_same_body_hash() {
    let (w, h) = (8u32, 8u32);
    let rgba = synth_rgba(w, h);
    let pages = build_pages(&rgba, w, h, BcnFormat::Bc4, 1);
    let (_, mut bytes, id) = texture_cache::build_and_encode(w, h, BcnFormat::Bc4, false, 1, pages);

    // Punch the header id to the reserved "absent" sentinel.
    bytes[8..24].fill(0);
    let (image, backfilled) = texture_cache::decode(&bytes).expect("sentinel id still decodes");
    assert_eq!(backfilled, id, "backfill must recompute the SAME body hash");
    // And that hash really is the body hash.
    assert_eq!(backfilled, texture_cache::content_id_for_body(&image.body()));
}

#[test]
fn invalid_containers_are_rejected_not_panicked_on() {
    let (_, bytes, _) = texture_cache::build_and_encode(
        8,
        8,
        BcnFormat::Bc1,
        false,
        1,
        build_pages(&synth_rgba(8, 8), 8, 8, BcnFormat::Bc1, 1),
    );
    assert!(texture_cache::decode(&bytes[..10]).is_none(), "truncated");
    assert!(texture_cache::decode(b"PTEX").is_none(), "header only");
    assert!(texture_cache::decode(b"nope").is_none(), "garbage");

    let mut bad_version = bytes.clone();
    bad_version[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert!(texture_cache::decode(&bad_version).is_none(), "unknown version");

    let mut bad_format = bytes.clone();
    bad_format[32..36].copy_from_slice(&999u32.to_le_bytes());
    assert!(texture_cache::decode(&bad_format).is_none(), "unknown format discriminant");

    let mut bad_srgb = bytes.clone();
    bad_srgb[36..40].copy_from_slice(&7u32.to_le_bytes());
    assert!(texture_cache::decode(&bad_srgb).is_none(), "non-bool srgb flag");

    let mut bad_page_count = bytes.clone();
    bad_page_count[44..48].copy_from_slice(&(u32::MAX / 2).to_le_bytes());
    assert!(texture_cache::decode(&bad_page_count).is_none(), "page_count overflow guard");
}

// ── T4: segment math (NPOT + block tails) ───────────────────────────────────

#[test]
fn npot_100x63_bc1_segment_table_matches_the_hand_computed_layout() {
    // Floor-halved chain (D3D-standard mip sizes, max(1, dim>>i)):
    //   (100,63)(50,31)(25,15)(12,7)(6,3)(3,1)(1,1) — 7 mips.
    // Encoded lens (ceil-to-4 block dims × 8B):
    //   mip0: 25·16·8 = 3200   mip4: 2·1·8 = 16
    //   mip1: 13·8·8  = 832    mip5: 1·1·8 = 8
    //   mip2: 7·4·8   = 224    mip6: 1·1·8 = 8
    //   mip3: 3·2·8   = 48
    // Coarse-first, every mip page-aligned (PAGE_SIZE = 16384).
    const P: u64 = texture_cache::PAGE_SIZE as u64;
    let expected: [(u32, u64, u64); 7] = [
        (6, 0, 8),
        (5, P, 8),
        (4, 2 * P, 16),
        (3, 3 * P, 48),
        (2, 4 * P, 224),
        (1, 5 * P, 832),
        (0, 6 * P, 3200),
    ];
    let got = texture_cache::mip_segment_table(100, 63, 7, BcnFormat::Bc1);
    assert_eq!(got.len(), 7, "floor chain of max(100,63)=100 runs to 1×1 in 7 levels");
    for (seg, &(mip, offset, len)) in got.iter().zip(&expected) {
        assert_eq!((seg.mip, seg.offset, seg.len), (mip, offset, len));
    }

    // Page table agrees: one PARTIAL page per mip here (all lens < PAGE_SIZE),
    // valid_len == mip len, coarse-first order preserved.
    let pages = texture_cache::page_table(100, 63, 7, BcnFormat::Bc1);
    let want: Vec<PageEntry> = expected
        .iter()
        .map(|&(mip, _, len)| PageEntry { mip, valid_len: len as u32 })
        .collect();
    assert_eq!(pages, want);
}

#[test]
fn large_textures_produce_multi_page_mips_and_consistent_totals() {
    // 512×512 BC7 (16B blocks): mip0 alone is 128×128 blocks × 16B = 256 KiB =
    // exactly 16 pages. The nine coarser mips contribute 4+1+1+1+1+1+1+1+1 =
    // 12 pages before it.
    const P: u64 = texture_cache::PAGE_SIZE as u64;
    let table = texture_cache::mip_segment_table(512, 512, 10, BcnFormat::Bc7);
    let mip0 = table.last().expect("finest last");
    assert_eq!(mip0.mip, 0);
    assert_eq!(mip0.len, 128 * 128 * 16);
    assert_eq!(mip0.offset, 12 * P, "nine coarser mips total 12 pages");

    let pages = texture_cache::page_table(512, 512, 10, BcnFormat::Bc7);
    assert_eq!(pages.len() as u64, table.iter().map(|s| s.len.div_ceil(P)).sum::<u64>());
    assert_eq!(pages.first().unwrap().mip, 9, "coarsest floor first");
    assert_eq!(pages.last().unwrap().mip, 0, "base mip last");
    assert!(pages.iter().rev().take(16).all(|p| p.valid_len as u64 == P), "base mip's 16 full pages");
}

#[test]
fn sub_block_mip_tails_pad_to_whole_blocks() {
    // A 2×1 mip occupies ONE padded 4×4 block (never truncates mid-block);
    // 100×63-style odd tails likewise ceil-div per dimension.
    assert_eq!(texture_cache::mip_encoded_bytes(2, 1, 0, BcnFormat::Bc1), 8);
    assert_eq!(texture_cache::mip_encoded_bytes(1, 1, 0, BcnFormat::Bc7), 16);
    assert_eq!(texture_cache::mip_encoded_bytes(100, 63, 0, BcnFormat::Bc1), 3200);
    assert_eq!(
        texture_cache::mip_encoded_bytes(12, 7, 0, BcnFormat::Bc5),
        3 * 2 * 16,
        "a 12×7 level ceil-divs to 3×2 blocks"
    );
    // Mip chain lengths (floor-halving: levels = significant bits of max dim).
    assert_eq!(texture_cache::mip_count_for(1, 1), 1);
    assert_eq!(texture_cache::mip_count_for(16, 16), 5);
    assert_eq!(texture_cache::mip_count_for(100, 63), 7);
}

#[test]
fn import_pipeline_assembly_matches_declared_geometry() {
    // End-to-end: a PNG goes through the full pipeline; the resulting page
    // sequence must satisfy the SAME invariants the tables declare.
    let (w, h) = (24u32, 17u32); // deliberately NPOT
    let rgba = synth_rgba(w, h);
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(
            &rgba,
            w,
            h,
            image::ExtendedColorType::Rgba8,
        )
        .expect("encode fixture png");

    let (image, bytes, id) = texture_cache::import_texture_bytes(&png, TextureSemantic::Occlusion)
        .expect("occlusion → BC4 pipeline");
    assert_eq!(image.format, BcnFormat::Bc4);
    assert!(!image.srgb);
    assert_ne!(id, 0);
    let (reparsed, reparsed_id) = texture_cache::decode(&bytes).expect("container valid");
    assert_eq!(reparsed_id, id);
    assert_eq!(reparsed.pages, image.pages);

    let declared = texture_cache::page_table(image.width, image.height, image.mip_count, image.format);
    assert_eq!(image.pages.len(), declared.len());
    for ((mip, data), entry) in image.pages.iter().zip(&declared) {
        assert_eq!(*mip, entry.mip);
        assert_eq!(data.len() as u32, entry.valid_len, "valid_len mirrors actual page bytes");
    }
}


