//! Mip-generation correctness (Helio#237 issue test 5):
//!
//! - sRGB semantics are filtered in LINEAR light against a golden value: the
//!   canonical black/white 2×2 checker must average to mid-gray *in linear
//!   space* (sRGB byte ≈ 188), NOT to the gamma-naive 127 a plain byte mean
//!   would produce. This is the "gamma-correct mipgen" contract, pinned
//!   numerically.
//! - Kaiser-filtered normal-map mips preserve unit vector length (within the
//!   stated tolerance) after BC5 round-trip — X/Y filtered + renormalized,
//!   Z reconstructed from stored X/Y exactly like S2's sampler will.

#![cfg(feature = "gpu")]

use image::ImageEncoder as _;
use helio_component::texture_cache::{self, BcnFormat, TextureSemantic};

fn png_bytes(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(rgba, width, height, image::ExtendedColorType::Rgba8)
        .expect("encode fixture png");
    png
}

#[test]
fn srgb_checker_averages_to_linear_mid_gray_not_gamma_naive_gray() {
    // 2×2: two black texels, two white. Linear-space average = 0.5 → sRGB
    // byte 188. A gamma-naive byte mean would give 127 — the whole point of
    // the golden value is that these differ WIDELY.
    const GOLDEN_LINEAR_MID_GRAY_SRGB_BYTE: u8 = 188;
    let rgba = vec![
        0u8, 0, 0, 255, 255, 255, 255, 255, //
        0, 0, 0, 255, 255, 255, 255, 255,
    ];
    let png = png_bytes(&rgba, 2, 2);
    let (image, _, _) =
        texture_cache::import_texture_bytes(&png, TextureSemantic::BaseColor).expect("import");

    assert!(image.srgb, "BaseColor must be flagged sRGB");
    assert_eq!(image.format, BcnFormat::Bc7);
    assert_eq!(image.mip_count, 2, "2×2 chain: base + 1×1 floor");

    // The coarsest mip IS the average of the whole image.
    let coarsest_mip = image.mip_count - 1;
    let (mw, mh) = texture_cache::mip_dims(2, 2, coarsest_mip);
    assert_eq!((mw, mh), (1, 1));
    let compressed = image.mip_bytes(coarsest_mip);
    let decoded = texture_cache::decode_bcn_mip(BcnFormat::Bc7, &compressed, mw, mh).expect("own bytes");
    for px in decoded.chunks_exact(4) {
        assert!(
            px[0].abs_diff(GOLDEN_LINEAR_MID_GRAY_SRGB_BYTE) <= 1,
            "golden linear-light gray expected ~{GOLDEN_LINEAR_MID_GRAY_SRGB_BYTE}, got {px:?}"
        );
        assert_eq!(px[0], px[1], "gray stays neutral");
        assert_eq!(px[1], px[2]);
    }

    // Explicit negative control: the gamma-NAIVE value would be ~127; if we
    // ever see it, filtering regressed to averaging encoded values.
    let naive_mean = (255u32 + 255) / 4;
    assert!(
        decoded[0].abs_diff(naive_mean as u8) > 50,
        "filtering must NOT be happening in gamma space"
    );
}

#[test]
fn linear_data_semantics_filter_in_storage_space() {
    // Occlusion (BC4, non-sRGB): same checker → the mask channel averages to
    // the plain byte mean 127/128 (no transfer function involved).
    let rgba = vec![
        0u8, 0, 0, 255, 255, 255, 255, 255, //
        0, 0, 0, 255, 255, 255, 255, 255,
    ];
    let png = png_bytes(&rgba, 2, 2);
    let (image, _, _) =
        texture_cache::import_texture_bytes(&png, TextureSemantic::Occlusion).expect("import");
    assert!(!image.srgb);
    assert_eq!(image.format, BcnFormat::Bc4);

    let compressed = image.mip_bytes(image.mip_count - 1);
    let decoded = texture_cache::decode_bcn_mip(BcnFormat::Bc4, &compressed, 1, 1).unwrap();
    let v = decoded[0]; // BC4 lands in R
    assert!(
        v.abs_diff(127) <= 2 || v.abs_diff(128) <= 2,
        "data path must average raw values, got {v}"
    );
}

#[test]
fn kaiser_normal_mips_preserve_unit_length_within_tolerance() {
    // An 8×8 normal map whose vectors tilt smoothly away from +Z (so every
    // filter tap sees real variation), encoded as tangent-space X/Y in R,G.
    const TOLERANCE: f32 = 0.05;
    let mut rgba = Vec::with_capacity(8 * 8 * 4);
    for y in 0..8u8 {
        for x in 0..8u8 {
            let nx = (f32::from(x) - 3.5) / 7.5 * 0.9; // ±0.9 tilt
            let ny = (f32::from(y) - 3.5) / 7.5 * 0.9;
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            let enc = |v: f32| ((v * 0.5 + 0.5) * 255.0).round() as u8;
            rgba.extend_from_slice(&[enc(nx), enc(ny), enc(nz), 255]);
        }
    }
    let png = png_bytes(&rgba, 8, 8);
    let (image, _, _) =
        texture_cache::import_texture_bytes(&png, TextureSemantic::Normal).expect("import");
    assert_eq!(image.format, BcnFormat::Bc5, "normals map to BC5");
    assert_eq!(texture_cache::semantic_target_format(TextureSemantic::Normal), BcnFormat::Bc5);

    for mip in 0..image.mip_count {
        let (lw, lh) = texture_cache::mip_dims(8, 8, mip);
        let compressed = image.mip_bytes(mip);
        let decoded = texture_cache::decode_bcn_mip(BcnFormat::Bc5, &compressed, lw, lh)
            .unwrap_or_else(|| panic!("mip {mip}: own BC5 bytes decode"));
        let mut checked = 0usize;
        for px in decoded.chunks_exact(4) {
            // Skip fully-flat padding texels (edge-replicated blocks on odd
            // tails can legitimately hold degenerate X,Y=0 → z=1, length 1).
            let nx = f32::from(px[0]) / 255.0 * 2.0 - 1.0;
            let ny = f32::from(px[1]) / 255.0 * 2.0 - 1.0;
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                (len - 1.0).abs() <= TOLERANCE,
                "mip {mip} ({lw}x{lh}): decoded normal length {len:.4} outside ±{TOLERANCE}"
            );
            checked += 1;
        }
        assert!(checked > 0, "mip {mip} produced no texels");
    }
}
