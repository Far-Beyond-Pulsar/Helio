//! Layout and interface-contract asserts.
//!
//! Everything here is a number that appears in two places — a Rust struct and a WGSL
//! struct, or this pass and `FoliagePlacePass` — with nothing at runtime to notice when
//! the two drift. A wrong uniform size does not error; the shader reads every field
//! after the mistake from the wrong offset and the grass comes out with garbage
//! dimensions, which is a long way from "someone added a `u32`".

use helio_pass_foliage_gbuffer::*;

// ── Uniform layouts ───────────────────────────────────────────────────────────

#[test]
fn foliage_globals_is_sixty_four_bytes() {
    // Mirrored by `struct FoliageGlobals` in foliage_gbuffer.wgsl. 64 is also a multiple
    // of 16, which a uniform-address-space struct has to be.
    assert_eq!(std::mem::size_of::<FoliageGlobals>(), 64);
    assert_eq!(std::mem::size_of::<FoliageGlobals>() % 16, 0);
}

#[test]
fn foliage_globals_field_offsets_match_the_wgsl_mirror() {
    assert_eq!(std::mem::offset_of!(FoliageGlobals, screen_size), 0);
    assert_eq!(std::mem::offset_of!(FoliageGlobals, frame), 8);
    assert_eq!(std::mem::offset_of!(FoliageGlobals, flags), 12);
    // The two vec4s must land on 16-byte boundaries or WGSL inserts padding the Rust
    // side does not have.
    assert_eq!(std::mem::offset_of!(FoliageGlobals, camera_ring), 16);
    assert_eq!(std::mem::offset_of!(FoliageGlobals, interaction_field), 32);
    assert_eq!(std::mem::offset_of!(FoliageGlobals, lod_quality_scale), 48);
    assert_eq!(std::mem::offset_of!(FoliageGlobals, scale_in_band), 52);
    assert_eq!(std::mem::offset_of!(FoliageGlobals, lod_fade_band), 56);
    assert_eq!(std::mem::offset_of!(FoliageGlobals, interaction_strength), 60);
}

#[test]
fn lod_uniform_is_thirty_two_bytes_and_fits_its_stride() {
    assert_eq!(std::mem::size_of::<FoliageLodUniform>(), 32);
    // The dynamic offset must be a multiple of `min_uniform_buffer_offset_alignment`,
    // which wgpu caps at 256. Using the cap is valid on every adapter; using the struct
    // size would not be.
    assert_eq!(LOD_UNIFORM_STRIDE, 256);
    assert!(std::mem::size_of::<FoliageLodUniform>() as u32 <= LOD_UNIFORM_STRIDE);
    assert_eq!(LOD_UNIFORM_STRIDE % 256, 0);
}

#[test]
fn draw_indirect_stride_matches_draw_indirect_args() {
    // `DrawIndirectArgs` is four u32: vertex_count, instance_count, first_vertex,
    // first_instance. The four LOD commands therefore sit at 0/16/32/48.
    assert_eq!(DRAW_INDIRECT_STRIDE, 16);
    let offsets: Vec<u64> = (0..4).map(|lod| lod as u64 * DRAW_INDIRECT_STRIDE).collect();
    assert_eq!(offsets, vec![0, 16, 32, 48]);
}

// ── "Vertex layout": there isn't one ──────────────────────────────────────────

#[test]
fn the_ladder_has_no_vertex_or_index_buffer_and_the_counts_are_pinned() {
    // The whole grass path is four non-indexed instanced strip draws. These vertex
    // counts are what `FoliagePlacePass` writes into the indirect buffer; if they change
    // here without changing there, a blade is drawn with a truncated strip.
    assert_eq!(LOD_VERTEX_COUNTS, [11, 7, 4, 4]);
    assert_eq!(LOD_SEGMENTS, [5, 3, 1, 1]);
    assert_eq!(LOD_IS_CARD, [false, false, true, true]);

    for lod in 0..LOD_COUNT {
        let expected = if LOD_IS_CARD[lod] {
            4 // two rows of two, no collapsed tip
        } else {
            2 * LOD_SEGMENTS[lod] + 1 // two per row, plus the collapsed tip
        };
        assert_eq!(
            LOD_VERTEX_COUNTS[lod], expected,
            "LOD {lod} vertex count disagrees with its own topology"
        );
        // A strip needs at least three vertices to make one triangle.
        assert!(LOD_VERTEX_COUNTS[lod] >= 3);
    }
}

#[test]
fn lod_uniforms_carry_the_ladder_verbatim() {
    let stride = 4096;
    let uniforms = FoliageGBufferPass::lod_uniforms(stride);
    for (lod, uniform) in uniforms.iter().enumerate() {
        assert_eq!(uniform.lod, lod as u32);
        assert_eq!(uniform.segments, LOD_SEGMENTS[lod]);
        assert_eq!(uniform.vertex_count, LOD_VERTEX_COUNTS[lod]);
        assert_eq!(uniform.region_base, lod as u32 * stride);
        assert_eq!(uniform.is_card, LOD_IS_CARD[lod] as u32);
        assert_eq!(uniform.width_scale, LOD_WIDTH_SCALE[lod]);
        assert_eq!(uniform.height_scale, LOD_HEIGHT_SCALE[lod]);
        // Reserved words must be zero so a future build can tell "unset" from
        // "predates the field".
        assert_eq!(uniform._pad, 0);
    }
}

// ── `visible_blades` entry encoding ───────────────────────────────────────────

#[test]
fn visible_blade_halves_do_not_bleed_into_each_other() {
    // This is the failure that draws grass from a neighbouring tile at a
    // plausible-looking position — no crash, no error, just a field that is subtly wrong
    // and does not move when you edit the tile it appears to belong to.
    assert_eq!(pack_visible_blade(0xffff, 0x0000), 0xffff_0000);
    assert_eq!(pack_visible_blade(0x0000, 0xffff), 0x0000_ffff);
    assert_eq!(VISIBLE_TILE_SHIFT, 16);
    assert_eq!(VISIBLE_LOCAL_MASK, 0xffff);

    for tile_slot in [0u32, 1, 4095, 4096, 65535] {
        for local in [0u32, 1, 2559, 65535] {
            assert_eq!(
                unpack_visible_blade(pack_visible_blade(tile_slot, local)),
                (tile_slot, local)
            );
        }
    }
}

#[test]
fn visible_blade_halves_cover_the_default_ring_and_tile_occupancy() {
    // 16 bits each has to be enough for the shipped presets, or the encoding silently
    // aliases. The default ring is 4096 slots; an 8 m tile at the reference 40 blades/m²
    // holds ~2560 blades.
    let ceiling = VISIBLE_LOCAL_MASK;
    assert!(helio_foliage_core::DEFAULT_TILE_RING_CAPACITY <= ceiling);
    let tile_area = helio_foliage_core::FOLIAGE_TILE_SIZE_METERS.powi(2);
    let blades_per_tile = (helio_foliage_core::GpuFoliageType::default().density
        * tile_area
        * helio_foliage_core::FoliageQuality::Ultra.density_multiplier())
        as u32;
    assert!(
        blades_per_tile <= ceiling,
        "{blades_per_tile} blades/tile at Ultra exceeds the 16-bit local index"
    );
}

#[test]
fn region_offsets_are_contiguous_in_lod_order() {
    for stride in [1u32, 64, 4096, 262_144] {
        let offsets: Vec<u32> = (0..4).map(|lod| visible_region_offset(lod, stride)).collect();
        assert_eq!(offsets, vec![0, stride, 2 * stride, 3 * stride]);
        // Strictly increasing, so no two LODs can read each other's region.
        assert!(offsets.windows(2).all(|w| w[1] > w[0]));
    }
}
