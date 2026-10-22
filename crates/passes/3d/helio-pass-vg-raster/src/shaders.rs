//! WGSL source assembly. There is no `#include`: each kernel is the
//! concatenation of the shared declarations and its own file.

pub const FORMAT: &str = helio_vg_format::WGSL;
pub const BINDINGS: &str = include_str!("../shaders/vg_bindings.wgsl");
/// THE position/index/attribute decode, shared by every VG pass
/// (`helio_vg_format::DECODE_WGSL`).
pub const DECODE: &str = helio_vg_format::DECODE_WGSL;
pub const RASTER_COMMON: &str = include_str!("../shaders/vg_raster_common.wgsl");
pub const HW_RASTER: &str = include_str!("../shaders/vg_hw_raster.wgsl");
pub const SW_RASTER: &str = include_str!("../shaders/vg_sw_raster.wgsl");
pub const SW_BIN: &str = include_str!("../shaders/vg_sw_bin.wgsl");
/// Reusable tiled shared-memory raster kernel (Tier 2 fallback, Tier 3 base).
pub const TILE_RASTER: &str = include_str!("../shaders/vg_tile_raster.wgsl");
pub const TILE_CLASSIFY: &str = include_str!("../shaders/vg_tile_classify.wgsl");

/// Group 0 bindings 1 (page pool) and 2 (asset quant) read by [`DECODE`].
pub fn decode_bindings() -> String {
    helio_vg_format::decode_bindings_wgsl(0, 1, 2)
}

fn join(parts: &[&str]) -> String {
    parts.join("\n")
}

fn scene(extra: &[&str]) -> String {
    let db = decode_bindings();
    let mut parts: Vec<&str> = vec![FORMAT, BINDINGS, &db, DECODE];
    parts.extend_from_slice(extra);
    join(&parts)
}

/// Tier 1 vertex + fragment module.
pub fn hw_raster_source() -> String {
    scene(&[HW_RASTER])
}
/// Tier 2 with 64-bit atomics (`sw_raster_main`).
pub fn sw_raster_source() -> String {
    scene(&[RASTER_COMMON, SW_RASTER])
}
/// Tier 2 fallback binner (`sw_bin_main`).
pub fn sw_bin_source() -> String {
    scene(&[RASTER_COMMON, SW_BIN])
}
/// Tiled raster kernel (`tile_raster_main`).
pub fn tile_raster_source() -> String {
    scene(&[RASTER_COMMON, TILE_RASTER])
}
/// `classify_main` and `dense_args_main` (standalone, no scene bindings).
pub fn tile_classify_source() -> String {
    TILE_CLASSIFY.to_string()
}

/// Every module with a label, for validation.
pub fn all_sources() -> Vec<(&'static str, String)> {
    vec![
        ("vg_hw_raster", hw_raster_source()),
        ("vg_sw_raster", sw_raster_source()),
        ("vg_sw_bin", sw_bin_source()),
        ("vg_tile_raster", tile_raster_source()),
        ("vg_tile_classify", tile_classify_source()),
    ]
}
