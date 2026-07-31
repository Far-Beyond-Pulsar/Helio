//! CPU mirror of the procedural blade geometry in `foliage_gbuffer.wgsl`.
//!
//! There is no vertex buffer and no index buffer anywhere in the grass path. Every
//! vertex of every blade is derived from `@builtin(vertex_index)` and the per-instance
//! [`GpuBladeInstance`](helio_foliage_core::GpuBladeInstance), which is what makes the
//! whole world's grass four `draw_indirect` calls (the plan's §6.3). The cost of that is
//! that the geometry only exists inside a shader, where nothing can look at it — so it is
//! written twice, here and in WGSL, and the two are pinned against each other by the
//! tests in `tests/geometry.rs`.
//!
//! This is the same arrangement `helio-foliage-core::placement` uses for the LOD maths,
//! and for the same reason: when a blade comes out inside-out or a strip has a fold in
//! it, the failure is a screenful of flickering polygons with no intermediate state to
//! inspect. A CPU mirror turns that into an assertion on three floats.
//!
//! # Strip topology
//!
//! Vertices are emitted in **row-major, side-minor** order:
//!
//! ```text
//!  index:  0        1        2        3        4        5   ...  2n
//!  row:    0        0        1        1        2        2        n (tip)
//!  side:  -1       +1       -1       +1       -1       +1        0
//! ```
//!
//! which is exactly a triangle strip up the blade. The final vertex of a blade LOD is a
//! single tip vertex (`side = 0`), so a 5-segment blade is 11 vertices rather than 12 —
//! collapsing the tip is what keeps the silhouette from ending in a flat chisel.
//!
//! Card LODs have no tip vertex: they are a plain 4-vertex quad strip (two rows of two),
//! which is why [`LOD_VERTEX_COUNTS`] is `[11, 7, 4, 4]` and not `2n + 1` throughout.
//!
//! Each instance is an independent strip. That is guaranteed by the WebGPU spec's
//! primitive-assembly algorithm — assembly runs per instance, and a strip is only split
//! on the restart value for *indexed* draws — so a non-indexed instanced strip draw can
//! never span an instance boundary and no degenerate stitch triangles are needed.

/// Number of drawable foliage LOD levels.
///
/// Mirrors `helio_foliage_core::FOLIAGE_LOD_COUNT` as a `usize`, so it can size the
/// tables below without a cast at every use. Named differently on purpose: the two are
/// the same number in different types, and a glob-importing caller that got the `u32`
/// where it wanted the `usize` would fail in a way that reads like a borrow error.
pub const LOD_COUNT: usize = 4;

/// Segments up the blade at each LOD (the plan's §6.3 ladder).
///
/// L2/L3 are cards, so their "segment" count is 1 in the sense of "one quad tall"; the
/// tip-collapse and the width taper are both switched off for them by [`LOD_IS_CARD`].
pub const LOD_SEGMENTS: [u32; LOD_COUNT] = [5, 3, 1, 1];

/// Vertices per instance at each LOD — the `vertex_count` the producer must write into
/// each `DrawIndirectArgs`.
///
/// `2 * segments + 1` for the blade LODs (tip collapsed to one vertex) and 4 for the
/// card LODs. These numbers are part of the interface with `FoliagePlacePass`: it writes
/// them into the indirect buffer, this pass never overrides them, and a mismatch draws
/// either a truncated blade or a strip that runs off the end of its own topology.
pub const LOD_VERTEX_COUNTS: [u32; LOD_COUNT] = [11, 7, 4, 4];

/// Whether each LOD draws a card (flat quad) rather than a tapered blade strip.
pub const LOD_IS_CARD: [bool; LOD_COUNT] = [false, false, true, true];

/// Forward curl of a blade at its tip, as a fraction of the blade's height.
///
/// A blade modelled as a flat vertical strip reads as a spike of plastic. The curl is
/// what gives grass its silhouette, and it also does real work for the wind model: the
/// arc-length correction in `helio_wind_offset` assumes the stem has length, so a blade
/// that is a straight vertical line has no lateral reach to trade against.
///
/// Applied as `curve * height_frac²`, i.e. a cantilever mode shape, so the root leaves
/// the ground vertically. A linear curl would visibly shear the blade out of the terrain
/// at its base, which is the same failure `helio_wind_sway` squares `height_frac` to
/// avoid.
pub const BLADE_CURVE_FRACTION: f32 = 0.35;

/// The LOD index of the clump card — the one LOD that draws a single instance per
/// *cluster* rather than per blade.
///
/// Must agree with `FOLIAGE_LOD_CLUMP` in `foliage_cull.wgsl`, which is what decides to
/// emit one index per cluster for this level. If the two ever disagree, one side emits
/// per-blade while the other sizes per-cluster and the far ring is off by the cluster
/// size — a density discontinuity, not a crash.
pub const CLUMP_LOD: usize = 3;

/// Width multiplier applied to the L3 clump card.
///
/// L3 is one card per 4×4 cluster (the plan's §6.3), so it must cover roughly the
/// footprint of sixteen blades rather than one. Getting this wrong is not subtle: too
/// narrow and the far ring visibly thins out at the L2→L3 boundary, too wide and the
/// ground turns into a solid mat.
pub const CLUMP_CARD_WIDTH_SCALE: f32 = 2.5;

/// Height multiplier applied to the L3 clump card.
///
/// Slightly taller than a single blade because a clump's silhouette is set by its
/// tallest members, not its mean.
pub const CLUMP_CARD_HEIGHT_SCALE: f32 = 1.35;

/// Per-LOD width multiplier, indexed by LOD. See [`CLUMP_CARD_WIDTH_SCALE`].
pub const LOD_WIDTH_SCALE: [f32; LOD_COUNT] = [1.0, 1.0, 1.0, CLUMP_CARD_WIDTH_SCALE];

/// Per-LOD height multiplier, indexed by LOD. See [`CLUMP_CARD_HEIGHT_SCALE`].
pub const LOD_HEIGHT_SCALE: [f32; LOD_COUNT] = [1.0, 1.0, 1.0, CLUMP_CARD_HEIGHT_SCALE];

/// One derived blade vertex, before any per-instance scaling, rotation or wind.
///
/// Deliberately dimensionless: everything here is a fraction, so the same values feed a
/// 0.15 m blade and a 0.45 m one. [`blade_local_position`] is where dimensions enter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BladeVertex {
    /// Row index up the blade, `0..=segments`.
    pub row: u32,
    /// Which edge of the strip: `-1.0`, `+1.0`, or `0.0` at a collapsed tip.
    pub side: f32,
    /// Normalised height along the blade, 0 at the root and 1 at the tip.
    ///
    /// This is the value every wind band scales with, so it must be exactly 0 at the
    /// root: `helio_wind_offset` returns identically zero there, which is what keeps the
    /// blade attached to the terrain instead of revealing the ground plane under it.
    pub height_frac: f32,
    /// Width taper at this row, as a fraction of the blade's full width.
    pub width_frac: f32,
    /// Whether this is the collapsed tip vertex.
    pub is_tip: bool,
}

/// Derive one vertex of the blade strip.
///
/// `lod` is clamped rather than asserted: this is called with a value that ultimately
/// came off the GPU-side LOD uniform, and the WGSL cannot panic. Both sides clamp so the
/// mirror stays a transcription.
///
/// A `vertex_index` past the LOD's vertex count clamps to the top row rather than
/// wrapping. That never happens in a correct draw — the vertex count is fixed by
/// [`LOD_VERTEX_COUNTS`] — but a producer writing the wrong `vertex_count` should draw a
/// degenerate sliver, not a blade folded back through itself.
pub fn blade_vertex(lod: u32, vertex_index: u32) -> BladeVertex {
    let lod = (lod as usize).min(LOD_COUNT - 1);
    let segments = LOD_SEGMENTS[lod];
    let is_card = LOD_IS_CARD[lod];

    // Cards have no collapsed tip. Using a sentinel that no legal vertex index can reach
    // keeps the two cases one branch apart instead of two code paths.
    let tip_index = if is_card { u32::MAX } else { 2 * segments };
    let is_tip = vertex_index == tip_index;

    let row = if is_tip {
        segments
    } else {
        (vertex_index >> 1).min(segments)
    };
    let side = if is_tip {
        0.0
    } else if vertex_index & 1 == 1 {
        1.0
    } else {
        -1.0
    };

    let height_frac = row as f32 / segments as f32;
    // Parabolic taper on blades, none on cards. The square (rather than a linear taper)
    // keeps the blade wide through its lower half, where the silhouette is actually
    // read, and pinches only near the tip.
    let width_frac = if is_card {
        1.0
    } else {
        1.0 - height_frac * height_frac
    };

    BladeVertex {
        row,
        side,
        height_frac,
        width_frac,
        is_tip,
    }
}

/// Blade-local position of one vertex, in metres, before yaw rotation and wind.
///
/// Local frame: `+X` across the blade, `+Y` up the blade from its root, `+Z` in the
/// direction the blade curls. The root is the origin, so translating by the blade's
/// world position is the whole of the model transform — there is no matrix anywhere in
/// this path.
///
/// `height` and `width` are the already-resolved per-instance dimensions (the type's
/// range lerped by the packed `height_scale` / `width_scale`, times the LOD's scale from
/// [`LOD_WIDTH_SCALE`] / [`LOD_HEIGHT_SCALE`], times the ring scale-in factor).
pub fn blade_local_position(lod: u32, vertex_index: u32, height: f32, width: f32) -> [f32; 3] {
    let clamped_lod = (lod as usize).min(LOD_COUNT - 1);
    let v = blade_vertex(lod, vertex_index);
    // Cards are flat by construction: a curled card would break the yaw-continuity rule
    // in the plan's §6.3, because the curl direction would swing the silhouette at the
    // L1→L2 boundary where the blade it replaces is still nearly straight.
    let curve = if LOD_IS_CARD[clamped_lod] {
        0.0
    } else {
        height * BLADE_CURVE_FRACTION
    };
    [
        v.side * 0.5 * width * v.width_frac,
        height * v.height_frac,
        curve * v.height_frac * v.height_frac,
    ]
}

/// Blade-local surface normal at one vertex, before yaw rotation.
///
/// Analytic rather than differenced: the strip is only two vertices wide, so a
/// finite-difference normal at the collapsed tip would divide by a zero-width edge. The
/// blade's along-stem tangent is `d/dt (0, height·t, curve·t²) = (0, height, 2·curve·t)`
/// and its across-stem tangent is `+X`, so the normal is their cross product.
///
/// The result faces `-Z` (against the curl) for a positive curve. Which face that is
/// does not matter downstream — the pipeline sets `cull_mode: None` because blades are
/// two-sided, and the fragment shader flips the normal by `@builtin(front_facing)`.
pub fn blade_local_normal(lod: u32, vertex_index: u32, height: f32) -> [f32; 3] {
    let clamped_lod = (lod as usize).min(LOD_COUNT - 1);
    let v = blade_vertex(lod, vertex_index);
    let curve = if LOD_IS_CARD[clamped_lod] {
        0.0
    } else {
        height * BLADE_CURVE_FRACTION
    };
    // cross((1,0,0), (0, height, 2·curve·t)) = (0, -2·curve·t, height)
    let n = [0.0f32, -2.0 * curve * v.height_frac, height.max(1.0e-6)];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / len, n[1] / len, n[2] / len]
}

/// The three vertex indices of triangle `triangle` in a `TriangleStrip`, in the order
/// the rasteriser assembles them.
///
/// WebGPU flips the first two vertices of every odd triangle so that consecutive
/// triangles in a strip share a winding direction. This is here because the winding test
/// would otherwise be asserting a property of the test's own assumption rather than of
/// the geometry.
pub fn strip_triangle(triangle: u32) -> [u32; 3] {
    if triangle % 2 == 0 {
        [triangle, triangle + 1, triangle + 2]
    } else {
        [triangle + 1, triangle, triangle + 2]
    }
}
