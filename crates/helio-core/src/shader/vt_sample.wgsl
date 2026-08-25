// Virtual-texturing sampling library — tier-transparent material sampling
// plus per-fragment demand feedback (Helio#238, texel-streaming S2).
//
// Prepended to any shader whose source contains `//!use helio_vt`.
// See helio_core::shader.
//
// ─────────────────────────────────────────────────────────────────────────────
// THE SAMPLING CONTRACT (first-class — mechanics exist to serve this)
// ─────────────────────────────────────────────────────────────────────────────
//
// Shaders sample textures WITHOUT EVER KNOWING TIERS EXIST. Exactly two rules
// govern every fetch, and nothing anywhere else branches on residency:
//
//   RULE 1 — WHOLE-MIP REGIME. Clamp the derivative-derived LOD to the slot's
//   floor scalar (`VtMetaRow.floor_flags.y`). One uniform-style load, zero
//   divergence: every fragment of a draw clamps against the same published
//   floor until the engine publishes a new one between frames.
//
//   RULE 2 — PAGED REGIME. Probe the always-resident page table (the meta
//   row's `mip_first_rank` LUT against its `resident_through_rank`) → fetch
//   the requested detail; a miss ⇒ the table's permanent low-detail fallback
//   entry — the same floor scalar rule 1 uses. The probe is two integer
//   compares.
//
// That is the whole contract. There is no third path, no tier enum in any
// signature, no branching on "is this streamed" anywhere else: the meta row's
// mode word picks which rule derives the final LOD, and both rules end in the
// identical `textureSampleLevel` either way. When every slot publishes an
// unrestricted floor (the default), the module degenerates to exactly today's
// sampling — the property the golden and statelessness gates pin (T1/T4,
// Helio#238).
//
// DEMAND FEEDBACK rides beside every sample: pixels are the only ground truth
// for perceptual demand, so each recorded sample also writes its wanted mip
// into the quarter-res density target (`vt_feedback_write`). Feedback is an
// observation — it never gates, delays, or alters the fetch itself.
//
// ─────────────────────────────────────────────────────────────────────────────
// MECHANICS
// ─────────────────────────────────────────────────────────────────────────────
//
// Bindings are MODULE-owned, group 2, identical across every including
// shader. Unlike the prelude — whose bindings stay the caller's business —
// the VT binding layout is part of the shared contract itself: five passes
// build the same group-2 layout, so drift surfaces at pipeline creation, not
// as silently misbound rows.
//
//     @group(2) @binding(0) var<storage, read> vt_meta: array<VtMetaRow>;
//     @group(2) @binding(1) var vt_density: texture_storage_2d<r32uint, write>;
//
// Physical-atlas scope line (honest): until SceneDB-owned materialization
// textures gain a bind path, the paged regime probes the table exactly as
// specified and then fetches through the slot's own bound mip-chained
// texture — today's whole-texture upload IS the physical page store. Only
// the fetch target inside `vt_fetch` changes when true page atlases land;
// neither the contract above nor any call site moves.
//
// Cell-write semantics, documented rather than hidden: WebGPU has no atomics
// on storage textures, so concurrent fragments mapping to one quarter-res
// cell race and the loser's store drops (plain `textureStore`, last-writer-
// wins). Overlapping fragments of one surface agree on the wanted mip within
// ±1, compaction still takes the max ACROSS cells, and the policy's
// hysteresis band exists precisely to absorb this noise — demand can be
// under-estimated by one intra-cell race, never fabricated.

/// Mirrors libhelio::GpuVtMetaRow field-for-field (64 bytes). A rename on
/// either side surfaces as garbage rows or validation errors, never silently.
struct VtMetaRow {
    /// [width, height, mip_count, format_discriminant]
    dims_xy_mips: vec4<u32>,
    /// [resident_through_rank, floor_mip, flags(mode<<1 | srgb), total_ranks]
    ///
    /// `resident_through_rank` is SceneDB's inclusive rank-prefix watermark;
    /// `0xFFFFFFFF` is the unrestricted sentinel ("everything resident").
    /// `floor_mip` is the finest mip whose ENTIRE page span lies under the
    /// watermark — the scalar both rules clamp/fall back to.
    floor_flags: vec4<u32>,
    /// First payload rank covering mip `index`, finest eight mips only
    /// (index 0 = base mip). Coarse-first canonical order guarantees every
    /// mip coarser than index 7 begins strictly before `mip_first_rank[7]`.
    mip_first_rank: array<u32, 8>,
}

@group(2) @binding(0) var<storage, read> vt_meta: array<VtMetaRow>;
@group(2) @binding(1) var vt_density: texture_storage_2d<r32uint, write>;

/// Per-invocation fragment coordinate, seeded once per fragment entry point
/// via [`vt_frame_begin`]. WGSL cannot source @builtin(position) inside an
/// arbitrary helper, and threading a parameter through every eval signature
/// would touch far more material-visible surface than this one assignment.
var<private> vt_frag_px: vec2<f32> = vec2<f32>(0.0);

/// Seeds the feedback address space. One inert call at the top of each
/// fragment entry point (`vt_frame_begin(in.clip_position.xy)`); carries no
/// shading meaning whatsoever.
fn vt_frame_begin(framebuffer_px: vec2<f32>) {
    vt_frag_px = framebuffer_px;
}

/// Derivative-based wanted-mip estimate. The CPU reference twin lives in
/// `libhelio::wanted_mip_from_derivatives`; the golden test pins them
/// together (T1).
fn vt_wanted_mip(row: VtMetaRow, uv: vec2<f32>) -> u32 {
    let w = f32(row.dims_xy_mips.x);
    let h = f32(row.dims_xy_mips.y);
    let dx = dpdx(uv);
    let dy = dpdy(uv);
    let fx = abs(dx.x) * w;
    let fy = abs(dx.y) * h;
    let gx = abs(dy.x) * w;
    let gy = abs(dy.y) * h;
    let footprint = max(max(fx, fy), max(gx, gy));
    // Round-to-nearest mirrors trilinear intent; never finer than exists.
    let chain_top = f32(max(row.dims_xy_mips.z, 1u) - 1u);
    return u32(clamp(round(log2(max(footprint, 1.0))), 0.0, chain_top));
}

/// Packs `(slot << 8) | (mip + 1)` — the +1 bias keeps 0 the untouched-cell
/// sentinel. Bit budget: slots own the high 24 bits, mips the low 8 (255 is
/// far beyond any NPOT chain's 32). Must match `libhelio::pack_feedback`
/// bit-for-bit.
fn vt_pack(slot: u32, wanted_mip: u32) -> u32 {
    return (slot << 8u) | ((min(wanted_mip, 254u) + 1u) & 0xFFu);
}

/// Records one sample's demand into the quarter-res density target.
/// Address = fragment pixel >> 2; value = packed slot/mip. Races drop the
/// loser's store (no texture atomics on WebGPU — module header explains why
/// that is acceptable).
fn vt_feedback_write(slot: u32, wanted_mip: u32) {
    if slot >= arrayLength(&vt_meta) {
        return;
    }
    let cell = vec2<u32>(vt_frag_px) >> vec2<u32>(2u, 2u);
    textureStore(vt_density, cell, vec4<u32>(vt_pack(slot, wanted_mip), 0u, 0u, 0u));
}

/// The ONE statement a raster sample site adds beside its existing fetch:
/// measures the fragment's wanted mip from UV derivatives and records it.
/// Pure observation — never gates or alters the fetch itself.
fn vt_record_demand(slot_index: u32, uv: vec2<f32>) {
    if slot_index >= arrayLength(&vt_meta) {
        return;
    }
    let row = vt_meta[slot_index];
    vt_feedback_write(slot_index, vt_wanted_mip(row, uv));
}

/// Analytic wanted-mip estimate for contexts with NO screen-space derivatives
/// (decal compute). APPROXIMATE by nature — flagged as such at every call
/// site: it assumes the surface is roughly camera-facing so one world-unit →
/// pixel conversion covers all four texel-footprint directions.
///
/// `extent_world` — the decal's world-space half extent;
/// `dist` — camera-to-surface distance;
/// `viewport_px` — render-target height in pixels;
/// `cot_half_fov` — projection matrix [1][1] (= 1/tan(fovY/2)).
fn vt_analytic_wanted_mip(row: VtMetaRow, extent_world: f32, dist: f32, viewport_px: f32, cot_half_fov: f32) -> u32 {
    // Pixels spanned per world unit at this depth (pinhole model).
    let pixels_per_world = viewport_px * 0.5 * cot_half_fov / max(dist, 1e-4);
    // Full decal extent across the screen, in texels of the BASE mip.
    let covered_texels = max(extent_world, 0.0) * 2.0 * pixels_per_world;
    let chain_top = f32(max(row.dims_xy_mips.z, 1u) - 1u);
    return u32(clamp(round(log2(max(covered_texels, 1.0))), 0.0, chain_top));
}

/// Rule 2's probe: is mip `m`'s ENTIRE page span inside the published rank
/// prefix?
///
/// The span END decides (not the start): mip m ends where the next-finer mip
/// begins, minus one; the base mip ends at the chain's last rank
/// (`total_ranks - 1`, carried in `floor_flags.w`). For the coarse tail past
/// the eight-entry LUT the nearest tabulated neighbor overstates the end —
/// conservative: it may demand slightly more residency than strictly needed,
/// never less, so an unsampleable mip is never fetched.
fn vt_mip_resident(row: VtMetaRow, m: u32) -> bool {
    let resident_through = row.floor_flags.x;
    if resident_through == 0xFFFFFFFFu {
        return true; // unrestricted sentinel
    }
    let mip_count = max(row.dims_xy_mips.z, 1u);
    let tabulated = min(mip_count, 8u);
    var end_exclusive: u32 = row.floor_flags.w;
    if m > 0u {
        // Next-finer entry: mip m-1 for m within the table; for the coarse
        // tail (m ≥ tabulated) the coarsest tabulated entry is the closest
        // proven boundary.
        let neighbor = min(m, tabulated);
        end_exclusive = row.mip_first_rank[min(neighbor - 1u, 7u)];
    }
    if end_exclusive == 0u {
        return true; // degenerate zero-length span
    }
    return (end_exclusive - 1u) <= resident_through;
}

/// THE sample function — both rules, one fetch, tier-invisible.
///
/// `tex`/`samp` are passed rather than bound (the caller indexes its own
/// bindless arrays; group/binding stay explicit at the call site like every
/// other shared module here). `row` is `vt_meta[slot]` — one storage load.
fn vt_sample(tex: texture_2d<f32>, samp: sampler, row: VtMetaRow, uv: vec2<f32>) -> vec4<f32> {
    return vt_fetch(tex, samp, row, uv, vt_wanted_mip(row, uv));
}

/// Explicit-level variant for compute contexts (no implicit derivatives):
/// the caller supplies the wanted mip — analytically for decals, or level 0.
fn vt_sample_level(
    tex: texture_2d<f32>,
    samp: sampler,
    row: VtMetaRow,
    uv: vec2<f32>,
    wanted_mip: u32,
) -> vec4<f32> {
    let chain_top = max(row.dims_xy_mips.z, 1u) - 1u;
    return vt_fetch(tex, samp, row, uv, clamp(wanted_mip, 0u, chain_top));
}

/// The single fetch both regimes collapse into.
fn vt_fetch(
    tex: texture_2d<f32>,
    samp: sampler,
    row: VtMetaRow,
    uv: vec2<f32>,
    wanted: u32,
) -> vec4<f32> {
    let floor_mip = f32(row.floor_flags.y);
    let streamed = ((row.floor_flags.z >> 1u) & 1u) == 1u;
    // Paged regime (rule 2): probe; miss ⇒ the permanent low-detail fallback
    // entry. Whole-mip regime (rule 1): straight to the floor clamp. Both
    // collapse to one scalar and one fetch — zero divergence beyond this.
    let hit = streamed && vt_mip_resident(row, wanted);
    let chain_top = max(row.dims_xy_mips.z, 1u) - 1u;
    let lod = select(floor_mip, f32(min(wanted, chain_top)), hit);
    return textureSampleLevel(tex, samp, uv, lod);
}

