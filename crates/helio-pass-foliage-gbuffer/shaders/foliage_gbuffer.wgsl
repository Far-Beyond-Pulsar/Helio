//!use helio_prelude
//!use helio_foliage_wind

// Foliage G-buffer rasterisation — blades (L0/L1) and cards (L2/L3).
//
// One pipeline, four `draw_indirect` calls, no vertex buffer and no index buffer.
// Every vertex is derived from `@builtin(vertex_index)` plus the packed 16-byte
// `BladeInstance` fetched through `visible_blades[]`. The CPU mirror of the geometry
// lives in `src/geometry.rs` and the two are pinned against each other in
// `tests/geometry.rs`; edit them together.
//
// ── Targets ─────────────────────────────────────────────────────────────────
//
// The pipeline declares all EIGHT G-buffer targets so it can be fused into the
// GBuffer subpass chain (a pipeline's fragment targets must match the render pass's
// attachments element-for-element). Grass writes five of them; the other three carry
// `ColorWrites::empty()` on the Rust side and are simply absent from
// `FoliageGBufferOutput` below:
//
//   0 albedo       Rgba8Unorm   written
//   1 normal       Rgba16Float  written
//   2 orm          Rgba8Unorm   written
//   3 emissive     Rgba16Float  written
//   4 lightmap_uv  Rg16Float    NOT written — grass has no lightmap UV
//   5 sss          Rgba16Float  NOT written
//   6 extra        Rgba16Float  NOT written
//   7 velocity     Rg16Float    written
//
// The empty write masks are mandatory, not cosmetic: a fragment target with no
// corresponding shader output has an UNDEFINED value, so without the mask those three
// channels would be filled with garbage wherever grass covers a pixel — and the
// deferred pass reads them for every pixel, not just the ones it recognises.
//
// ── Motion vectors ──────────────────────────────────────────────────────────
//
// `helio_wind_offset` is evaluated TWICE per vertex, at `wind.time_prev_time.x` and at
// `wind.time_prev_time.y`, producing a current and a previous world position. That pair
// is the whole reason `time` is an explicit parameter of the wind model and the whole
// reason the uniform carries two timestamps. Without it every blade reports zero motion,
// TAA reprojects moving grass onto the history texel of whatever was behind it, and the
// stochastic LOD cross-fade — which only resolves because TAA can integrate a dithered
// pattern that is temporally stable — turns into ghosting instead of a transition.
// See `helio-core/src/shader/foliage_wind.wgsl` and `libhelio::wind::GpuWind`.

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Mirror of `helio_foliage_core::FOLIAGE_TILE_SIZE_METERS`. Blade positions are
/// tile-local unorms, so this is the scale that turns them back into metres.
const FOLIAGE_TILE_SIZE: f32 = 8.0;

/// Mirror of `geometry::BLADE_CURVE_FRACTION`.
const FOLIAGE_BLADE_CURVE_FRACTION: f32 = 0.35;

/// `globals.flags` bit: the interaction field view is real, not the 1×1 placeholder.
const FOLIAGE_FLAG_INTERACTION_VALID: u32 = 1u;

/// Mirrors `FLAG_DEBUG_LOD` in `src/lib.rs`. Tints blades by LOD instead of shading them.
const FOLIAGE_FLAG_DEBUG_LOD: u32 = 2u;

/// Mirror of `helio_foliage_core::FOLIAGE_FLAG_RECEIVES_INTERACTION`.
const FOLIAGE_TYPE_FLAG_RECEIVES_INTERACTION: u32 = 1024u;

/// Shift/mask for the packed `visible_blades[]` entry. See `VISIBLE_TILE_SHIFT` in
/// `src/lib.rs` — this encoding is half of the contract with `FoliagePlacePass`.
const FOLIAGE_VISIBLE_TILE_SHIFT: u32 = 16u;

/// The clump-card LOD. Mirrors `CLUMP_LOD` in `src/geometry.rs` and `FOLIAGE_LOD_CLUMP`
/// in the producer's `foliage_cull.wgsl`: the one level that draws a single instance per
/// 4x4 cluster rather than per blade, and therefore the one that must not dither.
const FOLIAGE_LOD_CLUMP: u32 = 3u;
const FOLIAGE_VISIBLE_LOCAL_MASK: u32 = 0xffffu;

/// Largest finite f32, for the `is_finite` guards transcribed from
/// `helio_foliage_core::placement`.
const FOLIAGE_F32_MAX: f32 = 3.4028235e38;

// ═══════════════════════════════════════════════════════════════════════════════
// Uniforms and buffers
// ═══════════════════════════════════════════════════════════════════════════════

/// Mirror of `FoliageGlobals` in `src/lib.rs` (64 bytes).
struct FoliageGlobals {
    /// Render target size in pixels; the velocity target is in pixels/frame.
    screen_size: vec2<f32>,
    /// Frame index, the temporal axis of the cross-fade dither.
    frame: u32,
    /// `FOLIAGE_FLAG_*` bits.
    flags: u32,
    /// xyz unused (the camera position comes from `camera`), w = resident ring radius
    /// in metres — the input to the scale-in factor.
    camera_ring: vec4<f32>,
    /// xy = interaction field world-XZ origin, z = extent in metres, w = 1/extent.
    interaction_field: vec4<f32>,
    /// `FoliageQuality::lod_distance_scale`, already sanitised on the CPU.
    lod_quality_scale: f32,
    /// Ring-entry scale-in band width in metres.
    scale_in_band: f32,
    /// Width of the stochastic LOD cross-fade band in metres.
    lod_fade_band: f32,
    /// Global multiplier on the interaction bend.
    interaction_strength: f32,
}

/// Mirror of `FoliageLodUniform` in `src/lib.rs` (32 bytes), bound with a dynamic
/// offset so the four draws share one pipeline and one buffer.
struct FoliageLod {
    lod: u32,
    segments: u32,
    vertex_count: u32,
    /// First element of this LOD's region in `visible_blades[]`.
    region_base: u32,
    width_scale: f32,
    height_scale: f32,
    /// 1 when this LOD draws a flat card rather than a tapered blade strip.
    is_card: u32,
    _pad: u32,
}

/// Mirror of `helio_foliage_core::GpuFoliageType` (96 bytes).
///
/// **Every field is a scalar and none of them may become a vector.** `wind_response`
/// is the dangerous one: it sits at byte offset 52, and declaring it `vec3<f32>` would
/// give it WGSL's 16-byte alignment, push it to offset 64, and silently shift
/// `interaction_stiffness`, `material_id`, `density_layer`, `kind_and_flags` and
/// `mesh_or_impostor_id` off their Rust offsets. Nothing errors — foliage kinds resolve
/// to the wrong pipeline and materials come out random. See the WGSL mirroring section
/// on `GpuFoliageType`.
struct FoliageType {
    density: f32,
    height_min: f32,
    height_max: f32,
    width_min: f32,
    width_max: f32,
    slope_min: f32,
    slope_max: f32,
    altitude_min: f32,
    altitude_max: f32,
    lod0: f32,
    lod1: f32,
    lod2: f32,
    lod3: f32,
    // wind_response — three scalars, NEVER vec3<f32>.
    wind_trunk: f32,
    wind_branch: f32,
    wind_leaf: f32,
    interaction_stiffness: f32,
    material_id: u32,
    density_layer: u32,
    kind_and_flags: u32,
    mesh_or_impostor_id: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Mirror of `helio_foliage_core::GpuBladeInstance` (16 bytes).
struct BladeInstance {
    packed_pos: u32,
    packed_height_yaw: u32,
    packed_scale_type: u32,
    packed_tint_seed: u32,
}

/// Mirror of `helio_foliage_core::GpuFoliageTile` (32 bytes).
///
/// `vec2<i32>` is safe here and only here: it sits at offset 0, which is already
/// 8-byte aligned, so WGSL's vector alignment changes nothing.
struct FoliageTile {
    tile_coord: vec2<i32>,
    blade_offset: u32,
    blade_count: u32,
    bounds_center_y: f32,
    bounds_half_y: f32,
    state: u32,
    generation: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform> globals: FoliageGlobals;
@group(0) @binding(2) var<uniform> wind: Wind;
@group(0) @binding(3) var<storage, read> foliage_types: array<FoliageType>;
@group(0) @binding(4) var<storage, read> blade_arena: array<BladeInstance>;
@group(0) @binding(5) var<storage, read> tile_table: array<FoliageTile>;
@group(0) @binding(6) var<storage, read> visible_blades: array<u32>;
@group(0) @binding(7) var interaction_tex: texture_2d<f32>;
@group(0) @binding(8) var interaction_samp: sampler;

@group(1) @binding(0) var<uniform> lod_info: FoliageLod;

// ═══════════════════════════════════════════════════════════════════════════════
// Unpacking — bit-for-bit transcriptions of `helio_foliage_core::packing`
// ═══════════════════════════════════════════════════════════════════════════════
//
// These must match `unpack_blade` exactly. They are all one or two instructions, and
// the reciprocals are written as literal divisions of the *exact* denominators the Rust
// side uses (65535, 255, 65536) rather than rounded constants — a blade whose position
// drifts by a unorm step against the CPU reference fails the placement determinism test
// for reasons that look nothing like a packing bug.

fn foliage_unorm16(bits: u32) -> f32 {
    return f32(bits) * (1.0 / 65535.0);
}

fn foliage_unorm8(bits: u32) -> f32 {
    return f32(bits) * (1.0 / 255.0);
}

/// 16-bit turn to radians. Divides by 65536, not 65535: an angle wraps, so `0` and `2π`
/// are the same code and spending one on each would make the seam at zero non-uniform.
fn foliage_yaw(bits: u32) -> f32 {
    return f32(bits) * (6.28318530718 / 65536.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOD maths — transcriptions of `helio_foliage_core::placement`
// ═══════════════════════════════════════════════════════════════════════════════
//
// `FoliagePlacePass` reimplements the same functions to classify blades into the four
// `visible_blades` regions. If this copy and that one drift, the producer puts a blade
// in one region while this shader computes the cross-fade weight of another, and the
// two representations of the plant shear against each other across the fade band —
// exactly the artefact the fade exists to hide.

/// WGSL has no `isFinite`. NaN fails every comparison including with itself, and
/// infinities fail the magnitude test.
fn foliage_is_finite(x: f32) -> bool {
    return x == x && abs(x) <= FOLIAGE_F32_MAX;
}

/// Transcription of `helio_foliage_core::lod_fade_alpha`.
///
/// Smoothstep rather than a linear ramp: a linear dissolve is continuous in value but
/// not in slope, and the two slope discontinuities read as faint rings sweeping over the
/// ground at grazing angles.
fn foliage_lod_fade_alpha(d: f32, band_start: f32, band_end: f32) -> f32 {
    if !foliage_is_finite(d) || !foliage_is_finite(band_start) || !foliage_is_finite(band_end) {
        return 1.0;
    }
    if band_end <= band_start {
        return select(0.0, 1.0, d < band_start);
    }
    let t = clamp((d - band_start) / (band_end - band_start), 0.0, 1.0);
    return 1.0 - (t * t * (3.0 - 2.0 * t));
}

/// Transcription of `helio_foliage_core::scale_in_factor`.
///
/// `distance_from_edge` is how far *inside* the resident ring the blade sits. Zero at
/// the edge, so a tile that becomes resident grows its hundreds of blades out of the
/// ground instead of publishing them all at full height in one frame.
fn foliage_scale_in_factor(distance_from_edge: f32, band: f32) -> f32 {
    if !foliage_is_finite(distance_from_edge) {
        return 0.0;
    }
    if !foliage_is_finite(band) || band <= 0.0 {
        return select(0.0, 1.0, distance_from_edge > 0.0);
    }
    let t = clamp(distance_from_edge / band, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Upper distance bound of LOD `level`, with the same non-decreasing repair
/// `select_blade_lod` applies.
///
/// The repair is not defensive noise: a mis-authored ladder like `[8, 45, 20, 120]` must
/// degrade to an empty L2 rather than let a band be skipped, or the scene pops straight
/// from a 3-segment blade to a clump card with no fade in between.
fn foliage_lod_threshold(ty: FoliageType, level: u32, scale: f32) -> f32 {
    var ladder = array<f32, 4>(ty.lod0, ty.lod1, ty.lod2, ty.lod3);
    var threshold = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let scaled = ladder[i] * scale;
        if scaled > threshold {
            threshold = scaled;
        }
        if i >= level {
            break;
        }
    }
    return threshold;
}

/// Fraction of blades that survive the stochastic dither at this LOD and distance.
///
/// Symmetric by construction: at any threshold the near LOD's weight is `f` and the far
/// LOD's is `1 - f`, so the two sum to exactly one blade's worth of coverage everywhere
/// in the band. That property is what makes the dither resolve to constant density
/// rather than to a visible thinning or doubling ring.
fn foliage_cross_fade(ty: FoliageType, level: u32, distance: f32, scale: f32, band: f32) -> f32 {
    let upper = foliage_lod_threshold(ty, level, scale);
    var alpha = foliage_lod_fade_alpha(distance, upper - band, upper);
    if level > 0u {
        let lower = foliage_lod_threshold(ty, level - 1u, scale);
        alpha = min(alpha, 1.0 - foliage_lod_fade_alpha(distance, lower - band, lower));
    }
    return alpha;
}

/// Interleaved gradient noise — the screen-space half of the cross-fade dither.
///
/// Cheap, and more importantly *decorrelated across frames*, which is what lets TAA
/// integrate the dither away. A static dither pattern would resolve to a visible
/// stipple no matter how good the motion vectors are.
///
/// Alpha-to-coverage would be the obvious alternative and is not available: the
/// G-buffer is single-sampled everywhere (`MultisampleState::default()`, `count: 1`) and
/// WebGPU rejects `alphaToCoverageEnabled` when `count == 1`. Helio anti-aliases with
/// TAA/FXAA/SMAA post-passes, not MSAA.
fn foliage_dither(pixel: vec2<f32>, frame: u32) -> f32 {
    let p = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Geometry — transcription of `src/geometry.rs`
// ═══════════════════════════════════════════════════════════════════════════════

struct BladeVertex {
    side: f32,
    height_frac: f32,
    width_frac: f32,
}

fn foliage_blade_vertex(segments: u32, is_card: bool, vertex_index: u32) -> BladeVertex {
    // Cards have no collapsed tip; a sentinel no legal index can reach keeps the two
    // cases one branch apart rather than two code paths.
    var tip_index = 2u * segments;
    if is_card {
        tip_index = 0xffffffffu;
    }
    let is_tip = vertex_index == tip_index;

    var row = min(vertex_index >> 1u, segments);
    var side = select(-1.0, 1.0, (vertex_index & 1u) == 1u);
    if is_tip {
        row = segments;
        side = 0.0;
    }

    var out: BladeVertex;
    out.side = side;
    out.height_frac = f32(row) / f32(segments);
    // Parabolic taper on blades, none on cards.
    out.width_frac = select(1.0 - out.height_frac * out.height_frac, 1.0, is_card);
    return out;
}

/// Rotate a blade-local offset about +Y by `yaw`.
fn foliage_yaw_rotate(local: vec3<f32>, yaw: f32) -> vec3<f32> {
    let s = sin(yaw);
    let c = cos(yaw);
    return vec3<f32>(local.x * c + local.z * s, local.y, -local.x * s + local.z * c);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Interaction
// ═══════════════════════════════════════════════════════════════════════════════

/// Bend contributed by the interaction field, in world space.
///
/// TEMPORARY: `FoliageInteractionPass` does not exist yet (it is phase 3 of the plan),
/// so `frame.foliage_interaction` is normally unwritten and the pass binds a 1×1
/// placeholder with `FOLIAGE_FLAG_INTERACTION_VALID` clear. The early return below is
/// what makes that cost nothing — it is not a fallback that produces a wrong bend, it
/// produces no bend at all, which is exactly what "no interaction pass" should look
/// like.
///
/// Also temporary: the bend is applied identically at `t` and `t - dt`, so it
/// contributes nothing to the velocity target. The field has no history buffer to
/// evaluate a previous-frame bend against; when phase 3 lands with one, this should take
/// a time argument like the wind model does, for the same reason.
fn foliage_interaction_bend(world_pos: vec3<f32>, height_frac: f32, stiffness: f32) -> vec3<f32> {
    if (globals.flags & FOLIAGE_FLAG_INTERACTION_VALID) == 0u {
        return vec3<f32>(0.0);
    }
    let uv = (world_pos.xz - globals.interaction_field.xy) * globals.interaction_field.w;
    // Outside the field is unbent, with no seam: the field's edge is always well beyond
    // the distance at which a displacement is still non-zero.
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec3<f32>(0.0);
    }
    // `textureSampleLevel`, not `textureSample`: the vertex stage has no implicit
    // derivatives, so the LOD must be explicit.
    let field = textureSampleLevel(interaction_tex, interaction_samp, uv, 0.0);
    // RG = horizontal displacement, B = vertical crush (the plan's §9).
    let bend = vec3<f32>(field.r, -field.b, field.g);
    // Squared height fraction for the same cantilever reason the sway band uses it: the
    // root must not move or the blade detaches from the terrain.
    let response = globals.interaction_strength / max(stiffness, 0.25);
    return bend * (height_frac * height_frac * response);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Vertex stage
// ═══════════════════════════════════════════════════════════════════════════════

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) height_frac: f32,
    @location(2) prev_clip_position: vec4<f32>,
    @location(3) tint: vec2<f32>,
    /// Stochastic cross-fade weight. Flat: it is a per-blade quantity, and
    /// interpolating it would make the dither threshold vary across a single blade and
    /// eat its silhouette from one edge.
    @location(4) @interpolate(flat) fade: f32,
    /// Stable per-blade hash seed, the temporal anchor of the dither. Flat for the same
    /// reason, and because it is conceptually an integer.
    @location(5) @interpolate(flat) seed: u32,
    /// LOD this instance was drawn at, for the `FOLIAGE_FLAG_DEBUG_LOD` view.
    @location(6) @interpolate(flat) lod: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    // `cameras[0]`, not `@builtin(view_index)`: that builtin requires the MULTIVIEW
    // capability, which this engine does not request and most desktop adapters do not
    // expose, so a shader using it fails to create at all. Every other pass in the graph
    // (gbuffer, vg_gbuffer) indexes `cameras[0]`; stereo selection is a graph-wide change,
    // not something this pass gets to opt into unilaterally.
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    // ── Resolve the instance ──────────────────────────────────────────────────
    //
    // `instance_index` is relative to this draw, so the producer must write
    // `first_instance = 0` into every `DrawIndirectArgs`; the region offset is applied
    // here from the per-LOD uniform. See `VISIBLE_TILE_SHIFT` in `src/lib.rs` for the
    // packed reference encoding.
    let reference = visible_blades[lod_info.region_base + instance_index];
    let tile_slot = reference >> FOLIAGE_VISIBLE_TILE_SHIFT;
    let local_index = reference & FOLIAGE_VISIBLE_LOCAL_MASK;

    let tile = tile_table[tile_slot];
    let blade = blade_arena[tile.blade_offset + local_index];

    // ── Reconstruct the blade's world root ────────────────────────────────────
    // Mirror of `helio_foliage_core::blade_world_position`. Positions are tile-local so
    // a blade's encoding does not depend on where in the world its tile sits, which is
    // what lets placement be reproducible across GPUs.
    let tile_origin = vec2<f32>(f32(tile.tile_coord.x), f32(tile.tile_coord.y)) * FOLIAGE_TILE_SIZE;
    let local_uv = vec2<f32>(
        foliage_unorm16(blade.packed_pos & 0xffffu),
        foliage_unorm16(blade.packed_pos >> 16u),
    );
    // The height offset is an f16 in the low half; the yaw in the high half is NOT a
    // float, so it must not go through `unpack2x16float`.
    let height_offset = unpack2x16float(blade.packed_height_yaw).x;
    let yaw = foliage_yaw(blade.packed_height_yaw >> 16u);
    let root = vec3<f32>(
        tile_origin.x + local_uv.x * FOLIAGE_TILE_SIZE,
        tile.bounds_center_y + height_offset,
        tile_origin.y + local_uv.y * FOLIAGE_TILE_SIZE,
    );

    // ── Resolve the foliage type ──────────────────────────────────────────────
    let type_id = (blade.packed_scale_type >> 16u) & 0xffu;
    let ty = foliage_types[type_id];
    let height_lerp = foliage_unorm8(blade.packed_scale_type & 0xffu);
    let width_lerp = foliage_unorm8((blade.packed_scale_type >> 8u) & 0xffu);

    // ── Ring-entry scale-in and LOD cross-fade ────────────────────────────────
    // Mono index 0: the pipeline has no MULTIVIEW capability, so `view_index` is
    // unavailable. A future single-pass stereo path enables multiview and swaps
    // these to `cameras[view_index]`; the storage array is already dual-eye.
    let camera_pos = cameras[0].position_near.xyz;
    let distance_to_camera = distance(root, camera_pos);
    let scale_in = foliage_scale_in_factor(
        globals.camera_ring.w - distance_to_camera,
        globals.scale_in_band,
    );
    let fade = foliage_cross_fade(
        ty,
        lod_info.lod,
        distance_to_camera,
        globals.lod_quality_scale,
        globals.lod_fade_band,
    );

    // ── How this LOD cross-fades ──────────────────────────────────────────────
    //
    // The stochastic dither removes a whole *instance*. For a blade that is one thin
    // sliver among thousands and it resolves cleanly under TAA. For an L3 clump card it
    // does not: that card stands in for sixteen blades and is four times as wide, so
    // discarding one punches a sixteen-blade hole in the ground. Across the L2→L3 band,
    // where about half the cards are being discarded at any instant, those holes are the
    // visible gap at the LOD boundary — the fade itself creates it.
    //
    // So the clump LOD fades by *area* instead: the card shrinks toward zero across the
    // band and is never discarded. Coverage is the quantity that has to stay continuous,
    // and area goes as the square of the linear dimensions, hence `sqrt(fade)`. Blades
    // keep the dither, which is cheaper and correct at their size.
    var size_fade = 1.0;
    var dither_fade = fade;
    if lod_info.lod == FOLIAGE_LOD_CLUMP {
        size_fade = sqrt(clamp(fade, 0.0, 1.0));
        dither_fade = 1.0;
    }

    let height = mix(ty.height_min, ty.height_max, height_lerp)
        * lod_info.height_scale
        * scale_in
        * size_fade;
    let width = mix(ty.width_min, ty.width_max, width_lerp)
        * lod_info.width_scale
        * size_fade;

    // ── Derive the vertex ─────────────────────────────────────────────────────
    let is_card = lod_info.is_card != 0u;
    let v = foliage_blade_vertex(lod_info.segments, is_card, vertex_index);
    // Cards are flat: a curled card would swing its silhouette at the L1→L2 boundary
    // where the blade it replaces is still nearly straight, which is exactly the
    // continuity the plan's §6.3 card-orientation rule is protecting.
    let curve = select(height * FOLIAGE_BLADE_CURVE_FRACTION, 0.0, is_card);
    let local = vec3<f32>(
        v.side * 0.5 * width * v.width_frac,
        height * v.height_frac,
        curve * v.height_frac * v.height_frac,
    );
    let world_base = root + foliage_yaw_rotate(local, yaw);

    // Analytic normal: the strip is two vertices wide, so a differenced normal would
    // divide by a zero-width edge at the collapsed tip.
    let local_normal = normalize(vec3<f32>(
        0.0,
        -2.0 * curve * v.height_frac,
        max(height, 1.0e-6),
    ));

    // ── Wind, evaluated at BOTH timestamps ────────────────────────────────────
    //
    // This is the pair that fills the velocity target. Collapsing it to one evaluation
    // is the single change that would make every blade of grass in the frame smear under
    // TAA and take the LOD cross-fade down with it. See the file header.
    let response = vec3<f32>(ty.wind_trunk, ty.wind_branch, ty.wind_leaf);
    let seed = blade.packed_tint_seed >> 16u;
    let wind_now = helio_wind_offset(
        wind, world_base, root, v.height_frac, seed, response, wind.time_prev_time.x,
    );
    let wind_prev = helio_wind_offset(
        wind, world_base, root, v.height_frac, seed, response, wind.time_prev_time.y,
    );

    // The interaction bend is added to both positions, so it currently contributes no
    // velocity — see `foliage_interaction_bend`.
    var bend = vec3<f32>(0.0);
    if (ty.kind_and_flags & FOLIAGE_TYPE_FLAG_RECEIVES_INTERACTION) != 0u {
        bend = foliage_interaction_bend(world_base, v.height_frac, ty.interaction_stiffness);
    }

    let position_now = world_base + wind_now + bend;
    let position_prev = world_base + wind_prev + bend;

    var out: VertexOutput;
    out.clip_position = cameras[0].view_proj * vec4<f32>(position_now, 1.0);
    out.prev_clip_position = cameras[0].prev_view_proj * vec4<f32>(position_prev, 1.0);
    out.world_normal = foliage_yaw_rotate(local_normal, yaw);
    out.height_frac = v.height_frac;
    out.tint = vec2<f32>(
        foliage_unorm8(blade.packed_tint_seed & 0xffu),
        foliage_unorm8((blade.packed_tint_seed >> 8u) & 0xffu),
    );
    out.fade = dither_fade;
    out.seed = seed;
    out.lod = lod_info.lod;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fragment stage
// ═══════════════════════════════════════════════════════════════════════════════

/// Locations 4, 5 and 6 are deliberately absent — see the file header. wgpu's
/// fragment-interface check iterates the shader's outputs and does not demand one per
/// declared target, so the pipeline still validates against all eight attachments while
/// the three unwritten ones carry `ColorWrites::empty()`.
struct FoliageGBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) orm: vec4<f32>,
    @location(3) emissive: vec4<f32>,
    @location(7) velocity: vec2<f32>,
}

/// Grass is a dielectric; 0.04 is the standard non-metal F0, packed into the spare alpha
/// channels exactly the way `gbuffer.wgsl` does.
const FOLIAGE_F0: f32 = 0.04;

/// Screen-space motion in pixels per frame.
///
/// Identical in convention to `gbuffer.wgsl::compute_velocity` — `clip_position.xy` in a
/// fragment stage is already the framebuffer pixel coordinate, so only the previous
/// position needs the NDC→pixel mapping. Diverging here would put foliage velocity on a
/// different scale from every other surface and TAA would reject it as an outlier.
fn foliage_velocity(clip_position: vec2<f32>, prev_clip: vec4<f32>) -> vec2<f32> {
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_pixel = vec2<f32>(
        (prev_ndc.x * 0.5 + 0.5) * globals.screen_size.x,
        (0.5 - prev_ndc.y * 0.5) * globals.screen_size.y,
    );
    return clip_position - prev_pixel;
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> FoliageGBufferOutput {
    // ── Stochastic LOD cross-fade ─────────────────────────────────────────────
    //
    // A stable per-blade hash plus a per-pixel, per-frame dither. The blade hash is what
    // keeps a given blade making the *same* decision for the whole of its silhouette
    // (so a fading blade thins out of existence as a whole rather than dissolving from
    // one edge), and the screen/frame term is what makes the pattern integrate away
    // under TAA instead of freezing into a stipple.
    let threshold = fract(
        helio_wind_hash_unorm(input.seed) + foliage_dither(input.clip_position.xy, globals.frame),
    );
    if input.fade < threshold {
        discard;
    }

    // ── Shading ───────────────────────────────────────────────────────────────
    //
    // Procedural, not textured. This pass binds no material table and samples no
    // textures: card cutout atlases and the bindless material path arrive with the
    // impostor work in phase 5, and pulling `helio-pass-gbuffer`'s bindless material
    // bind group in now would make this crate depend on another pass crate for
    // infrastructure — the exact coupling the TODO on `create_material_bgl` asks not to
    // spread.
    let base = vec3<f32>(0.055, 0.115, 0.030);
    let tip = vec3<f32>(0.180, 0.320, 0.075);
    var albedo = mix(base, tip, input.height_frac);
    // Per-blade variation. Without it a field of grass is one flat colour and reads as
    // carpet no matter how good the geometry is.
    albedo *= vec3<f32>(
        mix(0.80, 1.20, input.tint.x),
        mix(0.90, 1.10, input.tint.y),
        mix(0.75, 1.25, input.tint.x),
    );

    // Blades are two-sided (`cull_mode: None`), so the geometric normal has to be
    // flipped for back faces or half of every blade lights as if it faced away from the
    // sun.
    let normal = normalize(select(-input.world_normal, input.world_normal, front_facing));

    // Ambient occlusion from the blade's own height: the base of a grass sward is deep
    // in its own canopy. This is cheap and it is what stops a lawn reading as a flat
    // green plane under ambient light.
    let ao = mix(0.35, 1.0, input.height_frac);

    var out: FoliageGBufferOutput;
    // LOD debug view: flat, unlit-looking colour per LOD so band boundaries are obvious.
    // Written into albedo with the emissive channel below left alone, so the bands stay
    // legible under any lighting.
    if (globals.flags & FOLIAGE_FLAG_DEBUG_LOD) != 0u {
        var lod_colour = vec3<f32>(1.0, 0.0, 0.0);
        if input.lod == 1u {
            lod_colour = vec3<f32>(0.0, 1.0, 0.0);
        } else if input.lod == 2u {
            lod_colour = vec3<f32>(0.0, 0.4, 1.0);
        } else if input.lod >= 3u {
            lod_colour = vec3<f32>(1.0, 0.9, 0.0);
        }
        albedo = lod_colour;
    }

    out.albedo = vec4<f32>(albedo, 1.0);
    out.normal = vec4<f32>(normal, FOLIAGE_F0);
    out.orm = vec4<f32>(ao, 0.75, 0.0, FOLIAGE_F0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, FOLIAGE_F0);
    out.velocity = foliage_velocity(input.clip_position.xy, input.prev_clip_position);
    return out;
}
