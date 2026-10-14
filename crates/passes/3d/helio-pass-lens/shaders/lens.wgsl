//!use helio_prelude

// ── Physically based screen-space lens ───────────────────────────────────────
//
// A view's lens is an ordered stack of optical elements reduced per pixel to
// four quantities: a ray map (sensor uv -> scene uv, per wavelength), a focus
// field s(uv), a pupil function and a transmission. The focus field and the
// thin-lens defocus it implies come from `focus_field`; the pupil function
// (`pupil_shape`, `pupil_profile`: iris, anamorphic squeeze, cat's eye,
// swirl, spherical aberration, apodization, obstruction) and the
// longitudinal chromatic aberration from the PUPIL and LONGITUDINAL_CA
// elements, resolved per gather pixel (`resolve_pupil`). The ray map
// (`ray_map`: projection, distortion, shift, keystone, breathing, prism,
// lateral chromatic aberration, underwater ports, composed in stack order)
// is applied in `cs_composite` after the defocus, which is computed in the
// undistorted scene; `cs_coc` scales the circles by the local magnification
// of the map. The transmission (`transmission`: vignetting, filters, T-stop)
// multiplies the composite.
//
// Stages, all GPU-sized from `cs_resolve` so a disabled lens costs one
// workgroup and a copy:
//
//   cs_resolve    match the view's lens row, autofocus, the ray map's
//                 overscan, size the dispatches
//   cs_coc        per-pixel signed circle of confusion from the focus field
//   cs_tiles      16x16 tile extent of the circles
//   cs_dilate     widen each tile's extent by the circles that reach it
//   cs_prefilter  half-resolution colour and circles, near class winning
//   cs_gather     near/far layers by scatter-as-gather
//   cs_composite  sharp, far and near layers back to full resolution
//
// Circles of confusion are diameters in input pixels; negative is in front
// of the focal surface (near field), positive behind it.

struct LensRow {
    view_id: u32,
    enabled: u32,
    focus_mode: u32,
    quality: u32,
    focal_length_mm: f32,
    f_number: f32,
    t_stop: f32,
    sensor_width_mm: f32,
    sensor_height_mm: f32,
    focus_distance_m: f32,
    af_u: f32,
    af_v: f32,
    af_speed: f32,
    max_coc_px: f32,
    element_first: u32,
    element_count: u32,
    tilt_angle: f32,
    tilt_axis: f32,
    field_curvature: f32,
    aperture_blades: u32,
    aperture_rotation: f32,
    af_radius: f32,
    _pad0: f32,
    _pad1: f32,
}

struct ElementRow {
    kind: u32,
    region_kind: u32,
    region_count: u32,
    flags: u32,
    feather: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    params: array<f32, 16>,
    region: array<f32, 16>,
}

// Mirrors `ResolvedLens` (gpu_types.rs); other passes read it as well.
struct Resolved {
    f_number: f32,
    focal_length_mm: f32,
    sensor_width_mm: f32,
    aperture_blades: u32,
    aperture_rotation: f32,
    natural_vignette: f32,
    enabled: u32,
    quality: u32,
    t_stop: f32,
    sensor_height_mm: f32,
    focus_distance_m: f32,
    max_coc_px: f32,
    field_curvature: f32,
    tilt_angle: f32,
    tilt_axis: f32,
    element_first: u32,
    element_count: u32,
    required_overscan: f32,
    map_flags: u32,
    spectral_samples: u32,
}

// Persistent between frames: the autofocus position in dioptres.
struct State {
    focus_dpt: f32,
    valid: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Frame {
    dt: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> lenses: array<LensRow>;
@group(0) @binding(1) var<storage, read> elements: array<ElementRow>;
@group(0) @binding(2) var<storage, read> cameras: array<Camera, 2>;
// The scene depth buffer, read as a plain float texture.
@group(0) @binding(3) var scene_depth: texture_2d<f32>;
@group(0) @binding(4) var scene_color: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> state: State;
@group(0) @binding(6) var<storage, read_write> resolved: Resolved;
@group(0) @binding(7) var<storage, read_write> args: array<u32, 18>;
@group(0) @binding(8) var<uniform> frame: Frame;
@group(0) @binding(9) var coc_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var coc_in: texture_2d<f32>;
@group(0) @binding(11) var<storage, read_write> tile_raw: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> tile_dilated: array<vec4<f32>>;
@group(0) @binding(13) var half_color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var half_coc_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(15) var half_color: texture_2d<f32>;
@group(0) @binding(16) var half_coc: texture_2d<f32>;
@group(0) @binding(17) var far_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(18) var near_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(19) var far_in: texture_2d<f32>;
@group(0) @binding(20) var near_in: texture_2d<f32>;
@group(0) @binding(21) var color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(22) var linear_samp: sampler;
// Custom iris shapes, one per layer, coverage in .r over the square that
// inscribes the pupil circle. A 1x1 white layer when the host has none.
@group(0) @binding(23) var aperture_shapes: texture_2d_array<f32>;

const PI: f32 = 3.141592653589793;
const GOLDEN_ANGLE: f32 = 2.399963229728653;
const HALF_MAX: f32 = 65504.0;
// Sky pixels sit at infinity; this keeps (s2 - s1) / s2 finite.
const SKY_DISTANCE: f32 = 1.0e5;
const MAX_DISTANCE: f32 = 1.0e6;
const DEFAULT_MOUNT_M: f32 = 0.08;
const DEFAULT_AF_RADIUS: f32 = 0.02;
const DEFAULT_MAX_COC_PX: f32 = 32.0;
const NO_ROW: u32 = 0xffffffffu;
const MAX_ELEMENTS: u32 = 32u;
const TILE: u32 = 16u;
// Circles smaller than this (pixels, half the near-field cutoff of the
// gather) are in focus for classification.
const NEAR_COC: f32 = 0.5;

const FOCUS_AUTO_POINT: u32 = 1u;

const ELEMENT_DIOPTER: u32 = 1u;
const ELEMENT_SWEET_SPOT: u32 = 2u;
const ELEMENT_PUPIL: u32 = 3u;
const ELEMENT_LONGITUDINAL_CA: u32 = 4u;
// Ray map elements are PROJECTION..=DOME_PORT, transmission elements
// NATURAL_VIGNETTE..=T_STOP_LOSS (components.rs `element_kind`).
const ELEMENT_PROJECTION: u32 = 5u;
const ELEMENT_BROWN_CONRADY: u32 = 6u;
const ELEMENT_DIVISION: u32 = 7u;
const ELEMENT_ANAMORPHIC_DISTORTION: u32 = 8u;
const ELEMENT_ANAMORPHIC_SQUEEZE: u32 = 9u;
const ELEMENT_SHIFT: u32 = 10u;
const ELEMENT_KEYSTONE: u32 = 11u;
const ELEMENT_FOCUS_BREATHING: u32 = 12u;
const ELEMENT_PRISM: u32 = 13u;
const ELEMENT_LATERAL_CA: u32 = 14u;
const ELEMENT_FLAT_PORT: u32 = 15u;
const ELEMENT_DOME_PORT: u32 = 16u;
const ELEMENT_NATURAL_VIGNETTE: u32 = 17u;
const ELEMENT_OPTICAL_VIGNETTE: u32 = 18u;
const ELEMENT_MECHANICAL_VIGNETTE: u32 = 19u;
const ELEMENT_ND_FILTER: u32 = 20u;
const ELEMENT_GRADUATED_ND: u32 = 21u;
const ELEMENT_COLOR_FILTER: u32 = 22u;
const ELEMENT_T_STOP_LOSS: u32 = 23u;

// `Resolved::map_flags` (gpu_types.rs).
const MAP_REMAP: u32 = 1u;
const MAP_SPECTRAL: u32 = 2u;
const MAP_TRANSMISSION: u32 = 4u;

// Spectral sampling: wavelengths 400..700 nm in equal bins, each with a
// position t = (lambda - 550) / 150 and per-channel sensor weights that sum
// to 1 over the samples.
const MAX_SPECTRAL: u32 = 16u;
const NO_COPY: u32 = 0xffffffffu;
const WATER_INDEX: f32 = 1.333;
// Largest rectilinear tangent a map may reach (about 85 degrees).
const MAX_ANGLE: f32 = 1.4835;
const OVERSCAN_CAP: f32 = 16.0;
const OVERSCAN_SAMPLES: u32 = 64u;

// Slots of a PUPIL element's params (components.rs `pupil`).
const P_BLADES: u32 = 0u;
const P_CURVATURE: u32 = 1u;
const P_ROTATION: u32 = 2u;
const P_ROUND_UNTIL_F: u32 = 3u;
const P_TEXTURE_LAYER: u32 = 4u;
const P_TEXTURE_AMOUNT: u32 = 5u;
const P_SQUEEZE: u32 = 6u;
const P_CATS_EYE: u32 = 7u;
const P_BARREL_RADIUS: u32 = 8u;
const P_SWIRL: u32 = 9u;
const P_SPHERICAL: u32 = 10u;
const P_ONION_AMPLITUDE: u32 = 11u;
const P_ONION_RINGS: u32 = 12u;
const P_APODIZATION: u32 = 13u;
const P_OBSTRUCTION: u32 = 14u;
const P_SA_ASYMMETRY: u32 = 15u;

// Samples that integrate a pupil's area and mean profile, and the most
// gather lanes (two seam configurations times three colour channels).
const PUPIL_SAMPLES: u32 = 96u;
const MAX_LANES: u32 = 6u;

const REGION_FULL_FRAME: u32 = 0u;
const REGION_HALF_PLANE: u32 = 1u;
const REGION_ELLIPSE: u32 = 2u;
const REGION_POLYGON: u32 = 3u;
const REGION_RADIAL_BAND: u32 = 4u;

// Indirect argument slot (x, y, z) of each stage in `args`.
const ARGS_COC: u32 = 0u;
const ARGS_TILES: u32 = 1u;
const ARGS_DILATE: u32 = 2u;
const ARGS_PREFILTER: u32 = 3u;
const ARGS_GATHER: u32 = 4u;
const ARGS_COMPOSITE: u32 = 5u;

fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
fn bounded(x: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if !finite(x) { return fallback; }
    return clamp(x, lo, hi);
}
fn radiance(c: vec3<f32>) -> vec3<f32> {
    return vec3(bounded(c.r, 0.0, HALF_MAX, 0.0),
                bounded(c.g, 0.0, HALF_MAX, 0.0),
                bounded(c.b, 0.0, HALF_MAX, 0.0));
}
fn ceil_div(a: vec2<u32>, b: u32) -> vec2<u32> {
    return (a + (b - 1u)) / b;
}

// ── Scene depth ──────────────────────────────────────────────────────────────

/// View-space distance along the optical axis at a pixel of an image of
/// `size`. Depth can be rendered below the image's resolution (TSR upscales
/// colour only): then the footprint's nearest of 2x2 texels is taken, so a
/// foreground edge is never given the background's focus.
fn scene_distance(px: vec2<u32>, size: vec2<u32>) -> f32 {
    let dsize = textureDimensions(scene_depth);
    let c = (vec2<f32>(px) + 0.5) * vec2<f32>(dsize) / vec2<f32>(size);
    let last = vec2<i32>(dsize) - 1;
    var d: f32;
    if dsize.x >= size.x && dsize.y >= size.y {
        d = textureLoad(scene_depth, clamp(vec2<i32>(floor(c)), vec2(0), last), 0).r;
    } else {
        let b = vec2<i32>(floor(c - 0.5));
        d = textureLoad(scene_depth, clamp(b, vec2(0), last), 0).r;
        d = min(d, textureLoad(scene_depth, clamp(b + vec2(1, 0), vec2(0), last), 0).r);
        d = min(d, textureLoad(scene_depth, clamp(b + vec2(0, 1), vec2(0), last), 0).r);
        d = min(d, textureLoad(scene_depth, clamp(b + vec2(1, 1), vec2(0), last), 0).r);
    }
    if !(d < 1.0) { return SKY_DISTANCE; }
    return helio_view_depth(d, cameras[0].position_near.w, cameras[0].forward_far.w);
}

// ── cs_resolve ───────────────────────────────────────────────────────────────

var<workgroup> picked_row: atomic<u32>;
var<workgroup> af_inverse: array<f32, 64>;
var<workgroup> af_hits: array<f32, 64>;
// The ray map's context for the other threads, and each thread's reach.
var<workgroup> wg_ctx: MapCtx;
var<workgroup> overscan_reach: array<f32, 64>;

fn write_args(slot: u32, groups: vec2<u32>, dispatch: bool) {
    args[slot * 3u] = select(0u, groups.x, dispatch);
    args[slot * 3u + 1u] = select(0u, groups.y, dispatch);
    args[slot * 3u + 2u] = select(0u, 1u, dispatch);
}

@compute @workgroup_size(64)
fn cs_resolve(@builtin(local_invocation_index) lid: u32) {
    if lid == 0u { atomicStore(&picked_row, NO_ROW); }
    workgroupBarrier();
    // The first enabled row of this view wins, like the camera post-process
    // baseline: SceneDB rows span the entity range, mostly empty.
    let view_id = bitcast<u32>(cameras[0].jitter_frame.w);
    let rows = arrayLength(&lenses);
    for (var i = lid; i < rows; i += 64u) {
        if lenses[i].enabled != 0u && lenses[i].view_id == view_id {
            atomicMin(&picked_row, i);
        }
    }
    workgroupBarrier();
    let index = atomicLoad(&picked_row);
    let found = index != NO_ROW;
    let row = lenses[select(0u, index, found)];
    let autofocus = found && row.focus_mode == FOCUS_AUTO_POINT;

    // Autofocus: a grid of depth taps around the point, averaged in
    // dioptres so near geometry dominates the way a rangefinder reads it.
    var inverse_distance = 0.0;
    var hit = 0.0;
    if autofocus {
        let dsize = textureDimensions(scene_depth);
        let radius = select(DEFAULT_AF_RADIUS, row.af_radius, row.af_radius > 0.0);
        let offset = (vec2<f32>(f32(lid % 8u), f32(lid / 8u)) / 7.0 - 0.5) * 2.0 * radius;
        let uv = clamp(vec2(row.af_u, row.af_v) + offset, vec2(0.0), vec2(1.0));
        let px = min(vec2<u32>(uv * vec2<f32>(dsize)), dsize - 1u);
        let d = textureLoad(scene_depth, vec2<i32>(px), 0).r;
        if d < 1.0 {
            inverse_distance = 1.0 / helio_view_depth(d, cameras[0].position_near.w, cameras[0].forward_far.w);
            hit = 1.0;
        }
    }
    af_inverse[lid] = inverse_distance;
    af_hits[lid] = hit;
    workgroupBarrier();

    let size = textureDimensions(scene_color);
    if lid == 0u {
        resolve_lens(row, found, autofocus, size);
    }
    workgroupBarrier();

    // The ray map's reach at the frame boundary, one sample per thread.
    overscan_reach[lid] = boundary_reach(lid, wg_ctx, size);
    workgroupBarrier();
    if lid != 0u { return; }
    var reach = 1.0;
    for (var i = 0u; i < OVERSCAN_SAMPLES; i++) {
        reach = max(reach, overscan_reach[i]);
    }
    resolved.required_overscan = min(reach, OVERSCAN_CAP);

    let half_size = (size + 1u) / 2u;
    let tiles = ceil_div(size, TILE);
    write_args(ARGS_COC, ceil_div(size, 8u), found);
    write_args(ARGS_TILES, tiles, found);
    write_args(ARGS_DILATE, ceil_div(tiles, 8u), found);
    write_args(ARGS_PREFILTER, ceil_div(half_size, 8u), found);
    write_args(ARGS_GATHER, ceil_div(half_size, 8u), found);
    // Always: a disabled lens copies the image through.
    write_args(ARGS_COMPOSITE, ceil_div(size, 8u), true);
}

/// Thread 0 of `cs_resolve`: sanitises the row, runs the autofocus easing and
/// writes the resolved lens and the ray map's context.
fn resolve_lens(row: LensRow, found: bool, autofocus: bool, size: vec2<u32>) {
    var sum = 0.0;
    var count = 0.0;
    for (var i = 0u; i < 64u; i++) {
        sum += af_inverse[i];
        count += af_hits[i];
    }

    // No row: neutral optics for consumers, lens off.
    var out: Resolved;
    out.f_number = 2.8;
    out.focal_length_mm = 50.0;
    out.sensor_width_mm = 36.0;
    out.aperture_blades = 6u;
    out.aperture_rotation = 0.0;
    out.natural_vignette = 0.0;
    out.enabled = 0u;
    out.quality = 0u;
    out.t_stop = 2.8;
    out.sensor_height_mm = 24.0;
    out.max_coc_px = DEFAULT_MAX_COC_PX;
    out.field_curvature = 0.0;
    out.tilt_angle = 0.0;
    out.tilt_axis = 0.0;
    out.element_first = 0u;
    out.element_count = 0u;
    out.required_overscan = 1.0;
    out.map_flags = 0u;
    out.spectral_samples = 1u;
    if found {
        out.f_number = bounded(row.f_number, 0.7, 32.0, 2.8);
        out.focal_length_mm = bounded(row.focal_length_mm, 8.0, 600.0, 50.0);
        out.sensor_width_mm = bounded(row.sensor_width_mm, 4.0, 70.0, 36.0);
        out.aperture_blades = select(clamp(row.aperture_blades, 3u, 16u), 0u, row.aperture_blades == 0u);
        out.aperture_rotation = bounded(row.aperture_rotation, -100.0, 100.0, 0.0);
        out.enabled = 1u;
        out.quality = min(row.quality, 2u);
        out.t_stop = bounded(row.t_stop, 0.7, 64.0, out.f_number);
        out.sensor_height_mm = select(
            out.sensor_width_mm * (2.0 / 3.0),
            bounded(row.sensor_height_mm, 3.0, 50.0, out.sensor_width_mm * (2.0 / 3.0)),
            row.sensor_height_mm > 0.0,
        );
        out.max_coc_px = select(DEFAULT_MAX_COC_PX, clamp(row.max_coc_px, 1.0, 128.0), row.max_coc_px > 0.0);
        out.field_curvature = bounded(row.field_curvature, -64.0, 64.0, 0.0);
        out.tilt_angle = bounded(row.tilt_angle, -1.4, 1.4, 0.0);
        out.tilt_axis = bounded(row.tilt_axis, -100.0, 100.0, 0.0);
        out.element_first = row.element_first;
        out.element_count = min(row.element_count, MAX_ELEMENTS);
    }

    // Focus in dioptres: manual follows the row; autofocus eases toward the
    // measured depth and holds where the point sees only sky.
    let focal_m = out.focal_length_mm * 1.0e-3;
    var target_dpt = 1.0 / max(bounded(row.focus_distance_m, 0.0, MAX_DISTANCE, MAX_DISTANCE), focal_m * 1.001);
    let has_history = state.valid != 0u;
    if autofocus {
        if count > 0.0 {
            target_dpt = sum / count;
        } else if has_history {
            target_dpt = state.focus_dpt;
        } else {
            target_dpt = 0.0;
        }
    }
    var dpt = target_dpt;
    if autofocus && has_history && row.af_speed > 0.0 {
        dpt = mix(state.focus_dpt, target_dpt, 1.0 - exp(-max(frame.dt, 0.0) * row.af_speed));
    }
    state.focus_dpt = dpt;
    state.valid = select(0u, 1u, found);
    out.focus_distance_m = clamp(1.0 / max(dpt, 1.0e-6), focal_m * 1.001, MAX_DISTANCE);

    // Which stages the stack needs, so a plain lens pays for none of them.
    var natural = 0.0;
    for (var i = 0u; i < out.element_count; i++) {
        let e = elements[out.element_first + i];
        if e.kind >= ELEMENT_PROJECTION && e.kind <= ELEMENT_DOME_PORT {
            out.map_flags |= MAP_REMAP;
        }
        // Lateral chromatic aberration, and the dispersion of the water
        // ports, make the map depend on the wavelength.
        if e.kind == ELEMENT_LATERAL_CA
            || (e.kind == ELEMENT_FLAT_PORT && abs(e.params[1]) > 0.0)
            || (e.kind == ELEMENT_DOME_PORT && abs(e.params[3]) > 0.0) {
            out.map_flags |= MAP_SPECTRAL;
        }
        if e.kind >= ELEMENT_NATURAL_VIGNETTE && e.kind <= ELEMENT_T_STOP_LOSS {
            out.map_flags |= MAP_TRANSMISSION;
        }
        if e.kind == ELEMENT_NATURAL_VIGNETTE {
            natural = max(natural, bounded(e.params[0], 0.0, 1.0, 0.0));
        }
    }
    out.natural_vignette = natural;
    if (out.map_flags & MAP_SPECTRAL) != 0u {
        out.map_flags |= MAP_REMAP;
        out.spectral_samples = spectral_sample_count(out.quality);
    }
    resolved = out;
    wg_ctx = make_ctx(out, size);
}

// ── Focus field ──────────────────────────────────────────────────────────────

/// Signed distance to the element's region (negative inside), in
/// image-height units.
fn polygon_distance(e: ElementRow, q: vec2<f32>) -> f32 {
    let n = min(e.region_count, 8u);
    if n < 3u { return 1.0e3; }
    // The signed area picks the outward side for either winding.
    var area = 0.0;
    for (var i = 0u; i < n; i++) {
        let a = vec2(e.region[2u * i], e.region[2u * i + 1u]);
        let j = (i + 1u) % n;
        let b = vec2(e.region[2u * j], e.region[2u * j + 1u]);
        area += a.x * b.y - b.x * a.y;
    }
    let side = select(-1.0, 1.0, area > 0.0);
    var d = -1.0e3;
    for (var i = 0u; i < n; i++) {
        let a = vec2(e.region[2u * i], e.region[2u * i + 1u]);
        let j = (i + 1u) % n;
        let b = vec2(e.region[2u * j], e.region[2u * j + 1u]);
        let edge = b - a;
        let outward = side * vec2(edge.y, -edge.x) / max(length(edge), 1.0e-6);
        d = max(d, dot(q - a, outward));
    }
    return d;
}

fn region_distance(e: ElementRow, q: vec2<f32>) -> f32 {
    switch e.region_kind {
        case REGION_FULL_FRAME: {
            return -1.0e3;
        }
        case REGION_HALF_PLANE: {
            let n = vec2(cos(e.region[0]), sin(e.region[0]));
            return e.region[1] - dot(q, n);
        }
        case REGION_ELLIPSE: {
            let radii = max(vec2(e.region[2], e.region[3]), vec2(1.0e-4));
            let p = q - vec2(e.region[0], e.region[1]);
            let c = cos(e.region[4]);
            let s = sin(e.region[4]);
            let local = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
            return (length(local / radii) - 1.0) * min(radii.x, radii.y);
        }
        case REGION_POLYGON: {
            return polygon_distance(e, q);
        }
        case REGION_RADIAL_BAND: {
            let r = length(q - vec2(e.region[2], e.region[3]));
            return max(e.region[0] - r, r - e.region[1]);
        }
        default: {
            // Mask channel: reserved, covers nothing.
            return 1.0e3;
        }
    }
}

/// Width of an element's soft edge in image-height units. A diopter sits in
/// front of the lens, out of focus, so its edge is blurred by the entrance
/// pupil seen from the filter: A / e radians, A the aperture diameter and e
/// the filter's distance. Authored widths replace that.
fn feather_width(e: ElementRow) -> f32 {
    if e.feather >= 0.0 { return e.feather; }
    // Only a diopter seam is optical; other regions have a nominal edge.
    if e.kind != ELEMENT_DIOPTER { return 0.002; }
    let focal_m = resolved.focal_length_mm * 1.0e-3;
    let aperture_m = focal_m / resolved.f_number;
    let mount_m = select(DEFAULT_MOUNT_M, e.params[2], e.params[2] > 0.0);
    return (aperture_m / mount_m) * (focal_m / (resolved.sensor_height_mm * 1.0e-3));
}

/// Coverage of the region at `q`: 1 inside, 0 outside, smooth across the
/// feather.
fn region_weight(e: ElementRow, q: vec2<f32>) -> f32 {
    let t = clamp(0.5 - region_distance(e, q) / max(feather_width(e), 1.0e-5), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

struct FocusSample {
    /// Focus in dioptres for the configuration that covers most of the
    /// pixel, and for the one across the strongest seam.
    inverse_a: f32,
    inverse_b: f32,
    /// Share of the pixel in configuration b: 0 away from seams, up to 0.5.
    omega: f32,
    /// Extra defocus from sweet spots, in pixels.
    extra: f32,
    transmission: f32,
}

/// The lens's focus field at a sensor position. Every element adds to the
/// base focus in dioptres: 1/s' = 1/s + sum(D). A diopter's region is
/// partial across its seam, where the pixel sees both its covered and
/// uncovered focus: the two strongest configurations are kept.
fn focus_field(uv: vec2<f32>, size: vec2<f32>) -> FocusSample {
    let ndc = helio_uv_to_ndc(uv);
    // Focal plane tilted about an axis through the focus point (Scheimpflug):
    // the ray through this pixel meets it at t = cos(a) s0 / (cos(a) + sin(a) (axis . d)).
    let direction = vec2(ndc.x / cameras[0].proj[0][0], ndc.y / cameras[0].proj[1][1]);
    let ca = cos(resolved.tilt_angle);
    let sa = sin(resolved.tilt_angle);
    let tilt = ca + sa * (cos(resolved.tilt_axis) * direction.x + sin(resolved.tilt_axis) * direction.y);
    let plane_distance = select(MAX_DISTANCE, ca * resolved.focus_distance_m / tilt, tilt > 0.05 * ca);
    let corner = length(size) * 0.5;
    let r = length((uv - 0.5) * size) / corner;
    var inverse = 1.0 / plane_distance + resolved.field_curvature * r * r;

    let q = (uv - 0.5) * vec2(size.x / size.y, 1.0);
    var strongest = 1.0;
    var seam_power = 0.0;
    var seam_covered = false;
    var seam_weight = 0.0;
    var extra = 0.0;
    var transmission = 1.0;
    for (var i = 0u; i < resolved.element_count; i++) {
        let e = elements[resolved.element_first + i];
        if e.kind == ELEMENT_DIOPTER {
            let w = region_weight(e, q);
            let covered = w >= 0.5;
            if covered { inverse += e.params[0]; }
            if w > 0.0 && w < 1.0 && abs(w - 0.5) < strongest {
                strongest = abs(w - 0.5);
                seam_power = e.params[0];
                seam_covered = covered;
                seam_weight = w;
            }
            // Darkest at the seam's centre.
            transmission = min(transmission, 1.0 - clamp(e.params[1], 0.0, 1.0) * (1.0 - abs(2.0 * w - 1.0)));
        } else if e.kind == ELEMENT_SWEET_SPOT {
            let outside = 1.0 - region_weight(e, q);
            extra = sqrt(extra * extra + (e.params[0] * outside) * (e.params[0] * outside));
        } else if e.kind == ELEMENT_DOME_PORT {
            // The dome's virtual image is a sphere of radius R n / (n - 1)
            // about its centre: its axial depth along a ray at angle theta
            // is rho cos(theta), i.e. 1/(rho cos) - 1/rho dioptres extra.
            let n = port_index(e.params[2]);
            let image_radius = max(e.params[0], 1.0e-3) * n / (n - 1.0);
            let secant = sqrt(1.0 + dot(direction, direction));
            inverse += region_weight(e, q) * bounded(e.params[4], 0.0, 4.0, 0.0) * (secant - 1.0) / image_radius;
        }
    }
    var out: FocusSample;
    out.inverse_a = inverse;
    out.inverse_b = inverse + select(seam_power, -seam_power, seam_covered);
    out.omega = select(0.0, min(seam_weight, 1.0 - seam_weight), strongest < 1.0);
    out.extra = extra;
    out.transmission = transmission;
    return out;
}

fn focus_distance(inverse: f32) -> f32 {
    let focal_m = resolved.focal_length_mm * 1.0e-3;
    return clamp(select(MAX_DISTANCE, 1.0 / inverse, inverse > 1.0e-6), focal_m * 1.001, MAX_DISTANCE);
}

/// Thin-lens circle of confusion diameter in pixels of an image `width`
/// pixels wide, signed: negative for geometry nearer than the focus.
///   c = A |s2 - s1| / s2 * f / (s1 - f),  A = f / N
fn circle_of_confusion(s1: f32, s2: f32, width: f32) -> f32 {
    let focal_m = resolved.focal_length_mm * 1.0e-3;
    let aperture_m = focal_m / resolved.f_number;
    let on_sensor = aperture_m * abs(s2 - s1) / s2 * focal_m / (s1 - focal_m);
    let px = on_sensor / (resolved.sensor_width_mm * 1.0e-3) * width;
    return select(px, -px, s2 < s1);
}

fn limited(coc: f32, extra: f32) -> f32 {
    var combined = coc;
    if extra > 0.0 {
        // Extra defocus adds in quadrature; a pixel otherwise in focus blurs
        // in the far direction.
        let magnitude = sqrt(coc * coc + extra * extra);
        combined = select(magnitude, -magnitude, coc < -NEAR_COC);
    }
    return clamp(combined, -resolved.max_coc_px, resolved.max_coc_px);
}

@compute @workgroup_size(8, 8)
fn cs_coc(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(coc_out);
    if any(id.xy >= size) { return; }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let s2 = scene_distance(id.xy, size);
    let field = focus_field(uv, vec2<f32>(size));
    // The circle is defocus on the sensor. It is gathered in the undistorted
    // scene and then mapped to the sensor, which magnifies it by
    // 1 / sqrt(det J) (J the map's Jacobian, scene per sensor), so the scene
    // circle is smaller by sqrt(det J) where the map zooms in.
    var magnification = 1.0;
    if (resolved.map_flags & MAP_REMAP) != 0u {
        let q = (uv - 0.5) * vec2(f32(size.x) / f32(size.y), 1.0);
        magnification = defocus_scale(q, make_ctx(resolved, size));
    }
    let coc_a = limited(circle_of_confusion(focus_distance(field.inverse_a), s2, f32(size.x)) * magnification, field.extra * magnification);
    var coc_b = coc_a;
    if field.omega > 0.0 {
        coc_b = limited(circle_of_confusion(focus_distance(field.inverse_b), s2, f32(size.x)) * magnification, field.extra * magnification);
    }
    textureStore(coc_out, id.xy, vec4(coc_a, coc_b, field.omega, field.transmission));
}

// ── Tiles ────────────────────────────────────────────────────────────────────
//
// Tile record: x largest circle diameter, y seam flag, z most negative and
// w most positive circle.

var<workgroup> tile_max: array<f32, 256>;
var<workgroup> tile_seam: array<f32, 256>;
var<workgroup> tile_min_signed: array<f32, 256>;
var<workgroup> tile_max_signed: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn cs_tiles(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let size = textureDimensions(coc_in);
    let px = wg.xy * TILE + local.xy;
    var largest = 0.0;
    var seam = 0.0;
    var lowest = 0.0;
    var highest = 0.0;
    if all(px < size) {
        let c = textureLoad(coc_in, px, 0);
        largest = max(abs(c.x), abs(c.y));
        seam = select(0.0, 1.0, c.z > 0.0);
        lowest = min(c.x, c.y);
        highest = max(c.x, c.y);
    }
    tile_max[lid] = largest;
    tile_seam[lid] = seam;
    tile_min_signed[lid] = lowest;
    tile_max_signed[lid] = highest;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            tile_max[lid] = max(tile_max[lid], tile_max[lid + stride]);
            tile_seam[lid] = max(tile_seam[lid], tile_seam[lid + stride]);
            tile_min_signed[lid] = min(tile_min_signed[lid], tile_min_signed[lid + stride]);
            tile_max_signed[lid] = max(tile_max_signed[lid], tile_max_signed[lid + stride]);
        }
        workgroupBarrier();
    }
    if lid == 0u {
        let tiles_x = ceil_div(size, TILE).x;
        tile_raw[wg.y * tiles_x + wg.x] = vec4(tile_max[0], tile_seam[0], tile_min_signed[0], tile_max_signed[0]);
    }
}

/// Widens each tile's record by the circles of nearby tiles that reach it,
/// so a tile's gather radius covers every source that can scatter into it.
/// x becomes the search radius in pixels.
@compute @workgroup_size(8, 8)
fn cs_dilate(@builtin(global_invocation_id) id: vec3<u32>) {
    // The tile grid is that of the circle texture.
    let size = textureDimensions(coc_in);
    let tiles = ceil_div(size, TILE);
    if any(id.xy >= tiles) { return; }
    let reach = i32(min(ceil(resolved.max_coc_px * 0.5 / f32(TILE)) + 1.0, 9.0));
    var radius = 0.0;
    var seam = 0.0;
    var lowest = 0.0;
    var highest = 0.0;
    for (var dv = -reach; dv <= reach; dv++) {
        for (var du = -reach; du <= reach; du++) {
            let n = vec2<i32>(id.xy) + vec2(du, dv);
            if any(n < vec2(0)) || any(vec2<u32>(n) >= tiles) { continue; }
            let raw = tile_raw[u32(n.y) * tiles.x + u32(n.x)];
            // Nearest pixel distance between the two tiles.
            let gap = max(f32(max(abs(du), abs(dv))) * f32(TILE) - f32(TILE - 1u), 0.0);
            let source_radius = raw.x * 0.5;
            if source_radius > 0.0 && gap <= source_radius {
                radius = max(radius, source_radius + 1.0);
                seam = max(seam, raw.y);
                lowest = min(lowest, raw.z);
                highest = max(highest, raw.w);
            }
        }
    }
    tile_dilated[id.y * tiles.x + id.x] = vec4(radius, seam, lowest, highest);
}

// ── Half resolution ──────────────────────────────────────────────────────────

/// Two-by-two blocks become one texel. A block with a near-field pixel takes
/// the near class so foreground blur is not lost to the sharp pixels beside
/// it; its colour is the mean of that class alone. Circles are stored in
/// half-resolution pixels.
@compute @workgroup_size(8, 8)
fn cs_prefilter(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = textureDimensions(half_color_out);
    if any(id.xy >= half_size) { return; }
    let full = textureDimensions(scene_color);
    var colors: array<vec3<f32>, 4>;
    var cocs: array<vec4<f32>, 4>;
    for (var k = 0u; k < 4u; k++) {
        let p = min(id.xy * 2u + vec2(k & 1u, k >> 1u), full - 1u);
        colors[k] = radiance(textureLoad(scene_color, p, 0).rgb);
        cocs[k] = textureLoad(coc_in, p, 0);
    }
    var nearest = 0u;
    var farthest = 0u;
    for (var k = 1u; k < 4u; k++) {
        if cocs[k].x < cocs[nearest].x { nearest = k; }
        if cocs[k].x > cocs[farthest].x { farthest = k; }
    }
    let near_class = cocs[nearest].x < -NEAR_COC;
    let representative = select(farthest, nearest, near_class);
    var sum = vec3(0.0);
    var count = 0.0;
    for (var k = 0u; k < 4u; k++) {
        if (cocs[k].x < -NEAR_COC) == near_class {
            sum += colors[k];
            count += 1.0;
        }
    }
    let c = cocs[representative];
    textureStore(half_color_out, id.xy, vec4(sum / count, 1.0));
    textureStore(half_coc_out, id.xy, vec4(c.x * 0.5, c.y * 0.5, c.z, c.w));
}

// ── Pupil function ───────────────────────────────────────────────────────────
//
// What a point source's blur disc looks like: P(rho, theta; field position,
// defocus sign, wavelength). The disc of a source with circle radius R is
// described in the pupil's own frame, u = (offset / R) after the field
// distortions below, so every shape lies inside the unit circle and the
// tile search radii stay valid.
//
//   shape    where light gets through (0..1): polygon iris with curved
//            blades or a custom texture, anamorphic squeeze, swirl, cat's
//            eye clip, central obstruction
//   profile  how bright the disc is across its radius: spherical
//            aberration (rim vs centre, flipped by the defocus sign),
//            onion rings, Gaussian apodization
//
// Pupil weights are normalised: the shape by its area fraction of the
// circle (so a uniform patch keeps its alpha under any shape), the profile
// by its mean over the shape (so brightness redistributes, never changes).
// The wavelength enters through the circle itself (`ca_half`).

struct Pupil {
    blades: u32,
    curvature: f32,
    rotation: f32,
    squeeze: f32,
    cats_eye: f32,
    barrel: f32,
    swirl: f32,
    spherical: f32,
    asymmetry: f32,
    onion: f32,
    rings: f32,
    apodization: f32,
    obstruction: f32,
    layer: i32,
    texture_amount: f32,
    // Unit vector pointing away from the optical axis at this pixel.
    radial: vec2<f32>,
    // Distance from the centre, 1 at the corner.
    field_r: f32,
    // Focus shift of red, green and blue relative to green, in dioptres.
    shifts: vec3<f32>,
    // False for a plain circle: no shape evaluation, no normalisation.
    shaped: bool,
    area: f32,
    mean_far: f32,
    mean_near: f32,
}

/// The pupil at a gather pixel: the lens's own polygon iris, replaced by the
/// last PUPIL element whose region covers the pixel, plus the summed
/// chromatic focus shifts. `uv` is the pixel in the unit square, `size` the
/// image in pixels.
fn resolve_pupil(uv: vec2<f32>, size: vec2<f32>) -> Pupil {
    var pp: Pupil;
    pp.blades = resolved.aperture_blades;
    pp.curvature = 0.0;
    pp.rotation = resolved.aperture_rotation;
    pp.squeeze = 1.0;
    pp.cats_eye = 0.0;
    pp.barrel = 1.0;
    pp.swirl = 0.0;
    pp.spherical = 0.0;
    pp.asymmetry = 0.0;
    pp.onion = 0.0;
    pp.rings = 6.0;
    pp.apodization = 0.0;
    pp.obstruction = 0.0;
    pp.layer = 0;
    pp.texture_amount = 0.0;
    pp.area = 1.0;
    pp.mean_far = 1.0;
    pp.mean_near = 1.0;
    var shifts = vec3(0.0);

    let q = (uv - 0.5) * vec2(size.x / size.y, 1.0);
    for (var i = 0u; i < resolved.element_count; i++) {
        let e = elements[resolved.element_first + i];
        if e.kind == ELEMENT_PUPIL {
            if region_weight(e, q) < 0.5 { continue; }
            let p = e.params;
            if p[P_BLADES] >= 0.0 {
                let n = u32(bounded(p[P_BLADES], 0.0, 16.0, 0.0) + 0.5);
                pp.blades = select(n, 0u, n < 3u);
                pp.rotation = bounded(p[P_ROTATION], -100.0, 100.0, 0.0);
            } else {
                pp.blades = resolved.aperture_blades;
                pp.rotation = resolved.aperture_rotation;
            }
            var curvature = bounded(p[P_CURVATURE], 0.0, 1.0, 0.0);
            let round_until = bounded(p[P_ROUND_UNTIL_F], 0.0, 64.0, 0.0);
            if round_until > 0.0 {
                // Wide open the blades are out of the way: a circle.
                curvature = mix(1.0, curvature, smoothstep(0.5 * round_until, round_until, resolved.f_number));
            }
            pp.curvature = curvature;
            pp.layer = i32(clamp(bounded(p[P_TEXTURE_LAYER], 0.0, 4096.0, 0.0), 0.0, f32(textureNumLayers(aperture_shapes) - 1u)));
            pp.texture_amount = bounded(p[P_TEXTURE_AMOUNT], 0.0, 1.0, 0.0);
            pp.squeeze = bounded(p[P_SQUEEZE], 1.0, 4.0, 1.0);
            pp.cats_eye = bounded(p[P_CATS_EYE], 0.0, 1.5, 0.0);
            pp.barrel = select(1.0, bounded(p[P_BARREL_RADIUS], 0.5, 4.0, 1.0), p[P_BARREL_RADIUS] > 0.0);
            pp.swirl = bounded(p[P_SWIRL], -0.6, 4.0, 0.0);
            pp.spherical = bounded(p[P_SPHERICAL], -1.0, 1.0, 0.0);
            pp.asymmetry = bounded(p[P_SA_ASYMMETRY], 0.0, 1.0, 0.0);
            pp.onion = bounded(p[P_ONION_AMPLITUDE], 0.0, 1.0, 0.0);
            pp.rings = select(6.0, bounded(p[P_ONION_RINGS], 1.0, 32.0, 6.0), p[P_ONION_RINGS] > 0.0);
            pp.apodization = select(0.0, bounded(p[P_APODIZATION], 0.1, 4.0, 0.0), p[P_APODIZATION] > 0.0);
            pp.obstruction = bounded(p[P_OBSTRUCTION], 0.0, 0.95, 0.0);
        } else if e.kind == ELEMENT_LONGITUDINAL_CA {
            let w = region_weight(e, q);
            if w > 0.0 {
                let d = vec3(
                    bounded(e.params[0], -8.0, 8.0, 0.0),
                    bounded(e.params[1], -8.0, 8.0, 0.0),
                    bounded(e.params[2], -8.0, 8.0, 0.0),
                );
                shifts += w * d;
            }
        }
    }
    // Green is the channel the focus field was resolved for.
    pp.shifts = shifts - vec3(shifts.y);
    pp.shaped = pp.blades >= 3u || pp.squeeze > 1.0001 || pp.cats_eye > 0.0 || pp.swirl != 0.0
        || pp.spherical != 0.0 || pp.onion > 0.0 || pp.apodization > 0.0
        || pp.obstruction > 0.0 || pp.texture_amount > 0.0;

    let offset = (uv - 0.5) * size;
    let reach = length(offset);
    pp.field_r = reach / (0.5 * length(size));
    pp.radial = select(vec2(1.0, 0.0), offset / max(reach, 1.0e-6), reach > 1.0e-4);
    return pp;
}

/// Shape and radial position of the pupil at `p`, the offset in pixels of the
/// destination from a source whose circle has radius `rs`. Returns
/// (coverage 0..1, radius in the pupil: 1 at its edge).
fn pupil_shape(p: vec2<f32>, rs: f32, pp: Pupil) -> vec2<f32> {
    let pn = p / rs;
    if !pp.shaped {
        let rho = length(pn);
        return vec2(clamp((1.0 - rho) * rs + 0.5, 0.0, 1.0), min(rho, 1.0));
    }
    let tangent = vec2(-pp.radial.y, pp.radial.x);
    // Swirl squashes the disc along the radial direction: it elongates
    // tangentially, increasingly toward the corners.
    let radial_part = dot(pn, pp.radial) * (1.0 + pp.swirl * pp.field_r);
    let u0 = pp.radial * radial_part + tangent * dot(pn, tangent);
    // Anamorphic: the pupil is `squeeze` times taller than wide.
    let u = vec2(u0.x * pp.squeeze, u0.y);
    let ru = length(u);

    var rho_iris = ru;
    if pp.blades >= 3u {
        let n = f32(pp.blades);
        let sector = 2.0 * PI / n;
        let t = (atan2(u.y, u.x) - pp.rotation) / sector;
        let a = (t - floor(t) - 0.5) * sector;
        let polygon = ru * cos(a) / cos(PI / n);
        rho_iris = mix(polygon, ru, pp.curvature);
    }
    var shape = clamp((1.0 - rho_iris) * rs + 0.5, 0.0, 1.0);
    var rho = rho_iris;
    if pp.texture_amount > 0.0 {
        let c = cos(pp.rotation);
        let s = sin(pp.rotation);
        let ut = vec2(c * u.x + s * u.y, -s * u.x + c * u.y);
        let sampled = textureSampleLevel(aperture_shapes, linear_samp, ut * 0.5 + 0.5, pp.layer, 0.0).r;
        // The texture covers the square around the pupil circle; the
        // circle bounds it.
        let textured = sampled * clamp((1.0 - ru) * rs + 0.5, 0.0, 1.0);
        shape = mix(shape, textured, pp.texture_amount);
        rho = mix(rho, ru, pp.texture_amount);
    }
    if pp.cats_eye > 0.0 {
        // The lens barrel clips the iris: a second circle displaced toward
        // the optical axis in proportion to the field radius.
        let centre = -pp.radial * min(pp.cats_eye * pp.field_r, 1.8);
        let rho_barrel = length(u - centre) / pp.barrel;
        shape = min(shape, clamp((1.0 - rho_barrel) * rs + 0.5, 0.0, 1.0));
        rho = max(rho, rho_barrel);
    }
    if pp.obstruction > 0.0 {
        shape *= clamp((ru - pp.obstruction) * rs + 0.5, 0.0, 1.0);
    }
    return vec2(shape, clamp(rho, 0.0, 1.0));
}

/// Radial brightness of the disc at pupil radius `rho`, `sign` +1 behind
/// the focus and -1 in front of it. Spherical aberration trades light
/// between the rim and the centre (3 rho^4 - 1 has a zero mean over the
/// disc); its sign flips with the defocus sign by `asymmetry`.
fn pupil_profile(rho: f32, sign: f32, pp: Pupil) -> f32 {
    let e = pp.spherical * mix(1.0, sign, pp.asymmetry);
    let r2 = rho * rho;
    var w = max(1.0 + e * (3.0 * r2 * r2 - 1.0), 0.03);
    if pp.onion > 0.0 {
        w *= max(1.0 + pp.onion * cos(2.0 * PI * pp.rings * rho), 0.0);
    }
    if pp.apodization > 0.0 {
        w *= exp(-r2 / (2.0 * pp.apodization * pp.apodization));
    }
    return w;
}

/// Integrates the pupil over a disc of radius `rs`: its area as a fraction
/// of the circle and its mean profile on each side of the focus.
fn normalised_pupil(pp_in: Pupil, rs: f32) -> Pupil {
    var pp = pp_in;
    if !pp.shaped { return pp; }
    var area = 0.0;
    var far = 0.0;
    var near = 0.0;
    for (var i = 0u; i < PUPIL_SAMPLES; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(PUPIL_SAMPLES));
        let angle = f32(i) * GOLDEN_ANGLE;
        let s = pupil_shape(vec2(cos(angle), sin(angle)) * (r * rs), rs, pp);
        area += s.x;
        far += s.x * pupil_profile(s.y, 1.0, pp);
        near += s.x * pupil_profile(s.y, -1.0, pp);
    }
    pp.area = max(area / f32(PUPIL_SAMPLES), 1.0e-3);
    pp.mean_far = far / max(area, 1.0e-6);
    pp.mean_near = near / max(area, 1.0e-6);
    return pp;
}

// ── Gather ───────────────────────────────────────────────────────────────────

fn interleaved_noise(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

/// Scatter-as-gather. Every source texel spreads its radiance over a disc of
/// its own circle of confusion, shaped by the pupil function; each
/// destination collects the sources whose disc reaches it, weighted by
/// pupil / disc area so the energy a source spreads is its own. Far-field
/// samples are averaged (a mean, so a uniform blur keeps the image's
/// energy); near-field samples form a layer whose alpha is the fraction of
/// the destination the blurred foreground covers, composited over the far
/// layer. Seam pixels gather twice, with the circles of each configuration,
/// and blend by the seam weight. With longitudinal chromatic aberration each
/// colour channel gathers with its own circle (a "lane" per seam
/// configuration and channel).
@compute @workgroup_size(8, 8)
fn cs_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = textureDimensions(half_color);
    if any(id.xy >= half_size) { return; }
    let tiles = ceil_div(half_size * 2u, TILE);
    let tile = tile_dilated[min(id.y * 2u / TILE, tiles.y - 1u) * tiles.x + min(id.x * 2u / TILE, tiles.x - 1u)];
    let center = textureLoad(half_color, id.xy, 0).rgb;
    var radius = tile.x * 0.5;
    if radius < 0.25 {
        textureStore(far_out, id.xy, vec4(center, 1.0));
        textureStore(near_out, id.xy, vec4(0.0));
        return;
    }
    let own = textureLoad(half_coc, id.xy, 0);
    let seams = select(1u, 2u, own.z > 0.0);
    let gather_near = tile.z < -NEAR_COC;

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(half_size);
    var pp = resolve_pupil(uv, vec2<f32>(half_size));
    // A channel's circle differs from green's by the circle of its focus
    // shift: dc = A f dD / (s1 - f), in half-resolution diameters.
    let focal_m = resolved.focal_length_mm * 1.0e-3;
    let per_dioptre = focal_m / resolved.f_number * focal_m / (resolved.sensor_width_mm * 1.0e-3)
        * f32(half_size.x * 2u) / max(1.0 - focal_m / resolved.focus_distance_m, 0.05);
    let chromatic = any(abs(pp.shifts) > vec3(1.0e-4));
    let colors = select(1u, 3u, chromatic);
    let lanes = seams * colors;
    let ca_half = pp.shifts * (0.5 * per_dioptre);
    let max_half = 0.5 * resolved.max_coc_px;
    if chromatic {
        radius += 0.5 * max(abs(ca_half.x), max(abs(ca_half.y), abs(ca_half.z)));
    }
    pp = normalised_pupil(pp, max(radius, 2.0));

    let max_taps = select(24u, 56u, resolved.quality != 0u);
    let taps = clamp(u32(ceil(f32(max_taps) * min(radius / 8.0, 1.0))), 8u, max_taps);
    let rotation = 2.0 * PI * interleaved_noise(vec2<f32>(id.xy));
    let last = vec2<i32>(half_size) - 1;

    var far_sum: array<vec3<f32>, MAX_LANES>;
    var far_weight: array<f32, MAX_LANES>;
    var near_sum: array<vec3<f32>, MAX_LANES>;
    var near_weight: array<f32, MAX_LANES>;
    var near_cover: array<f32, MAX_LANES>;
    // Unweighted far-class mean: stands in where no source's disc reaches,
    // i.e. behind a near-field blur, so its inner edge fades into the
    // surroundings instead of into the foreground's own colour.
    var fill_sum: array<vec3<f32>, MAX_LANES>;
    var fill_count: array<f32, MAX_LANES>;
    // Tap -1 is the destination's own texel, so a uniformly blurred region
    // weighs it like any other tap.
    for (var k = -1; k < i32(taps); k++) {
        var offset = vec2(0.0);
        if k >= 0 {
            let rr = radius * sqrt((f32(k) + 0.5) / f32(taps));
            let angle = f32(k) * GOLDEN_ANGLE + rotation;
            offset = vec2(cos(angle), sin(angle)) * rr;
        }
        let texel = clamp(vec2<i32>(id.xy) + vec2<i32>(floor(offset + 0.5)), vec2(0), last);
        let delta = vec2<f32>(texel - vec2<i32>(id.xy));
        let color = textureLoad(half_color, texel, 0).rgb;
        let circles = textureLoad(half_coc, texel, 0);
        for (var lane = 0u; lane < lanes; lane++) {
            let seam = lane / colors;
            let channel = lane % colors;
            var coc = select(circles.x, circles.y, seam == 1u);
            var masked = color;
            if chromatic {
                coc = clamp(coc + ca_half[channel], -max_half, max_half);
                masked = color * vec3(f32(channel == 0u), f32(channel == 1u), f32(channel == 2u));
            }
            let source_radius = 0.5 * abs(coc);
            let far_class = coc > -NEAR_COC;
            if far_class {
                fill_sum[lane] += masked;
                fill_count[lane] += 1.0;
            }
            // The destination's place in the source's disc.
            let rs = max(source_radius, 0.5);
            let lobe = pupil_shape(-delta, rs, pp);
            if lobe.x <= 0.0 { continue; }
            var weight = lobe.x / (rs * rs);
            if pp.shaped {
                let behind = coc >= 0.0;
                let norm = pp.area * select(pp.mean_near, pp.mean_far, behind);
                weight = weight * pupil_profile(lobe.y, select(-1.0, 1.0, behind), pp) / norm;
            }
            if far_class {
                far_sum[lane] += weight * masked;
                far_weight[lane] += weight;
            } else if gather_near {
                near_sum[lane] += weight * masked;
                near_weight[lane] += weight;
                near_cover[lane] += weight;
            }
        }
    }

    var far = vec4(0.0);
    var near = vec4(0.0);
    for (var seam = 0u; seam < seams; seam++) {
        var far_rgb = vec3(0.0);
        var far_alpha = 0.0;
        var near_rgb = vec3(0.0);
        var near_alpha = 0.0;
        for (var channel = 0u; channel < colors; channel++) {
            let lane = seam * colors + channel;
            if far_weight[lane] > 1.0e-9 {
                far_rgb += far_sum[lane] / far_weight[lane];
                far_alpha = 1.0;
            } else if fill_count[lane] > 0.0 {
                far_rgb += fill_sum[lane] / fill_count[lane];
                far_alpha = 1.0;
            }
            if near_weight[lane] > 1.0e-9 {
                let alpha = min(1.0, radius * radius / f32(taps + 1u) * near_cover[lane]);
                near_rgb += near_sum[lane] / near_weight[lane] * alpha;
                near_alpha += alpha;
            }
        }
        let share = select(1.0 - own.z, own.z, seam == 1u);
        far += share * vec4(far_rgb, far_alpha);
        near += share * vec4(near_rgb, near_alpha / f32(colors));
    }
    textureStore(far_out, id.xy, far);
    textureStore(near_out, id.xy, near);
}

// ── Ray map ──────────────────────────────────────────────────────────────────
//
// Sensor position -> scene position for one wavelength. The stack is applied
// in order to a point in focal-length units, x = (p_mm / f) with y down: the
// tangent coordinates of the rendered rectilinear image. Each element's
// region is tested at the sensor position and blends the element's map with
// the identity, so elements limited to a region fade in across its feather.
//
// The map says where on the *rendered* image the sensor looks. Beyond the
// rendered frame it clamps to the edge pixel; `required_overscan` says how
// much further than the frame the map reaches, for a camera to render.

struct MapCtx {
    // Focal-length units per image-height unit: sensor height / focal length.
    scale: f32,
    // The lens's focus in dioptres, for focus breathing.
    focus_dpt: f32,
    first: u32,
    count: u32,
}

fn make_ctx(r: Resolved, size: vec2<u32>) -> MapCtx {
    var c: MapCtx;
    let height_mm = r.sensor_width_mm * f32(size.y) / f32(size.x);
    c.scale = height_mm / r.focal_length_mm;
    c.focus_dpt = 1.0 / max(r.focus_distance_m, 1.0e-3);
    c.first = r.element_first;
    c.count = r.element_count;
    return c;
}

fn spectral_sample_count(quality: u32) -> u32 {
    return select(select(16u, 7u, quality == 1u), 3u, quality == 0u);
}

/// Index of refraction of the water of a port element.
fn port_index(n: f32) -> f32 {
    return select(WATER_INDEX, clamp(n, 1.05, 2.5), n > 0.0);
}

/// The rectilinear point of the ray at angle `theta` in the direction of `x`
/// (radius `r`).
fn along(x: vec2<f32>, r: f32, theta: f32) -> vec2<f32> {
    if r < 1.0e-5 { return x; }
    return x * (tan(clamp(theta, 0.0, MAX_ANGLE)) / r);
}

/// Angle at which a Kannala-Brandt lens records radius `r`: Newton on
/// theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9 = r.
fn kannala_brandt_angle(e: ElementRow, r: f32) -> f32 {
    var theta = r;
    for (var i = 0; i < 8; i++) {
        let t2 = theta * theta;
        let g = theta * (1.0 + t2 * (e.params[1] + t2 * (e.params[2] + t2 * (e.params[3] + t2 * e.params[4])))) - r;
        let d = 1.0 + t2 * (3.0 * e.params[1] + t2 * (5.0 * e.params[2] + t2 * (7.0 * e.params[3] + t2 * (9.0 * e.params[4]))));
        if abs(d) < 1.0e-4 { break; }
        theta = clamp(theta - g / d, 0.0, MAX_ANGLE);
    }
    return theta;
}

/// Panini: the sensor records X = (d + 1) sin(phi) / (d + cos(phi)) for a
/// ray of longitude phi, and the vertical scaled by (d + 1) / (d + cos(phi))
/// (mixed toward 1 by 1 - s).
fn panini_inverse(e: ElementRow, x: vec2<f32>) -> vec2<f32> {
    let d = clamp(e.params[1], 0.0, 8.0);
    let s = clamp(e.params[2], 0.0, 1.0);
    let k = x.x / (d + 1.0);
    let phi = clamp(asin(clamp(k * d / sqrt(1.0 + k * k), -1.0, 1.0)) + atan(k), -MAX_ANGLE, MAX_ANGLE);
    let c = cos(phi);
    let vertical = x.y / mix(1.0, (d + 1.0) / (d + c), s);
    return vec2(tan(phi), vertical / c);
}

/// A sensor point of the chosen projection -> the rectilinear image point.
fn apply_projection(e: ElementRow, x: vec2<f32>) -> vec2<f32> {
    let r = length(x);
    var theta = r;
    switch u32(bounded(e.params[0], 0.0, 15.0, 0.0) + 0.5) {
        case 1u: {
            theta = r;
        }
        case 2u: {
            theta = 2.0 * asin(min(r * 0.5, 1.0));
        }
        case 3u: {
            theta = asin(min(r, 1.0));
        }
        case 4u: {
            theta = 2.0 * atan(r * 0.5);
        }
        case 5u: {
            theta = kannala_brandt_angle(e, r);
        }
        case 6u: {
            return panini_inverse(e, x);
        }
        case 7u: {
            // Horizontal equidistant, vertical rectilinear.
            let phi = clamp(x.x, -MAX_ANGLE, MAX_ANGLE);
            return vec2(tan(phi), x.y / cos(phi));
        }
        default: {
            return x;
        }
    }
    return along(x, r, theta);
}

fn brown_conrady_forward(e: ElementRow, x: vec2<f32>) -> vec2<f32> {
    let r2 = dot(x, x);
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let numerator = 1.0 + e.params[0] * r2 + e.params[1] * r4 + e.params[2] * r6;
    var denominator = 1.0 + e.params[3] * r2 + e.params[4] * r4 + e.params[5] * r6;
    if abs(denominator) < 1.0e-3 { denominator = 1.0e-3; }
    let radial = numerator / denominator;
    let p1 = e.params[6];
    let p2 = e.params[7];
    return vec2(
        x.x * radial + 2.0 * p1 * x.x * x.y + p2 * (r2 + 2.0 * x.x * x.x),
        x.y * radial + p1 * (r2 + 2.0 * x.y * x.y) + 2.0 * p2 * x.x * x.y,
    );
}

fn anamorphic_forward(e: ElementRow, x: vec2<f32>) -> vec2<f32> {
    let r2 = dot(x, x);
    return vec2(
        x.x * (1.0 + e.params[0] * r2 + e.params[1] * r2 * r2),
        x.y * (1.0 + e.params[2] * r2 + e.params[3] * r2 * r2),
    );
}

fn distort(e: ElementRow, x: vec2<f32>) -> vec2<f32> {
    if e.kind == ELEMENT_BROWN_CONRADY { return brown_conrady_forward(e, x); }
    return anamorphic_forward(e, x);
}

/// The ideal point whose distorted image is `goal`: Newton's method on
/// `distort(x) = goal`, as the sensor-to-scene map needs the inverse of
/// the forward model.
fn undistort(e: ElementRow, goal: vec2<f32>) -> vec2<f32> {
    var x = goal;
    let h = 1.0e-3;
    for (var i = 0; i < 8; i++) {
        let f = distort(e, x) - goal;
        let jx = (distort(e, x + vec2(h, 0.0)) - distort(e, x - vec2(h, 0.0))) / (2.0 * h);
        let jy = (distort(e, x + vec2(0.0, h)) - distort(e, x - vec2(0.0, h))) / (2.0 * h);
        let det = jx.x * jy.y - jy.x * jx.y;
        if abs(det) < 1.0e-5 { break; }
        let step = vec2(jy.y * f.x - jy.x * f.y, jx.x * f.y - jx.y * f.x) / det;
        x -= clamp(step, vec2(-0.5), vec2(0.5));
    }
    return x;
}

/// A flat port: the camera records the ray at air angle theta, which left
/// the water at theta_w with sin(theta) = n sin(theta_w). Magnifies by about
/// n and pulls the edges in (pincushion); blue, with the higher index, more.
fn flat_port_map(e: ElementRow, x: vec2<f32>, t: f32) -> vec2<f32> {
    let n = max(port_index(e.params[0]) - e.params[1] * t, 1.01);
    let r = length(x);
    let sin_air = r / sqrt(1.0 + r * r);
    return along(x, r, asin(clamp(sin_air / n, 0.0, 1.0)));
}

/// A dome port of radius R with the pupil e from its centre. The ray at
/// angle theta meets the glass at incidence i, sin(i) = (e / R) sin(theta),
/// and refracts to sin(i_w) = sin(i) / n: it leaves along theta - i + i_w.
/// Centred (e = 0) the rays cross undeviated.
fn dome_port_map(e: ElementRow, x: vec2<f32>, t: f32) -> vec2<f32> {
    let n = max(port_index(e.params[2]) - e.params[3] * t, 1.01);
    let ratio = clamp(e.params[1] / max(e.params[0], 1.0e-3), -0.95, 0.95);
    let r = length(x);
    let theta = atan(r);
    let sin_incidence = ratio * sin(theta);
    return along(x, r, theta - asin(sin_incidence) + asin(sin_incidence / n));
}

fn map_element(e: ElementRow, x: vec2<f32>, t: f32, ctx: MapCtx) -> vec2<f32> {
    switch e.kind {
        case ELEMENT_PROJECTION: {
            return apply_projection(e, x);
        }
        case ELEMENT_BROWN_CONRADY, ELEMENT_ANAMORPHIC_DISTORTION: {
            return undistort(e, x);
        }
        case ELEMENT_DIVISION: {
            let r2 = dot(x, x);
            return x / max(1.0 + e.params[0] * r2 + e.params[1] * r2 * r2, 0.1);
        }
        case ELEMENT_ANAMORPHIC_SQUEEZE: {
            let s = select(1.0, clamp(e.params[0], 0.1, 8.0), e.params[0] > 0.0);
            return vec2(x.x * s, x.y);
        }
        case ELEMENT_SHIFT: {
            return x + vec2(e.params[0], e.params[1]) * ctx.scale;
        }
        case ELEMENT_KEYSTONE: {
            return x / max(1.0 + e.params[0] * x.x + e.params[1] * x.y, 0.1);
        }
        case ELEMENT_FOCUS_BREATHING: {
            return x / max(1.0 + e.params[0] * (ctx.focus_dpt - e.params[1]), 0.1);
        }
        case ELEMENT_LATERAL_CA: {
            return x / max(1.0 + t * (e.params[0] + t * e.params[1]), 0.1);
        }
        case ELEMENT_FLAT_PORT: {
            return flat_port_map(e, x, t);
        }
        case ELEMENT_DOME_PORT: {
            return dome_port_map(e, x, t);
        }
        default: {
            return x;
        }
    }
}

// ── Prism ────────────────────────────────────────────────────────────────────

fn prism_count(e: ElementRow) -> u32 {
    return clamp(u32(bounded(e.params[0], 2.0, 8.0, 2.0) + 0.5), 2u, 8u);
}

/// Where copy `copy` of a prism samples the (already mapped) point `x`.
fn prism_copy(e: ElementRow, x: vec2<f32>, copy: u32, ctx: MapCtx) -> vec2<f32> {
    let n = prism_count(e);
    let magnitude = bounded(e.params[1], -4.0, 4.0, 0.0) * ctx.scale;
    let angle = bounded(e.params[2], -100.0, 100.0, 0.0);
    var offset: vec2<f32>;
    if e.params[3] > 0.5 {
        let a = angle + 2.0 * PI * f32(copy) / f32(n);
        offset = vec2(cos(a), sin(a)) * magnitude;
    } else {
        offset = vec2(cos(angle), sin(angle)) * ((f32(copy) - 0.5 * f32(n - 1u)) * magnitude);
    }
    let spin = bounded(e.params[4], -100.0, 100.0, 0.0) * f32(copy);
    let c = cos(spin);
    let s = sin(spin);
    return vec2(c * x.x - s * x.y, s * x.x + c * x.y) + offset;
}

/// Weight of a copy: `ratio^copy`, normalised over the copies.
fn prism_weight(e: ElementRow, copy: u32) -> f32 {
    let n = prism_count(e);
    let ratio = select(1.0, bounded(e.params[5], 0.05, 20.0, 1.0), e.params[5] > 0.0);
    var total = 0.0;
    var mine = 1.0;
    var w = 1.0;
    for (var i = 0u; i < n; i++) {
        if i == copy { mine = w; }
        total += w;
        w *= ratio;
    }
    return mine / total;
}

struct PrismSel {
    // Element index within the lens's stack, NO_ROW with none.
    index: u32,
    weight: f32,
}

/// The first prism element covering the sensor position `q`.
fn find_prism(q: vec2<f32>, ctx: MapCtx) -> PrismSel {
    var sel: PrismSel;
    sel.index = NO_ROW;
    sel.weight = 0.0;
    for (var i = 0u; i < ctx.count; i++) {
        let e = elements[ctx.first + i];
        if e.kind == ELEMENT_PRISM {
            let w = region_weight(e, q);
            if w > 0.0 {
                sel.index = i;
                sel.weight = w;
                return sel;
            }
        }
    }
    return sel;
}

/// The scene position (image-height units about the frame centre) the sensor
/// position `q` looks at, for the wavelength position `t` (-1 blue .. 1
/// red). `prism` and `copy` select one copy of the prism `find_prism`
/// returned; `NO_COPY` skips prisms.
fn ray_map(q: vec2<f32>, t: f32, prism: u32, copy: u32, ctx: MapCtx) -> vec2<f32> {
    var x = q * ctx.scale;
    for (var i = 0u; i < ctx.count; i++) {
        let e = elements[ctx.first + i];
        if e.kind < ELEMENT_PROJECTION || e.kind > ELEMENT_DOME_PORT { continue; }
        if e.kind == ELEMENT_PRISM {
            if i == prism && copy != NO_COPY {
                x = prism_copy(e, x, copy, ctx);
            }
            continue;
        }
        let w = region_weight(e, q);
        if w <= 0.0 { continue; }
        x = mix(x, map_element(e, x, t, ctx), w);
    }
    let scene = x / ctx.scale;
    return select(q, scene, all(abs(scene) < vec2(1.0e4)));
}

/// The ray map's magnification at the scene position `s`: sqrt(det J) of the
/// sensor-to-scene map at the sensor point that looks at `s` (found by
/// Newton), i.e. scene length per sensor length. Isotropic: it scales a
/// circle, not an ellipse.
fn defocus_scale(s: vec2<f32>, ctx: MapCtx) -> f32 {
    var q = s;
    let h = 2.0e-3;
    var det = 1.0;
    for (var i = 0u; i < 4u; i++) {
        let m = ray_map(q, 0.0, NO_ROW, NO_COPY, ctx);
        let jx = (ray_map(q + vec2(h, 0.0), 0.0, NO_ROW, NO_COPY, ctx) - ray_map(q - vec2(h, 0.0), 0.0, NO_ROW, NO_COPY, ctx)) / (2.0 * h);
        let jy = (ray_map(q + vec2(0.0, h), 0.0, NO_ROW, NO_COPY, ctx) - ray_map(q - vec2(0.0, h), 0.0, NO_ROW, NO_COPY, ctx)) / (2.0 * h);
        det = jx.x * jy.y - jy.x * jx.y;
        if abs(det) < 1.0e-4 { break; }
        let f = m - s;
        q -= clamp(vec2(jy.y * f.x - jy.x * f.y, jx.x * f.y - jx.y * f.x) / det, vec2(-0.25), vec2(0.25));
    }
    let scale = sqrt(abs(det));
    return select(1.0, clamp(scale, 0.25, 4.0), finite(scale));
}

/// How far beyond the frame the map reaches at boundary sample `lid` (0..64,
/// sixteen per edge, corners included): the largest scene extent over the
/// frame's half extent, over the spectrum's ends and every prism copy.
fn boundary_reach(lid: u32, ctx: MapCtx, size: vec2<u32>) -> f32 {
    let aspect = f32(size.x) / f32(size.y);
    let k = f32(lid % 16u) / 16.0;
    var uv = vec2(0.0, 1.0 - k);
    switch lid / 16u {
        case 0u: {
            uv = vec2(k, 0.0);
        }
        case 1u: {
            uv = vec2(1.0, k);
        }
        case 2u: {
            uv = vec2(1.0 - k, 1.0);
        }
        default: {}
    }
    let q = (uv - 0.5) * vec2(aspect, 1.0);
    let prism = find_prism(q, ctx);
    var copies = 1u;
    if prism.index != NO_ROW { copies = prism_count(elements[ctx.first + prism.index]); }
    var reach = 1.0;
    for (var c = 0u; c < copies; c++) {
        let copy = select(NO_COPY, c, prism.index != NO_ROW);
        for (var j = 0u; j < 3u; j++) {
            let s = ray_map(q, f32(j) - 1.0, prism.index, copy, ctx);
            reach = max(reach, max(abs(s.x) / (0.5 * aspect), abs(s.y) / 0.5));
        }
    }
    return select(OVERSCAN_CAP, reach, finite(reach));
}

// ── Transmission ─────────────────────────────────────────────────────────────

/// Area of the unit circle inside a circle of radius `b` whose centre is `d`
/// away, as a fraction of the unit circle's area.
fn circle_overlap(d: f32, b: f32) -> f32 {
    if d >= 1.0 + b { return 0.0; }
    if d <= abs(1.0 - b) { return min(1.0, b * b); }
    let a1 = acos(clamp((d * d + 1.0 - b * b) / (2.0 * d), -1.0, 1.0));
    let a2 = acos(clamp((d * d + b * b - 1.0) / (2.0 * d * b), -1.0, 1.0));
    let k = sqrt(max((-d + 1.0 + b) * (d + 1.0 - b) * (d - 1.0 + b) * (d + 1.0 + b), 0.0));
    return (a1 + b * b * a2 - 0.5 * k) / PI;
}

/// Light that reaches the sensor position `q` (image-height units): natural
/// cos^4 from the ray's angle after the map, the barrel's optical
/// vignetting, hood, filters and the T-stop loss, per colour channel.
fn transmission(q: vec2<f32>, ctx: MapCtx, size: vec2<u32>) -> vec3<f32> {
    let aspect = f32(size.x) / f32(size.y);
    let ray = ray_map(q, 0.0, NO_ROW, NO_COPY, ctx) * ctx.scale;
    let cos_squared = 1.0 / (1.0 + dot(ray, ray));
    // 1 at the corner, as in the pupil's field position.
    let field_r = length(q) / (0.5 * sqrt(1.0 + aspect * aspect));
    var t = vec3(1.0);
    for (var i = 0u; i < ctx.count; i++) {
        let e = elements[ctx.first + i];
        if e.kind < ELEMENT_NATURAL_VIGNETTE || e.kind > ELEMENT_T_STOP_LOSS { continue; }
        let w = region_weight(e, q);
        if w <= 0.0 { continue; }
        switch e.kind {
            case ELEMENT_NATURAL_VIGNETTE: {
                let natural = mix(1.0, cos_squared * cos_squared, bounded(e.params[0], 0.0, 1.0, 0.0));
                t *= mix(1.0, natural, w);
            }
            case ELEMENT_OPTICAL_VIGNETTE: {
                var barrel = select(1.0, bounded(e.params[1], 0.5, 4.0, 1.0), e.params[1] > 0.0);
                // The barrel is a fixed size: against a smaller iris (a
                // higher f-number) it clips less.
                if e.params[2] > 0.0 { barrel *= resolved.f_number / bounded(e.params[2], 0.7, 64.0, resolved.f_number); }
                let d = min(bounded(e.params[0], 0.0, 1.5, 0.0) * field_r, 1.8);
                t *= mix(1.0, circle_overlap(d, barrel), w);
            }
            case ELEMENT_MECHANICAL_VIGNETTE: {
                let feather = max(bounded(e.params[3], 0.0, 4.0, 0.02), 1.0e-3);
                var sd = length(q) - e.params[1];
                if e.params[0] > 0.5 {
                    let c = cos(e.params[4]);
                    let s = sin(e.params[4]);
                    let local = vec2(c * q.x + s * q.y, -s * q.x + c * q.y);
                    let dd = abs(local) - vec2(e.params[1], e.params[2]);
                    sd = length(max(dd, vec2(0.0))) + min(max(dd.x, dd.y), 0.0);
                }
                let u = clamp(0.5 - sd / feather, 0.0, 1.0);
                t *= mix(1.0, u * u * (3.0 - 2.0 * u), w);
            }
            case ELEMENT_ND_FILTER: {
                t *= exp2(-bounded(e.params[0], 0.0, 32.0, 0.0) * w);
            }
            case ELEMENT_GRADUATED_ND: {
                let soft = max(bounded(e.params[3], 0.0, 8.0, 0.0), 0.002);
                let n = vec2(cos(e.params[1]), sin(e.params[1]));
                let g = smoothstep(-0.5 * soft, 0.5 * soft, dot(q, n) - e.params[2]);
                let tint = clamp(vec3(e.params[4], e.params[5], e.params[6]), vec3(0.0), vec3(1.0));
                let full = tint * exp2(-bounded(e.params[0], 0.0, 32.0, 0.0));
                t *= mix(vec3(1.0), full, g * w);
            }
            case ELEMENT_COLOR_FILTER: {
                let tinted = clamp(vec3(e.params[0], e.params[1], e.params[2]), vec3(0.0), vec3(1.0));
                t *= mix(vec3(1.0), tinted, bounded(e.params[3], 0.0, 1.0, 0.0) * w);
            }
            case ELEMENT_T_STOP_LOSS: {
                let ratio = resolved.f_number / max(resolved.t_stop, 0.1);
                t *= mix(1.0, ratio * ratio, bounded(e.params[0], 0.0, 1.0, 0.0) * w);
            }
            default: {}
        }
    }
    return t;
}

// ── Spectral reconstruction ──────────────────────────────────────────────────

/// Sensor response of the red, green and blue channels at `lambda` nm:
/// broad positive lobes, so every sample reaches the channels it overlaps.
fn sensor_weights(lambda: f32) -> vec3<f32> {
    let r = (lambda - 610.0) / 40.0;
    let g = (lambda - 545.0) / 35.0;
    let b = (lambda - 460.0) / 30.0;
    return vec3(exp(-0.5 * r * r), exp(-0.5 * g * g), exp(-0.5 * b * b));
}

// ── Composite ────────────────────────────────────────────────────────────────

/// The sharp, far and near layers of one position: in-focus pixels keep
/// their own full-resolution value; blurred ones take the far layer, which
/// is the backdrop behind any foreground blur.
fn shade(c: vec3<f32>, k: vec4<f32>, far: vec4<f32>, near: vec4<f32>) -> vec3<f32> {
    let blur = mix(smoothstep(0.5, 1.5, abs(k.x)), smoothstep(0.5, 1.5, abs(k.y)), k.z);
    let backdrop = select(c, far.rgb / far.a, far.a > 0.05);
    var rgb = mix(c, backdrop, blur);
    rgb = rgb * (1.0 - near.a) + near.rgb;
    return rgb * k.w;
}

/// Bilinear scene colour. A position within a thousandth of a texel of a
/// texel's centre takes exactly that texel, so an identity map reproduces
/// the image bit for bit.
fn scene_bilinear(uv: vec2<f32>) -> vec3<f32> {
    let dims = textureDimensions(scene_color);
    let last = vec2<i32>(dims) - 1;
    let raw = uv * vec2<f32>(dims) - 0.5;
    let nearest = round(raw);
    let pos = select(raw, nearest, abs(raw - nearest) < vec2(1.0e-3));
    let base = floor(pos);
    let f = pos - base;
    let p = vec2<i32>(base);
    let a = radiance(textureLoad(scene_color, clamp(p, vec2(0), last), 0).rgb);
    let b = radiance(textureLoad(scene_color, clamp(p + vec2(1, 0), vec2(0), last), 0).rgb);
    let c = radiance(textureLoad(scene_color, clamp(p + vec2(0, 1), vec2(0), last), 0).rgb);
    let d = radiance(textureLoad(scene_color, clamp(p + vec2(1, 1), vec2(0), last), 0).rgb);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// The defocused image at a scene position `uv` (unit square): clamped to the
/// edge outside the rendered frame.
fn defocused_at(uv: vec2<f32>) -> vec3<f32> {
    let p = clamp(uv, vec2(0.0), vec2(1.0));
    let dims = textureDimensions(coc_in);
    let texel = min(vec2<u32>(p * vec2<f32>(dims)), dims - 1u);
    let k = textureLoad(coc_in, texel, 0);
    let far = textureSampleLevel(far_in, linear_samp, p, 0.0);
    let near = textureSampleLevel(near_in, linear_samp, p, 0.0);
    return shade(scene_bilinear(p), k, far, near);
}

/// The wavelength-weighted image seen from sensor position `q` through one
/// prism copy (or none).
fn spectral_color(
    q: vec2<f32>,
    prism: u32,
    copy: u32,
    ctx: MapCtx,
    n: u32,
    ts: ptr<function, array<f32, 16>>,
    ws: ptr<function, array<vec3<f32>, 16>>,
    aspect: f32,
) -> vec3<f32> {
    var acc = vec3(0.0);
    for (var j = 0u; j < n; j++) {
        let s = ray_map(q, (*ts)[j], prism, copy, ctx);
        acc += (*ws)[j] * defocused_at(s / vec2(aspect, 1.0) + 0.5);
    }
    return acc;
}

/// The image through the ray map at sensor position `q`. Each of the `n`
/// wavelength samples has its own map; the channels take their spectral
/// weights of it, which sum to 1 over the samples, so equal maps give the
/// image back. A prism averages its copies by weight, fading in across its
/// region.
fn mapped_color(q: vec2<f32>, ctx: MapCtx, size: vec2<u32>) -> vec3<f32> {
    let aspect = f32(size.x) / f32(size.y);
    let n = select(1u, clamp(resolved.spectral_samples, 1u, MAX_SPECTRAL), (resolved.map_flags & MAP_SPECTRAL) != 0u);
    var ts: array<f32, 16>;
    var ws: array<vec3<f32>, 16>;
    ts[0] = 0.0;
    ws[0] = vec3(1.0);
    if n > 1u {
        var total = vec3(0.0);
        for (var j = 0u; j < n; j++) {
            let lambda = 400.0 + (f32(j) + 0.5) * 300.0 / f32(n);
            ts[j] = (lambda - 550.0) / 150.0;
            ws[j] = sensor_weights(lambda);
            total += ws[j];
        }
        for (var j = 0u; j < n; j++) {
            ws[j] = ws[j] / total;
        }
    }
    let prism = find_prism(q, ctx);
    if prism.index == NO_ROW {
        return spectral_color(q, NO_ROW, NO_COPY, ctx, n, &ts, &ws, aspect);
    }
    let e = elements[ctx.first + prism.index];
    var copies = vec3(0.0);
    for (var c = 0u; c < prism_count(e); c++) {
        copies += prism_weight(e, c) * spectral_color(q, prism.index, c, ctx, n, &ts, &ws, aspect);
    }
    if prism.weight >= 1.0 { return copies; }
    return mix(spectral_color(q, NO_ROW, NO_COPY, ctx, n, &ts, &ws, aspect), copies, prism.weight);
}

@compute @workgroup_size(8, 8)
fn cs_composite(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(color_out);
    if any(id.xy >= size) { return; }
    let c = textureLoad(scene_color, id.xy, 0);
    // No enabled lens for this view: the image passes through untouched.
    if resolved.enabled == 0u {
        textureStore(color_out, id.xy, c);
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let q = (uv - 0.5) * vec2(f32(size.x) / f32(size.y), 1.0);
    let ctx = make_ctx(resolved, size);
    var rgb: vec3<f32>;
    if (resolved.map_flags & MAP_REMAP) == 0u {
        // Texel for texel: no map, no spectrum.
        let k = textureLoad(coc_in, id.xy, 0);
        let far = textureSampleLevel(far_in, linear_samp, uv, 0.0);
        let near = textureSampleLevel(near_in, linear_samp, uv, 0.0);
        rgb = shade(c.rgb, k, far, near);
    } else {
        rgb = mapped_color(q, ctx, size);
    }
    if (resolved.map_flags & MAP_TRANSMISSION) != 0u {
        rgb *= transmission(q, ctx, size);
    }
    textureStore(color_out, id.xy, vec4(rgb, c.a));
}
