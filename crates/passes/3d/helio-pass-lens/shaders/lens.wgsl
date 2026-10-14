//!use helio_prelude

// ── Physically based screen-space lens ───────────────────────────────────────
//
// A view's lens is an ordered stack of optical elements reduced per pixel to
// four quantities: a ray map (sensor uv -> scene uv, per wavelength), a focus
// field s(uv), a pupil function and a transmission. This phase resolves the
// focus field and the thin-lens defocus it implies; the ray map is the
// identity, the pupil a circle (`pupil_coverage`) and the transmission 1
// except for diopter seam darkening. Later phases add element kinds to
// `focus_field` and replace those functions.
//
// Stages, all GPU-sized from `cs_resolve` so a disabled lens costs one
// workgroup and a copy:
//
//   cs_resolve    match the view's lens row, autofocus, size the dispatches
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
    if lid != 0u { return; }

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
    if found {
        out.f_number = bounded(row.f_number, 0.7, 32.0, 2.8);
        out.focal_length_mm = bounded(row.focal_length_mm, 8.0, 600.0, 50.0);
        out.sensor_width_mm = bounded(row.sensor_width_mm, 4.0, 70.0, 36.0);
        out.aperture_blades = select(clamp(row.aperture_blades, 3u, 16u), 0u, row.aperture_blades == 0u);
        out.aperture_rotation = bounded(row.aperture_rotation, -100.0, 100.0, 0.0);
        out.enabled = 1u;
        out.quality = min(row.quality, 1u);
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
    resolved = out;

    let size = textureDimensions(scene_color);
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
    let coc_a = limited(circle_of_confusion(focus_distance(field.inverse_a), s2, f32(size.x)), field.extra);
    var coc_b = coc_a;
    if field.omega > 0.0 {
        coc_b = limited(circle_of_confusion(focus_distance(field.inverse_b), s2, f32(size.x)), field.extra);
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

// ── Gather ───────────────────────────────────────────────────────────────────

/// Coverage of a source's blur disc at an offset from its centre, soft by
/// half a pixel each side. The pupil is a circle in this phase; iris shapes
/// and optical vignetting replace this function.
fn pupil_coverage(offset: vec2<f32>, radius: f32) -> f32 {
    return clamp(radius - length(offset) + 0.5, 0.0, 1.0);
}

fn interleaved_noise(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

/// Scatter-as-gather. Every source texel spreads its radiance over a disc of
/// its own circle of confusion; each destination collects the sources whose
/// disc reaches it, weighted by 1 / disc area so the energy a source spreads
/// is its own. Far-field samples are averaged (a mean, so a uniform blur
/// keeps the image's energy); near-field samples form a layer whose alpha is
/// the fraction of the destination the blurred foreground covers, composited
/// over the far layer. Seam pixels gather twice, with the circles of each
/// configuration, and blend by the seam weight.
@compute @workgroup_size(8, 8)
fn cs_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = textureDimensions(half_color);
    if any(id.xy >= half_size) { return; }
    let tiles = ceil_div(half_size * 2u, TILE);
    let tile = tile_dilated[min(id.y * 2u / TILE, tiles.y - 1u) * tiles.x + min(id.x * 2u / TILE, tiles.x - 1u)];
    let center = textureLoad(half_color, id.xy, 0).rgb;
    let radius = tile.x * 0.5;
    if radius < 0.25 {
        textureStore(far_out, id.xy, vec4(center, 1.0));
        textureStore(near_out, id.xy, vec4(0.0));
        return;
    }
    let own = textureLoad(half_coc, id.xy, 0);
    let channels = select(1u, 2u, own.z > 0.0);
    let gather_near = tile.z < -NEAR_COC;
    let max_taps = select(24u, 56u, resolved.quality != 0u);
    let taps = clamp(u32(ceil(f32(max_taps) * min(radius / 8.0, 1.0))), 8u, max_taps);
    let rotation = 2.0 * PI * interleaved_noise(vec2<f32>(id.xy));
    let last = vec2<i32>(half_size) - 1;

    var far_sum = array<vec3<f32>, 2>(vec3(0.0), vec3(0.0));
    var far_weight = array<f32, 2>(0.0, 0.0);
    var near_sum = array<vec3<f32>, 2>(vec3(0.0), vec3(0.0));
    var near_weight = array<f32, 2>(0.0, 0.0);
    var near_cover = array<f32, 2>(0.0, 0.0);
    // Unweighted far-class mean: stands in where no source's disc reaches,
    // i.e. behind a near-field blur, so its inner edge fades into the
    // surroundings instead of into the foreground's own colour.
    var fill_sum = array<vec3<f32>, 2>(vec3(0.0), vec3(0.0));
    var fill_count = array<f32, 2>(0.0, 0.0);
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
        for (var ch = 0u; ch < channels; ch++) {
            let coc = select(circles.x, circles.y, ch == 1u);
            let source_radius = 0.5 * abs(coc);
            let far_class = coc > -NEAR_COC;
            if far_class {
                fill_sum[ch] += color;
                fill_count[ch] += 1.0;
            }
            let cover = pupil_coverage(delta, source_radius);
            if cover <= 0.0 { continue; }
            let area = PI * max(source_radius, 0.5) * max(source_radius, 0.5);
            let weight = cover / area;
            if far_class {
                far_sum[ch] += weight * color;
                far_weight[ch] += weight;
            } else if gather_near {
                near_sum[ch] += weight * color;
                near_weight[ch] += weight;
                near_cover[ch] += cover / (max(source_radius, 0.5) * max(source_radius, 0.5));
            }
        }
    }

    var far = vec4(0.0);
    var near = vec4(0.0);
    for (var ch = 0u; ch < channels; ch++) {
        var far_layer = vec4(0.0);
        if far_weight[ch] > 1.0e-9 {
            far_layer = vec4(far_sum[ch] / far_weight[ch], 1.0);
        } else if fill_count[ch] > 0.0 {
            far_layer = vec4(fill_sum[ch] / fill_count[ch], 1.0);
        }
        var near_layer = vec4(0.0);
        if near_weight[ch] > 1.0e-9 {
            let alpha = min(1.0, radius * radius / f32(taps + 1u) * near_cover[ch]);
            near_layer = vec4(near_sum[ch] / near_weight[ch] * alpha, alpha);
        }
        let share = select(1.0 - own.z, own.z, ch == 1u);
        far += share * far_layer;
        near += share * near_layer;
    }
    textureStore(far_out, id.xy, far);
    textureStore(near_out, id.xy, near);
}

// ── Composite ────────────────────────────────────────────────────────────────

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
    let k = textureLoad(coc_in, id.xy, 0);
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let far = textureSampleLevel(far_in, linear_samp, uv, 0.0);
    let near = textureSampleLevel(near_in, linear_samp, uv, 0.0);
    // In-focus pixels keep their own full-resolution value; blurred ones
    // take the far layer, which is the backdrop behind any foreground blur.
    let blur = mix(smoothstep(0.5, 1.5, abs(k.x)), smoothstep(0.5, 1.5, abs(k.y)), k.z);
    let backdrop = select(c.rgb, far.rgb / far.a, far.a > 0.05);
    var rgb = mix(c.rgb, backdrop, blur);
    rgb = rgb * (1.0 - near.a) + near.rgb;
    textureStore(color_out, id.xy, vec4(rgb * k.w, c.a));
}
