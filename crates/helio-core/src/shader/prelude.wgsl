// Helio shader prelude — the canonical camera layout and the depth/G-buffer
// conventions that go with it.
//
// Prepended to any shader that opts in with the `helio_prelude` marker (the
// `MARKER` of helio_core::shader). The marker is never written out here: text
// that holds it would opt in again when a composed source is resolved.
// See helio_core::shader.
//
// This exists because every pass used to re-derive this math from scratch, and
// they drifted: the NDC y-flip was present in some shaders and absent in others,
// one pass used the OpenGL [-1,1] depth convention against a wgpu [0,1] buffer,
// and two passes "decoded" G-buffer normals that were never encoded. Each bug
// was invisible in isolation and only showed up as a wrong picture. Depth math
// is not per-pass policy, so it does not live in per-pass code.

// ── Camera ──────────────────────────────────────────────────────────────────
// Field-for-field mirror of helio_core `GpuCameraUniforms` (368 B).
// Shaders declare their OWN binding (the group/binding varies per pass) but must
// use this struct rather than redeclaring it:
//
//     (opt in with the helio_prelude marker)
//     @group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
//
struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    /// Camera world position (xyz) + near plane (w).
    position_near:  vec4<f32>,
    /// Camera forward direction (xyz) + effect range (w) — not a clip plane,
    /// the projection has none (see Depth below).
    forward_far:    vec4<f32>,
    /// TAA jitter (xy) + frame index (z) + padding (w).
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

// ── Screen space ────────────────────────────────────────────────────────────

/// UV (y down, origin top-left) to NDC (y up).
///
/// The flip is the whole point. wgpu NDC has y+ up while framebuffer/UV space has
/// y+ down, so `uv * 2 - 1` is right for x and wrong for y. Getting this wrong is
/// deceptively survivable: if a shader also inverts it without the flip, the
/// round-trip is self-consistent and only the *world* position it derives is
/// mirrored — which reads as a plausible-looking but wrong image.
fn helio_uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

/// NDC (y up) back to UV (y down). Inverse of `helio_uv_to_ndc`.
fn helio_ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// ── Depth ───────────────────────────────────────────────────────────────────
// Reversed Z with an infinite far plane (helio_core::depth): the near plane is
// depth 1, infinity is depth 0, the buffer clears to 0 and nearer surfaces win
// with `greater`. NDC z is wgpu's [0,1], never OpenGL's [-1,1] — depth-buffer
// values go into `view_proj_inv` as-is; `depth * 2 - 1` silently skews every
// reconstructed position.
//
// Compare depths through these helpers rather than with a bare `<` or `max`:
// which of two depths is nearer is the convention's business, not the pass's.
//
// `forward_far.w` is not a clip plane. It is the range of distance-limited
// effects (froxels, fog); geometry beyond it still rasterizes.

/// Smallest depth reconstruction treats as a surface. Sky (0) would unproject
/// to w = 0; clamping to this puts it at a huge but finite distance along the
/// same view ray instead (near / 1e-12 — tens of millions of km).
const HELIO_DEPTH_MIN: f32 = 1e-12;

/// Whether a depth-buffer sample is sky: nothing was drawn there.
fn helio_is_sky(depth: f32) -> bool {
    return depth <= 0.0;
}

/// Whether depth `a` is nearer the camera than depth `b`.
fn helio_depth_is_nearer(a: f32, b: f32) -> bool {
    return a > b;
}

/// The nearer of two depths.
fn helio_depth_nearest(a: f32, b: f32) -> f32 {
    return max(a, b);
}

/// The farther of two depths.
fn helio_depth_farthest(a: f32, b: f32) -> f32 {
    return min(a, b);
}

/// Relative margin of a Hi-Z occlusion test (see `helio_hiz_occludes`).
const HELIO_HIZ_MARGIN: f32 = 1.0 / 65536.0;

/// Whether an occluder whose farthest depth over a footprint is `farthest`
/// hides a bound whose nearest point has depth `nearest`.
///
/// The margin is relative: reversed depth is `near / distance`, so a relative
/// margin is the same fraction of distance at every range, where a fixed one
/// would be kilometres wide at the horizon and nothing up close.
fn helio_hiz_occludes(farthest: f32, nearest: f32) -> bool {
    return helio_depth_is_nearer(farthest, nearest * (1.0 + HELIO_HIZ_MARGIN));
}

/// World position from a depth-buffer sample.
/// `inv_view_proj` is `cameras[0].view_proj_inv`; `depth` is the raw buffer value.
/// Sky comes back as a far point along its view ray, never as inf/NaN.
fn helio_world_from_depth(inv_view_proj: mat4x4<f32>, uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let world = inv_view_proj * vec4<f32>(helio_uv_to_ndc(uv), max(depth, HELIO_DEPTH_MIN), 1.0);
    return world.xyz / world.w;
}

/// Raw depth to positive view-space depth (distance along the view axis).
///
/// `proj` is `cameras[0].proj`. Inverts only the projection's own z and w
/// rows, so it needs no near/far pair kept in sync with it and holds for
/// perspective (finite or infinite) and orthographic projections, reversed or
/// standard. Sky reads as a huge but finite distance.
fn helio_view_depth(proj: mat4x4<f32>, depth: f32) -> f32 {
    let d = max(depth, HELIO_DEPTH_MIN);
    return (proj[3][2] - d * proj[3][3]) / (proj[2][2] - d * proj[2][3]);
}

// ── G-buffer ────────────────────────────────────────────────────────────────

/// Decode a G-buffer normal.
///
/// There is nothing to decode: the G-buffer normal target is Rgba16Float and
/// gbuffer.wgsl writes `surface.normal` into it directly, in world space, already
/// signed. A `* 2 - 1` here is undoing an encoding that never happened — it turns
/// a floor's (0,1,0) into a normalized (-1,1,-1). This function exists so that
/// intent is stated in one place instead of guessed at per pass.
fn helio_gbuffer_normal(raw: vec3<f32>) -> vec3<f32> {
    return normalize(raw);
}

// Surface tags (`helio_frame_api::gbuffer::SURFACE`). The surface channel
// (Rg16Float) holds a tag (x < 0) saying what the lighting must know about a
// surface, or the payload one feature requested for the graph (x >= 0, which
// that feature reads). Drawers write a tag by these names; readers test it
// through the predicates below. Nothing writes or compares a raw tag value.

/// An ordinary surface with no payload.
const GBUFFER_TAG_PLAIN: vec2<f32> = vec2<f32>(-1.0, -1.0);
/// Dense geometry that carries its own occlusion (a meadow's blades): its
/// many thin layers fill a screen-space AO kernel and would turn whole fields
/// black, so the lighting applies none.
const GBUFFER_TAG_OWN_OCCLUSION: vec2<f32> = vec2<f32>(-1.0, -1.5);
/// A surface a drawer traced rather than rasterized (a voxel terrain), with
/// its occlusion its own: no screen-space AO (a coarse lattice's edges
/// overwhelm the kernel at distance). Debug views of the visible surface's
/// source show it as traced.
const GBUFFER_TAG_TRACED: vec2<f32> = vec2<f32>(-1.0, -2.0);

/// The tag of content drawn through view-remap chain `chain`
/// (`helio_frame_api::lighting::VIEW_REMAPS`): (-1, -3 - chain).
fn gbuffer_tag_view_remap(chain: u32) -> vec2<f32> {
    return vec2<f32>(-1.0, -3.0 - f32(chain));
}

/// Whether a surface texel holds the requested payload rather than a tag.
fn gbuffer_has_payload(surface: vec2<f32>) -> bool {
    return surface.x >= 0.0;
}

/// Whether a surface texel is tag `tag`.
fn gbuffer_tagged(surface: vec2<f32>, tag: vec2<f32>) -> bool {
    return surface.x == tag.x && surface.y == tag.y;
}

/// Whether the surface's occlusion is its own: the lighting adds no
/// screen-space AO.
fn gbuffer_own_occlusion(surface: vec2<f32>) -> bool {
    return gbuffer_tagged(surface, GBUFFER_TAG_TRACED)
        || gbuffer_tagged(surface, GBUFFER_TAG_OWN_OCCLUSION);
}

/// The view-remap chain a pixel belongs to, or -1.
fn view_remap_of(surface: vec2<f32>) -> i32 {
    if surface.x > -1.5 && surface.x < -0.5 && surface.y < -2.5 {
        return min(i32(round(-3.0 - surface.y)), 255);
    }
    return -1;
}

// ── Volumetric scattering ───────────────────────────────────────────────────

const HELIO_PI: f32 = 3.14159265359;

/// Henyey-Greenstein phase function.
///
/// `cos_theta` is `dot(ray_dir, dir_to_light)` — 1.0 when looking straight down
/// the light's path toward it. `g` in (-1, 1): 0 isotropic, >0 forward-scattering
/// (the bright halo around a low sun), <0 back-scattering. |g| = 1 is a
/// singularity — the caller is expected to have clamped.
fn helio_hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * HELIO_PI * pow(max(denom, 1e-4), 1.5));
}
