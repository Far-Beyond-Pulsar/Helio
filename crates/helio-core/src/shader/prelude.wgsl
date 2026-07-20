// Helio shader prelude — the canonical camera layout and the depth/G-buffer
// conventions that go with it.
//
// Prepended to any shader whose first lines contain `//!use helio_prelude`.
// See helio_core::shader.
//
// This exists because every pass used to re-derive this math from scratch, and
// they drifted: the NDC y-flip was present in some shaders and absent in others,
// one pass used the OpenGL [-1,1] depth convention against a wgpu [0,1] buffer,
// and two passes "decoded" G-buffer normals that were never encoded. Each bug
// was invisible in isolation and only showed up as a wrong picture. Depth math
// is not per-pass policy, so it does not live in per-pass code.

// ── Camera ──────────────────────────────────────────────────────────────────
// Field-for-field mirror of helio_core / libhelio `GpuCameraUniforms` (256 B).
// Shaders declare their OWN binding (the group/binding varies per pass) but must
// use this struct rather than redeclaring it:
//
//     //!use helio_prelude
//     @group(0) @binding(0) var<uniform> camera: Camera;
//
struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    /// Camera world position (xyz) + near plane (w).
    position_near:  vec4<f32>,
    /// Camera forward direction (xyz) + far plane (w).
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
// The engine builds its projection with glam `Mat4::perspective_rh`, so NDC z is
// [0,1] (near..far) — the wgpu/D3D convention, NOT OpenGL's [-1,1]. Depth-buffer
// values therefore go into `view_proj_inv` as-is; remapping with `depth * 2 - 1`
// is an OpenGL habit that silently skews every reconstructed position.

/// World position from a depth-buffer sample.
/// `inv_view_proj` is `camera.view_proj_inv`; `depth` is the raw [0,1] buffer value.
fn helio_world_from_depth(inv_view_proj: mat4x4<f32>, uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let world = inv_view_proj * vec4<f32>(helio_uv_to_ndc(uv), depth, 1.0);
    return world.xyz / world.w;
}

/// Raw [0,1] depth to positive view-space distance (near..far).
///
/// `near`/`far` are `camera.position_near.w` / `camera.forward_far.w`.
fn helio_view_depth(depth: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - depth * (far - near));
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

// ── Shadows ─────────────────────────────────────────────────────────────────
// The *math* of shadow lookup, shared; the texture sample itself stays in the
// consumer. WGSL cannot take a storage-array pointer as a plain function
// parameter, so a full `shadow_factor()` here would have to hard-code the atlas
// binding — and the whole point of the prelude is that group/binding stays the
// caller's business. Cascade selection and projection are the parts that
// actually drifted, and they are pure functions.

/// Which cascade(s) a view distance falls in, and how much to blend between them.
struct HelioCsmSelection {
    /// Primary cascade index (0..3).
    cascade_a: u32,
    /// Cascade to blend toward; equals `cascade_a` outside a blend zone.
    cascade_b: u32,
    /// 0 = use `cascade_a` alone, >0 = mix toward `cascade_b`.
    blend: f32,
}

/// Fraction of a split distance spent cross-fading between adjacent cascades.
/// Without it, a cascade boundary is a hard line across the floor.
const HELIO_CSM_BLEND_ZONE: f32 = 0.1;

/// Pick CSM cascades for a view distance.
///
/// `splits` is `Globals.csm_splits`: xyz are the three cascade boundaries
/// (cascade 3 runs from splits.z to the far plane). `view_dist` is the distance
/// from the camera to the shaded point, not its depth-buffer value.
fn helio_csm_select(view_dist: f32, splits: vec4<f32>) -> HelioCsmSelection {
    var s: HelioCsmSelection;
    s.cascade_a = 3u;
    s.cascade_b = 3u;
    s.blend = 0.0;

    let half_zone = HELIO_CSM_BLEND_ZONE * 0.5;

    if view_dist < splits.x * (1.0 - half_zone) {
        s.cascade_a = 0u;
    } else if view_dist < splits.x * (1.0 + half_zone) {
        s.cascade_a = 0u;
        s.cascade_b = 1u;
        s.blend = smoothstep(splits.x * (1.0 - half_zone), splits.x * (1.0 + half_zone), view_dist);
    } else if view_dist < splits.y * (1.0 - half_zone) {
        s.cascade_a = 1u;
    } else if view_dist < splits.y * (1.0 + half_zone) {
        s.cascade_a = 1u;
        s.cascade_b = 2u;
        s.blend = smoothstep(splits.y * (1.0 - half_zone), splits.y * (1.0 + half_zone), view_dist);
    } else if view_dist < splits.z * (1.0 - half_zone) {
        s.cascade_a = 2u;
    } else if view_dist < splits.z * (1.0 + half_zone) {
        s.cascade_a = 2u;
        s.cascade_b = 3u;
        s.blend = smoothstep(splits.z * (1.0 - half_zone), splits.z * (1.0 + half_zone), view_dist);
    }

    return s;
}

/// A world position projected into a shadow map.
struct HelioShadowProj {
    /// Atlas UV (y down), only meaningful when `valid`.
    uv: vec2<f32>,
    /// [0,1] depth to compare against, only meaningful when `valid`.
    depth: f32,
    /// False when the point is behind the light or outside the map — callers
    /// must treat that as *lit*, not shadowed, or geometry outside the cascade
    /// goes black.
    valid: bool,
}

/// Project a world position into a shadow map's UV + depth.
///
/// `light_view_proj` is `shadow_matrices[layer].mat`. The y-flip goes through
/// `helio_ndc_to_uv` for the same reason it does everywhere else — shadow UV is
/// framebuffer-oriented (y down) while NDC is y up.
fn helio_shadow_project(light_view_proj: mat4x4<f32>, world_pos: vec3<f32>) -> HelioShadowProj {
    var s: HelioShadowProj;
    s.uv = vec2<f32>(0.0);
    s.depth = 0.0;
    s.valid = false;

    let clip = light_view_proj * vec4<f32>(world_pos, 1.0);
    // Behind the light's near plane: the perspective divide would mirror the
    // point back into frame and shadow it against unrelated depths.
    if clip.w <= 0.0 { return s; }

    let ndc = clip.xyz / clip.w;
    let uv = helio_ndc_to_uv(ndc.xy);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return s;
    }

    s.uv = uv;
    s.depth = ndc.z;
    s.valid = true;
    return s;
}

// ── Froxel grid ─────────────────────────────────────────────────────────────
// The volumetric fog grid is a view-space 3D texture with exponentially spaced
// depth slices — dense near the camera, sparse far away, which is where the
// detail actually matters.
//
// Two shaders must agree on this mapping exactly: the fog pass writes froxels by
// slice index, and the post-process composite reads them back by pixel depth. A
// mismatch does not error, it just samples the wrong slice and puts the fog at
// the wrong distance. Hence: here, once, rather than once per shader.

/// Near plane of the froxel grid.
///
/// Deliberately not the camera near plane. Exponential spacing from 0.1 would
/// spend most of the 64 slices within a couple of metres of the camera and leave
/// almost none for the distance the fog is actually read at.
const HELIO_FROXEL_NEAR: f32 = 0.5;

/// View depth (not radial distance) → normalized slice coordinate in [0,1].
fn helio_froxel_slice_from_view_depth(view_depth: f32, far: f32) -> f32 {
    let d = max(view_depth, HELIO_FROXEL_NEAR);
    return log(d / HELIO_FROXEL_NEAR) / log(max(far, HELIO_FROXEL_NEAR * 2.0) / HELIO_FROXEL_NEAR);
}

/// Normalized slice coordinate in [0,1] → view depth. Inverse of the above.
fn helio_froxel_view_depth_from_slice(slice_norm: f32, far: f32) -> f32 {
    return HELIO_FROXEL_NEAR * pow(max(far, HELIO_FROXEL_NEAR * 2.0) / HELIO_FROXEL_NEAR, slice_norm);
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
