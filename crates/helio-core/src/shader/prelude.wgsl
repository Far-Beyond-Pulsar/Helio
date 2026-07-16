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
