//! Editor-only portal-surface indicator.
//!
//! In game mode a portal is meant to be perfectly seamless — nothing marks
//! where its surface actually is, only the duplicated content shows (see
//! `gbuffer_portal.wgsl` / `portal_mask.wgsl`). That's exactly right for
//! play, but invisible trigger-plane-sized geometry is miserable to work
//! with in an editor: you can't see it, click it, or tell where its bounds
//! are relative to the room around it.
//!
//! This pass draws a checkerboard over each portal's opening, alpha-blended
//! into `pre_aa` *after* deferred lighting (same slot Corona/Billboard/
//! LensFlare use for their own overlays) — one checker color is fully
//! transparent (`discard`, zero cost, the seamless look is untouched there),
//! the other is a low-alpha darkening tint. The result reads as a faint
//! marker grid over the portal's true footprint without ever fully hiding
//! the duplicated content behind it. Only runs when the pass's `editor_mode`
//! flag is set — zero draws, zero cost, in game builds.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

/// Must match libhelio::GpuPortalView (144 bytes) — see gbuffer_portal.wgsl.
struct GpuPortalView {
    transform:         mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
    half_extent:       vec2<f32>,
    coordinate_space:  u32,
    _pad:              u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> portal_views: array<GpuPortalView>;

const LOCAL_CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

// Checkerboard cell size, in world units (metres). Tune to taste — this is
// an editor-visualization constant, not something scenes need to configure.
const CELL_SIZE: f32 = 0.5;
// Alpha of the darkened checker cell; the other cell is fully transparent.
const DARKEN_ALPHA: f32 = 0.18;

struct OverlayOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_xy: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> OverlayOutput {
    let portal = portal_views[instance_index];
    let local = LOCAL_CORNERS[vertex_index] * portal.half_extent;
    let world_pos = portal.transform * vec4<f32>(local, 0.0, 1.0);

    var out: OverlayOutput;
    out.clip_position = cameras[0].view_proj * world_pos;
    out.local_xy = local;
    return out;
}

@fragment
fn fs_main(input: OverlayOutput) -> @location(0) vec4<f32> {
    let cell = vec2<i32>(floor(input.local_xy / CELL_SIZE));
    // Bitwise AND on the low bit — well-defined for negative i32 in WGSL,
    // unlike `%` which can return negative results and break the parity.
    let checker = (cell.x ^ cell.y) & 1;
    if checker == 0 {
        discard;
    }
    return vec4<f32>(0.0, 0.0, 0.0, DARKEN_ALPHA);
}
