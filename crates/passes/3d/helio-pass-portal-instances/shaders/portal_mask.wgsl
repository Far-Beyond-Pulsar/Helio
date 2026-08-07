//! Screen-space portal-opening mask for one recursion *level* — two tiny
//! sub-passes run back to back (see
//! `helio-pass-portal-instances::mask::PortalMaskPass::execute`), one
//! `PortalMaskPass` instance per level `1..=MAX_CHAIN_DEPTH`, interleaved
//! with a matching per-level `PortalInstancePass` draw
//! (Mask(1),Draw(1),Mask(2),Draw(2),Mask(3),Draw(3) — see
//! `helio-default-graphs`).
//!
//! # Why per-level, not once for everything
//!
//! A depth-1 portal is a real surface the camera can look at directly, so
//! its screen footprint can be stamped straight from the real camera
//! against the real (already-drawn) depth buffer — that's the whole trick
//! `gbuffer_portal.wgsl`'s module doc describes. A depth-2 "portal seen
//! through a portal" has no such real surface: it's only ever visible
//! *through* its parent, so the only correct way to know its screen
//! footprint is to map its real quad through the parent chain's composed
//! transform and depth-test that against whatever's *actually been drawn
//! there so far* — which includes the parent level's own duplicated content
//! (a nearer wall reflection legitimately blocks a farther one, exactly
//! like a real mirror maze). That dependency is why this can't be one flat
//! pass: level 2's stamp needs level 1's draw to have already happened.
//!
//! 1. **Stamp** (`vs_stamp`/`fs_stamp`): for every chain whose `depth`
//!    equals this pass's `level`, draws its *last* portal's real opening
//!    quad, placed at that portal's own true position/size and then mapped
//!    through the composed transform of the chain's parent prefix (portals
//!    `0..depth-1`; identity for level 1, so this reduces to exactly the
//!    old single-portal stamp there). Depth-tested (read-only) against
//!    whatever's currently in the depth buffer. A surviving fragment writes
//!    this *specific chain's* index (not just its portal's — two different
//!    chains can reach the same final portal from different parents,
//!    landing in different screen positions, so the portal alone can't
//!    identify which one a pixel belongs to) into `portal_mask`.
//!
//! 2. **Reset** (`vs_reset`/`fs_reset`): identical role to the pre-recursion
//!    version — a full-screen triangle that writes the far-plane depth
//!    wherever this level's mask is non-zero, so this level's own duplicate
//!    draw self-occludes correctly instead of comparing against whatever
//!    (unrelated) real depth happened to be sitting behind the opening.

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

/// Must match libhelio::GpuPortalChain (16 bytes at MAX_CHAIN_DEPTH=3).
struct GpuPortalChain {
    portals: array<u32, 3>,
    depth:   u32,
}

struct StampUniform {
    level: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ── Stamp ────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> portal_views: array<GpuPortalView>;
@group(0) @binding(2) var<storage, read> portal_chains: array<GpuPortalChain>;
@group(0) @binding(3) var<uniform> stamp: StampUniform;
// Coordinate-space transforms (current frame) — slot `portal.coordinate_space`
// holds that portal's `pair_map_inverse`. Composing *these* (not
// `portal.transform`, which only places a portal's own quad at its own real
// position) is what maps content through a chain of portals, same as
// portal_cull.wgsl / gbuffer_portal.wgsl.
@group(0) @binding(4) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;

// Two triangles covering [-1,1]^2 in the portal's own local X/Y, scaled by
// half_extent in the vertex shader — the portal's real opening quad.
const LOCAL_CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

struct StampOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) chain_index: u32,
}

@vertex
fn vs_stamp(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> StampOutput {
    let chain = portal_chains[instance_index];

    var out: StampOutput;
    if chain.depth != stamp.level {
        // Not this level's chain — degenerate the triangle to nothing
        // rather than branch the draw call itself; simplest way to keep one
        // fixed-size `draw(0..6, 0..chain_count)` call covering every
        // level's stamps without a separate compacted list per level.
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.chain_index = instance_index;
        return out;
    }

    // Parent prefix's composed transform — identity at level 1 (no
    // parent), otherwise portals[depth-2] downto portals[0], same
    // deepest-first composition gbuffer_portal.wgsl's vertex shader uses
    // for an instance's own coordinate space.
    var parent = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    for (var i = chain.depth - 1u; i > 0u; i--) {
        let p = portal_views[chain.portals[i - 1u]];
        parent = coordinate_spaces[p.coordinate_space] * parent;
    }

    let last_portal = portal_views[chain.portals[chain.depth - 1u]];
    let local = LOCAL_CORNERS[vertex_index] * last_portal.half_extent;
    let own_world_pos = last_portal.transform * vec4<f32>(local, 0.0, 1.0);
    let world_pos = parent * own_world_pos;

    out.clip_position = cameras[0].view_proj * world_pos;
    out.chain_index = instance_index;
    return out;
}

@fragment
fn fs_stamp(input: StampOutput) -> @location(0) u32 {
    return input.chain_index + 1u;
}

// ── Reset ────────────────────────────────────────────────────────────────────
// Separate pipeline / bind group layout from the stamp pass above — WGSL
// group/binding numbers here are independent of the stamp entry point's.

@group(0) @binding(0) var portal_mask: texture_2d<u32>;

// Depth written wherever the mask is non-zero. Deliberately the *exact* far
// value (matches GBufferPass's own depth clear, see helio-pass-gbuffer) —
// not just "very far" — because deferred lighting distinguishes real geometry
// from empty background by comparing against that same clear value. Content
// inside the portal opening that isn't covered by any duplicated surface
// (e.g. the open interior of a hollow duplicated corridor) must read back as
// ordinary background there, exactly as if there were no portal, rather than
// as a very-distant-but-technically-real surface with stale G-buffer data —
// the latter previously showed up as a faint lit-looking rectangle over the
// whole opening. Any real duplicate content still wins the instance pass's
// LessEqual test against this, since legitimate scene depth is always
// strictly less than the far plane's exact NDC depth.
const RESET_DEPTH: f32 = 1.0;

@vertex
fn vs_reset(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Standard full-screen-triangle trick: 3 vertices covering the whole
    // clip-space square and then some, no vertex/index buffer needed.
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, -y, RESET_DEPTH, 1.0);
}

@fragment
fn fs_reset(@builtin(position) pos: vec4<f32>) {
    let mask_value = textureLoad(portal_mask, vec2<i32>(pos.xy), 0).r;
    if mask_value == 0u {
        discard;
    }
    // Depth write happens via the pipeline's normal depth-write path using
    // this fragment's interpolated position.z (== RESET_DEPTH, constant
    // across the triangle) — no @builtin(frag_depth) needed.
}
