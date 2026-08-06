// Proxy composite — merge one secondary G-buffer (portal eye or sublevel
// camera) into the main G-buffer, per docs/portals_and_sublevels.md §6.
//
// One full-screen triangle per active view, scissored to `region_rect`
// (CPU-side, via `pass.set_scissor_rect`). Per pixel: reconstruct world
// position from the secondary depth + secondary camera, map through
// `space_transform` (identity for sublevels, the portal's `pair_map_inverse`
// for portals — see `helio::scene::secondary_views` for why the inverse),
// reproject through the *main* camera, and write `@builtin(frag_depth)` so
// the pipeline's ordinary fixed-function depth test against the
// already-`Load`ed main depth buffer does the "closer than existing
// geometry" test for free — a fragment that loses the test never reaches
// the color/depth store. This is the load-bearing correctness detail beyond
// the design doc's pseudocode: composited content must land in the real
// depth buffer, or SSAO/shadows/TAA downstream see a hole.

const CAMERA_SLOTS: u32 = 7u;

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

/// Per-draw composite parameters. A CPU-side transcode of
/// `helio_secondary_core::GpuSecondaryView` into a uniform-address-space-safe
/// layout (that struct's own tightly-packed field order isn't valid as a
/// WGSL `<uniform>` binding — see `ProxyCompositePass::prepare`'s comment)
/// carrying only what this shader needs: `region_rect`/`clip_plane` stay
/// CPU-side (scissor / not implemented in v1, respectively).
struct CompositeParams {
    camera_slot:     u32,
    viewport_width:  f32,
    viewport_height: f32,
    _pad0:           u32,
    space_transform: mat4x4<f32>,
}

struct GBufferOutput {
    @location(0) albedo:      vec4<f32>,
    @location(1) normal:      vec4<f32>,
    @location(2) orm:         vec4<f32>,
    @location(3) emissive:    vec4<f32>,
    @location(4) lightmap_uv: vec2<f32>,
    @location(5) sss:         vec4<f32>,
    @location(6) extra:       vec4<f32>,
    @location(7) velocity:    vec2<f32>,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0)  var<storage, read> cameras: array<Camera, CAMERA_SLOTS>;
@group(0) @binding(1)  var<uniform>       params:  CompositeParams;
@group(0) @binding(2)  var sec_albedo:      texture_2d<f32>;
@group(0) @binding(3)  var sec_normal:      texture_2d<f32>;
@group(0) @binding(4)  var sec_orm:         texture_2d<f32>;
@group(0) @binding(5)  var sec_emissive:    texture_2d<f32>;
@group(0) @binding(6)  var sec_lightmap_uv: texture_2d<f32>;
@group(0) @binding(7)  var sec_sss:         texture_2d<f32>;
@group(0) @binding(8)  var sec_extra:       texture_2d<f32>;
@group(0) @binding(9)  var sec_depth:       texture_depth_2d;
@group(0) @binding(10) var sec_sampler:     sampler;

fn helio_uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

fn helio_world_from_depth(inv_view_proj: mat4x4<f32>, uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let world = inv_view_proj * vec4<f32>(helio_uv_to_ndc(uv), depth, 1.0);
    return world.xyz / world.w;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Standard 3-vertex full-screen triangle; no vertex buffer.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> GBufferOutput {
    let viewport = vec2<f32>(params.viewport_width, params.viewport_height);
    let uv = frag_coord.xy / viewport;

    // Depth textures take an i32 mip level for `textureSampleLevel`, unlike
    // the f32 level regular sampled textures use below — a WGSL asymmetry,
    // not a typo.
    let depth_sec = textureSampleLevel(sec_depth, sec_sampler, uv, 0);
    let sec_cam = cameras[params.camera_slot];
    let world = helio_world_from_depth(sec_cam.view_proj_inv, uv, depth_sec);

    // Identity for sublevels (already placed world positions); the portal's
    // B→A `pair_map_inverse` for portals, mapping the eye's real destination-
    // side geometry back into the frame the main camera actually occupies.
    let world2 = (params.space_transform * vec4<f32>(world, 1.0)).xyz;

    let main_cam = cameras[0];
    let main_clip = main_cam.view_proj * vec4<f32>(world2, 1.0);
    if main_clip.w <= 0.0 {
        discard;
    }
    let main_ndc = main_clip.xyz / main_clip.w;
    if main_ndc.z < 0.0 || main_ndc.z > 1.0 {
        discard;
    }

    // Rotate (not translate) the stored normal through `space_transform` —
    // `pair_map`/`pair_map_inverse` are rigid, so `w = 0` cancels translation
    // and leaves a pure rotation, which is what a *direction* needs.
    let sec_normal_ws = textureSampleLevel(sec_normal, sec_sampler, uv, 0.0).xyz;
    let normal_ws = normalize((params.space_transform * vec4<f32>(sec_normal_ws, 0.0)).xyz);

    // Velocity against the *main* camera (design doc §6), not the secondary
    // one — but this approximation reuses `world2` for both this frame and
    // last (it does not track how `space_transform` itself changed
    // frame-to-frame, e.g. a sublevel mid-move). Static or slow-moving
    // portals/sublevels motion-vector correctly; a fast-moving sublevel's
    // composited pixels get a one-frame TAA history mismatch in the
    // direction of motion, which TAA's temporal clamp already tolerates for
    // moderate error. Tracked as a follow-up, not a v1 requirement.
    let prev_clip = main_cam.prev_view_proj * vec4<f32>(world2, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_pixel = vec2<f32>(
        (prev_ndc.x * 0.5 + 0.5) * viewport.x,
        (0.5 - prev_ndc.y * 0.5) * viewport.y,
    );

    var out: GBufferOutput;
    out.albedo      = textureSampleLevel(sec_albedo, sec_sampler, uv, 0.0);
    out.normal      = vec4<f32>(normal_ws, 0.0);
    out.orm         = textureSampleLevel(sec_orm, sec_sampler, uv, 0.0);
    out.emissive    = textureSampleLevel(sec_emissive, sec_sampler, uv, 0.0);
    out.lightmap_uv = textureSampleLevel(sec_lightmap_uv, sec_sampler, uv, 0.0).xy;
    out.sss         = textureSampleLevel(sec_sss, sec_sampler, uv, 0.0);
    out.extra       = textureSampleLevel(sec_extra, sec_sampler, uv, 0.0);
    out.velocity    = frag_coord.xy - prev_pixel;
    out.depth       = main_ndc.z;
    return out;
}
