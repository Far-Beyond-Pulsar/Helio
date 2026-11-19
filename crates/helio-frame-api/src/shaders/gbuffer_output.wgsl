// What a G-buffer drawer's fragment writes: the frame's targets
// (`helio_frame_api::gbuffer::TARGETS`) in location order. A drawer with
// nothing for a target writes zero to it (or masks it in its pipeline). A
// drawer fills the surface's targets from a `SurfaceData` with
// `gbuffer_output` / `gbuffer_depth_output` (below; they need
// `//!use frame_surface`).
struct GBufferOutput {
    @location(0) albedo:      vec4<f32>,
    @location(1) normal:      vec4<f32>,
    @location(2) orm:         vec4<f32>,
    @location(3) emissive:    vec4<f32>,
    // The surface channel: a tag (x < 0) or the requested payload
    // (`helio_frame_api::gbuffer::SURFACE`).
    @location(4) surface:     vec2<f32>,
    @location(5) sss:         vec4<f32>,
    @location(6) extra:       vec4<f32>,
    // Motion vectors: motion.xy, view-depth residual z, flags w
    // (`helio_frame_api::gbuffer::VELOCITY`).
    @location(7) velocity:    vec4<f32>,
}

// The same targets for a drawer that also writes the fragment's depth (a
// ray-marched surface).
struct GBufferDepthOutput {
    @location(0) albedo:      vec4<f32>,
    @location(1) normal:      vec4<f32>,
    @location(2) orm:         vec4<f32>,
    @location(3) emissive:    vec4<f32>,
    // The surface channel: a tag (x < 0) or the requested payload
    // (`helio_frame_api::gbuffer::SURFACE`).
    @location(4) surface:     vec2<f32>,
    @location(5) sss:         vec4<f32>,
    @location(6) extra:       vec4<f32>,
    @location(7) velocity:    vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// A drawer's fragment output for surface `s` (`//!use frame_surface`: the
// G-buffer's encoding, `gbuffer_surface_texels`), its surface channel
// `surface` (a tag or the requested payload) and motion `velocity`.
fn gbuffer_output(s: SurfaceData, surface: vec2<f32>, velocity: vec4<f32>) -> GBufferOutput {
    let t = gbuffer_surface_texels(s);
    return GBufferOutput(t.albedo, t.normal, t.orm, t.emissive, surface, t.sss, t.extra, velocity);
}

// The same with the fragment's depth.
fn gbuffer_depth_output(s: SurfaceData, surface: vec2<f32>, velocity: vec4<f32>, depth: f32) -> GBufferDepthOutput {
    let t = gbuffer_surface_texels(s);
    return GBufferDepthOutput(t.albedo, t.normal, t.orm, t.emissive, surface, t.sss, t.extra, velocity, depth);
}
