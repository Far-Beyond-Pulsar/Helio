// What a G-buffer drawer's fragment writes: the frame's targets
// (`helio_frame_api::gbuffer::TARGETS`) in location order. A drawer with
// nothing for a target writes zero to it (or masks it in its pipeline).
struct GBufferOutput {
    @location(0) albedo:      vec4<f32>,
    @location(1) normal:      vec4<f32>,
    @location(2) orm:         vec4<f32>,
    @location(3) emissive:    vec4<f32>,
    @location(4) lightmap_uv: vec2<f32>,
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
    @location(4) lightmap_uv: vec2<f32>,
    @location(5) sss:         vec4<f32>,
    @location(6) extra:       vec4<f32>,
    @location(7) velocity:    vec4<f32>,
    @builtin(frag_depth) depth: f32,
}
