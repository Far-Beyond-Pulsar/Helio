@group(0) @binding(8) var<storage, read> surfaces: array<Surface>;

// GBuffer publication (fullscreen triangle, depth-tested against meshes).
struct GOut {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) orm: vec4<f32>,
    @location(3) emissive: vec4<f32>,
    @location(4) lightmap_uv: vec2<f32>,
    @location(5) sss: vec4<f32>,
    @location(6) extra: vec4<f32>,
    @location(7) velocity: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn fullscreen(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((v << 1u) & 2u), f32(v & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn gbuffer(@builtin(position) pixel: vec4<f32>) -> GOut {
    let s = surfaces[pixel_index(vec2<u32>(pixel.xy))];
    if (s.flags & 3u) != 1u { discard; }
    let d = pixel_ray(pixel.xy);
    let position = camera.position_near.xyz + s.t * d;
    let clip = camera.view_proj * vec4<f32>(position, 1.0);
    let depth = clip.z / clip.w;
    if clip.w <= 0.0 || depth < 0.0 || depth > 1.0 { discard; }
    let a = unpack4x8unorm(s.albedo_ao);
    let material = (s.flags >> 8u) & 255u;
    let normal = oct_decode(s.normal);
    let roughness = frame.palette[min(material, 15u)].w;
    let f0 = 0.03;
    let albedo = pow(a.rgb, vec3<f32>(2.2));
    var out: GOut;
    out.albedo = vec4<f32>(albedo, 1.0);
    out.normal = vec4<f32>(normal, f0);
    out.orm = vec4<f32>(a.a, roughness, 0.0, f0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, f0);
    // Terrain tag consumed by deferred lighting (skips screen-space AO).
    out.lightmap_uv = vec2<f32>(-1.0, -2.0);
    out.sss = vec4<f32>(0.0);
    out.extra = vec4<f32>(0.0);
    let previous = camera.prev_view_proj * vec4<f32>(position, 1.0);
    var velocity = vec2<f32>(0.0);
    if previous.w > 0.0 {
        let ndc = previous.xy / previous.w;
        velocity = pixel.xy - (vec2<f32>(ndc.x, -ndc.y) * 0.5 + 0.5) * frame.screen.xy;
    }
    out.velocity = vec4<f32>(velocity, 0.0, 0.0);
    out.depth = depth;
    return out;
}
