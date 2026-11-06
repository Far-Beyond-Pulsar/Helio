// The performance overlay's stand-in surface for its cost probe when no
// materials kind offers one (`helio_frame_api::diagnostics::SurfaceProbe`):
// Lambert, the reflected radiance per unit light.
fn surface_probe(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
) -> vec3<f32> {
    return base_color * (1.0 - metallic) * max(dot(n, l), 0.0) / 3.14159265359;
}
