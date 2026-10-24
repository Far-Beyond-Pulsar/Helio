// The sky over a surface anywhere in the atmosphere, from the sky table
// (`atmosphere_sky_table`, `SKY_TABLE_*` in atmosphere_common.wgsl). Needs
// the `atmosphere` snippet and a binding named `atmosphere_sky_table`
// (array<vec4<f32>>, uniform or storage).

// The table's harmonics at `from_camera` (km, relative to the camera)
// evaluated along `n`, each band scaled by `bands` (1, 1, 1: irradiance /
// pi for a surface facing `n`; 1, 3/2, 4: the sky's radiance along `n`,
// undoing the cosine convolution). In units of the sun's illuminance; above
// the atmosphere the planet's glow falls with its solid angle. Zero without
// an atmosphere.
fn atmosphere_sky_table_eval(frame: AtmosphereFrame, from_camera: vec3<f32>, n: vec3<f32>, bands: vec3<f32>) -> vec3<f32> {
    if frame.planet.w < 0.5 { return vec3<f32>(0.0); }
    let bottom = frame.params.bottom_radius;
    let top = frame.params.top_radius;
    let pos = from_camera - frame.planet.xyz;
    let r = max(length(pos), bottom);
    let up = normalize(pos);
    let sun = frame.sun.xyz;
    let mu_s = dot(up, sun);
    var x = sun - up * mu_s;
    if dot(x, x) < 1e-8 { x = cross(up, select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(up.x) > 0.9)); }
    x = normalize(x);
    let z = cross(x, up);
    let l = vec3<f32>(dot(n, x), dot(n, up), dot(n, z));
    let height = clamp((r - bottom) / (top - bottom), 0.0, 1.0);
    let fr = sqrt(height) * f32(SKY_TABLE_R - 1u);
    let fm = clamp((mu_s - SKY_TABLE_MU_LO) / (1.0 - SKY_TABLE_MU_LO), 0.0, 1.0) * f32(SKY_TABLE_MU - 1u);
    let r0 = min(u32(fr), SKY_TABLE_R - 2u);
    let m0 = min(u32(fm), SKY_TABLE_MU - 2u);
    let w = vec2<f32>(fr - f32(r0), fm - f32(m0));
    let basis = array<f32, 6>(
        0.282095 * bands.x,
        0.488603 * l.y * bands.y,
        0.488603 * l.x * bands.y,
        1.092548 * l.x * l.y * bands.z,
        0.315392 * (3.0 * l.z * l.z - 1.0) * bands.z,
        0.546274 * (l.x * l.x - l.y * l.y) * bands.z,
    );
    var e = vec3<f32>(0.0);
    for (var k = 0u; k < SKY_TABLE_COEFFS; k++) {
        let a = (r0 * SKY_TABLE_MU + m0) * SKY_TABLE_COEFFS + k;
        let b = a + SKY_TABLE_COEFFS;
        let c = a + SKY_TABLE_MU * SKY_TABLE_COEFFS;
        let d = c + SKY_TABLE_COEFFS;
        let v = mix(mix(atmosphere_sky_table[a].rgb, atmosphere_sky_table[b].rgb, w.y),
            mix(atmosphere_sky_table[c].rgb, atmosphere_sky_table[d].rgb, w.y), w.x);
        e += v * basis[k];
    }
    let above = length(pos) / top;
    let falloff = select(1.0, 1.0 / (above * above), above > 1.0);
    return max(e, vec3<f32>(0.0)) * frame.sun_illuminance.rgb * falloff;
}
