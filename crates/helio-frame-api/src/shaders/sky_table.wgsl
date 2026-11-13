// The sky over a surface anywhere in the air, from the sky table
// (`sky_table`, `SKY_TABLE_*` in frame_sky.wgsl). Needs the `frame_sky`
// snippet and two bindings by name: the frame as `sky_frame` (SkyFrame) and
// the table as `sky_table` (array<vec4<f32>>), each uniform or storage.
//
// Both are read through their bindings, never passed in by value: the loop
// below indexes the frame's directional lights by a runtime index, and an
// array reached through a by-value argument cannot be indexed in place, so
// the compiler copies all of it (1.5 KiB) into per-invocation memory on every
// call. WGSL allows no uniform or storage pointer parameters, so the
// bindings' names are the contract.

// The table's harmonics at `from_camera` (km, relative to the camera)
// evaluated along `n`, each band scaled by `bands` (1, 1, 1: irradiance /
// pi for a surface facing `n`; 1, 3/2, 4: the sky's radiance along `n`,
// undoing the cosine convolution). The result sums each directional light's
// contribution; above the atmosphere the planet's glow falls with its solid
// angle. Zero without an atmosphere.
fn sky_table_eval(from_camera: vec3<f32>, n: vec3<f32>, bands: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return vec3<f32>(0.0); }
    let bottom = sky_frame.params.bottom_radius;
    let top = sky_frame.params.top_radius;
    let pos = from_camera - sky_frame.planet.xyz;
    let r = max(length(pos), bottom);
    let up = normalize(pos);
    let height = clamp((r - bottom) / (top - bottom), 0.0, 1.0);
    let fr = sqrt(height) * f32(SKY_TABLE_R - 1u);
    let r0 = min(u32(fr), SKY_TABLE_R - 2u);
    let wr = fr - f32(r0);
    var result = vec3<f32>(0.0);
    for (var i = 0u; i < sky_frame.directional_light_count.x; i++) {
        let base = i * 3u;
        let sun = sky_frame.directional_lights[base].xyz;
        let energy = sky_frame.directional_lights[base + 1u].rgb;
        let mu_s = dot(up, sun);
        var x = sun - up * mu_s;
        if dot(x, x) < 1e-8 { x = cross(up, select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(up.x) > 0.9)); }
        x = normalize(x);
        let z = cross(x, up);
        let l = vec3<f32>(dot(n, x), dot(n, up), dot(n, z));
        let fm = clamp((mu_s - SKY_TABLE_MU_LO) / (1.0 - SKY_TABLE_MU_LO), 0.0, 1.0) * f32(SKY_TABLE_MU - 1u);
        let m0 = min(u32(fm), SKY_TABLE_MU - 2u);
        let wm = fm - f32(m0);
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
            let v = mix(mix(sky_table[a].rgb, sky_table[b].rgb, wm),
                mix(sky_table[c].rgb, sky_table[d].rgb, wm), wr);
            e += v * basis[k];
        }
        result += max(e, vec3<f32>(0.0)) * energy;
    }
    let above = length(pos) / top;
    let falloff = select(1.0, 1.0 / (above * above), above > 1.0);
    return result * falloff;
}
