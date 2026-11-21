// The sky's irradiance on a surface anywhere in the air, from the sky table
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
//
// Irradiance only: low-order harmonics hold a cosine-weighted integral of
// the sky well, not the sky along a direction. A ray takes the sky from the
// radiance table (`sky_radiance_eval`).

// Irradiance / pi at `from_camera` (km, relative to the camera) on a
// surface facing `n`: a Lambertian surface of albedo a reflects a * this.
// The result sums each directional light's contribution; above the
// atmosphere the planet's glow falls with its solid angle. Zero without an
// atmosphere.
fn sky_table_eval(from_camera: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return vec3<f32>(0.0); }
    let pos = from_camera - sky_frame.planet.xyz;
    let r = max(length(pos), sky_frame.params.bottom_radius);
    let up = normalize(pos);
    let fr = sky_table_row(sky_frame.params, r);
    let r0 = min(u32(fr), SKY_TABLE_R - 2u);
    let wr = fr - f32(r0);
    var result = vec3<f32>(0.0);
    for (var i = 0u; i < sky_frame.directional_light_count.x; i++) {
        let base = i * 3u;
        let sun = sky_frame.directional_lights[base].xyz;
        let energy = sky_frame.directional_lights[base + 1u].rgb;
        let l = transpose(sky_table_frame(up, sun)) * n;
        let fm = sky_table_column(dot(up, sun));
        let m0 = min(u32(fm), SKY_TABLE_MU - 2u);
        let wm = fm - f32(m0);
        let basis = array<f32, 6>(
            0.282095,
            0.488603 * l.y,
            0.488603 * l.x,
            1.092548 * l.x * l.y,
            0.315392 * (3.0 * l.z * l.z - 1.0),
            0.546274 * (l.x * l.x - l.y * l.y),
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
    let above = length(pos) / sky_frame.params.top_radius;
    let falloff = select(1.0, 1.0 / (above * above), above > 1.0);
    return result * falloff;
}
