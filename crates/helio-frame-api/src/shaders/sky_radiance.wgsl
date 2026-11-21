// The sky's radiance along a direction from anywhere, from the sky radiance
// table (`sky_radiance`, `SKY_RADIANCE_*` in frame_sky.wgsl): what a ray
// that meets nothing takes. Needs the `frame_sky` snippet and three
// bindings by name: the frame as `sky_frame` (SkyFrame, uniform or
// storage), the table as `sky_radiance` (texture_3d<f32>) and a filtering,
// clamping sampler as `sky_radiance_sampler`
// (`helio_frame_api::lighting::sky::sky_radiance_sampler`).
//
// Seen from the eye it is the sky the sky pass draws (the sky-view LUT) to
// the table's resolution: the air's single and multiple scattering and the
// lit ground below the horizon, without the suns' discs (each sun is a
// light its samplers sample), the stars, a panorama or the clouds.

// The log of the radiance table at `uvw`, bilinear within its slice.
fn sky_radiance_read(uvw: vec3<f32>) -> vec3<f32> {
    let v = textureSampleLevel(sky_radiance, sky_radiance_sampler, uvw, 0.0).rgb;
    return log(max(v, vec3<f32>(1e-9)));
}

// The sky's radiance along unit `d` from `from_camera` (km, relative to the
// camera), summed over the frame's directional lights. From above the air,
// the sky where `d` enters it (nothing where it misses). Zero without an
// atmosphere.
fn sky_radiance_eval(from_camera: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return vec3<f32>(0.0); }
    let p = sky_frame.params;
    var pos = from_camera - sky_frame.planet.xyz;
    if length(pos) > p.top_radius {
        // No air outside the shell: the radiance along `d` is the one where
        // it enters.
        let t = ray_sphere(pos, d, p.top_radius);
        if t.x < 0.0 { return vec3<f32>(0.0); }
        pos += d * t.x;
    }
    // Points below the planet's surface (valleys, caves) see the sky of a
    // point on it.
    let r = clamp(length(pos), p.bottom_radius, p.top_radius);
    let up = normalize(pos);
    let fr = sky_table_row(p, r);
    let r0 = min(u32(fr), SKY_TABLE_R - 2u);
    let wr = fr - f32(r0);
    var result = vec3<f32>(0.0);
    for (var i = 0u; i < sky_frame.directional_light_count.x; i++) {
        let base = i * 3u;
        let sun = sky_frame.directional_lights[base].xyz;
        let energy = sky_frame.directional_lights[base + 1u].rgb;
        let l = transpose(sky_table_frame(up, sun)) * d;
        let fm = sky_table_column(dot(up, sun));
        let m0 = min(u32(fm), SKY_TABLE_MU - 2u);
        let wm = fm - f32(m0);
        // The four entries about (r, mu_s), each read at the coordinate of
        // `d` at the actual radius (each row's map is laid out about its own
        // horizon, so the horizon lines up between rows).
        let uvw = sky_radiance_uvw(p, r, l, f32(r0 * SKY_TABLE_MU + m0));
        let slice = 1.0 / f32(SKY_TABLE_R * SKY_TABLE_MU);
        let row = f32(SKY_TABLE_MU) * slice;
        let a = sky_radiance_read(uvw);
        let b = sky_radiance_read(uvw + vec3<f32>(0.0, 0.0, slice));
        let c = sky_radiance_read(uvw + vec3<f32>(0.0, 0.0, row));
        let e = sky_radiance_read(uvw + vec3<f32>(0.0, 0.0, row + slice));
        // Between entries geometrically: the sky falls exponentially with
        // altitude (the air's scale heights) and with the sun's path through
        // the air (towards and below the horizon), which a linear blend
        // overestimates by tens of percent between rows kilometres apart and
        // in twilight.
        result += exp(mix(mix(a, b, wm), mix(c, e, wm), wr)) * energy;
    }
    return result;
}
