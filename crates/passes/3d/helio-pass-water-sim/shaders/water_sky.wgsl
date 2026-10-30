// The sky as the water sees it, shared by the surface and the underwater
// medium so the underside and the fog are one colour. Needs the `atmosphere`
// snippet, an `atmosphere: AtmosphereFrame` binding, and
// `fn water_sky_lut(uv: vec2f) -> vec3f` sampling the sky-view LUT.

/// Sky radiance along `ray` (sun-illuminance units, as the scene is lit in):
/// the sky-view LUT the sky itself is drawn from. Downward rays mirror up, as
/// Ultra-Water's `R.y = abs(R.y)` does, so a tilted facet never reflects
/// black ground.
fn sky_color(ray: vec3f, light_dir: vec3f) -> vec3f {
    if atmosphere.planet.w >= 0.5 {
        let dir = normalize(vec3f(ray.x, abs(ray.y), ray.z));
        let angles = sky_view_direction_angles(atmosphere, dir);
        let uv = sky_view_uv(atmosphere.params, length(atmosphere.eye.xyz), angles.x, angles.y);
        return water_sky_lut(uv) * atmosphere.sun_illuminance.rgb;
    }
    let up      = clamp(ray.y, 0.0, 1.0);
    let horizon = vec3f(0.80, 0.90, 1.00);
    let zenith  = vec3f(0.10, 0.30, 0.80);
    let sky     = mix(horizon, zenith, up * up);
    let spec    = pow(max(0.0, dot(normalize(light_dir), ray)), 5000.0);
    return sky + vec3f(spec) * vec3f(10.0, 8.0, 6.0);
}

/// The sun's radiance at the sea (Ultra-Water's `uSunColor`: ground
/// illuminance / pi), reddened by the air it crosses at a low sun.
fn water_sun_radiance(sun_dir: vec3f) -> vec3f {
    if atmosphere.planet.w >= 0.5 {
        return atmosphere.sun_illuminance.rgb
            * atmosphere_transmittance_towards(atmosphere, vec3f(0.0), sun_dir) / HELIO_PI;
    }
    return vec3f(3.0, 2.82, 2.46) / HELIO_PI;
}

/// The sky's mean radiance over the whole sphere: what Ultra-Water reads from
/// the lowest mip of its sky cube to light the water column. The zenith alone
/// is far too bright at a low sun (the dome is dark overhead and bright only
/// near the horizon), so average the dome (zenith plus a ring of horizon
/// samples round the sun) and halve it for the unlit lower hemisphere.
fn water_sky_ambient(sun_dir: vec3f) -> vec3f {
    let zenith = sky_color(vec3f(0.0, 1.0, 0.0), sun_dir);
    let h = normalize(vec2f(sun_dir.x, sun_dir.z) + vec2f(1e-4, 0.0));
    let ring = (sky_color(normalize(vec3f(h.x, 0.1, h.y)), sun_dir)
        + sky_color(normalize(vec3f(-h.x, 0.1, -h.y)), sun_dir)
        + sky_color(normalize(vec3f(-h.y, 0.1, h.x)), sun_dir)
        + sky_color(normalize(vec3f(h.y, 0.1, -h.x)), sun_dir)) * 0.25;
    return 0.5 * (0.3 * zenith + 0.7 * ring);
}
