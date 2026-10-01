// Shared encoding for the planetary sky-view LUT. Half its rows resolve the
// surface-facing sky, half resolve the limb/sky above the geometric horizon.
// Squared angular distance concentrates both halves at the horizon, and an
// exterior eye spends no rows on directions that miss the atmosphere.
fn planet_sky_basis() -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if dot(planetary_eye.xyz, planetary_eye.xyz) > 1e-10 { up = normalize(planetary_eye.xyz); }
    var tangent = sky.sun_direction - up * dot(sky.sun_direction, up);
    if dot(tangent, tangent) < 1e-6 {
        let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(up.y) > 0.9);
        tangent = cross(axis, up);
    }
    let x = normalize(tangent);
    return mat3x3<f32>(x, -up, cross(x, -up));
}

fn planet_sky_angles() -> vec2<f32> {
    let distance = max(length(planetary_eye.xyz), sky.earth_radius);
    let horizon = atan2(sky.earth_radius, sqrt(max((distance - sky.earth_radius) * (distance + sky.earth_radius), 0.0)));
    var outer = 3.14159265358979;
    if distance > sky.atm_radius {
        outer = atan2(sky.atm_radius, sqrt((distance - sky.atm_radius) * (distance + sky.atm_radius)));
    }
    return vec2<f32>(horizon, outer);
}

fn planet_sky_direction(uv: vec2<f32>) -> vec3<f32> {
    let angles = planet_sky_angles();
    let lower = 1.0 - 2.0 * uv.y;
    let upper = 2.0 * uv.y - 1.0;
    let theta = select(angles.x * (1.0 - lower * lower),
        angles.x + upper * upper * (angles.y - angles.x), uv.y >= 0.5);
    let azimuth = (uv.x - 0.5) * 6.28318530717958;
    return planet_sky_basis() * vec3<f32>(sin(theta) * cos(azimuth), cos(theta), sin(theta) * sin(azimuth));
}

fn planet_sky_uv(direction: vec3<f32>) -> vec2<f32> {
    let local = transpose(planet_sky_basis()) * direction;
    let theta = atan2(length(local.xz), local.y);
    let angles = planet_sky_angles();
    let lower = 0.5 * (1.0 - sqrt(clamp(1.0 - theta / angles.x, 0.0, 1.0)));
    let upper = 0.5 + 0.5 * sqrt(clamp((theta - angles.x) / max(angles.y - angles.x, 1e-7), 0.0, 1.0));
    // The fullscreen LUT raster flips the direction-coordinate V in texture space.
    return vec2<f32>(atan2(local.z, local.x) / 6.28318530717958 + 0.5,
        1.0 - select(lower, upper, theta >= angles.x));
}
