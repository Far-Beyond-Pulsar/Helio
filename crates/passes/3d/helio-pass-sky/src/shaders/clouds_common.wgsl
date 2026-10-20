// The atmosphere's cloud layer (`CloudLayer`): a band of altitudes over the
// planet's surface where the weather gathers clouds, billows shape them and
// wisps erode their edges (after Schneider's Nubis, Horizon Zero Dawn), lit
// by the sun through the air with multiple scattering (Wrenninge's octaves,
// as in Hillaire 2016). Needs the `atmosphere` snippet and bindings named
// `cloud_weather`, `cloud_shape`, `cloud_detail` (texture_3d<f32>, the
// noise `cloud_noise.wgsl` builds) and `cloud_noise_sampler` (linear,
// repeating, mipmapped). Positions are kilometres relative to the planet's
// centre, like the atmosphere's.

// Texels along each side of the noise volumes.
const CLOUD_WEATHER_TEXELS: f32 = 64.0;
const CLOUD_SHAPE_TEXELS: f32 = 128.0;
const CLOUD_DETAIL_TEXELS: f32 = 32.0;

// The layer's parameters as the density reads them, gathered once per
// invocation (the frame is too large to pass to every density sample).
struct CloudLayerParams {
    // Radii of the planet, the layer's bottom and top (km); 1 / thickness.
    planet_radius: f32,
    inner: f32,
    outer: f32,
    inv_thickness: f32,
    coverage: f32,
    kind: f32,
    extinction: f32,
    erosion: f32,
    inv_weather: f32,
    inv_shape: f32,
    inv_detail: f32,
    // The wisps' mip relative to the billows'.
    detail_lod_bias: f32,
    weather_offset: vec3<f32>,
    climate_offset: vec3<f32>,
    shape_offset: vec3<f32>,
    detail_offset: vec3<f32>,
    albedo: vec3<f32>,
}

fn cloud_layer_params(frame: AtmosphereFrame) -> CloudLayerParams {
    let p = frame.params;
    var c: CloudLayerParams;
    c.planet_radius = p.bottom_radius;
    c.inner = p.bottom_radius + p.cloud_bottom;
    c.outer = p.bottom_radius + max(p.cloud_top, p.cloud_bottom + 1e-3);
    c.inv_thickness = 1.0 / (c.outer - c.inner);
    c.coverage = clamp(p.cloud_coverage, 0.0, 1.0);
    c.kind = clamp(p.cloud_type, 0.0, 1.0);
    c.extinction = max(p.cloud_extinction, 0.0);
    c.erosion = clamp(p.cloud_erosion, 0.0, 1.0);
    c.inv_weather = 1.0 / max(p.cloud_weather_scale, 1e-3);
    c.inv_shape = 1.0 / max(p.cloud_shape_scale, 1e-3);
    c.inv_detail = 1.0 / max(p.cloud_detail_scale, 1e-3);
    c.detail_lod_bias = log2((max(p.cloud_shape_scale, 1e-3) / CLOUD_SHAPE_TEXELS)
        / (max(p.cloud_detail_scale, 1e-3) / CLOUD_DETAIL_TEXELS));
    c.weather_offset = frame.cloud_weather_offset.xyz;
    c.climate_offset = frame.cloud_climate_offset.xyz;
    c.shape_offset = frame.cloud_shape_offset.xyz;
    c.detail_offset = frame.cloud_detail_offset.xyz;
    c.albedo = clamp(p.cloud_albedo, vec3<f32>(0.0), vec3<f32>(1.0));
    return c;
}

fn cloud_remap(x: f32, lo: f32, hi: f32) -> f32 {
    return (x - lo) / max(hi - lo, 1e-5);
}

// Density over the layer's height `h` (0 bottom, 1 top) for a form `kind`:
// stratus is a thin sheet low in the layer, cumulus rises through it, flat
// at the base and rounded at the crown.
fn cloud_height_profile(h: f32, kind: f32) -> f32 {
    let top = mix(0.35, 1.0, kind);
    let base = smoothstep(0.0, mix(0.08, 0.14, kind), h);
    let crown = 1.0 - smoothstep(top * mix(0.55, 0.3, kind), top, h);
    return base * crown;
}

// How much cloud the weather allows over `anchor` (the point's direction at
// the layer's bottom, km): 0 clear sky, 1 the billows' full extent. x: that
// coverage, y: the local form.
fn cloud_weather_at(c: CloudLayerParams, anchor: vec3<f32>) -> vec2<f32> {
    let weather = textureSampleLevel(cloud_weather, cloud_noise_sampler, anchor * c.inv_weather + c.weather_offset, 0.0);
    let climate = textureSampleLevel(cloud_weather, cloud_noise_sampler,
        anchor * (c.inv_weather / CLOUD_CLIMATE) + c.climate_offset, 0.0);
    let field = weather.r * 0.6 + climate.g * 0.4;
    // The coverage's quantile of the field: the share of the sky with clouds.
    let threshold = mix(0.78, 0.22, c.coverage);
    let cover = smoothstep(threshold - 0.12, threshold + 0.12, field);
    let kind = clamp(c.kind + (weather.b - 0.5) * 0.6, 0.0, 1.0);
    return vec2<f32>(cover, kind);
}

// Cloud density (0..1, times `extinction` per km) at planet-relative point
// `x`, `r` from the centre at height `h` in the layer (0..1), under
// `weather` (`cloud_weather_at`), with the billows at mip `lod`; the wisps
// only when `detail`.
fn cloud_density_under(c: CloudLayerParams, x: vec3<f32>, h: f32, weather: vec2<f32>, lod: f32, detail: bool) -> f32 {
    if h <= 0.0 || h >= 1.0 || weather.x <= 0.0 { return 0.0; }
    let profile = cloud_height_profile(h, weather.y);
    if profile <= 0.0 { return 0.0; }
    let s = textureSampleLevel(cloud_shape, cloud_noise_sampler, x * c.inv_shape + c.shape_offset, lod);
    let worley = s.g * 0.625 + s.b * 0.25 + s.a * 0.125;
    let billows = clamp(cloud_remap(s.r, worley - 1.0, 1.0), 0.0, 1.0);
    // Coverage carves the billows (thin cover keeps only their cores) and
    // scales what is left.
    var d = clamp(cloud_remap(billows * profile, 1.0 - weather.x, 1.0), 0.0, 1.0) * weather.x;
    if d <= 0.0 || !detail { return d; }
    let n = textureSampleLevel(cloud_detail, cloud_noise_sampler, x * c.inv_detail + c.detail_offset,
        max(lod + c.detail_lod_bias, 0.0));
    let wisps = n.r * 0.625 + n.g * 0.25 + n.b * 0.125;
    // Wispy at the base, billowing towards the crown.
    let erode = mix(wisps, 1.0 - wisps, clamp(h * 4.0, 0.0, 1.0)) * c.erosion;
    return clamp(cloud_remap(d, erode, 1.0), 0.0, 1.0);
}

// The layer's height (0..1) of planet-relative point `x`.
fn cloud_height(c: CloudLayerParams, x: vec3<f32>) -> f32 {
    return (length(x) - c.inner) * c.inv_thickness;
}

// Cloud density at `x` under its own weather. x: the density, y: the height
// in the layer, zw: the weather there (`cloud_weather_at`; zero outside).
fn cloud_density(c: CloudLayerParams, x: vec3<f32>, lod: f32, detail: bool) -> vec4<f32> {
    let r = length(x);
    let h = (r - c.inner) * c.inv_thickness;
    if h <= 0.0 || h >= 1.0 { return vec4<f32>(0.0, h, 0.0, 0.0); }
    let weather = cloud_weather_at(c, x * (c.inner / r));
    return vec4<f32>(cloud_density_under(c, x, h, weather, lod, detail), h, weather);
}

// The span of a ray from `o` along `d` (planet-relative, km) inside the
// layer: x start, y end (y <= x: it misses). From below, the span up
// through the layer (none if the ground comes first); from inside, up to
// where the ray leaves; from above, from the top down to the bottom or out.
// Only the first span: a ray that passes under the layer and rises into it
// again beyond the horizon stops at its bottom.
fn cloud_layer_segment(c: CloudLayerParams, o: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    let outer = ray_sphere(o, d, c.outer);
    if outer.y <= 0.0 { return vec2<f32>(0.0, -1.0); }
    let inner = ray_sphere(o, d, c.inner);
    let r = length(o);
    if r < c.inner {
        let ground = ray_sphere(o, d, c.planet_radius);
        if ground.x > 0.0 { return vec2<f32>(0.0, -1.0); }
        return vec2<f32>(inner.y, outer.y);
    }
    let start = select(0.0, outer.x, r > c.outer);
    let end = select(outer.y, inner.x, inner.x > 0.0);
    return vec2<f32>(start, end);
}

// Henyey-Greenstein.
fn cloud_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * ATMOSPHERE_PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));
}

// Light scattered towards the eye at a point whose optical depth towards
// the sun is `od`, per unit of scattering and of sun: Wrenninge's multiple
// scattering octaves (each bounce weaker, less extinguished and less
// forward), over a phase of a strong forward lobe and a backward one (the
// silver lining and the darkened back of a cloud seen against the sun).
// Four octaves, each 0.6 of the last's light, 0.3 of its extinction: thick
// clouds keep most of a diffuse reflector's light (seen from above, their
// tops are brighter than the ground) at the price of thin ones scattering
// about twice their single scattering.
const CLOUD_OCTAVES: u32 = 4u;

fn cloud_sun_scattering(cos_theta: f32, od: f32) -> f32 {
    var total = 0.0;
    var a = 1.0;
    var b = 1.0;
    var c = 1.0;
    for (var k = 0u; k < CLOUD_OCTAVES; k++) {
        let phase = mix(cloud_hg(cos_theta, 0.8 * c), cloud_hg(cos_theta, -0.3 * c), 0.3);
        total += a * phase * exp(-od * b);
        a *= 0.6;
        b *= 0.3;
        c *= 0.5;
    }
    return total;
}
