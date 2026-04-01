// update.frag.wgsl — one step of the shallow-water wave propagation.
//
// Texture layout (Rgba16Float):
//   R = height
//   G = velocity
//   B = normal.x  (written by normal pass)
//   A = normal.z  (written by normal pass)

@group(0) @binding(0) var water_texture: texture_2d<f32>;
@group(0) @binding(1) var water_sampler: sampler;

struct UpdateUniforms {
    /// Texel size: (1 / texture_width, 1 / texture_height)
    delta: vec2<f32>,
    /// Wave spring constant — restoring force toward the mean height.
    /// Lower (~1.0) feels like fluid; higher (~2.0) feels jelly-like.
    spring: f32,
    /// Per-step energy damping multiplier (0..1).
    /// Closer to 1.0 = waves persist longer. Closer to 0.9 = waves die quickly.
    damping: f32,
    /// Wind direction in XZ sim-texture space (pre-normalised; zero = no wind).
    wind_dir: vec2<f32>,
    /// Wind strength multiplier. 0 = calm; ~1 = gentle ripples; ~5 = choppy.
    wind_strength: f32,
    /// Elapsed simulation time (seconds) — scrolls the wind-noise pattern.
    time: f32,
}
@group(0) @binding(2) var<uniform> u: UpdateUniforms;

// ---------------------------------------------------------------------------
// Value noise helpers for wind-driven turbulence
// ---------------------------------------------------------------------------

fn hash2(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(127.1, 311.7));
    q += dot(q, q + vec2<f32>(19.19, 74.39));
    return fract(q.x * q.y) * 2.0 - 1.0;  // [-1, 1]
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);  // smoothstep
    return mix(
        mix(hash2(i),                    hash2(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y,
    );
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var info = textureSample(water_texture, water_sampler, uv);

    let dx = vec2<f32>(u.delta.x, 0.0);
    let dy = vec2<f32>(0.0, u.delta.y);

    // Average of the four cardinal neighbours' heights
    let avg = (
        textureSample(water_texture, water_sampler, uv - dx).r +
        textureSample(water_texture, water_sampler, uv - dy).r +
        textureSample(water_texture, water_sampler, uv + dx).r +
        textureSample(water_texture, water_sampler, uv + dy).r
    ) * 0.25;

    // Velocity = displacement toward mean (spring) + energy damping
    info.g += (avg - info.r) * u.spring;
    info.g *= u.damping;

    // Wind: scroll a two-octave noise pattern in the wind direction and use it
    // to inject turbulent velocity impulses each sim step.
    if u.wind_strength > 0.001 {
        let scroll = u.wind_dir * u.time * 0.04;
        let p0 = uv * 6.0 + scroll;
        let p1 = uv * 12.7 + scroll * 1.3 + vec2<f32>(5.31, 1.73);
        let turbulence = value_noise(p0) * 0.65 + value_noise(p1) * 0.35;
        info.g += turbulence * u.wind_strength * 0.003;
    }

    // Euler-integrate height
    info.r += info.g;

    return info;
}
