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
    /// Wave spring constant (scales wave propagation speed in SWE).
    spring: f32,
    /// Per-step energy damping multiplier (0..1).
    damping: f32,
    /// Wind direction in XZ sim-texture space (pre-normalised; zero = no wind).
    wind_dir: vec2<f32>,
    /// Wind strength multiplier. 0 = calm; ~1 = gentle ripples; ~5 = choppy.
    wind_strength: f32,
    /// Elapsed simulation time -- drives traveling wave phases.
    time: f32,
    /// Wave scale: smaller = shorter wavelengths (chop); larger = long swells.
    wave_scale: f32,
    /// Fixed sim-step duration for stable injection magnitude regardless of wave_speed.
    time_step: f32,
}
@group(0) @binding(2) var<uniform> u: UpdateUniforms;

// ---------------------------------------------------------------------------
// Wind: traveling sinusoidal wave trains, simplified JONSWAP-inspired spectrum.
//
// Each octave is a plane wave W(uv, t) = sin(dot(uv, dir) * k - omega * t).
// We inject the delta  W(t_old) - W(t_new)  directly into info.r, identical to
// the hitbox.frag.wgsl sign convention.  This is the wave's own time-derivative
// ( ~= omega * dt * cos(...) ), which the SWE spring propagates into radiating
// rings.  Spatial mean of each sin() term is 0 -- no DC height drift.
//
// Octave spread: primary swell in wind direction; secondary at +18 deg; cross-
// chop at -30 deg; short ripples at +50 deg.  Amplitudes follow ~1/n^1.5 to
// match the high-frequency roll-off of a real ocean spectrum.
// ---------------------------------------------------------------------------
fn twave(uv: vec2<f32>, t: f32, k: f32, omega: f32, dir: vec2<f32>) -> f32 {
    return sin(dot(uv, dir) * k - omega * t);
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
    // Euler-integrate height
    info.r += info.g;

    // Traveling wave injection -- only when wind is active and normalised.
    if u.wind_strength > 0.001 && dot(u.wind_dir, u.wind_dir) > 0.5 {
        let perp   = vec2<f32>(-u.wind_dir.y, u.wind_dir.x);
        // Smaller wave_scale = larger k = shorter wavelengths (choppy sea).
        // Larger wave_scale = smaller k = long swells.
        let inv_ws = 1.0 / max(u.wave_scale, 0.01);
        let t_old  = u.time - u.time_step;

        var dh = 0.0;

        // Octave 0 -- primary swell, strict wind direction (50% of energy)
        let k0 = 6.2832 * 1.5 * inv_ws;
        dh += (twave(uv, t_old,  k0, 0.65, u.wind_dir) -
               twave(uv, u.time, k0, 0.65, u.wind_dir)) * 0.50;

        // Octave 1 -- secondary swell +18 deg off wind
        let d1 = normalize(u.wind_dir + perp * 0.3249);   // tan(18 deg)
        let k1 = 6.2832 * 2.8 * inv_ws;
        dh += (twave(uv, t_old,  k1, 1.10, d1) -
               twave(uv, u.time, k1, 1.10, d1)) * 0.28;

        // Octave 2 -- cross-chop -30 deg
        let d2 = normalize(u.wind_dir - perp * 0.5774);   // tan(30 deg)
        let k2 = 6.2832 * 5.3 * inv_ws;
        dh += (twave(uv, t_old,  k2, 2.00, d2) -
               twave(uv, u.time, k2, 2.00, d2)) * 0.14;

        // Octave 3 -- short ripples +50 deg
        let d3 = normalize(u.wind_dir + perp * 1.1918);   // tan(50 deg)
        let k3 = 6.2832 * 9.5 * inv_ws;
        dh += (twave(uv, t_old,  k3, 3.60, d3) -
               twave(uv, u.time, k3, 3.60, d3)) * 0.08;

        info.r += dh * u.wind_strength * 0.05;
    }

    return info;
}