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
    /// Wave spring constant -- restoring force toward the mean height.
    spring: f32,
    /// Per-step energy damping multiplier (0..1).
    damping: f32,
    /// Wind direction in XZ sim-texture space (pre-normalised; zero = no wind).
    wind_dir: vec2<f32>,
    /// Wind strength multiplier. 0 = calm; ~1 = gentle ripples; ~5 = choppy.
    wind_strength: f32,
    /// Elapsed simulation time (seconds) -- drives gust-centre trajectories.
    time: f32,
    /// Wave spatial scale. 1.0 = default size; 0.25 = quarter-size (fine ripples).
    wave_scale: f32,
    /// Elapsed time for one sim step (seconds). Controls gust-centre velocity and wave speed.
    time_step: f32,
}
@group(0) @binding(2) var<uniform> u: UpdateUniforms;

// ---------------------------------------------------------------------------
// Wind: Lissajous gust-centre trajectory (stays smoothly in UV [0, 1]^2)
// ---------------------------------------------------------------------------
fn gust_pos(t: f32, idx: i32, wind_dir: vec2<f32>) -> vec2<f32> {
    let fi = f32(idx);
    let fx = 0.17 + fi * 0.083;
    let fy = 0.21 + fi * 0.067;
    let px = fi * 1.2345;
    let py = fi * 2.3456 + 1.5708;
    let base = vec2<f32>(
        sin(t * fx + px) * 0.38 + 0.5,
        sin(t * fy + py) * 0.38 + 0.5,
    );
    return clamp(base + wind_dir * 0.05, vec2<f32>(0.05), vec2<f32>(0.95));
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

    // Wind: N traveling gust centres, hitbox-style conservative delta encoding.
    // wave_scale shrinks/enlarges the gust footprint: smaller = finer ripples.
    if u.wind_strength > 0.001 {
        let prev_t = u.time - u.time_step;
        let gust_radius = 0.10 * clamp(u.wave_scale, 0.05, 4.0);
        let scale = u.wind_strength * 0.10;

        for (var i: i32 = 0; i < 6; i++) {
            let p_new = gust_pos(u.time, i, u.wind_dir);
            let p_old = gust_pos(prev_t, i, u.wind_dir);

            let d_new = max(0.0, 1.0 - length(p_new - uv) / gust_radius);
            let d_old = max(0.0, 1.0 - length(p_old - uv) / gust_radius);
            let w_new = 0.5 - cos(d_new * 3.14159265) * 0.5;
            let w_old = 0.5 - cos(d_old * 3.14159265) * 0.5;

            info.r += (w_old - w_new) * scale;
        }
    }

    return info;
}