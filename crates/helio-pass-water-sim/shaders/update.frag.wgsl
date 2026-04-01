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
    /// Lower (~1.0) feels like fluid; higher (~2.0) feels jelly-like.
    spring: f32,
    /// Per-step energy damping multiplier (0..1).
    /// Closer to 1.0 = waves persist longer. Closer to 0.9 = waves die quickly.
    damping: f32,
    /// Wind direction in XZ sim-texture space (pre-normalised; zero = no wind).
    wind_dir: vec2<f32>,
    /// Wind strength multiplier. 0 = calm; ~1 = gentle ripples; ~5 = choppy.
    wind_strength: f32,
    /// Elapsed simulation time (seconds) -- drives gust-centre trajectories.
    time: f32,
}
@group(0) @binding(2) var<uniform> u: UpdateUniforms;

// ---------------------------------------------------------------------------
// Wind: Lissajous gust-centre trajectory (stays smoothly in UV [0, 1]^2)
// ---------------------------------------------------------------------------
fn gust_pos(t: f32, idx: i32, wind_dir: vec2<f32>) -> vec2<f32> {
    let fi = f32(idx);
    // Each source has a unique natural frequency and starting phase so they
    // spread across the surface and never synchronise.
    let fx = 0.17 + fi * 0.083;
    let fy = 0.21 + fi * 0.067;
    let px = fi * 1.2345;
    let py = fi * 2.3456 + 1.5708;
    let base = vec2<f32>(
        sin(t * fx + px) * 0.38 + 0.5,
        sin(t * fy + py) * 0.38 + 0.5,
    );
    // Slight persistent drift in wind direction so gusts feel directional
    // without dominating the Lissajous variation.
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
    //
    // Each gust centre follows a Lissajous trajectory across the surface.
    // Writing the height delta (old_weight - new_weight) rather than a raw
    // impulse is conservative: the net displacement integrated over the whole
    // surface is zero every step, so the global mean height never drifts.
    // This mirrors exactly how hitbox.frag.wgsl creates waves.
    if u.wind_strength > 0.001 {
        // One sim step back in time (2 steps/frame * 60 fps = ~120 steps/s).
        let prev_t = u.time - 0.00833;
        let gust_radius = 0.10;
        let scale = u.wind_strength * 0.10;

        for (var i: i32 = 0; i < 6; i++) {
            let p_new = gust_pos(u.time, i, u.wind_dir);
            let p_old = gust_pos(prev_t, i, u.wind_dir);

            // Cosine-bell weight: 0 outside radius, 1 at centre -- same profile
            // as drop.frag.wgsl so the excited waves look identical.
            let d_new = max(0.0, 1.0 - length(p_new - uv) / gust_radius);
            let d_old = max(0.0, 1.0 - length(p_old - uv) / gust_radius);
            let w_new = 0.5 - cos(d_new * 3.14159265) * 0.5;
            let w_old = 0.5 - cos(d_old * 3.14159265) * 0.5;

            // Same sign convention as hitbox.frag.wgsl:
            //   where gust was  -> height rises  (surface springs back up)
            //   where gust is   -> height falls   (gust drags surface down)
            info.r += (w_old - w_new) * scale;
        }
    }

    return info;
}