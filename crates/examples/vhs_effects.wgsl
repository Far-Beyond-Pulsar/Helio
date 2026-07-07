// User shader snippet injected into the post-process pipeline.
// Uses noise_tex, noise_samp, and pp_custom from the core bindings.
//
// v2 changes vs original, aimed at "found VHS footage" realism rather than
// a generic retro filter:
//   1. Lens identity: barrel distortion + vignette + edge chromatic aberration
//      (cheap camcorder wide-angle glass, not a flat digital frame)
//   2. CCD highlight bloom/smear: bright sources streak vertically, a classic
//      tell of consumer CCD sensors that's otherwise completely absent
//   3. Grain now samples noise_tex with a slow scroll instead of pure hash
//      noise -> spatially/temporally correlated grain instead of TV static
//   4. Head-switching noise bar near the bottom of frame (real decks show
//      this on every single frame - previously missing entirely)
//   5. Rare, severe tracking tears layered on top of the existing smooth
//      jitter, so damage reads as bursty/eventful instead of constant
// Everything from the original (YIQ separation, chroma phase drift, rolling
// scanline error, per-line jitter, crushed blacks/highlights, dot crawl,
// flicker) is kept - it was solid and just needed a physical frame around it.
//
fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 127.1) * 43758.5453123);
}

fn yiq2rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(1.0,  0.956,  0.621)),
        dot(c, vec3<f32>(1.0, -0.272, -0.647)),
        dot(c, vec3<f32>(1.0, -1.106,  1.703)),
    );
}

fn rgb2yiq(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.299, 0.587, 0.114)),
        dot(c, vec3<f32>(0.596, -0.274, -0.322)),
        dot(c, vec3<f32>(0.211, -0.523, 0.312)),
    );
}

fn blur_ring(uv: vec2<f32>, radius: f32, n: u32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    let inv = 1.0 / f32(n);
    for (var i = 0u; i < n; i++) {
        let a = 6.2832 * f32(i) * inv;
        acc += textureSampleLevel(hdr_input, linear_samp, uv + vec2<f32>(cos(a), sin(a)) * radius, 0.0).rgb;
    }
    return acc * inv;
}

// Cheap camcorder wide-angle glass: mild barrel distortion around center.
fn barrel_distort(uv: vec2<f32>, amount: f32) -> vec2<f32> {
    let c = uv - vec2<f32>(0.5, 0.5);
    let r2 = dot(c, c);
    return uv + c * r2 * amount;
}

// Vertical bloom/smear from bright sources - classic cheap-CCD artifact.
// Samples upward along the column and lets bright pixels "bleed" up the frame.
fn ccd_smear(uv: vec2<f32>, px: vec2<f32>) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = 1u; i < 6u; i++) {
        let o = f32(i) * px.y * 1.5;
        let s = textureSampleLevel(hdr_input, linear_samp, uv - vec2<f32>(0.0, o), 0.0).rgb;
        let luma = dot(s, vec3<f32>(0.299, 0.587, 0.114));
        let bright = smoothstep(0.92, 1.0, luma); // was 0.75 - way too easy to trigger on lit scenes
        let w = bright / f32(i);
        acc += s * w;
        wsum += w;
    }
    if wsum > 0.0001 {
        return clamp(acc / wsum, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec3<f32>(0.0);
}

// Single tracking-tear event generator. Called twice with different periods/
// seeds from user_effects so tears don't share one clock - see call site.
fn tear_pass(time: f32, line: f32, period: f32, seed_base: f32) -> f32 {
    let event_seed = floor(time / period);
    let event_roll = hash11(event_seed + seed_base);
    let event_active = step(0.55, event_roll); // more frequent than before (was 0.72)
    let event_phase = fract(time / period);
    let hold = 0.05 + hash11(event_seed + seed_base + 0.5) * 0.06;
    let window = step(0.0, event_phase) * (1.0 - step(hold, event_phase)); // hard on/off, no ease
    let band_seed = event_seed * 13.7 + floor(line / 18.0) + seed_base;
    let band_active = step(0.5, hash11(band_seed));
    return (hash11(band_seed + 0.33) * 2.0 - 1.0) * 40.0 * event_active * window * band_active;
}

fn user_effects(color: vec3<f32>, uv_in: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {
    let tape_jitter  = pp_custom[0].y;
    let jitter_freq  = pp_custom[0].z;
    let flicker_amt  = pp_custom[0].w;
    let noise_amt    = pp_custom[1].x;
    let time         = pp_custom[1].y;

    let px = 1.0 / dims;
    let frame = floor(time * 60.0);

    // ── Lens: barrel distortion up front, everything downstream samples through it ──
    let uv = barrel_distort(uv_in, 0.06);

    let line = uv.y * dims.y;

    // ── Vertical bounce (VCR servo instability) ───────────────────────────
    let v_bounce = sin(time * 1.73) * 0.5 + sin(time * 3.11) * 0.25;
    let vu = uv + vec2<f32>(0.0, v_bounce * px.y);

    // ── Rolling scanline offset error ─────────────────────────────────────
    let roll_pos = fract(time * 0.025 + 0.3);
    let roll_width = 0.06;
    let roll_dist = abs(vu.y - roll_pos);
    let roll_weight = 1.0 - smoothstep(0.0, roll_width, roll_dist);
    let roll_shift = sin(line * 50.0 + time * 2.0) * roll_weight * 2.5;

    // ── Per-scanline tape jitter (continuous, low-amplitude "always there") ──
    let jit = (sin(line * jitter_freq + time * 3.7) * 5.0
             + sin(line * 17.0 + time * 5.3) * 2.0) * tape_jitter;

    // ── Rare severe tracking tear ──────────────────────────────────────────
    // Bursty, not periodic. Two independent event generators (different
    // periods/seeds, see tear_pass above) so tears don't all share one clock
    // - stacking them makes the timing feel genuinely random rather than
    // "every N seconds." Envelope is a hard rectangular pulse, not a
    // smoothstep ease: real tracking tears snap instantly to full
    // displacement and hold, they don't slide/animate into position.
    let tear_shift = tear_pass(time, line, 1.6, 0.0) + tear_pass(time, line, 2.3, 91.0);

    let ju = vu + vec2<f32>((jit + roll_shift + tear_shift) * px.x, 0.0);

    // ── YIQ separation with per-channel blur ─────────────────────────────
    let yuv = blur_ring(ju, 0.5 * px.x, 5u);
    let y = rgb2yiq(yuv).x;

    let i_uv = ju + vec2<f32>(0.6 * px.x, 0.0);
    let i_base = rgb2yiq(blur_ring(i_uv, 3.0 * px.x, 9u)).y;

    let q_uv = ju + vec2<f32>(1.2 * px.x, 0.0);
    let q_base = rgb2yiq(blur_ring(q_uv, 1.5 * px.x, 9u)).z;

    // ── Chroma phase drift (VHS color decoder instability) ────────────────
    let phase = sin(time * 0.37) * 0.08 + sin(time * 0.73) * 0.04;
    let cp = cos(phase);
    let sp = sin(phase);
    let i = i_base * cp - q_base * sp;
    let q = i_base * sp + q_base * cp;

    var result = yiq2rgb(vec3<f32>(y, i, q));

    // ── CCD highlight bloom/smear ──────────────────────────────────────────
    let smear = ccd_smear(ju, px);
    result += smear * 0.25;

    // ── Rolling glitch noise bands ────────────────────────────────────────
    // Each band now gets its own independent speed, spacing, opacity, and
    // on/off state (driven by per-band hashed values) instead of all 4
    // bands sitting at a fixed 0.25 interval and scrolling in lockstep.
    // The old version was perfectly evenly-spaced and synchronized, which
    // reads as a mechanical grid rather than tape dropout.
    for (var g = 0u; g < 4u; g++) {
        let gf = f32(g);
        let g_speed  = 0.008 + hash11(gf * 3.1 + 1.0) * 0.03;
        let g_offset = hash11(gf * 7.7 + 2.0); // irregular starting position, not g*0.25
        let g_on_period = 2.0 + hash11(gf * 5.3 + 3.0) * 3.0;
        let g_on_seed = floor(time / g_on_period) + gf * 91.7;
        let g_active = step(0.45, hash11(g_on_seed)); // band randomly absent this window
        let g_opacity = 0.4 + hash11(gf * 2.3 + 4.0) * 0.4;

        let gp = fract(g_offset + time * g_speed);
        let gd = abs(vu.y - gp);
        let gw = (1.0 - smoothstep(0.0, 0.02 + hash11(gf) * 0.015, gd)) * g_active;
        if gw > 0.005 {
            let gn = hash21(vec2<f32>(line + gf * 100.0, frame));
            result = mix(result, vec3<f32>(gn * 0.4 + 0.1), gw * g_opacity);
        }
    }

    // ── Head-switching noise bar ────────────────────────────────────────────
    // Every real VHS frame shows a thin noisy strip near the bottom edge
    // where the video heads switch. Fixed position, subtle roll, always present.
    let hs_pos = 0.965 + sin(time * 0.6) * 0.006;
    let hs_dist = uv.y - hs_pos;
    let hs_band = smoothstep(-0.02, 0.0, hs_dist) * (1.0 - smoothstep(0.0, 0.035, hs_dist));
    if hs_band > 0.001 {
        let noise_dims_hs = vec2<f32>(textureDimensions(noise_tex));
        let hn_px = vec2<f32>(uv.x * dims.x * 0.4 + time * 90.0, frame * 0.7);
        let hn = textureSampleLevel(noise_tex, noise_samp, hn_px / noise_dims_hs, 0.0).r;
        result = mix(result, vec3<f32>(hn), hs_band * 0.85);
        result += vec2<f32>(hash11(line + frame), 0.0).x * hs_band * 0.15;
    }

    // ── VHS lighting: crushed blacks, clipped highlights ─────────────────
    result = pow(max(result, vec3<f32>(0.0)), vec3<f32>(1.4));
    result = 1.0 - pow(max(1.0 - result, vec3<f32>(0.0)), vec3<f32>(1.6));
    result = mix(result, result * result * result, 0.15);

    // ── Scanlines ─────────────────────────────────────────────────────────
    let scan = sin(uv.y * dims.y * 3.14159);
    result *= 1.0 - 0.06 * (1.0 - scan * scan);

    // ── Textured grain (replaces pure hash noise) ──────────────────────────
    // Sampling an actual noise texture with a slow per-axis scroll gives
    // grain that stays spatially coherent frame-to-frame instead of
    // reshuffling every pixel every frame like white TV static.
    //
    // IMPORTANT: tile at the noise texture's *native texel size*, not an
    // arbitrary constant. The previous version scaled by a magic 0.01 with
    // no relation to the texture's actual resolution, which aliased into a
    // small number of large repeating bands across the screen instead of
    // fine grain (this was the "8 evenly-spaced stripes" bug).
    let noise_dims = vec2<f32>(textureDimensions(noise_tex));
    // Per-frame random jump instead of continuous scroll: continuous scroll
    // makes the whole grain pattern slide across screen as one rigid block.
    // Jumping to a new random offset every frame keeps the same spatial
    // correlation *within* a frame (so it still looks like grain, not white
    // noise) while decorrelating frame-to-frame so it flickers/sparkles
    // instead of sliding.
    let jump = vec2<f32>(hash11(frame * 1.7 + 0.3), hash11(frame * 2.3 + 5.1)) * noise_dims;
    let grain_px = uv * dims + jump;
    let grain_uv = grain_px / noise_dims;
    let tex_grain = textureSampleLevel(noise_tex, noise_samp, grain_uv, 0.0).r;
    let px_id = floor(uv * dims);
    let seed = frame + hash21(px_id) * 1000.0;
    let r1 = hash21(vec2<f32>(seed + 1.0, seed * 0.3 + 2.0));

    let luma = dot(result, vec3<f32>(0.299, 0.587, 0.114));
    let noise_strength = (0.015 + 0.03 * (1.0 - luma)) * noise_amt;
    let grain = (tex_grain * 2.0 - 1.0) * noise_strength;
    result += grain;

    // ── Chroma noise ──────────────────────────────────────────────────────
    // Rebalanced: the green coefficient previously had the largest magnitude
    // AND always took the opposite sign from red/blue, so any time cn went
    // negative, green spiked harder than the other channels and scaled up
    // in dark areas (most of this scene) - that was the visible green
    // speckle artifact. Smaller, more even coefficients here.
    let cn = (r1 * 2.0 - 1.0) * 0.018 * (1.0 - luma) * noise_amt;
    result += vec3<f32>(cn * 0.35, cn * -0.2, cn * 0.4);

    // ── Dot crawl on sharp edges ──────────────────────────────────────────
    let edge = length(fwidth(result));
    let crawl = sin(uv.x * dims.x * 0.5 + time * 50.0) * edge * 0.06;
    result += vec3<f32>(crawl * 0.4, -crawl * 0.25, crawl * 0.6);

    // ── Edge chromatic aberration (lens, distinct from tape chroma error) ──
    let center_dist = length(uv - vec2<f32>(0.5, 0.5));
    let ca_amt = center_dist * center_dist * 0.0025;
    let ca_r = textureSampleLevel(hdr_input, linear_samp, ju + vec2<f32>(ca_amt, 0.0), 0.0).r;
    let ca_b = textureSampleLevel(hdr_input, linear_samp, ju - vec2<f32>(ca_amt, 0.0), 0.0).b;
    result = mix(result, vec3<f32>(ca_r, result.g, ca_b), 0.6);

    // ── Flicker ───────────────────────────────────────────────────────────
    let fl = hash21(vec2<f32>(frame * 0.01, 0.5));
    result *= 1.0 - 0.025 * flicker_amt * (fl * 2.0 - 1.0);

    // ── Vignette (camcorder lens falloff) ──────────────────────────────────
    let vig = 1.0 - smoothstep(0.35, 0.85, center_dist) * 0.4;
    result *= vig;

    return clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));
}
