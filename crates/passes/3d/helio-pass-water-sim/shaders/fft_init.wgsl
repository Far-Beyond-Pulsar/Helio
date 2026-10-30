// fft_init.wgsl — the ocean's initial spectrum h0(k), four cascades stacked in a
// 256 x 1024 target. Ported from Ultra-Water's OceanSim.js (JONSWAP wind sea
// plus a long swell, Donelan-Banner directional spreading).
//
// Output (RGBA32F): xy = h0(k), zw = conj(h0(-k)).
//
// Re-run every frame: the spectrum follows water volume row 0 (its wind and
// wave amplitude), which only the GPU can read.

struct Fft {
    lengths: vec4<f32>,
    cut:     vec4<f32>,
    time: f32, cap_fade: f32, lambda: f32, dt: f32,
    foam_recovery: f32, seed: u32, sub: i32, vertical: i32,
    cascade: i32, level: i32, _p0: i32, _p1: i32,
}
@group(0) @binding(3) var<uniform> u: Fft;

struct Row { v: array<vec4<f32>, 16> }
@group(1) @binding(0) var<storage, read> rows: array<Row>;

const N: i32 = 256;
const PI: f32 = 3.14159265359;
const G: f32 = 9.81;
const ST: f32 = 7.4e-5;       // surface tension / density: w^2 = g k + (s/p) k^3

struct Spec {
    scale: f32, angle: f32, spread: f32, swell: f32,
    alpha: f32, peak: f32, gamma: f32, fade: f32,
}

fn omega(k: f32) -> f32 { return sqrt(G * k + ST * k * k * k); }

fn pcg2d(v: vec2<u32>) -> vec2<u32> {
    var x = v * 1664525u + 1013904223u;
    x.x += x.y * 1664525u; x.y += x.x * 1664525u;
    x = x ^ (x >> vec2<u32>(16u));
    x.x += x.y * 1664525u; x.y += x.x * 1664525u;
    x = x ^ (x >> vec2<u32>(16u));
    return x;
}

fn gauss(id: vec2<i32>, c: i32) -> vec2<f32> {
    let h = pcg2d(vec2<u32>(id) + vec2<u32>(u32(c) * 7919u, u32(c) * 104729u) + vec2<u32>(u.seed));
    let u1 = (f32(h.x) + 1.0) / 4294967296.0;
    let u2 = f32(h.y) / 4294967296.0;
    let r = sqrt(-2.0 * log(u1));
    return r * vec2<f32>(cos(2.0 * PI * u2), sin(2.0 * PI * u2));
}

fn jonswap(w: f32, sp: Spec) -> f32 {
    let sigma = select(0.09, 0.07, w <= sp.peak);
    let r = exp(-(w - sp.peak) * (w - sp.peak) / (2.0 * sigma * sigma * sp.peak * sp.peak));
    let iw = 1.0 / w;
    let q = sp.peak * iw;
    return sp.scale * sp.alpha * G * G * pow(iw, 5.0) * exp(-1.25 * q * q * q * q) * pow(sp.gamma, r);
}

fn normalisation(s: f32) -> f32 {
    let s2 = s * s; let s3 = s2 * s; let s4 = s3 * s;
    if s < 5.0 {
        return -0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * s + 0.163;
    }
    return -4.80e-08 * s4 + 1.07e-05 * s3 - 9.53e-04 * s2 + 5.90e-02 * s + 3.93e-01;
}

fn directional(theta: f32, w: f32, sp: Spec) -> f32 {
    let x = w / sp.peak;
    var s = select(6.97 * pow(x, 5.0), 9.77 * pow(x, -2.5), x > 1.0);
    s += 16.0 * tanh(min(x, 20.0)) * sp.swell * sp.swell;
    let t = theta - sp.angle;
    let cosine2s = normalisation(s) * pow(abs(cos(0.5 * t)), 2.0 * s);
    return mix(2.0 / PI * cos(t) * cos(t), cosine2s, sp.spread);
}

fn h0(id: vec2<i32>, c: i32, specs: array<Spec, 2>) -> vec2<f32> {
    let L = u.lengths[c];
    let dk = 2.0 * PI / L;
    let k = vec2<f32>(id - vec2<i32>(N / 2)) * dk;
    let kl = length(k);
    let lo = select(u.cut[max(c - 1, 0)], 1e-4, c == 0);
    let hi = select(u.cut[min(c, 2)], 1e4, c == 3);
    if kl < lo || kl > hi { return vec2<f32>(0.0); }
    let w = omega(kl);
    let dwdk = (G + 3.0 * ST * kl * kl) / (2.0 * w);
    let theta = atan2(k.y, k.x);
    var S = 0.0;
    for (var i = 0; i < 2; i++) {
        let sp = specs[i];
        if sp.scale <= 0.0 { continue; }
        let fade = select(sp.fade, u.cap_fade, c == 3);
        S += jonswap(w, sp) * directional(theta, w, sp) * exp(-fade * fade * kl * kl);
    }
    return gauss(id, c) * sqrt(2.0 * S * dwdk / kl * dk * dk);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // Row 0 sets the sea for every volume: waves tile in world space, so one
    // spectrum is shared. Wind falls back to the wave direction when calm.
    let row = rows[0];
    var dir = row.v[14].xy;
    if dot(dir, dir) < 1e-8 { dir = row.v[3].xy; }
    if dot(dir, dir) < 1e-8 { dir = vec2<f32>(1.0, 0.0); }
    let wind_angle = atan2(dir.y, dir.x);
    // Wind strength 0..10 -> 4..24 m/s; wave_amplitude 0.5 is the reference sea.
    let wind_speed = 4.0 + 2.0 * max(row.v[14].z, 0.0);
    let amp = clamp(row.v[2].x / 0.5, 0.05, 6.0);

    let fetch = 120000.0;
    var specs: array<Spec, 2>;
    specs[0] = Spec(
        amp * amp, wind_angle, 0.85, 0.25,
        0.076 * pow(G * fetch / (wind_speed * wind_speed), -0.22),
        22.0 * pow(wind_speed * fetch / (G * G), -0.33),
        3.3, 0.04);
    let swell_fetch = 900000.0;
    let swell_speed = 7.0;
    specs[1] = Spec(
        0.7 * amp * amp, wind_angle - 0.61, 1.0, 1.0,
        0.076 * pow(G * swell_fetch / (swell_speed * swell_speed), -0.22),
        22.0 * pow(swell_speed * swell_fetch / (G * G), -0.33),
        3.3, 0.04);

    let p = vec2<i32>(pos.xy);
    let c = p.y / N;
    let id = vec2<i32>(p.x, p.y - c * N);
    let a = h0(id, c, specs);
    let b = h0(vec2<i32>((N - id.x) % N, (N - id.y) % N), c, specs);
    return vec4<f32>(a, b.x, -b.y);
}
