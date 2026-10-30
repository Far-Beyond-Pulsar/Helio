// fft_evolve.wgsl — advance h0(k) to time t and emit the spectra that the
// inverse FFT turns into displacement and slopes. Two real fields are packed
// into one complex spectrum, so IFFT(A + iB) = a + ib.

struct Fft {
    lengths: vec4<f32>,
    cut:     vec4<f32>,
    time: f32, cap_fade: f32, lambda: f32, dt: f32,
    foam_recovery: f32, seed: u32, sub: i32, vertical: i32,
    cascade: i32, level: i32, _p0: i32, _p1: i32,
}
@group(0) @binding(0) var h0_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> u: Fft;

const N: i32 = 256;
const PI: f32 = 3.14159265359;
const G: f32 = 9.81;
const ST: f32 = 7.4e-5;
const LOOP: f32 = 300.0;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn omega(k: f32) -> f32 { return sqrt(G * k + ST * k * k * k); }
fn pack2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x - b.y, a.y + b.x); }

struct Out {
    @location(0) o0: vec4<f32>,
    @location(1) o1: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> Out {
    let p = vec2<i32>(pos.xy);
    let c = p.y / N;
    let k = vec2<f32>(vec2<i32>(p.x, p.y - c * N) - vec2<i32>(N / 2)) * (2.0 * PI / u.lengths[c]);
    let kl = length(k);
    let ik = select(0.0, 1.0 / kl, kl > 1e-6);
    // w = n * W0, so the sea loops every LOOP seconds and the phase stays small
    let w0 = 2.0 * PI / LOOP;
    let ph = 2.0 * PI * fract(floor(omega(kl) / w0) * (u.time / LOOP));
    let e = vec2<f32>(cos(ph), sin(ph));
    let s = textureLoad(h0_tex, p, 0);
    let h = cmul(s.xy, e) + cmul(s.zw, vec2<f32>(e.x, -e.y));
    let ih = vec2<f32>(-h.y, h.x);

    let dx = ih * k.x * ik;
    let dz = ih * k.y * ik;
    let dxdx = -h * k.x * k.x * ik;
    let dzdz = -h * k.y * k.y * ik;
    let dxdz = -h * k.x * k.y * ik;
    let dydx = ih * k.x;
    let dydz = ih * k.y;

    var o: Out;
    o.o0 = vec4<f32>(pack2(dx, dz), pack2(h, dxdz));
    o.o1 = vec4<f32>(pack2(dydx, dydz), pack2(dxdx, dzdz));
    return o;
}
