// fft_stage.wgsl — one Stockham radix-2 inverse FFT stage (e^{+i}), over all
// four stacked cascades at once. `u.vertical` picks the axis, `u.sub` the
// sub-transform size (2..N).

struct Fft {
    lengths: vec4<f32>,
    cut:     vec4<f32>,
    time: f32, cap_fade: f32, lambda: f32, dt: f32,
    foam_recovery: f32, seed: u32, sub: i32, vertical: i32,
    cascade: i32, level: i32, _p0: i32, _p1: i32,
}
@group(0) @binding(0) var in0: texture_2d<f32>;
@group(0) @binding(1) var in1: texture_2d<f32>;
@group(0) @binding(3) var<uniform> u: Fft;

const N: i32 = 256;
const PI: f32 = 3.14159265359;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

struct Out {
    @location(0) o0: vec4<f32>,
    @location(1) o1: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> Out {
    let p = vec2<i32>(pos.xy);
    let vertical = u.vertical == 1;
    let i = select(p.x, p.y % N, vertical);
    let hs = u.sub >> 1u;
    let e = (i / u.sub) * hs + (i & (hs - 1));
    var pe: vec2<i32>;
    var po: vec2<i32>;
    if vertical {
        let base = p.y - i;
        pe = vec2<i32>(p.x, base + e);
        po = vec2<i32>(p.x, base + e + N / 2);
    } else {
        pe = vec2<i32>(e, p.y);
        po = vec2<i32>(e + N / 2, p.y);
    }
    let a = 2.0 * PI * f32(i & (u.sub - 1)) / f32(u.sub);
    let tw = vec2<f32>(cos(a), sin(a));
    let e0 = textureLoad(in0, pe, 0);
    let o0 = textureLoad(in0, po, 0);
    let e1 = textureLoad(in1, pe, 0);
    let o1 = textureLoad(in1, po, 0);
    var o: Out;
    o.o0 = vec4<f32>(e0.xy + cmul(tw, o0.xy), e0.zw + cmul(tw, o0.zw));
    o.o1 = vec4<f32>(e1.xy + cmul(tw, o1.xy), e1.zw + cmul(tw, o1.zw));
    return o;
}
