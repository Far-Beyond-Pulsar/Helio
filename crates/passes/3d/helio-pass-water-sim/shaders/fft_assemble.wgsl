// fft_assemble.wgsl — turn the inverse-FFT result of one cascade into the
// textures the surface reads.
//
//   disp  (RGBA16F) = (lambda Dx, Dy, lambda Dz, turbulence)
//   slope (RGBA16F) = (sx, sz, sx^2, sz^2)
//
// Slopes are divided out at full resolution so their mips hold the mean and the
// second moment, whose difference is the slope variance (roughness). The
// turbulence channel is the foam: it follows the Jacobian of the horizontal
// displacement down at once (the surface pinches) and recovers at a fixed rate.

struct Fft {
    lengths: vec4<f32>,
    cut:     vec4<f32>,
    time: f32, cap_fade: f32, lambda: f32, dt: f32,
    foam_recovery: f32, seed: u32, sub: i32, vertical: i32,
    cascade: i32, level: i32, _p0: i32, _p1: i32,
}
@group(0) @binding(0) var in0: texture_2d<f32>;
@group(0) @binding(1) var in1: texture_2d<f32>;
@group(0) @binding(2) var prev: texture_2d<f32>;
@group(0) @binding(3) var<uniform> u: Fft;

const N: i32 = 256;

struct Out {
    @location(0) disp: vec4<f32>,
    @location(1) slope: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> Out {
    let id = vec2<i32>(pos.xy);
    let p = id + vec2<i32>(0, u.cascade * N);
    // undo the centred-k frequency shift
    let s = select(-1.0, 1.0, ((id.x + id.y) & 1) == 0);
    let a = textureLoad(in0, p, 0) * s;   // Dx, Dz, Dy, Dxz
    let b = textureLoad(in1, p, 0) * s;   // Dyx, Dyz, Dxx, Dzz
    let dxx = u.lambda * b.z;
    let dzz = u.lambda * b.w;
    let dxz = u.lambda * a.w;
    let jac = (1.0 + dxx) * (1.0 + dzz) - dxz * dxz;
    let slope = b.xy / max(vec2<f32>(1.0 + dxx, 1.0 + dzz), vec2<f32>(0.2));
    let last = textureLoad(prev, id, 0).w;
    let turb = min(jac, last + u.dt * u.foam_recovery / max(jac, 0.5));
    var o: Out;
    o.disp = vec4<f32>(u.lambda * a.x, a.z, u.lambda * a.y, turb);
    o.slope = vec4<f32>(slope, slope * slope);
    return o;
}
