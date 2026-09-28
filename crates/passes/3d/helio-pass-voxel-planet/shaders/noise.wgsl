// Bit-exact integer noise; mirror of src/noise.rs. Terrain programs build
// on these so CPU and GPU fields agree to the bit.
const NOISE_ONE: i32 = 65536; // noise unit (Q16)

fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    var v = (bitcast<u32>(x) * 0x8da6b343u) ^ (bitcast<u32>(y) * 0xd8163841u)
        ^ (bitcast<u32>(z) * 0xcb1ab31fu) ^ seed;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    return v ^ (v >> 16u);
}

fn grad(hash: u32, x: i32, y: i32, z: i32) -> i32 {
    let h = hash & 15u;
    let u = select(y, x, h < 8u);
    let v = select(select(z, x, h == 12u || h == 14u), y, h < 4u);
    return select(-u, u, (h & 1u) == 0u) + select(-v, v, (h & 2u) == 0u);
}

fn mul16(a: i32, w: i32) -> i32 {
    return (a * (w >> 8u) + ((a * (w & 255)) >> 8u)) >> 8u;
}

fn fade(t: i32) -> i32 {
    let tu = bitcast<u32>(t);
    let t2 = (tu * tu) >> 16u;
    let t3 = (t2 * tu) >> 16u;
    let inner = 6 * i32(t2) - 15 * t + 10 * NOISE_ONE;
    return mul16(i32(t3), inner);
}

fn lerp_q16(a: i32, b: i32, w: i32) -> i32 {
    return a + mul16(b - a, w);
}

fn to_q12(v: i32, shift: u32) -> i32 {
    if shift >= 16u { return v >> (shift - 16u); }
    return v << (16u - shift);
}

fn noise(p: vec3<i32>, shift: u32, seed: u32) -> i32 {
    let mask = (1 << shift) - 1;
    let c = p >> vec3<u32>(shift);
    let f = vec3<i32>(to_q12(p.x & mask, shift), to_q12(p.y & mask, shift), to_q12(p.z & mask, shift));
    let w = vec3<i32>(fade(f.x), fade(f.y), fade(f.z));
    let g000 = grad(hash3(c.x, c.y, c.z, seed), f.x, f.y, f.z);
    let g100 = grad(hash3(c.x + 1, c.y, c.z, seed), f.x - NOISE_ONE, f.y, f.z);
    let g010 = grad(hash3(c.x, c.y + 1, c.z, seed), f.x, f.y - NOISE_ONE, f.z);
    let g110 = grad(hash3(c.x + 1, c.y + 1, c.z, seed), f.x - NOISE_ONE, f.y - NOISE_ONE, f.z);
    let g001 = grad(hash3(c.x, c.y, c.z + 1, seed), f.x, f.y, f.z - NOISE_ONE);
    let g101 = grad(hash3(c.x + 1, c.y, c.z + 1, seed), f.x - NOISE_ONE, f.y, f.z - NOISE_ONE);
    let g011 = grad(hash3(c.x, c.y + 1, c.z + 1, seed), f.x, f.y - NOISE_ONE, f.z - NOISE_ONE);
    let g111 = grad(hash3(c.x + 1, c.y + 1, c.z + 1, seed), f.x - NOISE_ONE, f.y - NOISE_ONE, f.z - NOISE_ONE);
    let x00 = lerp_q16(g000, g100, w.x);
    let x10 = lerp_q16(g010, g110, w.x);
    let x01 = lerp_q16(g001, g101, w.x);
    let x11 = lerp_q16(g011, g111, w.x);
    let y0 = lerp_q16(x00, x10, w.y);
    let y1 = lerp_q16(x01, x11, w.y);
    return clamp(lerp_q16(y0, y1, w.z), -NOISE_ONE, NOISE_ONE);
}

fn scale_q16(n: i32, amplitude: i32) -> i32 {
    return n * (amplitude >> 16u) + ((n * ((amplitude & 0xffff) >> 4u)) >> 12u);
}

fn div_floor(a: i32, b: i32) -> i32 {
    let q = a / b;
    return select(q, q - 1, (a % b != 0) && ((a < 0) != (b < 0)));
}

fn rem_floor(a: i32, b: i32) -> i32 {
    let r = a % b;
    return select(r, r + b, r < 0);
}
