//! Bit-exact integer noise, mirrored by `shaders/noise.wgsl`.
//!
//! Terrain generators build their fields from these primitives so the CPU
//! (collision, ray casts, gameplay queries) and the GPU (streaming,
//! rendering) agree to the bit. Every operation is wrapping two's-complement
//! integer arithmetic; WGSL names are the same except `lerp` (`lerp_q16`)
//! and `scale` (`scale_q16`).
use glam::IVec3;

/// Noise unit (Q16, `NOISE_ONE` in WGSL): [`noise`] returns values in
/// about `[-ONE, ONE]`.
pub const ONE: i32 = 65_536;

/// Integer hash of a lattice point.
#[inline]
pub fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut v = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f)
        ^ seed;
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846c_a68b);
    v ^ (v >> 16)
}

#[inline]
fn grad(hash: u32, x: i32, y: i32, z: i32) -> i32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { u.wrapping_neg() }).wrapping_add(if h & 2 == 0 {
        v
    } else {
        v.wrapping_neg()
    })
}

/// `a * w >> 16` for `|a| < 2^19`, `0 <= w <= 65536` with 32-bit intermediates.
#[inline]
pub fn mul16(a: i32, w: i32) -> i32 {
    a.wrapping_mul(w >> 8)
        .wrapping_add(a.wrapping_mul(w & 255) >> 8)
        >> 8
}

#[inline]
fn fade(t: i32) -> i32 {
    // 6t^5 - 15t^4 + 10t^3 in Q16.
    let tu = t as u32;
    let t2 = tu.wrapping_mul(tu) >> 16;
    let t3 = t2.wrapping_mul(tu) >> 16;
    let inner = (6 * t2 as i32).wrapping_sub(15 * t).wrapping_add(10 * ONE);
    mul16(t3 as i32, inner)
}

/// `a + (b - a) * w` for a Q16 weight `w` in `[0, ONE]`.
#[inline]
pub fn lerp(a: i32, b: i32, w: i32) -> i32 {
    a.wrapping_add(mul16(b.wrapping_sub(a), w))
}

/// Gradient noise on a lattice of spacing `2^shift` domain units, with
/// 16-bit fractions and output in about `[-ONE, ONE]`. Its gradient is at
/// most about 5.3 `ONE` per lattice spacing.
pub fn noise(p: IVec3, shift: u32, seed: u32) -> i32 {
    let mask = (1i32 << shift) - 1;
    let c = [p.x >> shift, p.y >> shift, p.z >> shift];
    let f = [p.x & mask, p.y & mask, p.z & mask].map(|v| {
        if shift >= 16 {
            v >> (shift - 16)
        } else {
            v << (16 - shift)
        }
    });
    let w = f.map(fade);
    let corner = |dx: i32, dy: i32, dz: i32| {
        grad(
            hash3(c[0].wrapping_add(dx), c[1].wrapping_add(dy), c[2].wrapping_add(dz), seed),
            f[0] - dx * ONE,
            f[1] - dy * ONE,
            f[2] - dz * ONE,
        )
    };
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), w[0]);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), w[0]);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), w[0]);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), w[0]);
    let y0 = lerp(x00, x10, w[1]);
    let y1 = lerp(x01, x11, w[1]);
    lerp(y0, y1, w[2]).clamp(-ONE, ONE)
}

/// `n * amplitude / ONE` without overflow (`|n| <= 2^17`, `|amplitude| < 2^27`).
#[inline]
pub fn scale(n: i32, amplitude: i32) -> i32 {
    n.wrapping_mul(amplitude >> 16)
        .wrapping_add(n.wrapping_mul((amplitude & 0xffff) >> 4) >> 12)
}

/// Unit of the fine noise (Q24, `FINE_ONE` in WGSL).
pub const FINE_ONE: i32 = 1 << 24;

/// `a * b / 2^24` rounded toward zero, for `|a| < 2^28` and `|b| <= 2^24`,
/// in 32-bit arithmetic (12-bit limbs; within two units of exact).
#[inline]
pub fn mul_fine(a: i32, b: i32) -> i32 {
    let negative = (a < 0) != (b < 0);
    let (a, b) = (a.unsigned_abs(), b.unsigned_abs());
    let (ah, al) = (a >> 12, a & 0xfff);
    let (bh, bl) = (b >> 12, b & 0xfff);
    let m = (ah * bh + ((ah * bl + al * bh + ((al * bl) >> 12)) >> 12)) as i32;
    if negative { -m } else { m }
}

#[inline]
fn fade_q24(t: i32) -> i32 {
    let t2 = mul_fine(t, t);
    let t3 = mul_fine(t2, t);
    mul_fine(6 * t2 - 15 * t + 10 * FINE_ONE, t3)
}

/// [`noise`] with 24-bit fractions and output in about `[-FINE_ONE,
/// FINE_ONE]` (about 256 times [`noise`]). Low-frequency octaves whose
/// value scales large quantities (continents, mountain regions, domain
/// warp) use it: with 16-bit fractions they would be constant over tens of
/// metres and step between neighbouring cells.
pub fn noise_fine(p: IVec3, shift: u32, seed: u32) -> i32 {
    let mask = (1i32 << shift) - 1;
    let c = [p.x >> shift, p.y >> shift, p.z >> shift];
    let f = [p.x & mask, p.y & mask, p.z & mask].map(|v| {
        if shift >= 24 {
            v >> (shift - 24)
        } else {
            v << (24 - shift)
        }
    });
    let w = f.map(fade_q24);
    let corner = |dx: i32, dy: i32, dz: i32| {
        grad(
            hash3(c[0].wrapping_add(dx), c[1].wrapping_add(dy), c[2].wrapping_add(dz), seed),
            f[0] - dx * FINE_ONE,
            f[1] - dy * FINE_ONE,
            f[2] - dz * FINE_ONE,
        )
    };
    let lerp = |a: i32, b: i32, w: i32| a + mul_fine(b - a, w);
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), w[0]);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), w[0]);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), w[0]);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), w[0]);
    let y0 = lerp(x00, x10, w[1]);
    let y1 = lerp(x01, x11, w[1]);
    lerp(y0, y1, w[2]).clamp(-FINE_ONE, FINE_ONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_bounded_and_continuous_at_lattice_points() {
        let mut max = 0;
        for x in -300..300 {
            let p = IVec3::new(x * 37, x * -11 + 5, x * 3);
            let v = noise(p, 6, 99);
            max = max.max(v.abs());
            let a = noise(IVec3::new(64 * x, 0, 0), 6, 5);
            let b = noise(IVec3::new(64 * x + 1, 0, 0), 6, 5);
            assert!((a - b).abs() < 6400, "{a} {b}");
        }
        assert!(max > 16000 && max <= 65536, "{max}");
    }

    #[test]
    fn fine_noise_matches_noise_and_never_steps() {
        for shift in [8u32, 16, 22, 26, 29] {
            // Two domain units at the gradient bound (5.3 per lattice spacing)
            // plus rounding: the 16-bit noise steps far beyond this.
            let bound = (2.0 * 5.3 * f64::from(FINE_ONE) / f64::from(1u32 << shift)).ceil() as i32 + 8;
            for x in -2000..2000 {
                // Mid-cell: at a lattice point the noise may be flat along x.
                let p = IVec3::new((1 << (shift - 1)) + x * 2 + 1, 99_614_720, (1 << (shift - 2)) - 81);
                let fine = noise_fine(p, shift, 7);
                let coarse = noise(p, shift, 7) * 256;
                // Agrees with the 16-bit noise to within its truncation error.
                assert!((fine - coarse).abs() <= 16 * 256, "shift {shift} x {x}: fine {fine} coarse {coarse}");
                let step = (noise_fine(p + IVec3::X * 2, shift, 7) - fine).abs();
                assert!(step <= bound, "shift {shift} x {x}: step {step} > {bound}");
            }
        }
        assert_eq!(mul_fine(-(1 << 27), 1 << 23), -(1 << 26));
        assert_eq!(mul_fine(12_345_678, FINE_ONE), 12_345_678);
    }

    #[test]
    fn noise_is_continuous_at_fine_steps() {
        for shift in [8u32, 12, 16, 19, 20, 24] {
            let mut worst = 0;
            for x in -600_000..-300_000 {
                let p = IVec3::new(x * 3 + 7, 99_614_720, -5_975_683);
                let d = (noise(p, shift, 12345) - noise(p + IVec3::X * 2, shift, 12345)).abs();
                worst = worst.max(d);
            }
            // Two domain units at the measured gradient bound (5.3 per lattice
            // spacing) plus rounding.
            let bound = (2.0 * 5.3 * 65536.0 / f64::from(1u32 << shift)).ceil() as i32 + 8;
            assert!(worst <= bound, "shift {shift}: {worst} > {bound}");
        }
    }
}
