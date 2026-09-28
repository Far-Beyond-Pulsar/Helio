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
