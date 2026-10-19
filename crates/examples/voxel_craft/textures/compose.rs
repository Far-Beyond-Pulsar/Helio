//! Composing block faces from the pack's raw tiles: the turf tinted green,
//! grass fringes over dirt, and dense leaf tiles built from single sprigs.

use image::{Rgba, RgbaImage};

/// Multiplies every pixel's colour by `tint` (alpha is kept).
pub fn tint(image: &RgbaImage, tint: [f32; 3]) -> RgbaImage {
    let mut out = image.clone();
    for p in out.pixels_mut() {
        for c in 0..3 {
            p[c] = (f32::from(p[c]) * tint[c]).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// `top` over `bottom` (straight alpha); the result is opaque where `bottom` is.
pub fn over(bottom: &mut RgbaImage, top: &RgbaImage, dx: i32, dy: i32, rotation: u8) {
    let (w, h) = bottom.dimensions();
    for y in 0..h {
        for x in 0..w {
            // The top tile as rotated by `rotation` quarter turns and wrapped
            // around by the offset (tiles repeat).
            let (sx, sy) = ((x as i32 - dx).rem_euclid(w as i32) as u32, (y as i32 - dy).rem_euclid(h as i32) as u32);
            let (sx, sy) = match rotation % 4 {
                0 => (sx, sy),
                1 => (sy, w - 1 - sx),
                2 => (w - 1 - sx, h - 1 - sy),
                _ => (h - 1 - sy, sx),
            };
            let s = top.get_pixel(sx.min(top.width() - 1), sy.min(top.height() - 1));
            let a = f32::from(s[3]) / 255.0;
            if a <= 0.0 {
                continue;
            }
            let d = bottom.get_pixel_mut(x, y);
            for c in 0..3 {
                d[c] = (f32::from(s[c]) * a + f32::from(d[c]) * (1.0 - a)) as u8;
            }
            d[3] = 255;
        }
    }
}

/// The pack's turf, tinted.
pub fn grass_top(turf: &RgbaImage, green: [f32; 3]) -> RgbaImage {
    tint(turf, green)
}

/// Dirt with the pack's grass fringe (an overlay of alpha) tinted over it.
pub fn grass_side(dirt: &RgbaImage, overlay: &RgbaImage, green: [f32; 3]) -> RgbaImage {
    let mut out = dirt.clone();
    over(&mut out, &tint(overlay, green), 0, 0, 0);
    out
}

/// A cheap deterministic hash in [0, 1) for a pixel.
fn noise(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x.wrapping_mul(0x85eb_ca6b) ^ y.wrapping_mul(0xc2b2_ae35) ^ seed.wrapping_mul(0x27d4_eb2f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65536.0
}

/// A dense leaf block from one sprig: a dark mottled canopy floor with the
/// sprig laid over it several times, turned and shifted, then tinted.
pub fn leaves(sprig: &RgbaImage, green: [f32; 3], seed: u32) -> RgbaImage {
    let (w, h) = sprig.dimensions();
    // The sprig's mean colour, darkened, is the shade between leaves.
    let (mut sum, mut n) = ([0.0f32; 3], 0.0f32);
    for p in sprig.pixels().filter(|p| p[3] > 128) {
        for c in 0..3 {
            sum[c] += f32::from(p[c]);
        }
        n += 1.0;
    }
    let mean = if n > 0.0 { [sum[0] / n, sum[1] / n, sum[2] / n] } else { [60.0, 90.0, 40.0] };
    let mut out = RgbaImage::from_fn(w, h, |x, y| {
        let shade = 0.26 + 0.22 * noise(x / 3, y / 3, seed) + 0.1 * noise(x, y, seed ^ 0x55);
        Rgba([(mean[0] * shade) as u8, (mean[1] * shade) as u8, (mean[2] * shade) as u8, 255])
    });
    // Laid over in a fixed order so the tile is the same on every run.
    let layers: [(i32, i32, u8); 6] = [(0, 0, 0), (w as i32 / 2, h as i32 / 3, 2), (w as i32 / 3, -(h as i32) / 2, 1), (-(w as i32) / 3, h as i32 / 2, 3), (w as i32 / 5, h as i32 / 5, 1), (-(w as i32) / 4, -(h as i32) / 4, 2)];
    for (i, (dx, dy, rotation)) in layers.into_iter().enumerate() {
        // Later sprigs are lit brighter: depth in the canopy.
        let lift = 0.72 + 0.1 * i as f32;
        over(&mut out, &tint(sprig, [lift; 3]), dx, dy, rotation);
    }
    tint(&out, green)
}

/// The mean colour of a tile's opaque pixels (sRGB bytes), for flat fallbacks.
pub fn average(image: &RgbaImage) -> [f32; 3] {
    let (mut sum, mut n) = ([0.0f32; 3], 0.0f32);
    for p in image.pixels().filter(|p| p[3] > 0) {
        for c in 0..3 {
            sum[c] += f32::from(p[c]);
        }
        n += 1.0;
    }
    if n == 0.0 {
        return [0.5; 3];
    }
    [sum[0] / n / 255.0, sum[1] / n / 255.0, sum[2] / n / 255.0]
}
