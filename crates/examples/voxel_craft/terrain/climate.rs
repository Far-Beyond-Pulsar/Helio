//! Deterministic hash noise for placing things: forest regions, tree mixes
//! and which spots in a region get a feature. Pure functions of the world
//! seed and a column, so a region decorates identically every time.

/// A well-mixed 64-bit hash of up to four integers and the seed.
pub fn hash(seed: u64, a: i64, b: i64, c: i64) -> u64 {
    let mut h = seed ^ 0x9e37_79b9_7f4a_7c15;
    for v in [a, b, c] {
        h ^= (v as u64).wrapping_add(0x9e37_79b9_7f4a_7c15).wrapping_add(h << 6).wrapping_add(h >> 2);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
    }
    h
}

/// Uniform in [0, 1) from a hash.
pub fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Smooth value noise in [0, 1) of a face's column at wavelength `scale` cells.
pub fn smooth(seed: u64, face: u8, i: i32, j: i32, scale: f64) -> f64 {
    let (x, y) = (f64::from(i) / scale, f64::from(j) / scale);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let corner = |dx: f64, dy: f64| unit(hash(seed, i64::from(face), (x0 + dx) as i64, (y0 + dy) as i64));
    let top = corner(0.0, 0.0) * (1.0 - sx) + corner(1.0, 0.0) * sx;
    let bottom = corner(0.0, 1.0) * (1.0 - sx) + corner(1.0, 1.0) * sx;
    top * (1.0 - sy) + bottom * sy
}

/// How wooded a column's surroundings are, 0..1 (large rolling regions).
pub fn forest_density(seed: u64, face: u8, i: i32, j: i32) -> f64 {
    let broad = smooth(seed ^ 0x51, face, i, j, 260.0);
    let local = smooth(seed ^ 0x52, face, i, j, 70.0);
    (broad * 0.75 + local * 0.25 - 0.28).max(0.0) * 1.8
}

/// A second field choosing the forest's kind, 0..1.
pub fn forest_kind(seed: u64, face: u8, i: i32, j: i32) -> f64 {
    smooth(seed ^ 0x77, face, i, j, 420.0)
}
