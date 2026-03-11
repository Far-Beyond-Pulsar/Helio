//! CPU-side 3D value noise matching the WGSL implementation (IQ-style).
//!
//! Used for terrain-aware brick classification and CPU-side picking (ray march).
//! The noise algorithm matches `hash3` + `noise3` in the WGSL evaluate shaders.

use glam::Vec3;

use super::terrain::TerrainConfig;

// ─── 3D Value Noise (IQ-style, matching GPU) ────────────────────────────────

/// Hash function matching the WGSL `hash3()`.
/// Maps a lattice coordinate to a pseudo-random value in [0, 1].
fn hash3(px: f32, py: f32, pz: f32) -> f32 {
    let mut qx = (px * 0.3183099 + 0.1).fract();
    let mut qy = (py * 0.3183099 + 0.1).fract();
    let mut qz = (pz * 0.3183099 + 0.1).fract();
    // Handle negative fracts to match WGSL behavior (always positive)
    if qx < 0.0 { qx += 1.0; }
    if qy < 0.0 { qy += 1.0; }
    if qz < 0.0 { qz += 1.0; }
    qx *= 17.0;
    qy *= 17.0;
    qz *= 17.0;
    let v = qx * qy * qz * (qx + qy + qz);
    let r = v.fract();
    if r < 0.0 { r + 1.0 } else { r }
}

/// 3D value noise with quintic interpolation, matching WGSL `noise3()`.
/// Returns value in [-1, 1].
pub fn noise3(px: f32, py: f32, pz: f32) -> f32 {
    let ix = px.floor();
    let iy = py.floor();
    let iz = pz.floor();

    let fx = px - ix;
    let fy = py - iy;
    let fz = pz - iz;

    // Quintic interpolant: 6t^5 - 15t^4 + 10t^3
    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
    let uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0);

    // Sample 8 corners of the lattice cell
    let a = hash3(ix,       iy,       iz      );
    let b = hash3(ix + 1.0, iy,       iz      );
    let c = hash3(ix,       iy + 1.0, iz      );
    let d = hash3(ix + 1.0, iy + 1.0, iz      );
    let e = hash3(ix,       iy,       iz + 1.0);
    let f = hash3(ix + 1.0, iy,       iz + 1.0);
    let g = hash3(ix,       iy + 1.0, iz + 1.0);
    let h = hash3(ix + 1.0, iy + 1.0, iz + 1.0);

    // Trilinear interpolation
    fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
    let val = lerp(
        lerp(lerp(a, b, ux), lerp(c, d, ux), uy),
        lerp(lerp(e, f, ux), lerp(g, h, ux), uy),
        uz,
    );

    // Remap from [0,1] to [-1,1]
    val * 2.0 - 1.0
}

/// Domain rotation matrix (matching WGSL FBM_ROT constants).
/// Pure rotation component of IQ's rational-entry matrix (divided by 2 from original).
/// Multiply result by lacunarity to get the combined rotation + frequency scaling.
fn fbm_rotate(px: f32, py: f32, pz: f32, lac: f32) -> (f32, f32, f32) {
    // Row-major: same dot product order as WGSL
    // Row 0: (0.00, 0.80, 0.60)
    // Row 1: (-0.80, 0.36, -0.48)
    // Row 2: (-0.60, -0.48, 0.64)
    let rx = lac * ( 0.00 * px + 0.80 * py + 0.60 * pz);
    let ry = lac * (-0.80 * px + 0.36 * py - 0.48 * pz);
    let rz = lac * (-0.60 * px - 0.48 * py + 0.64 * pz);
    (rx, ry, rz)
}

/// 2D FBM (samples noise3 at y=0 plane).
pub fn fbm2(x: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut max_amp = 0.0;
    let mut sx = x;
    let mut sy = 0.0_f32;
    let mut sz = z;

    for _ in 0..octaves {
        value += amplitude * noise3(sx, sy, sz);
        max_amp += amplitude;
        amplitude *= persistence;
        let (rx, ry, rz) = fbm_rotate(sx, sy, sz, lacunarity);
        sx = rx;
        sy = ry;
        sz = rz;
    }

    value / max_amp
}

// ─── Terrain SDF evaluation (CPU side) ──────────────────────────────────────

/// Evaluate the terrain SDF at a world position using the given config.
/// Returns the signed distance (negative = inside terrain, positive = above).
pub fn terrain_sdf(pos: Vec3, config: &TerrainConfig) -> f32 {
    let n = fbm2(
        pos.x * config.frequency,
        pos.z * config.frequency,
        config.octaves,
        config.lacunarity,
        config.persistence,
    );
    pos.y - (config.height + n * config.amplitude)
}

/// Estimate the terrain height range within a brick's XZ footprint.
pub fn terrain_height_range(
    brick_min: Vec3,
    brick_max: Vec3,
    config: &TerrainConfig,
) -> (f32, f32) {
    let cx = (brick_min.x + brick_max.x) * 0.5;
    let cz = (brick_min.z + brick_max.z) * 0.5;

    let samples = [
        terrain_height_at(cx, cz, config),
        terrain_height_at(brick_min.x, brick_min.z, config),
        terrain_height_at(brick_max.x, brick_min.z, config),
        terrain_height_at(brick_min.x, brick_max.z, config),
        terrain_height_at(brick_max.x, brick_max.z, config),
    ];

    let min_h = samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_h = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let brick_size = brick_max.x - brick_min.x;
    let margin = config.amplitude * config.frequency * brick_size;
    (min_h - margin, max_h + margin)
}

fn terrain_height_at(x: f32, z: f32, config: &TerrainConfig) -> f32 {
    let n = fbm2(
        x * config.frequency,
        z * config.frequency,
        config.octaves,
        config.lacunarity,
        config.persistence,
    );
    config.height + n * config.amplitude
}
