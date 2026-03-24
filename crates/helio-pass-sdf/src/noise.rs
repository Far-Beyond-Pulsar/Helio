//! CPU noise functions matching the WGSL shaders (for brick classification).

use super::terrain::TerrainConfig;
use glam::Vec3;

/// IQ-style hash: maps integer lattice coordinate to pseudo-random value in [0,1].
fn hash3(px: f32, py: f32, pz: f32) -> f32 {
    let mut q = Vec3::new(
        (px * 0.3183099 + 0.1).fract(),
        (py * 0.3183099 + 0.1).fract(),
        (pz * 0.3183099 + 0.1).fract(),
    ) * 17.0;
    (q.x * q.y * q.z * (q.x + q.y + q.z)).fract()
}

/// 3D value noise with quintic interpolation. Returns value in [-1, 1].
fn noise3(px: f32, py: f32, pz: f32) -> f32 {
    let ix = px.floor();
    let iy = py.floor();
    let iz = pz.floor();
    let fx = px - ix;
    let fy = py - iy;
    let fz = pz - iz;

    // Quintic interpolant.
    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
    let uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0);

    let a = hash3(ix, iy, iz);
    let b = hash3(ix + 1.0, iy, iz);
    let c = hash3(ix, iy + 1.0, iz);
    let d = hash3(ix + 1.0, iy + 1.0, iz);
    let e = hash3(ix, iy, iz + 1.0);
    let f = hash3(ix + 1.0, iy, iz + 1.0);
    let g = hash3(ix, iy + 1.0, iz + 1.0);
    let h = hash3(ix + 1.0, iy + 1.0, iz + 1.0);

    let val = lerp(
        lerp(lerp(a, b, ux), lerp(c, d, ux), uy),
        lerp(lerp(e, f, ux), lerp(g, h, ux), uy),
        uz,
    );
    val * 2.0 - 1.0
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// IQ domain rotation (matches WGSL fbm_rotate). Base lacunarity scale.
fn fbm_rotate(p: Vec3, lac: f32) -> Vec3 {
    let r0 = Vec3::new(0.00, -0.80, -0.60);
    let r1 = Vec3::new(0.80, 0.36, -0.48);
    let r2 = Vec3::new(0.60, -0.48, 0.64);
    lac * Vec3::new(
        p.dot(Vec3::new(r0.x, r1.x, r2.x)),
        p.dot(Vec3::new(r0.y, r1.y, r2.y)),
        p.dot(Vec3::new(r0.z, r1.z, r2.z)),
    )
}

/// 2D FBM evaluated at (x, 0, z) — CPU equivalent of the WGSL `terrain_fbm2`.
pub fn fbm2(x: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut max_amp = 0.0f32;
    let mut p = Vec3::new(x, 0.0, z);
    for _ in 0..octaves {
        value += amplitude * noise3(p.x, p.y, p.z);
        max_amp += amplitude;
        amplitude *= persistence;
        p = fbm_rotate(p, lacunarity);
    }
    if max_amp > 0.0 {
        value / max_amp
    } else {
        0.0
    }
}

/// Evaluate the terrain SDF at a world position — matches WGSL `terrain_sdf`.
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

/// Returns the (min_height, max_height) range of the terrain SDF within a brick AABB.
/// Used by `BrickMap` to decide whether a brick intersects the terrain surface.
pub fn terrain_height_range(
    brick_min: Vec3,
    brick_max: Vec3,
    config: &TerrainConfig,
) -> (f32, f32) {
    let samples = [
        Vec3::new(brick_min.x, 0.0, brick_min.z),
        Vec3::new(brick_max.x, 0.0, brick_min.z),
        Vec3::new(brick_min.x, 0.0, brick_max.z),
        Vec3::new(brick_max.x, 0.0, brick_max.z),
        Vec3::new(
            (brick_min.x + brick_max.x) * 0.5,
            0.0,
            (brick_min.z + brick_max.z) * 0.5,
        ),
    ];
    let heights: Vec<f32> = samples
        .iter()
        .map(|p| {
            let n = fbm2(
                p.x * config.frequency,
                p.z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            config.height + n * config.amplitude
        })
        .collect();
    let min_h = heights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_h = heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min_h, max_h)
}

