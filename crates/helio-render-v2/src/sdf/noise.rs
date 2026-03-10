//! CPU-side 3D simplex noise matching the WGSL implementation.
//!
//! Used for terrain-aware brick classification: determines which bricks
//! intersect the terrain surface without requiring GPU readback.

use glam::Vec3;

use super::terrain::{TerrainConfig, TerrainStyle};

// ─── 3D Simplex Noise ───────────────────────────────────────────────────────

/// Permutation table (256 entries, repeated).
const PERM: [u8; 512] = {
    const BASE: [u8; 256] = [
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
        140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
        247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
        57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
        74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
        60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
        65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
        200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
        52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
        207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
        119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
        129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
        218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
        81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
        184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
        222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    ];
    let mut out = [0u8; 512];
    let mut i = 0;
    while i < 512 {
        out[i] = BASE[i & 255];
        i += 1;
    }
    out
};

/// Gradient vectors for 3D simplex noise.
const GRAD3: [[f32; 3]; 12] = [
    [1.0,1.0,0.0], [-1.0,1.0,0.0], [1.0,-1.0,0.0], [-1.0,-1.0,0.0],
    [1.0,0.0,1.0], [-1.0,0.0,1.0], [1.0,0.0,-1.0], [-1.0,0.0,-1.0],
    [0.0,1.0,1.0], [0.0,-1.0,1.0], [0.0,1.0,-1.0], [0.0,-1.0,-1.0],
];

fn dot3(g: [f32; 3], x: f32, y: f32, z: f32) -> f32 {
    g[0] * x + g[1] * y + g[2] * z
}

/// 3D simplex noise. Returns value in [-1, 1].
pub fn simplex3(x: f32, y: f32, z: f32) -> f32 {
    const F3: f32 = 1.0 / 3.0;
    const G3: f32 = 1.0 / 6.0;

    let s = (x + y + z) * F3;
    let i = (x + s).floor() as i32;
    let j = (y + s).floor() as i32;
    let k = (z + s).floor() as i32;

    let t = (i + j + k) as f32 * G3;
    let x0 = x - (i as f32 - t);
    let y0 = y - (j as f32 - t);
    let z0 = z - (k as f32 - t);

    // Determine simplex we're in
    let (i1, j1, k1, i2, j2, k2);
    if x0 >= y0 {
        if y0 >= z0 {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        } else if x0 >= z0 {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
        } else {
            i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
        }
    } else {
        if y0 < z0 {
            i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
        } else if x0 < z0 {
            i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
        } else {
            i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        }
    }

    let x1 = x0 - i1 as f32 + G3;
    let y1 = y0 - j1 as f32 + G3;
    let z1 = z0 - k1 as f32 + G3;
    let x2 = x0 - i2 as f32 + 2.0 * G3;
    let y2 = y0 - j2 as f32 + 2.0 * G3;
    let z2 = z0 - k2 as f32 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let kk = (k & 255) as usize;

    let gi0 = (PERM[ii + PERM[jj + PERM[kk] as usize] as usize] % 12) as usize;
    let gi1 = (PERM[ii + i1 as usize + PERM[jj + j1 as usize + PERM[kk + k1 as usize] as usize] as usize] % 12) as usize;
    let gi2 = (PERM[ii + i2 as usize + PERM[jj + j2 as usize + PERM[kk + k2 as usize] as usize] as usize] % 12) as usize;
    let gi3 = (PERM[ii + 1 + PERM[jj + 1 + PERM[kk + 1] as usize] as usize] % 12) as usize;

    let mut n0 = 0.0;
    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if t0 > 0.0 {
        let t02 = t0 * t0;
        n0 = t02 * t02 * dot3(GRAD3[gi0], x0, y0, z0);
    }

    let mut n1 = 0.0;
    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if t1 > 0.0 {
        let t12 = t1 * t1;
        n1 = t12 * t12 * dot3(GRAD3[gi1], x1, y1, z1);
    }

    let mut n2 = 0.0;
    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if t2 > 0.0 {
        let t22 = t2 * t2;
        n2 = t22 * t22 * dot3(GRAD3[gi2], x2, y2, z2);
    }

    let mut n3 = 0.0;
    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if t3 > 0.0 {
        let t32 = t3 * t3;
        n3 = t32 * t32 * dot3(GRAD3[gi3], x3, y3, z3);
    }

    32.0 * (n0 + n1 + n2 + n3)
}

/// Fractional Brownian Motion using 3D simplex noise.
pub fn fbm3(x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut freq = 1.0;
    let mut max_amp = 0.0;

    for _ in 0..octaves {
        value += amplitude * simplex3(x * freq, y * freq, z * freq);
        max_amp += amplitude;
        freq *= lacunarity;
        amplitude *= persistence;
    }

    value / max_amp // normalize to [-1, 1]
}

/// 2D FBM (samples simplex3 at z=0 plane).
pub fn fbm2(x: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    fbm3(x, 0.0, z, octaves, lacunarity, persistence)
}

// ─── Terrain SDF evaluation (CPU side) ──────────────────────────────────────

/// Evaluate the terrain SDF at a world position using the given config.
/// Returns the signed distance (negative = inside terrain, positive = above).
pub fn terrain_sdf(pos: Vec3, config: &TerrainConfig) -> f32 {
    match config.style {
        TerrainStyle::Flat => {
            let n = fbm2(
                pos.x * config.frequency,
                pos.z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            pos.y - (config.height + n * config.amplitude)
        }
        TerrainStyle::Rolling => {
            let n = fbm2(
                pos.x * config.frequency,
                pos.z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            pos.y - (config.height + n * config.amplitude)
        }
        TerrainStyle::Mountains => {
            let n = fbm2(
                pos.x * config.frequency,
                pos.z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            // Ridge noise: fold the output for sharper peaks
            let ridge = 1.0 - n.abs();
            pos.y - (config.height + ridge * config.amplitude)
        }
        TerrainStyle::Caves => {
            // Full 3D density field with vertical gradient
            let density = fbm3(
                pos.x * config.frequency,
                pos.y * config.frequency,
                pos.z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            let height_bias = (pos.y - config.height) / config.amplitude;
            height_bias - density
        }
        TerrainStyle::Islands => {
            let n = fbm2(
                pos.x * config.frequency,
                pos.z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            // Radial falloff from origin
            let dist_xz = (pos.x * pos.x + pos.z * pos.z).sqrt();
            let falloff = (1.0 - (dist_xz * 0.02).min(1.0)).max(0.0);
            pos.y - (config.height + n * config.amplitude * falloff)
        }
    }
}

/// Estimate the terrain height range within a brick's XZ footprint.
/// Returns (min_height, max_height).
pub fn terrain_height_range(
    brick_min: Vec3,
    brick_max: Vec3,
    config: &TerrainConfig,
) -> (f32, f32) {
    // Sample terrain at brick center and corners (XZ only)
    let cx = (brick_min.x + brick_max.x) * 0.5;
    let cz = (brick_min.z + brick_max.z) * 0.5;

    // Sample 5 points: center + 4 corners of XZ footprint
    let samples = [
        terrain_height_at(cx, cz, config),
        terrain_height_at(brick_min.x, brick_min.z, config),
        terrain_height_at(brick_max.x, brick_min.z, config),
        terrain_height_at(brick_min.x, brick_max.z, config),
        terrain_height_at(brick_max.x, brick_max.z, config),
    ];

    let min_h = samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_h = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Add conservative margin for noise variation within the brick
    let brick_size = brick_max.x - brick_min.x;
    let margin = config.amplitude * config.frequency * brick_size;
    (min_h - margin, max_h + margin)
}

/// Terrain height at a specific XZ position (for heightfield styles).
fn terrain_height_at(x: f32, z: f32, config: &TerrainConfig) -> f32 {
    match config.style {
        TerrainStyle::Flat | TerrainStyle::Rolling => {
            let n = fbm2(
                x * config.frequency,
                z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            config.height + n * config.amplitude
        }
        TerrainStyle::Mountains => {
            let n = fbm2(
                x * config.frequency,
                z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            let ridge = 1.0 - n.abs();
            config.height + ridge * config.amplitude
        }
        TerrainStyle::Islands => {
            let n = fbm2(
                x * config.frequency,
                z * config.frequency,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            let dist_xz = (x * x + z * z).sqrt();
            let falloff = (1.0 - (dist_xz * 0.02).min(1.0)).max(0.0);
            config.height + n * config.amplitude * falloff
        }
        TerrainStyle::Caves => {
            // For caves, return a wide range since it's 3D
            config.height
        }
    }
}
