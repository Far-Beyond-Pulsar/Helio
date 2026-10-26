//! CPU-side 3D value noise matching the WGSL implementation (IQ-style).
//!
//! Used for terrain-aware brick classification and CPU-side picking.

use crate::terrain::TerrainConfig;
use glam::Vec3;

/// Hash function matching the WGSL `hash3()`.
fn hash3(px: f32, py: f32, pz: f32) -> f32 {
    let mut qx = (px * 0.3183099 + 0.1).fract();
    let mut qy = (py * 0.3183099 + 0.1).fract();
    let mut qz = (pz * 0.3183099 + 0.1).fract();
    if qx < 0.0 {
        qx += 1.0;
    }
    if qy < 0.0 {
        qy += 1.0;
    }
    if qz < 0.0 {
        qz += 1.0;
    }
    qx *= 17.0;
    qy *= 17.0;
    qz *= 17.0;
    let v = qx * qy * qz * (qx + qy + qz);
    let r = v.fract();
    if r < 0.0 {
        r + 1.0
    } else {
        r
    }
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

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
    let val = lerp(
        lerp(lerp(a, b, ux), lerp(c, d, ux), uy),
        lerp(lerp(e, f, ux), lerp(g, h, ux), uy),
        uz,
    );
    val * 2.0 - 1.0
}

/// Domain rotation matrix (matching WGSL FBM_ROT constants).
fn fbm_rotate(px: f32, py: f32, pz: f32, lac: f32) -> (f32, f32, f32) {
    let rx = lac * (0.00 * px + 0.80 * py + 0.60 * pz);
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

/// Sample FBM at a 2D (x, z) point — used internally for domain warping.
fn fbm2_at(x: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    fbm2(x, z, octaves, lacunarity, persistence)
}

/// Domain-warped FBM following Inigo Quilez's technique.
///
/// Two warp layers:
///   warp1 = vec2(fbm(p), fbm(p + (5.2, 1.3)))
///   warp2 = vec2(fbm(p + warp_amount * warp1), fbm(p + warp_amount * warp1 + (1.7, 9.2)))
///   result = fbm(p + warp_amount * warp2)
///
/// Returns `(warped_x, warped_z, noise_value)`.
pub fn warped_fbm3(
    x: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    warp_amount: f32,
) -> (f32, f32, f32) {
    // First warp layer
    let w1x = fbm2_at(x, z, octaves, lacunarity, persistence);
    let w1z = fbm2_at(x + 5.2, z + 1.3, octaves, lacunarity, persistence);

    let p1x = x + warp_amount * w1x;
    let p1z = z + warp_amount * w1z;

    // Second warp layer
    let w2x = fbm2_at(p1x, p1z, octaves, lacunarity, persistence);
    let w2z = fbm2_at(p1x + 1.7, p1z + 9.2, octaves, lacunarity, persistence);

    let p2x = x + warp_amount * w2x;
    let p2z = z + warp_amount * w2z;

    // Final sample
    let value = fbm2_at(p2x, p2z, octaves, lacunarity, persistence);
    (p2x, p2z, value)
}

/// Evaluate the terrain SDF at a world position.
pub fn terrain_sdf(pos: Vec3, config: &TerrainConfig) -> f32 {
    terrain_sdf_styled(pos, config)
}

/// Evaluate the terrain SDF with style-specific noise characteristics.
pub fn terrain_sdf_styled(pos: Vec3, config: &TerrainConfig) -> f32 {
    use crate::terrain::TerrainStyle;
    let fx = pos.x * config.frequency;
    let fz = pos.z * config.frequency;
    let terrain_height = match config.style {
        TerrainStyle::Rolling => {
            // Gentle rolling hills with balanced FBM.
            fbm2(
                fx,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
            ) * config.amplitude
        }
        TerrainStyle::Mountains => {
            // Taller, sharper mountains with tighter ridges.
            let n = fbm2(
                fx * 1.5,
                fz * 1.5,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            n * config.amplitude * 1.3
        }
        TerrainStyle::Canyons => {
            // Eroded canyon shapes with extra carved detail.
            let n = fbm2(
                fx,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            let detail = fbm2(fx * 3.0, fz * 3.0, 3, config.lacunarity, 0.4);
            n * config.amplitude + detail * 3.0
        }
        TerrainStyle::Dunes => {
            // Wind-swept dunes with elongated directional structure.
            let stretch = 3.0;
            let n = fbm2(
                fx * stretch,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            n * config.amplitude
        }
        TerrainStyle::Warped => {
            // Domain-warped organic terrain (IQ two-layer warp).
            let (_, _, val) = warped_fbm3(
                fx,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
                config.warp_amount,
            );
            val * config.amplitude
        }
    };

    pos.y - (config.height + terrain_height)
}

/// Estimate the terrain height range within a brick's XZ footprint.
///
/// Uses a 3×3 grid of samples across the brick's XZ extent to capture
/// terrain features that a coarse 5-point (center + corners) pattern
/// would miss — particularly at higher LOD levels where bricks span
/// large world areas.
pub fn terrain_height_range(
    brick_min: Vec3,
    brick_max: Vec3,
    config: &TerrainConfig,
) -> (f32, f32) {
    let mut min_h = f32::INFINITY;
    let mut max_h = f32::NEG_INFINITY;

    // 3×3 grid across XZ footprint (corners + edge midpoints + center)
    for iz in 0..3u32 {
        let fz = iz as f32 * 0.5;
        let z = brick_min.z + (brick_max.z - brick_min.z) * fz;
        for ix in 0..3u32 {
            let fx = ix as f32 * 0.5;
            let x = brick_min.x + (brick_max.x - brick_min.x) * fx;
            let h = terrain_height_at(x, z, config);
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }

    let brick_size = brick_max.x - brick_min.x;
    let margin = config.amplitude * config.frequency * brick_size;
    (min_h - margin, max_h + margin)
}

fn terrain_height_at(x: f32, z: f32, config: &TerrainConfig) -> f32 {
    use crate::terrain::TerrainStyle;
    let fx = x * config.frequency;
    let fz = z * config.frequency;
    let terrain_height = match config.style {
        TerrainStyle::Rolling => {
            fbm2(
                fx,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
            ) * config.amplitude
        }
        TerrainStyle::Mountains => {
            let n = fbm2(
                fx * 1.5,
                fz * 1.5,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            n * config.amplitude * 1.3
        }
        TerrainStyle::Canyons => {
            let n = fbm2(
                fx,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            let detail = fbm2(fx * 3.0, fz * 3.0, 3, config.lacunarity, 0.4);
            n * config.amplitude + detail * 3.0
        }
        TerrainStyle::Dunes => {
            let stretch = 3.0;
            let n = fbm2(
                fx * stretch,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
            );
            n * config.amplitude
        }
        TerrainStyle::Warped => {
            let (_, _, val) = warped_fbm3(
                fx,
                fz,
                config.octaves,
                config.lacunarity,
                config.persistence,
                config.warp_amount,
            );
            val * config.amplitude
        }
    };
    config.height + terrain_height
}
