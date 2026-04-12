//! CPU direct-light lightmap baker.
//!
//! Replaces Nebula's GPU path tracer for the [`BakeConfig::fast`] preset.
//!
//! ## Design goals
//! 1. **Correctness**: uses the *exact* attenuation formula from
//!    `pbr_direct_light` in `deferred_lighting.wgsl` — no convention mismatch.
//! 2. **Zero noise**: direct lighting only; one deterministic shadow ray per
//!    light per texel.  No Monte Carlo randomness, no denoiser needed.
//! 3. **Speed**: parallelised over rows using `std::thread::scope`.
//!
//! ## Convention
//! The deferred shader reads: `lightmap_indirect = lightmap_sample * albedo`
//! For this to match the Lambertian diffuse term from `pbr_direct_light`:
//!   `Lo_diffuse = (albedo / π) · Σ radiance_i · NdotL_i`
//! we store:
//!   `texel = Σ (radiance_i · NdotL_i) / π`
//! so that `texel * albedo = (albedo / π) · Σ radiance_i · NdotL_i` ✓
//!
//! ## Attenuation
//! Matches the runtime shader exactly:
//!   `ratio   = d / range`
//!   `falloff = max(0, 1 - ratio²)`
//!   `atten   = falloff²`  — i.e. `(1-(d/r)²)²`

use std::sync::Arc;

use nebula::core::scene::{BakeMesh, LightSource, LightSourceKind};
use nebula::prelude::SceneGeometry;

use crate::cache::{CachedAtlasRegion, CachedLightmap};

// ── Constants ─────────────────────────────────────────────────────────────────

const PI: f32 = std::f32::consts::PI;
const EPSILON: f32 = 1e-7;
const SHADOW_BIAS: f32 = 2e-3;

// ── Vector math (avoids pulling in a dep just for 3-component ops) ─────────────

#[inline(always)] fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
#[inline(always)] fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline(always)] fn scale(a: [f32; 3], s: f32) -> [f32; 3] { [a[0]*s, a[1]*s, a[2]*s] }
#[inline(always)] fn dot(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline(always)] fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
#[inline(always)] fn len(a: [f32; 3]) -> f32 { dot(a, a).sqrt() }
#[inline(always)] fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = len(a);
    if l < EPSILON { [0.0, 1.0, 0.0] } else { scale(a, 1.0/l) }
}
#[inline(always)] fn lerp3(a: [f32; 3], b: [f32; 3], c: [f32; 3], w: f32, u: f32, v: f32) -> [f32; 3] {
    [a[0]*w+b[0]*u+c[0]*v, a[1]*w+b[1]*u+c[1]*v, a[2]*w+b[2]*u+c[2]*v]
}

/// GLSL-style `smoothstep(edge0, edge1, x)`.
#[inline(always)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < EPSILON {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Runtime-matching smooth quadratic falloff: `(max(0, 1 - (d/r)²))²`.
#[inline(always)]
fn smooth_atten(dist: f32, range: f32) -> f32 {
    if dist >= range { return 0.0; }
    let ratio = dist / range;
    let f = (1.0 - ratio * ratio).max(0.0);
    f * f
}

// ── Scene representation ──────────────────────────────────────────────────────

/// One triangle's world-space geometry + lightmap UVs.
struct Triangle {
    p:  [[f32; 3]; 3],
    n:  [[f32; 3]; 3],
    lm: [[f32; 2]; 3],
}

/// All triangles for one mesh plus a world-space AABB for ray pre-filtering.
struct MeshTris {
    tris:     Vec<Triangle>,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

/// A light flattened into a simple enum (avoids lifetime constraints on Nebula types).
enum CpuLight {
    Directional {
        direction: [f32; 3],   // unit vector pointing AWAY from the light source
        color:     [f32; 3],
        intensity: f32,
        shadows:   bool,
    },
    Point {
        position:  [f32; 3],
        range:     f32,
        color:     [f32; 3],
        intensity: f32,
        shadows:   bool,
    },
    Spot {
        position:  [f32; 3],
        direction: [f32; 3],   // unit vector pointing away from the spotlight
        range:     f32,
        cos_inner: f32,        // cosine of inner half-angle
        cos_outer: f32,        // cosine of outer half-angle
        color:     [f32; 3],
        intensity: f32,
        shadows:   bool,
    },
}

// ── Build helpers ─────────────────────────────────────────────────────────────

fn build_mesh_tris(meshes: &[BakeMesh]) -> Vec<MeshTris> {
    meshes.iter().map(|mesh| {
        let fallback: Vec<[f32; 2]> = mesh.uvs.clone();
        let lm_uvs = mesh.lightmap_uvs.as_ref().unwrap_or(&fallback);

        let mut aabb_min = [f32::MAX; 3];
        let mut aabb_max = [f32::MIN; 3];

        let tris: Vec<Triangle> = mesh.indices.chunks(3).filter_map(|chunk| {
            if chunk.len() < 3 { return None; }
            let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
            if i0 >= mesh.positions.len() || i1 >= mesh.positions.len() || i2 >= mesh.positions.len() {
                return None;
            }
            let p = [mesh.positions[i0], mesh.positions[i1], mesh.positions[i2]];
            for pi in &p {
                for k in 0..3 {
                    aabb_min[k] = aabb_min[k].min(pi[k]);
                    aabb_max[k] = aabb_max[k].max(pi[k]);
                }
            }
            Some(Triangle {
                p,
                n: [
                    mesh.normals.get(i0).copied().unwrap_or([0.0, 1.0, 0.0]),
                    mesh.normals.get(i1).copied().unwrap_or([0.0, 1.0, 0.0]),
                    mesh.normals.get(i2).copied().unwrap_or([0.0, 1.0, 0.0]),
                ],
                lm: [
                    lm_uvs.get(i0).copied().unwrap_or([0.0; 2]),
                    lm_uvs.get(i1).copied().unwrap_or([0.0; 2]),
                    lm_uvs.get(i2).copied().unwrap_or([0.0; 2]),
                ],
            })
        }).collect();

        MeshTris { tris, aabb_min, aabb_max }
    }).collect()
}

fn convert_lights(scene: &SceneGeometry) -> Vec<CpuLight> {
    scene.lights.iter().filter(|l| l.bake_enabled).filter_map(|l| {
        match &l.kind {
            LightSourceKind::Directional { direction } => Some(CpuLight::Directional {
                direction: *direction,
                color: l.color,
                intensity: l.intensity,
                shadows: l.casts_shadows,
            }),
            LightSourceKind::Point { position, range } => Some(CpuLight::Point {
                position: *position,
                range: *range,
                color: l.color,
                intensity: l.intensity,
                shadows: l.casts_shadows,
            }),
            LightSourceKind::Spot { position, direction, range, inner_angle, outer_angle } => {
                Some(CpuLight::Spot {
                    position: *position,
                    direction: *direction,
                    range: *range,
                    // inner_angle / outer_angle are in radians (set in build_static_bake_scene)
                    cos_inner: inner_angle.cos(),
                    cos_outer: outer_angle.cos(),
                    color: l.color,
                    intensity: l.intensity,
                    shadows: l.casts_shadows,
                })
            }
            _ => None, // Area lights not yet supported
        }
    }).collect()
}

// ── Ray tracing ───────────────────────────────────────────────────────────────

/// Möller-Trumbore ray-triangle intersection. Returns `t > 0` on hit.
#[inline]
fn ray_tri(ro: [f32; 3], rd: [f32; 3], p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> f32 {
    let e1  = sub(p1, p0);
    let e2  = sub(p2, p0);
    let h   = cross(rd, e2);
    let det = dot(e1, h);
    if det.abs() < EPSILON { return -1.0; }
    let inv = 1.0 / det;
    let s   = sub(ro, p0);
    let u   = inv * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q = cross(s, e1);
    let v = inv * dot(rd, q);
    if v < 0.0 || (u + v) > 1.0 { return -1.0; }
    let t = inv * dot(e2, q);
    if t > EPSILON { t } else { -1.0 }
}

/// Ray-AABB slab test — returns true if the ray might intersect the box within `[0, max_t]`.
#[inline]
fn ray_aabb(ro: [f32; 3], rd: [f32; 3], aabb_min: [f32; 3], aabb_max: [f32; 3], max_t: f32) -> bool {
    let mut tmin = 0.0_f32;
    let mut tmax = max_t;
    for i in 0..3 {
        if rd[i].abs() < EPSILON {
            if ro[i] < aabb_min[i] || ro[i] > aabb_max[i] { return false; }
        } else {
            let inv = 1.0 / rd[i];
            let t1 = (aabb_min[i] - ro[i]) * inv;
            let t2 = (aabb_max[i] - ro[i]) * inv;
            tmin = tmin.max(t1.min(t2));
            tmax = tmax.min(t1.max(t2));
        }
    }
    tmin <= tmax
}

/// Shadow test: returns `true` if anything blocks the path from `origin` to `light_pos`.
/// `origin` should already be offset by a normal bias.
fn is_occluded(origin: [f32; 3], light_pos: [f32; 3], mesh_tris: &[MeshTris]) -> bool {
    let d = sub(light_pos, origin);
    let max_t = len(d);
    if max_t < SHADOW_BIAS { return false; }
    let rd = scale(d, 1.0 / max_t);
    let max_t = max_t - SHADOW_BIAS; // don't hit the light source itself

    for mesh in mesh_tris {
        if !ray_aabb(origin, rd, mesh.aabb_min, mesh.aabb_max, max_t) { continue; }
        for tri in &mesh.tris {
            let t = ray_tri(origin, rd, tri.p[0], tri.p[1], tri.p[2]);
            if t > 0.0 && t < max_t { return true; }
        }
    }
    false
}

// ── Lighting ──────────────────────────────────────────────────────────────────

/// Compute the total diffuse irradiance/π at a world-space surface point.
///
/// Stores `Σ (L_i · NdotL_i) / π` so that the runtime formula
/// `lightmap_sample * albedo = (albedo/π) · Σ L_i · NdotL_i` matches
/// the Lambertian diffuse term from `pbr_direct_light`.
fn eval_direct(
    pos:       [f32; 3],
    nor:       [f32; 3],
    lights:    &[CpuLight],
    mesh_tris: &[MeshTris],
    ambient:   [f32; 3],
) -> [f32; 3] {
    let mut acc = ambient;
    // Surface origin offset by normal to avoid self-shadowing
    let biased_pos = add(pos, scale(nor, SHADOW_BIAS * 5.0));

    for light in lights {
        let (l_dir, radiance, light_pos, casts_shadows): ([f32; 3], [f32; 3], [f32; 3], bool) =
            match light {
                CpuLight::Directional { direction, color, intensity, shadows } => {
                    // direction points away from the light; negate for the L vector
                    let l = normalize([-direction[0], -direction[1], -direction[2]]);
                    let radiance = scale(*color, *intensity);
                    // Shadow ray goes towards a "light at infinity"
                    let light_pos = add(pos, scale(l, 10_000.0));
                    (l, radiance, light_pos, *shadows)
                }
                CpuLight::Point { position, range, color, intensity, shadows } => {
                    let to_l = sub(*position, pos);
                    let dist = len(to_l);
                    if dist >= *range { continue; }
                    let l = scale(to_l, 1.0 / dist);
                    let atten = smooth_atten(dist, *range);
                    let radiance = scale(*color, intensity * atten);
                    (l, radiance, *position, *shadows)
                }
                CpuLight::Spot { position, direction, range, cos_inner, cos_outer, color, intensity, shadows } => {
                    let to_l = sub(*position, pos);
                    let dist = len(to_l);
                    if dist >= *range { continue; }
                    let l = scale(to_l, 1.0 / dist);
                    // Cone test (matching runtime smoothstep)
                    let cos_spot = dot([-direction[0], -direction[1], -direction[2]], l);
                    let cone = smoothstep(*cos_outer, *cos_inner, cos_spot);
                    if cone <= 0.0 { continue; }
                    let atten = smooth_atten(dist, *range);
                    let radiance = scale(*color, intensity * atten * cone);
                    (l, radiance, *position, *shadows)
                }
            };

        let ndotl = dot(nor, l_dir).max(0.0);
        if ndotl <= 0.0 { continue; }

        // Shadow test
        let shadow = if casts_shadows && is_occluded(biased_pos, light_pos, mesh_tris) {
            0.0
        } else {
            1.0
        };

        // Accumulate irradiance/π (matches runtime: lightmap_sample * albedo = albedo/π · Σ L·NdotL)
        acc[0] += radiance[0] * ndotl * shadow / PI;
        acc[1] += radiance[1] * ndotl * shadow / PI;
        acc[2] += radiance[2] * ndotl * shadow / PI;
    }

    acc
}

// ── Texel lookup ──────────────────────────────────────────────────────────────

/// Barycentric coordinates of a 2-D point `p` relative to triangle `(uv0,uv1,uv2)`.
/// Returns `Some((u, v))` with `w = 1-u-v ≥ 0` when the point is inside.
#[inline]
fn bary2(uv0: [f32; 2], uv1: [f32; 2], uv2: [f32; 2], p: [f32; 2]) -> Option<(f32, f32)> {
    let d1 = [uv1[0]-uv0[0], uv1[1]-uv0[1]];
    let d2 = [uv2[0]-uv0[0], uv2[1]-uv0[1]];
    let dp = [p[0]-uv0[0], p[1]-uv0[1]];
    let det = d1[0]*d2[1] - d1[1]*d2[0];
    if det.abs() < EPSILON { return None; }
    let inv = 1.0 / det;
    let u = (dp[0]*d2[1] - dp[1]*d2[0]) * inv;
    let v = (d1[0]*dp[1] - d1[1]*dp[0]) * inv;
    // Allow small epsilon for edge texels
    if u >= -1e-4 && v >= -1e-4 && (u + v) <= 1.0 + 1e-4 {
        Some((u.max(0.0), v.max(0.0)))
    } else {
        None
    }
}

/// Find the world-space position and normal for a texel in a single mesh.
fn find_texel_world(mesh: &MeshTris, lm_uv: [f32; 2]) -> Option<([f32; 3], [f32; 3])> {
    for tri in &mesh.tris {
        if let Some((u, v)) = bary2(tri.lm[0], tri.lm[1], tri.lm[2], lm_uv) {
            let w = (1.0 - u - v).max(0.0);
            let pos = lerp3(tri.p[0], tri.p[1], tri.p[2], w, u, v);
            let nor = normalize(lerp3(tri.n[0], tri.n[1], tri.n[2], w, u, v));
            return Some((pos, nor));
        }
    }
    None
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the CPU lightmap baker and return a ready-to-cache result.
///
/// # Parameters
/// - `scene`: scene geometry produced by [`build_static_bake_scene`](crate::SceneGeometry)
/// - `resolution`: atlas width & height in texels (clamped to at least 64)
/// - `ambient`: constant irradiance/π added to every lit texel — prevents
///   pitch-black shadows; think of it as a very dim sky-light fill.
pub fn bake_lightmap(scene: &SceneGeometry, resolution: u32, ambient: [f32; 3]) -> CachedLightmap {
    let resolution = resolution.max(64);
    let n = scene.meshes.len();

    if n == 0 {
        log::warn!("[cpu-lightmap] Scene has no meshes — returning empty lightmap.");
        return CachedLightmap {
            width: resolution, height: resolution,
            channels: 4, is_f32: true,
            texels: vec![0u8; (resolution * resolution * 4 * 4) as usize],
            atlas_regions: vec![],
        };
    }

    // Equal-area atlas grid — identical to Nebula's build_atlas_regions
    let cols = (n as f32).sqrt().ceil() as u32;
    let rows = (n as u32).div_ceil(cols);
    let cell_w = 1.0_f32 / cols as f32;
    let cell_h = 1.0_f32 / rows as f32;

    let light_count = scene.lights.iter().filter(|l| l.bake_enabled).count();
    log::info!(
        "[cpu-lightmap] Baking {}×{} atlas — {} mesh(es) in {}×{} grid, {} light(s)",
        resolution, resolution, n, cols, rows, light_count
    );

    // Atlas region descriptors (same UUID encoding as bake.rs)
    let atlas_regions: Vec<CachedAtlasRegion> = scene.meshes.iter().enumerate().map(|(i, mesh)| {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;
        let bytes = mesh.id.as_bytes();
        let hi = u64::from_le_bytes(bytes[ ..8].try_into().unwrap());
        let lo = u64::from_le_bytes(bytes[8..].try_into().unwrap());
        CachedAtlasRegion {
            mesh_id:   [hi, lo],
            uv_offset: [col as f32 * cell_w, row as f32 * cell_h],
            uv_scale:  [cell_w, cell_h],
        }
    }).collect();

    // Build per-mesh triangle lists (positions are already world-space)
    let mesh_tris = Arc::new(build_mesh_tris(&scene.meshes));
    let lights    = Arc::new(convert_lights(scene));

    let res     = resolution as usize;
    let cols_us = cols as usize;
    let rows_us = rows as usize;

    // Split work into per-thread horizontal bands
    let nthreads   = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let band_h     = (res + nthreads - 1) / nthreads;

    // Collect each band's pixels then flatten
    let bands: Vec<Vec<[f32; 4]>> = std::thread::scope(|s| {
        (0..nthreads).map(|t| {
            let y0 = t * band_h;
            let y1 = (y0 + band_h).min(res);
            let mesh_tris = Arc::clone(&mesh_tris);
            let lights    = Arc::clone(&lights);
            s.spawn(move || {
                let mut band = vec![[0.0_f32; 4]; (y1 - y0) * res];
                for ty in y0..y1 {
                    let lm_v = (ty as f32 + 0.5) / res as f32;
                    // Determine which atlas row this texel falls in
                    let cell_row = ((lm_v * rows_us as f32) as usize).min(rows_us - 1);
                    for tx in 0..res {
                        let lm_u = (tx as f32 + 0.5) / res as f32;
                        let cell_col = ((lm_u * cols_us as f32) as usize).min(cols_us - 1);
                        let mesh_idx = cell_row * cols_us + cell_col;
                        if mesh_idx >= mesh_tris.len() { continue; }

                        if let Some((pos, nor)) =
                            find_texel_world(&mesh_tris[mesh_idx], [lm_u, lm_v])
                        {
                            let irr = eval_direct(pos, nor, &lights, &mesh_tris, ambient);
                            band[(ty - y0) * res + tx] = [irr[0], irr[1], irr[2], 1.0];
                        }
                    }
                }
                band
            })
        }).collect::<Vec<_>>()
         .into_iter()
         .map(|h| h.join().unwrap_or_default())
         .collect()
    });

    // Reassemble bands into a flat pixel buffer
    let mut pixels: Vec<[f32; 4]> = vec![[0.0; 4]; res * res];
    for (t, band) in bands.into_iter().enumerate() {
        let y0  = t * band_h;
        let y1  = (y0 + band_h).min(res);
        let dst = y0 * res;
        let len = (y1 - y0) * res;
        pixels[dst..dst + len].copy_from_slice(&band[..len]);
    }

    log::info!("[cpu-lightmap] Done. {} texels written.", res * res);

    CachedLightmap {
        width:         resolution,
        height:        resolution,
        channels:      4,
        is_f32:        true,
        texels:        bytemuck::cast_slice(&pixels).to_vec(),
        atlas_regions,
    }
}
