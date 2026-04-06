use std::sync::Arc;

use glam::Vec3;
use nebula::prelude::{BakeContext, BakePass, NullReporter, SceneGeometry};

use crate::cache::{BakeCache, CachedAo, CachedAtlasRegion, CachedLightmap, CachedProbes, CachedPvs};
use crate::config::BakeConfig;
use crate::data::BakedData;

// ── Error type ─────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum BakeError {
    #[error("Nebula bake error: {0}")]
    Nebula(#[from] nebula::prelude::NebulaError),

    #[error("Cache I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cache serialization error: {0}")]
    Serialize(#[from] Box<bincode::ErrorKind>),

    #[error("Cache deserialization error: {0}")]
    Deserialize(Box<bincode::ErrorKind>),
}

// ── Public entry point ─────────────────────────────────────────────────────────

/// Run all configured bake passes, using the on-disk cache where available.
///
/// **This function blocks** the calling thread until every cache-miss bake pass
/// completes on the GPU. The renderer calls it synchronously before drawing frame 1.
///
/// # Cache behaviour
/// - If a `.bin` cache file exists with the correct magic + version, its data is
///   loaded directly — no GPU work is done for that pass.
/// - On a cache miss (first run, or after deleting the file) the Nebula GPU baker
///   runs and the result is persisted to disk before returning.
///
/// # Device sharing
/// Nebula creates its own headless `HighPerformance` wgpu device for baking.
/// After baking the raw texel bytes are uploaded to `helio_device` — the
/// lifetime of those textures is tied to the returned [`BakedData`].
pub fn run_bake_blocking(
    helio_device: &Arc<wgpu::Device>,
    helio_queue: &Arc<wgpu::Queue>,
    scene: &SceneGeometry,
    config: &BakeConfig,
) -> Result<BakedData, BakeError> {
    let cache = BakeCache::new(&config.cache_dir, &config.scene_name);
    cache.ensure_dir()?;

    // ── Determine which passes need GPU work ───────────────────────────────────
    let need_ao = config.ao.is_some() && cache.load_ao()?.is_none();
    let need_lm = config.lightmap.is_some() && cache.load_lightmap()?.is_none();
    let need_probes = config.probes.is_some() && cache.load_probes()?.is_none();
    let need_pvs = config.pvs.is_some() && cache.load_pvs()?.is_none();

    // ── GPU baking (all cache misses in one Nebula context) ────────────────────
    if !need_ao && !need_lm && !need_probes && !need_pvs {
        log::info!(
            "[helio-bake] '{}' — all passes loaded from disk cache, no GPU bake needed.",
            config.scene_name
        );
    }
    if need_ao || need_lm || need_probes || need_pvs {
        log::info!(
            "[helio-bake] Starting GPU bake for '{}' (ao={}, lightmap={}, probes={}, pvs={})",
            config.scene_name, need_ao, need_lm, need_probes, need_pvs
        );

        pollster::block_on(async {
            let ctx = BakeContext::new().await?;

            // AO
            if need_ao {
                let ao_cfg = config.ao.as_ref().unwrap();
                log::info!("[helio-bake] Baking AO ({}×{}, {} rays)…",
                    ao_cfg.resolution, ao_cfg.resolution, ao_cfg.ray_count);
                let out = nebula::ao::AoBaker.execute(scene, ao_cfg, &ctx, &NullReporter).await?;
                let cached = CachedAo {
                    width: out.width,
                    height: out.height,
                    texels: out.texels,
                };
                cache.save_ao(&cached)?;
                log::info!("[helio-bake] AO bake complete.");
            }

            // Lightmap
            if need_lm {
                let lm_cfg = config.lightmap.as_ref().unwrap();
                log::info!("[helio-bake] Baking lightmap ({}×{}, {}spp)…",
                    lm_cfg.resolution, lm_cfg.resolution, lm_cfg.samples_per_texel);
                let out = nebula::light::LightmapBaker.execute(scene, lm_cfg, &ctx, &NullReporter).await?;
                let regions: Vec<CachedAtlasRegion> = out.atlas_regions.iter().map(|r| {
                    let bytes = r.mesh_id.as_bytes();
                    let hi = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                    let lo = u64::from_le_bytes(bytes[8..].try_into().unwrap());
                    CachedAtlasRegion {
                        mesh_id: [hi, lo],
                        uv_offset: r.uv_offset,
                        uv_scale: r.uv_scale,
                    }
                }).collect();
                let cached = CachedLightmap {
                    width: out.width,
                    height: out.height,
                    channels: out.channels,
                    is_f32: out.is_f32,
                    texels: out.texels,
                    atlas_regions: regions,
                };
                cache.save_lightmap(&cached)?;
                log::info!("[helio-bake] Lightmap bake complete.");
            }

            // Probes
            if need_probes {
                let spec = config.probes.as_ref().unwrap();
                log::info!("[helio-bake] Baking {} probe(s) ({}px faces, {} mips)…",
                    spec.positions.len(), spec.config.face_resolution, spec.config.specular_mip_levels);

                let mut all_face_data: Vec<u8> = Vec::new();
                let mut bytes_per_probe = 0usize;
                let mut all_sh: Vec<Vec<[f32; 3]>> = Vec::new();
                // Metadata extracted from the first probe result; fall back to config values.
                let mut face_resolution = spec.config.face_resolution;
                let mut mip_levels = spec.config.specular_mip_levels;
                let mut is_rgbe = false;

                for (i, &pos) in spec.positions.iter().enumerate() {
                    let pos_v = Vec3::from(pos);
                    let (refl, irr) = nebula::probe::ProbeBaker::bake_at(
                        pos_v, scene, &spec.config, &ctx, &NullReporter,
                    ).await?;

                    if i == 0 {
                        bytes_per_probe  = refl.face_data.len();
                        face_resolution  = refl.face_resolution;
                        mip_levels       = refl.mip_levels;
                        is_rgbe          = refl.is_rgbe;
                    }
                    all_face_data.extend_from_slice(&refl.face_data);

                    let sh: Vec<[f32; 3]> = irr.coefficients.iter()
                        .map(|c| [c.r, c.g, c.b])
                        .collect();
                    all_sh.push(sh);
                }

                let cached = CachedProbes {
                    face_resolution,
                    mip_levels,
                    is_rgbe,
                    face_data: all_face_data,
                    bytes_per_probe,
                    positions: spec.positions.clone(),
                    irradiance_sh: all_sh,
                };
                cache.save_probes(&cached)?;
                log::info!("[helio-bake] Probe bake complete.");
            }

            // PVS
            if need_pvs {
                let pvs_cfg = config.pvs.as_ref().unwrap();
                log::info!("[helio-bake] Baking PVS (cell_size={}, ray_budget={})…",
                    pvs_cfg.cell_size, pvs_cfg.ray_budget);
                let out = nebula::visibility::PvsBaker.execute(scene, pvs_cfg, &ctx, &NullReporter).await?;
                let cached = CachedPvs {
                    world_min: out.world_min,
                    world_max: out.world_max,
                    grid_dims: out.grid_dims,
                    cell_size: out.cell_size,
                    cell_count: out.cell_count,
                    words_per_cell: out.words_per_cell,
                    bits: out.bits,
                };
                cache.save_pvs(&cached)?;
                log::info!("[helio-bake] PVS bake complete.");
            }

            Ok::<(), BakeError>(())
        })?;
    } else {
        log::info!("[helio-bake] All passes for '{}' loaded from cache.", config.scene_name);
    }

    // ── Upload cached data to Helio's GPU device ───────────────────────────────
    let ao = if config.ao.is_some() { cache.load_ao()? } else { None };
    let lm = if config.lightmap.is_some() { cache.load_lightmap()? } else { None };
    let probes = if config.probes.is_some() { cache.load_probes()? } else { None };
    let pvs = if config.pvs.is_some() { cache.load_pvs()? } else { None };

    BakedData::upload_to_gpu(helio_device, helio_queue, ao, lm, probes, pvs)
}
