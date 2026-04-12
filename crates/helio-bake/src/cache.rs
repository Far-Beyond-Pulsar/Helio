use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bake::BakeError;

/// Magic header written at the start of every cache file.
const CACHE_MAGIC: &[u8; 8] = b"HLBKCACH";
/// Bump when the serialized layout changes so stale files are rejected.
const CACHE_VERSION: u32 = 3;

// ── On-disk representations ────────────────────────────────────────────────────

/// Cached ambient occlusion bake.
#[derive(Serialize, Deserialize)]
pub(crate) struct CachedAo {
    pub width: u32,
    pub height: u32,
    /// R32F texels, row-major, `width * height * 4` bytes.
    pub texels: Vec<u8>,
}

/// Cached lightmap atlas bake.
#[derive(Serialize, Deserialize)]
pub(crate) struct CachedLightmap {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// `true` = RGBA32F (4 × f32/texel), `false` = RGBA16F (4 × f16/texel).
    pub is_f32: bool,
    pub texels: Vec<u8>,
    pub atlas_regions: Vec<CachedAtlasRegion>,
}

/// Per-mesh UV region inside the lightmap atlas.
#[derive(Serialize, Deserialize, Clone)]
pub struct CachedAtlasRegion {
    /// UUID of the mesh this region covers (as `[u64; 2]` for serde-free storage).
    pub mesh_id: [u64; 2],
    /// Top-left corner in atlas UV space `[0, 1]`.
    pub uv_offset: [f32; 2],
    /// Width/height in atlas UV space.
    pub uv_scale: [f32; 2],
}

/// Cached reflection + irradiance probe (all probes concatenated).
#[derive(Serialize, Deserialize)]
pub(crate) struct CachedProbes {
    pub face_resolution: u32,
    pub mip_levels: u32,
    pub is_rgbe: bool,
    /// Concatenated face data for all probes: probe_count × face_data_per_probe bytes.
    pub face_data: Vec<u8>,
    /// Number of bytes per probe in `face_data`.
    pub bytes_per_probe: usize,
    /// Baked positions (parallel to the probe entries in `face_data`).
    pub positions: Vec<[f32; 3]>,
    /// SH coefficients per probe: 9 RGB triplets (L2 irradiance).
    /// Outer vec = probes, inner = 9 `[f32; 3]` entries.
    pub irradiance_sh: Vec<Vec<[f32; 3]>>,
}

/// Cached potentially-visible set.
#[derive(Serialize, Deserialize)]
pub(crate) struct CachedPvs {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub cell_count: u32,
    pub words_per_cell: u32,
    pub bits: Vec<u64>,
}

// ── Cache manager ──────────────────────────────────────────────────────────────

/// On-disk bake cache rooted at a configurable directory.
///
/// Files are named `{dir}/{scene_name}_{pass}.bin` and begin with a magic
/// header + version so stale entries from old formats are automatically
/// invalidated and re-baked.
pub(crate) struct BakeCache {
    dir: PathBuf,
    scene_name: String,
}

impl BakeCache {
    pub fn new(dir: &Path, scene_name: &str) -> Self {
        Self {
            dir: dir.to_owned(),
            scene_name: scene_name.to_owned(),
        }
    }

    fn path(&self, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}_{}.bin", self.scene_name, suffix))
    }

    /// Ensure the cache directory exists.
    pub fn ensure_dir(&self) -> Result<(), BakeError> {
        fs::create_dir_all(&self.dir).map_err(BakeError::Io)
    }

    // ── Generic helpers ───────────────────────────────────────────────────────

    fn write<T: Serialize>(&self, suffix: &str, value: &T) -> Result<(), BakeError> {
        let payload = bincode::serialize(value).map_err(BakeError::Serialize)?;
        let path = self.path(suffix);
        let mut file = fs::File::create(&path).map_err(BakeError::Io)?;
        file.write_all(CACHE_MAGIC).map_err(BakeError::Io)?;
        file.write_all(&CACHE_VERSION.to_le_bytes()).map_err(BakeError::Io)?;
        file.write_all(&payload).map_err(BakeError::Io)?;
        log::info!("[helio-bake] Wrote cache: {}", path.display());
        Ok(())
    }

    fn read<T: for<'de> Deserialize<'de>>(&self, suffix: &str) -> Result<Option<T>, BakeError> {
        let path = self.path(suffix);
        if !path.exists() {
            return Ok(None);
        }
        let mut file = match fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                log::warn!("Could not open bake cache {}: {}", path.display(), e);
                return Ok(None);
            }
        };

        // Validate magic + version
        let mut magic = [0u8; 8];
        let mut ver_bytes = [0u8; 4];
        if file.read_exact(&mut magic).is_err()
            || &magic != CACHE_MAGIC
            || file.read_exact(&mut ver_bytes).is_err()
            || u32::from_le_bytes(ver_bytes) != CACHE_VERSION
        {
            log::info!(
                "Bake cache {} is stale (wrong magic/version). Will re-bake.",
                path.display()
            );
            return Ok(None);
        }

        let mut payload = Vec::new();
        file.read_to_end(&mut payload).map_err(BakeError::Io)?;
        let value: T = bincode::deserialize(&payload).map_err(|e| {
            log::warn!("Bake cache {} failed to deserialize: {}. Will re-bake.", path.display(), e);
            BakeError::Deserialize(e)
        })?;
        log::info!("[helio-bake] Loaded from cache: {}", path.display());
        Ok(Some(value))
    }

    // ── AO ────────────────────────────────────────────────────────────────────

    pub fn load_ao(&self) -> Result<Option<CachedAo>, BakeError> {
        self.read("ao")
    }

    pub fn save_ao(&self, data: &CachedAo) -> Result<(), BakeError> {
        self.write("ao", data)
    }

    // ── Lightmap ──────────────────────────────────────────────────────────────

    pub fn load_lightmap(&self) -> Result<Option<CachedLightmap>, BakeError> {
        self.read("lightmap")
    }

    pub fn save_lightmap(&self, data: &CachedLightmap) -> Result<(), BakeError> {
        self.write("lightmap", data)
    }

    // ── Probes ────────────────────────────────────────────────────────────────

    pub fn load_probes(&self) -> Result<Option<CachedProbes>, BakeError> {
        self.read("probes")
    }

    pub fn save_probes(&self, data: &CachedProbes) -> Result<(), BakeError> {
        self.write("probes", data)
    }

    // ── PVS ───────────────────────────────────────────────────────────────────

    pub fn load_pvs(&self) -> Result<Option<CachedPvs>, BakeError> {
        self.read("pvs")
    }

    pub fn save_pvs(&self, data: &CachedPvs) -> Result<(), BakeError> {
        self.write("pvs", data)
    }
}
