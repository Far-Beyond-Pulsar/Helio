use std::path::PathBuf;

/// Configuration for offline baking.
///
/// Controls which passes run, their quality settings, and the cache directory.
/// Use [`BakeConfig::fast`] or [`BakeConfig::ultra`] for common presets,
/// or construct manually for custom settings.
#[derive(Clone, Debug)]
pub struct BakeConfig {
    /// Unique name for this scene's bake artifacts (used as the cache key).
    ///
    /// Cache files will be named `{cache_dir}/{scene_name}_ao.bin`, etc.
    /// Example: `"outdoor_rocks"`, `"indoor_corridor"`.
    pub scene_name: String,

    /// Directory where cache files are written on first bake and read on subsequent runs.
    ///
    /// The directory is created automatically if it does not exist.
    /// Default: `"bake_cache"` (relative to the working directory at runtime).
    pub cache_dir: PathBuf,

    /// Bake ambient occlusion.
    ///
    /// The resulting R32F texture replaces screen-space SSAO, giving stable,
    /// pre-computed AO that reads correctly on first frame.
    /// Set to `None` to skip (SSAO still runs at runtime).
    pub ao: Option<nebula::ao::AoConfig>,

    /// Bake full-scene PBR lightmaps (direct + multi-bounce indirect illumination).
    ///
    /// Produces an RGBA atlas texture with per-mesh UV region mapping.
    /// Set to `None` to skip (real-time lighting only).
    pub lightmap: Option<nebula::light::LightmapConfig>,

    /// Bake reflection + irradiance probes at one or more world positions.
    ///
    /// Each probe produces:
    /// - A pre-filtered RGBA32F cubemap mip chain (specular IBL)
    /// - L2 spherical-harmonic coefficients (diffuse irradiance) — 9 RGB values
    ///
    /// Set to `None` to skip (runtime IBL from sky LUT only).
    pub probes: Option<ProbeSpec>,

    /// Bake potentially-visible sets (PVS) for CPU-side visibility culling.
    ///
    /// Provides a fast `is_visible(from_cell, to_cell)` query at runtime,
    /// which can gate draw calls before they reach GPU culling.
    /// Set to `None` to skip.
    pub pvs: Option<nebula::visibility::PvsConfig>,
}

/// Probe bake spec: where probes are placed and at what quality.
#[derive(Clone, Debug)]
pub struct ProbeSpec {
    /// World-space positions to bake.
    ///
    /// In a typical scene: one probe per room/zone, placed at head height.
    /// The baked probes are stored sequentially and can be indexed by position
    /// using the closest-probe logic in [`BakedData`](crate::BakedData).
    pub positions: Vec<[f32; 3]>,

    /// Quality / resolution settings for probe capture.
    pub config: nebula::probe::ProbeConfig,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            scene_name: "scene".into(),
            cache_dir: "bake_cache".into(),
            ao: Some(nebula::ao::AoConfig::default()),
            lightmap: Some(nebula::light::LightmapConfig::fast()),
            probes: None,
            pvs: None,
        }
    }
}

impl BakeConfig {
    /// Fast preset: AO + lightmap at reduced quality, no probes or PVS.
    ///
    /// Good for development iteration. Bake times are seconds to low minutes.
    pub fn fast(scene_name: impl Into<String>) -> Self {
        Self {
            scene_name: scene_name.into(),
            cache_dir: "bake_cache".into(),
            ao: Some(nebula::ao::AoConfig::fast()),
            lightmap: Some(nebula::light::LightmapConfig::fast()),
            probes: None,
            pvs: None,
        }
    }

    /// Ultra preset: all passes at maximum quality.
    ///
    /// Bake once for shipping. Bake times are minutes.
    pub fn ultra(scene_name: impl Into<String>) -> Self {
        Self {
            scene_name: scene_name.into(),
            cache_dir: "bake_cache".into(),
            ao: Some(nebula::ao::AoConfig::ultra()),
            lightmap: Some(nebula::light::LightmapConfig::ultra()),
            probes: Some(ProbeSpec {
                positions: vec![[0.0, 2.0, 0.0]],
                config: nebula::probe::ProbeConfig::ultra(),
            }),
            pvs: Some(nebula::visibility::PvsConfig::default()),
        }
    }

    /// Override the cache directory (builder pattern).
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }

    /// Add a probe spec (builder pattern).
    pub fn with_probes(mut self, probe_spec: ProbeSpec) -> Self {
        self.probes = Some(probe_spec);
        self
    }

    /// Enable PVS baking with default settings (builder pattern).
    pub fn with_pvs(mut self) -> Self {
        self.pvs = Some(nebula::visibility::PvsConfig::default());
        self
    }
}
