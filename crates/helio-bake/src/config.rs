use std::path::PathBuf;

/// Quality settings for the built-in CPU lightmap baker.
///
/// The CPU baker uses the **exact** attenuation formula from the runtime
/// deferred shader, so baked and dynamic lights match in brightness.
/// It only evaluates direct lighting (one shadow ray per light per texel),
/// giving clean, noise-free results without a denoiser.
#[derive(Clone, Debug)]
pub struct CpuLightmapConfig {
    /// Atlas width and height in texels.  Higher = more texel density.
    pub resolution: u32,
    /// Constant irradiance/π added to every lit texel.
    ///
    /// Prevents pitch-black shadows by acting as a dim ambient fill.
    /// Stored in the same units as the lightmap (pre-divided by π).
    /// Default: `[0.06, 0.06, 0.06]` (≈ 6 % white fill).
    pub ambient_fill: [f32; 3],
}

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

    /// Bake full-scene PBR lightmaps using the CPU direct-light baker.
    ///
    /// Uses the same attenuation formula as the runtime deferred shader;
    /// no Nebula GPU path tracer is involved.  Set to `None` to skip.
    pub cpu_lightmap: Option<CpuLightmapConfig>,

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
            cpu_lightmap: Some(CpuLightmapConfig {
                resolution:    1024,
                ambient_fill:  [0.06, 0.06, 0.06],
            }),
            lightmap: None,
            probes: None,
            pvs: None,
        }
    }
}

impl BakeConfig {
    /// Fast preset: CPU direct-light baker, no probes or PVS.
    ///
    /// Evaluates direct lighting with the same attenuation formula as the
    /// runtime deferred shader.  One deterministic shadow ray per light per
    /// texel — zero noise, no denoiser, sub-second bake times for typical scenes.
    pub fn fast(scene_name: impl Into<String>) -> Self {
        Self {
            scene_name: scene_name.into(),
            cache_dir: "bake_cache".into(),
            ao: Some(nebula::ao::AoConfig::fast()),
            cpu_lightmap: Some(CpuLightmapConfig {
                resolution:   1024,
                ambient_fill: [0.06, 0.06, 0.06],
            }),
            lightmap: None,
            probes: None,
            pvs: None,
        }
    }

    /// Medium preset: multi-bounce GI lightmap with denoising.
    ///
    /// 64 spp, 2 bounces, spatial denoising — good balance of quality and
    /// bake time (seconds to low minutes).  Suitable for playtesting.
    pub fn medium(scene_name: impl Into<String>) -> Self {
        Self {
            scene_name: scene_name.into(),
            cache_dir: "bake_cache".into(),
            ao: Some(nebula::ao::AoConfig::default()),
            cpu_lightmap: None,
            lightmap: Some(nebula::light::LightmapConfig {
                resolution:         1024,
                samples_per_texel:  64,
                bounce_count:       2,
                denoise:            true,
                ..nebula::light::LightmapConfig::default()
            }),
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
            cpu_lightmap: None,
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
