//! Tier-3 tuning knobs, the generated WGSL constant prelude, and shader assembly.

/// Compile-time (pipeline-creation-time) configuration of the dense kernels.
/// Every field becomes a WGSL `const`, so changing it means rebuilding the
/// pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseConfig {
    /// Tile edge in pixels (8 or 16). One workgroup of `tile * tile` threads
    /// per dense tile.
    pub tile: u32,
    /// Depth bucket resolution: `1 << bucket_bits` buckets over reversed-Z
    /// [0, 1]. The plan specifies 12.
    pub bucket_bits: u32,
    /// Peel-layer slab size: a layer consumes the nearest buckets until at
    /// least this many triangles are covered.
    pub slab_tris: u32,
    /// Nominal peel depth K (plan: K <= 4; 2 on low shared memory).
    pub nominal_layers: u32,
    /// Hard layer cap. Layers in `nominal_layers..max_layers` are the "extra
    /// peel iterations"; reaching the cap clamps the tile.
    pub max_layers: u32,
    /// Per-tile cluster list capacity (clusters past it are dropped and
    /// counted in `DenseStats::bin_overflow`).
    pub cluster_cap: u32,
    /// Triangles set up per shared-memory batch.
    pub tri_batch: u32,
    /// Upper bound on dense tiles processed per frame (further ones are
    /// dropped and counted in `DenseStats::slot_overflow`).
    pub max_dense_slots: u32,
}

impl Default for DenseConfig {
    fn default() -> Self {
        Self {
            tile: 16,
            bucket_bits: 12,
            slab_tris: 1024,
            nominal_layers: 4,
            max_layers: 16,
            cluster_cap: 256,
            tri_batch: 64,
            max_dense_slots: 8192,
        }
    }
}

impl DenseConfig {
    /// Plan §6 "Mobile / low shared memory (< 16 KiB)": 8x8 tiles, K = 2.
    pub fn low_shared_memory() -> Self {
        Self {
            tile: 8,
            bucket_bits: 10,
            slab_tris: 512,
            nominal_layers: 2,
            max_layers: 8,
            cluster_cap: 128,
            tri_batch: 32,
            max_dense_slots: 4096,
        }
    }

    /// Pick the default config when the device offers enough workgroup
    /// storage, else the low-memory one.
    pub fn for_workgroup_storage(max_workgroup_storage_bytes: u32) -> Self {
        let d = Self::default();
        if d.shared_memory_bytes() <= max_workgroup_storage_bytes {
            d
        } else {
            Self::low_shared_memory()
        }
    }

    pub fn threads(&self) -> u32 {
        self.tile * self.tile
    }

    pub fn buckets(&self) -> u32 {
        1 << self.bucket_bits
    }

    /// Workgroup storage the peel kernel declares: histogram, staged cluster
    /// vertices, triangle setup batch + boxes, and ~16 scalars.
    pub fn shared_memory_bytes(&self) -> u32 {
        self.buckets() / 2 * 4 + 128 * 16 + self.tri_batch * 48 + self.tri_batch * 4 + 16 * 4
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.tile != 8 && self.tile != 16 {
            return Err(format!("tile must be 8 or 16, got {}", self.tile));
        }
        if !(2..=12).contains(&self.bucket_bits) {
            return Err("bucket_bits must be in 2..=12".into());
        }
        if self.tri_batch == 0 || self.tri_batch > self.threads() {
            return Err("tri_batch must be in 1..=tile*tile".into());
        }
        if self.nominal_layers == 0 || self.max_layers < self.nominal_layers {
            return Err("need 1 <= nominal_layers <= max_layers".into());
        }
        if self.slab_tris == 0 || self.cluster_cap == 0 || self.max_dense_slots == 0 {
            return Err("slab_tris, cluster_cap and max_dense_slots must be nonzero".into());
        }
        // Histogram counts are u16 halves of u32 words.
        if self.cluster_cap * helio_vg_format::CLUSTER_MAX_TRIS >= 65536 {
            return Err("cluster_cap * 128 must stay below 65536 (u16 bucket counts)".into());
        }
        Ok(())
    }

    /// WGSL constants every Tier-3 kernel expects.
    pub fn prelude(&self) -> String {
        format!(
            "const TILE: u32 = {tile}u;\n\
             const WG_THREADS: u32 = {threads}u;\n\
             const BUCKET_BITS: u32 = {bits}u;\n\
             const NB: u32 = {nb}u;\n\
             const HIST_WORDS: u32 = {hw}u;\n\
             const SLAB_TRIS: u32 = {slab}u;\n\
             const NOMINAL_LAYERS: u32 = {nom}u;\n\
             const MAX_LAYERS: u32 = {max}u;\n\
             const CLUSTER_CAP: u32 = {cap}u;\n\
             const TRI_BATCH: u32 = {batch}u;\n\
             const TRI_BATCH_X3: u32 = {batch3}u;\n\
             const CULL_RANGES: u32 = {cr}u;\n\
             const CULL_RANGE_COUNT: u32 = {rc}u;\n",
            tile = self.tile,
            threads = self.threads(),
            bits = self.bucket_bits,
            nb = self.buckets(),
            hw = self.buckets() / 2,
            slab = self.slab_tris,
            nom = self.nominal_layers,
            max = self.max_layers,
            cap = self.cluster_cap,
            batch = self.tri_batch,
            batch3 = self.tri_batch * 3,
            cr = helio_vg_format::CULL_STATE_RANGES,
            rc = 2 * helio_vg_format::CULL_BIN_COUNT,
        )
    }
}

const COMMON: &str = include_str!("../shaders/common.wgsl");
const INPUTS: &str = include_str!("../shaders/inputs.wgsl");

/// Group 0 bindings 3 (page pool) and 4 (asset quant) read by the shared decode.
fn decode_bindings() -> String {
    helio_vg_format::decode_bindings_wgsl(0, 3, 4)
}
const TILES: &str = include_str!("../shaders/tiles.wgsl");
const ARGS: &str = include_str!("../shaders/args.wgsl");
const BIN: &str = include_str!("../shaders/bin.wgsl");
const PEEL: &str = include_str!("../shaders/peel.wgsl");
const RT: &str = include_str!("../shaders/rt.wgsl");

fn assemble(parts: &[&str]) -> String {
    parts.join("\n")
}

/// Dense-tile compaction (`compact`, `compact_tiles_only`).
pub fn tiles_source(cfg: &DenseConfig) -> String {
    assemble(&[&cfg.prelude(), helio_vg_format::WGSL, COMMON, TILES])
}

/// Indirect-args kernels (`make_args`, `make_args_rt`).
pub fn args_source(cfg: &DenseConfig) -> String {
    assemble(&[&cfg.prelude(), helio_vg_format::WGSL, COMMON, ARGS])
}

/// Cluster-to-tile binning (`bin`).
pub fn bin_source(cfg: &DenseConfig) -> String {
    assemble(&[&cfg.prelude(), helio_vg_format::WGSL, COMMON, &decode_bindings(), helio_vg_format::DECODE_WGSL, INPUTS, BIN])
}

/// Depth peel (`peel`).
pub fn peel_source(cfg: &DenseConfig) -> String {
    assemble(&[&cfg.prelude(), helio_vg_format::WGSL, COMMON, &decode_bindings(), helio_vg_format::DECODE_WGSL, INPUTS, PEEL])
}

/// Ray-query primary visibility (`trace`). Requires `EXPERIMENTAL_RAY_QUERY`.
pub fn rt_source(cfg: &DenseConfig) -> String {
    // `enable` directives must precede every declaration.
    let mut s = String::from("enable wgpu_ray_query;\n");
    s.push_str(&assemble(&[&cfg.prelude(), helio_vg_format::WGSL, COMMON, RT]));
    s
}

/// Every WGSL source this crate compiles, named (for validation tests).
pub fn validation_sources(cfg: &DenseConfig) -> Vec<(&'static str, String)> {
    vec![
        ("tiles", tiles_source(cfg)),
        ("args", args_source(cfg)),
        ("bin", bin_source(cfg)),
        ("peel", peel_source(cfg)),
        ("rt", rt_source(cfg)),
    ]
}
