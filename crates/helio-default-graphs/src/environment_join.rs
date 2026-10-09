//! The environment join: the frontend's authored fog volumes, post-process
//! volumes, camera post-process baselines, water volumes, foliage,
//! atmospheres, decals and particle emitters to the rows the volumetric fog,
//! post-process, water, foliage, sky, decal and Corona passes read, on the
//! GPU (Pulsar-Native#1035, Phase 4).
//!
//! The same contract as [`crate::scene_join`]: each component instance
//! derives a source row in its own space (`helio_component`'s
//! `environment_rows`), keyed by the instance entity. Each frame one of its
//! inputs changed, the join copies every *placed* source row into its pass
//! buffer, at the same row, or packed into the leading rows for a pass that
//! reads a fixed number of them:
//!
//! | Source key | Published as | Placed when |
//! |---|---|---|
//! | `global_fog` | `"global_fog_media"` | attached, enabled, owner current and visible |
//! | `local_fog` | `"local_fog_media"` | as above; bounds = the owner-oriented box's world AABB |
//! | `post_process_volumes` | `"post_process_volumes"` | as above; bounds as above |
//! | `camera_post_process` | `"camera_postprocess"` | attached, enabled, owner current (visibility does not apply) |
//! | `water_volumes` | `"water_volumes"`, packed into [`MAX_WATER_VOLUMES`] rows | as for volumes; the surface height follows the owner's Y; the sun is the scene's first directional light |
//! | `foliage` | `"foliage_types"`, `"foliage_layers"`, `"foliage_wind"`, each packed | attached, enabled, owner current and visible, with a density |
//! | `wind` | `"foliage_wind"`, over the foliage components' own wind; the wind of every placed `"water_volumes"` row that does not opt out | attached, enabled, owner current (visibility does not apply); a foliage component that opts out of the global wind takes precedence |
//! | `atmospheres` | `"atmospheres"` | attached, enabled, owner current (visibility does not apply); a planet placed at its owner is centred on the owner's position |
//! | `decals` | `"decals"`, packed into [`MAX_DECALS`] rows | attached, enabled, owner current and visible, with a non-zero box; the transform maps world space into the owner-placed box |
//! | `corona_emitters` | `"corona_emitters"`, packed into [`MAX_CORONA_EMITTERS`] rows | attached, enabled, owner current and visible, requesting particles; placed at the owner's transform, with a range of the Corona particle pool |
//!
//! Particle emitters share the Corona pass's particle pool
//! ([`CORONA_POOL_PARTICLES`]): as the join packs them it gives each, in
//! row order, the next contiguous range of the size it requests (rounded up
//! to [`CORONA_RANGE_ALIGNMENT`]), a prefix sum on the GPU.
//! A range that does not fit is clamped to what is left of the pool, so an
//! emitter past the end of a full pool has none and draws nothing. Each row
//! also carries its emitter's identity (its source row + 1, in the
//! `spawn_cursor` word, which the pass keeps itself): the pass restarts a
//! row's particles when another emitter takes it.
//!
//! A water row's sun (`sun_direction`, toward the sun, `w` 1) is the first
//! directional light with any intensity in the scene join's
//! `"scene_lights"` ([`LIGHTS_KEY`]), as the sky's atmosphere picks its sun;
//! straight up with `w` 0 when the scene has none. A water row's wind is
//! the level's global wind unless the volume opts out (`wind_params.w`):
//! its direction's XZ and [`WATER_WIND_STRENGTH_PER_SPEED`] times its speed.
//!
//! Every other row is zero, which each pass treats as inert. A volume's
//! bounds follow its owner's position, rotation and scale; a foliage layer is
//! a world-aligned square centred on its owner. The passes and their
//! fallbacks are unchanged.
//!
//! A table that places nothing from the owners' transforms is re-derived
//! only when its own source, the owners, generations or visibility change,
//! so moving an unrelated object does not re-derive it (and does not make
//! the foliage passes re-grow their tiles).

pub use helio_core::{
    BufferHandle, BufferKey, SceneBufferProjection, SceneDerivation, SceneDerivationContext,
    SceneDerivationOutput,
};

use crate::scene_join::{
    GENERATION_ROW_BYTES, HIDDEN_ROW_BYTES, LIGHTS_KEY, OWNER_ROW_BYTES, TRANSFORM_ROW_BYTES,
};

/// `GlobalFogSourceRow`: the `GlobalFogComponent` pass row (16 words).
pub const GLOBAL_FOG_SOURCE_ROW_BYTES: u64 = 16 * 4;
/// `LocalFogSourceRow`: local size vec4, medium (16), edge fade vec4.
pub const LOCAL_FOG_SOURCE_ROW_BYTES: u64 = 24 * 4;
/// `PostProcessVolumeSourceRow`: local size vec4, then the pass row after
/// its bounds (156 words).
pub const POST_PROCESS_VOLUME_SOURCE_ROW_BYTES: u64 = 160 * 4;
/// `CameraPostProcessSourceRow`: the `CameraPostProcessComponent` pass row.
pub const CAMERA_POST_PROCESS_SOURCE_ROW_BYTES: u64 = 152 * 4;
/// `WaterVolumeSourceRow`: local size (`w`: surface height above the
/// owner), then the water volume row after its bounds (56 words).
pub const WATER_VOLUME_SOURCE_ROW_BYTES: u64 = 60 * 4;
/// `FoliageSourceRow`: the foliage type row (24 words; its density first),
/// the layer (half extent, infinite flag, altitude min and max) and the
/// wind row (12 words).
pub const FOLIAGE_SOURCE_ROW_BYTES: u64 = 40 * 4;
/// `AtmosphereSourceRow`: the `helio_pass_sky::AtmosphereComponent` pass
/// row, its centre in the owner's space.
pub const ATMOSPHERE_SOURCE_ROW_BYTES: u64 = 28 * 4;
/// `DecalSourceRow`: local size vec4, then the `helio_pass_decal`
/// `DecalComponent` row (32 words) with its transform left to the join.
pub const DECAL_SOURCE_ROW_BYTES: u64 = 36 * 4;
/// Rows the decal pass reads (`helio_pass_decal::MAX_DECALS`): placed decals
/// beyond these are not drawn.
pub const MAX_DECALS: u32 = helio_pass_decal::MAX_DECALS;
/// `CoronaEmitterSourceRow`: the `helio_pass_corona` emitter row (60 words)
/// with its transform and particle offset left to the join and the
/// requested particles in `particle_count`.
pub const CORONA_EMITTER_SOURCE_ROW_BYTES: u64 = 60 * 4;
/// Emitter rows the Corona pass reads: placed emitters beyond these are not
/// drawn.
pub const MAX_CORONA_EMITTERS: u32 = helio_pass_corona::CORONA_MAX_EMITTERS;
/// The Corona particle pool the join allocates emitter ranges from.
pub const CORONA_POOL_PARTICLES: u32 = helio_pass_corona::CORONA_MAX_PARTICLES;
/// Every emitter range starts on a multiple of this.
pub const CORONA_RANGE_ALIGNMENT: u32 = helio_pass_corona::CORONA_RANGE_ALIGNMENT;
/// `particle_count`: the requested range in the source, the allocated one
/// out (`environment_join.wgsl`'s `EMITTER_COUNT_WORD`).
const CORONA_COUNT_WORD: u32 = 46;
/// Rows the water passes read (`helio_pass_water_sim::MAX_SIM_VOLUMES`):
/// placed water volumes beyond these are not drawn.
pub const MAX_WATER_VOLUMES: u32 = helio_pass_water_sim::MAX_SIM_VOLUMES;
/// `GlobalWindSourceRow`: the foliage wind row (12 words), marked global.
pub const WIND_SOURCE_ROW_BYTES: u64 = 12 * 4;
/// A water row's wind strength per m/s of the global wind's speed
/// (`environment_join.wgsl`'s `WATER_WIND_PER_SPEED`): a 2 m/s breeze makes
/// gentle ripples (1), 10 m/s a choppy surface (5).
pub const WATER_WIND_STRENGTH_PER_SPEED: f32 = 0.5;
/// The wind row's `_pad[0]`: 1 in a global wind's row.
const GLOBAL_WIND_MARK_WORD: u32 = 10;
/// A foliage source row's wind `_pad[0]` (28 + 10): 1 when the component
/// opts out of the global wind.
const FOLIAGE_OWN_WIND_WORD: u32 = 38;
/// Foliage types the join publishes. Placement draws each candidate's type
/// from the live leading rows, so the table's capacity costs only a short
/// per-workgroup scan.
pub const MAX_FOLIAGE_TYPES: u32 = 64;
/// Foliage layers the join publishes (`helio_pass_foliage_place`'s table).
pub const MAX_FOLIAGE_LAYERS: u32 = helio_pass_foliage_place::MAX_FOLIAGE_LAYERS;

pub const GLOBAL_FOG_KEY: BufferKey = BufferKey::of("global_fog_media");
pub const LOCAL_FOG_KEY: BufferKey = BufferKey::of("local_fog_media");
pub const POST_PROCESS_VOLUMES_KEY: BufferKey = BufferKey::of("post_process_volumes");
pub const CAMERA_POST_PROCESS_KEY: BufferKey = BufferKey::of("camera_postprocess");
pub const WATER_VOLUMES_KEY: BufferKey = BufferKey::of("water_volumes");
pub const FOLIAGE_TYPES_KEY: BufferKey = BufferKey::of("foliage_types");
pub const FOLIAGE_LAYERS_KEY: BufferKey = BufferKey::of("foliage_layers");
pub const FOLIAGE_WIND_KEY: BufferKey = BufferKey::of("foliage_wind");
pub const ATMOSPHERES_KEY: BufferKey = BufferKey::of("atmospheres");
pub const DECALS_KEY: BufferKey = BufferKey::of("decals");
pub const CORONA_EMITTERS_KEY: BufferKey = BufferKey::of("corona_emitters");

const WORKGROUP: u32 = 64;
const SPATIAL: u32 = 1;
const GATE_HIDDEN: u32 = 2;
const SURFACE: u32 = 4;
const LAYER: u32 = 8;
const CENTERED: u32 = 16;
const DECAL: u32 = 32;
const EMITTER: u32 = 64;
const SUN: u32 = 128;
const WATER_WIND: u32 = 256;
const NO_GATE_WORD: u32 = u32::MAX;

/// Where the frontend's rows live.
#[derive(Clone, Copy, Debug)]
pub struct EnvironmentJoinKeys {
    pub owners: BufferKey,
    pub generations: BufferKey,
    pub hidden: BufferKey,
    pub transforms: BufferKey,
    pub global_fog: BufferKey,
    pub local_fog: BufferKey,
    pub post_process_volumes: BufferKey,
    pub camera_post_process: BufferKey,
    pub water_volumes: BufferKey,
    pub foliage: BufferKey,
    pub atmospheres: BufferKey,
    pub decals: BufferKey,
    pub corona_emitters: BufferKey,
    pub wind: BufferKey,
}

/// The source buffers, in [`EnvironmentJoinKeys`] order.
#[derive(Clone, Copy)]
enum Source {
    GlobalFog,
    LocalFog,
    PostProcessVolumes,
    CameraPostProcess,
    WaterVolumes,
    Foliage,
    Atmospheres,
    Decals,
    CoronaEmitters,
    Wind,
}

impl Source {
    const ALL: [Self; 10] = [
        Self::GlobalFog,
        Self::LocalFog,
        Self::PostProcessVolumes,
        Self::CameraPostProcess,
        Self::WaterVolumes,
        Self::Foliage,
        Self::Atmospheres,
        Self::Decals,
        Self::CoronaEmitters,
        Self::Wind,
    ];

    fn key(self, keys: &EnvironmentJoinKeys) -> BufferKey {
        match self {
            Self::GlobalFog => keys.global_fog,
            Self::LocalFog => keys.local_fog,
            Self::PostProcessVolumes => keys.post_process_volumes,
            Self::CameraPostProcess => keys.camera_post_process,
            Self::WaterVolumes => keys.water_volumes,
            Self::Foliage => keys.foliage,
            Self::Atmospheres => keys.atmospheres,
            Self::Decals => keys.decals,
            Self::CoronaEmitters => keys.corona_emitters,
            Self::Wind => keys.wind,
        }
    }

    fn row_bytes(self) -> u64 {
        match self {
            Self::GlobalFog => GLOBAL_FOG_SOURCE_ROW_BYTES,
            Self::LocalFog => LOCAL_FOG_SOURCE_ROW_BYTES,
            Self::PostProcessVolumes => POST_PROCESS_VOLUME_SOURCE_ROW_BYTES,
            Self::CameraPostProcess => CAMERA_POST_PROCESS_SOURCE_ROW_BYTES,
            Self::WaterVolumes => WATER_VOLUME_SOURCE_ROW_BYTES,
            Self::Foliage => FOLIAGE_SOURCE_ROW_BYTES,
            Self::Atmospheres => ATMOSPHERE_SOURCE_ROW_BYTES,
            Self::Decals => DECAL_SOURCE_ROW_BYTES,
            Self::CoronaEmitters => CORONA_EMITTER_SOURCE_ROW_BYTES,
            Self::Wind => WIND_SOURCE_ROW_BYTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    rows: u32,
    source_words: u32,
    output_words: u32,
    flags: u32,
    capacity: u32,
    source_offset: u32,
    copy_words: u32,
    gate_word: u32,
    /// Particles the emitter ranges are allocated from (`EMITTER` tables).
    pool: u32,
    _pad: [u32; 3],
}

/// What one pass buffer is made from.
struct Spec {
    label: &'static str,
    key: BufferKey,
    source: Source,
    output_words: u32,
    flags: u32,
    /// Packed into this many leading rows (`cs_compact_rows`) instead of
    /// keeping the source row index.
    capacity: Option<u32>,
    /// The word of the source row this table starts reading at.
    source_offset: u32,
    /// Words copied after the headers; 0 copies as many as both rows hold.
    copy_words: u32,
    /// A source word (from the start of the row) that must be non-zero for
    /// the row to be placed, or [`NO_GATE_WORD`].
    gate_word: u32,
    /// Sources written over this one (see [`Spec::over`]).
    layers: Vec<Spec>,
}

impl Spec {
    fn new(
        label: &'static str,
        key: BufferKey,
        source: Source,
        output_words: u32,
        flags: u32,
    ) -> Self {
        Self {
            label,
            key,
            source,
            output_words,
            flags,
            capacity: None,
            source_offset: 0,
            copy_words: 0,
            gate_word: NO_GATE_WORD,
            layers: Vec::new(),
        }
    }

    fn packed(self, capacity: u32) -> Self {
        Self {
            capacity: Some(capacity),
            ..self
        }
    }

    fn slice(self, source_offset: u32, copy_words: u32) -> Self {
        Self {
            source_offset,
            copy_words,
            ..self
        }
    }

    fn gated_on(self, gate_word: u32) -> Self {
        Self { gate_word, ..self }
    }

    /// Also write `layer`'s placed rows into this table, after this spec's
    /// own and any earlier layer's: a packed table's leading rows then hold
    /// the last source that placed any, so later layers take precedence.
    /// `layer` keeps its own source, slice, gate and flags; its key and
    /// capacity are this table's.
    fn over(mut self, layer: Spec) -> Self {
        self.layers.push(Spec {
            key: self.key,
            capacity: self.capacity,
            output_words: self.output_words,
            ..layer
        });
        self
    }

    fn reads_lights(&self) -> bool {
        self.flags & SUN != 0
    }

    fn reads_transforms(&self) -> bool {
        self.flags & (SPATIAL | LAYER | CENTERED | DECAL | EMITTER) != 0
    }

    fn output_row_bytes(&self) -> u64 {
        u64::from(self.output_words) * 4
    }
}

/// One pass buffer and its derivation state.
struct Table {
    spec: Spec,
    uniforms: wgpu::Buffer,
    /// Further sources written into the same rows after the table's own,
    /// in order, each over the last where it places a row (see
    /// [`Spec::over`]); each with its own uniforms.
    layers: Vec<(Spec, wgpu::Buffer)>,
    buffer: wgpu::Buffer,
    rows: u32,
    epoch: u64,
    content_generation: u64,
    last_inputs: Option<u64>,
}

impl Table {
    fn new(device: &wgpu::Device, mut spec: Spec) -> Self {
        let rows = spec.capacity.unwrap_or(1);
        let uniforms = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let layers = std::mem::take(&mut spec.layers)
            .into_iter()
            .map(|layer| {
                let buffer = uniforms(layer.label);
                (layer, buffer)
            })
            .collect();
        Self {
            uniforms: uniforms(spec.label),
            layers,
            buffer: allocate(device, spec.label, spec.output_row_bytes(), rows),
            rows,
            epoch: 0,
            content_generation: 0,
            last_inputs: None,
            spec,
        }
    }

    fn handle(&self) -> BufferHandle {
        BufferHandle {
            buffer: self.buffer.clone(),
            epoch: self.epoch,
            row_bytes: self.spec.output_row_bytes(),
            content_generation: self.content_generation,
        }
    }
}

fn allocate(device: &wgpu::Device, label: &str, row_bytes: u64, rows: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: row_bytes * u64::from(rows.max(1)),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// The environment join as a scene derivation. See the module doc.
pub struct EnvironmentJoin {
    keys: EnvironmentJoinKeys,
    pipeline: wgpu::ComputePipeline,
    compact: wgpu::ComputePipeline,
    empty: wgpu::Buffer,
    tables: Vec<Table>,
    reported: Vec<BufferKey>,
}

impl EnvironmentJoin {
    pub fn new(device: &wgpu::Device, keys: EnvironmentJoinKeys) -> Self {
        let common = include_str!("../shaders/scene_join_common.wgsl");
        let source = include_str!("../shaders/environment_join.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Environment Join"),
            source: wgpu::ShaderSource::Wgsl(format!("{common}\n{source}").into()),
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Environment Join"),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let volume = SPATIAL | GATE_HIDDEN;
        let specs = [
            Spec::new(
                "Environment Join Global Fog",
                GLOBAL_FOG_KEY,
                Source::GlobalFog,
                16,
                GATE_HIDDEN,
            ),
            Spec::new(
                "Environment Join Local Fog",
                LOCAL_FOG_KEY,
                Source::LocalFog,
                28,
                volume,
            ),
            Spec::new(
                "Environment Join Post-Process Volumes",
                POST_PROCESS_VOLUMES_KEY,
                Source::PostProcessVolumes,
                164,
                volume,
            ),
            Spec::new(
                "Environment Join Camera Post-Process",
                CAMERA_POST_PROCESS_KEY,
                Source::CameraPostProcess,
                152,
                0,
            ),
            // Each placed volume takes the scene's sun, and the level's
            // global wind unless it opts out (its `wind_params.w`).
            Spec::new(
                "Environment Join Water Volumes",
                WATER_VOLUMES_KEY,
                Source::WaterVolumes,
                64,
                volume | SURFACE | SUN,
            )
            .packed(MAX_WATER_VOLUMES)
            .over(
                Spec::new(
                    "Environment Join Water Global Wind",
                    WATER_VOLUMES_KEY,
                    Source::Wind,
                    64,
                    WATER_WIND,
                )
                .gated_on(GLOBAL_WIND_MARK_WORD),
            ),
            // One foliage source row feeds three tables; each takes the rows
            // whose type has a density (word 0).
            Spec::new(
                "Environment Join Foliage Types",
                FOLIAGE_TYPES_KEY,
                Source::Foliage,
                24,
                GATE_HIDDEN,
            )
            .packed(MAX_FOLIAGE_TYPES)
            .slice(0, 24)
            .gated_on(0),
            Spec::new(
                "Environment Join Foliage Layers",
                FOLIAGE_LAYERS_KEY,
                Source::Foliage,
                8,
                GATE_HIDDEN | LAYER,
            )
            .packed(MAX_FOLIAGE_LAYERS)
            .slice(24, 0)
            .gated_on(0),
            // The foliage passes read one wind row: the first foliage
            // component's own wind, under the level's global wind, under
            // the first foliage component that opts out of the global wind
            // (its wind's `_pad[0]`, source word 38).
            Spec::new(
                "Environment Join Foliage Wind",
                FOLIAGE_WIND_KEY,
                Source::Foliage,
                12,
                GATE_HIDDEN,
            )
            .packed(1)
            .slice(28, 12)
            .gated_on(0)
            .over(
                Spec::new(
                    "Environment Join Global Wind",
                    FOLIAGE_WIND_KEY,
                    Source::Wind,
                    12,
                    0,
                )
                .gated_on(GLOBAL_WIND_MARK_WORD),
            )
            .over(
                Spec::new(
                    "Environment Join Foliage Own Wind",
                    FOLIAGE_WIND_KEY,
                    Source::Foliage,
                    12,
                    GATE_HIDDEN,
                )
                .slice(28, 12)
                .gated_on(FOLIAGE_OWN_WIND_WORD),
            ),
            Spec::new(
                "Environment Join Atmospheres",
                ATMOSPHERES_KEY,
                Source::Atmospheres,
                28,
                CENTERED,
            ),
            // The pass walks a fixed number of rows.
            Spec::new(
                "Environment Join Decals",
                DECALS_KEY,
                Source::Decals,
                32,
                GATE_HIDDEN | DECAL,
            )
            .packed(MAX_DECALS),
            // The pass reads a fixed number of rows; each placed emitter
            // takes the next range of the particle pool.
            Spec::new(
                "Environment Join Corona Emitters",
                CORONA_EMITTERS_KEY,
                Source::CoronaEmitters,
                60,
                GATE_HIDDEN | EMITTER,
            )
            .packed(MAX_CORONA_EMITTERS)
            .gated_on(CORONA_COUNT_WORD),
        ];
        Self {
            keys,
            pipeline: pipeline("cs_join_rows"),
            compact: pipeline("cs_compact_rows"),
            empty: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Environment Join Empty Input"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            tables: specs
                .into_iter()
                .map(|spec| Table::new(device, spec))
                .collect(),
            reported: Vec::new(),
        }
    }

    fn input<'a>(
        &mut self,
        inputs: &'a SceneBufferProjection,
        key: BufferKey,
        row_bytes: u64,
    ) -> Option<&'a BufferHandle> {
        let handle = inputs.get(key)?;
        if handle.row_bytes != 0 && handle.row_bytes != row_bytes {
            if !self.reported.contains(&key) {
                self.reported.push(key);
                log::error!(
                    "environment join: buffer {key:?} has {}-byte rows, expected {row_bytes}; \
                     the rows it feeds are left empty",
                    handle.row_bytes
                );
            }
            return None;
        }
        Some(handle)
    }
}

fn signature(handles: &[Option<&BufferHandle>], extra: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    };
    for handle in handles {
        match handle {
            Some(handle) => {
                mix(handle.epoch);
                mix(handle.content_generation);
                mix(handle.buffer.size());
            }
            None => mix(u64::MAX),
        }
    }
    mix(extra);
    hash
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl SceneDerivation for EnvironmentJoin {
    fn name(&self) -> &'static str {
        "EnvironmentJoin"
    }

    fn derive(
        &mut self,
        ctx: &SceneDerivationContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> SceneDerivationOutput {
        let keys = self.keys;
        let inputs = ctx.inputs;
        let owners = self.input(inputs, keys.owners, OWNER_ROW_BYTES);
        let generations = self.input(inputs, keys.generations, GENERATION_ROW_BYTES);
        let hidden = self.input(inputs, keys.hidden, HIDDEN_ROW_BYTES);
        let transforms = self.input(inputs, keys.transforms, TRANSFORM_ROW_BYTES);
        // The scene join's light rows, for the water's sun. Its output, so
        // its row size is the pass's, not a frontend layout.
        let lights = inputs.get(LIGHTS_KEY);
        let sources: Vec<Option<&BufferHandle>> = Source::ALL
            .iter()
            .map(|source| self.input(inputs, source.key(&keys), source.row_bytes()))
            .collect();
        let mut recorded = false;
        for table in &mut self.tables {
            let spec = &table.spec;
            let source_row_bytes = spec.source.row_bytes();
            let source = sources[spec.source as usize];
            let rows = source.map_or(0, |s| {
                (s.buffer.size() / source_row_bytes).min(u64::from(u32::MAX)) as u32
            });
            let mut grew = false;
            if spec.capacity.is_none() && rows > table.rows {
                table.rows = rows.next_power_of_two();
                table.buffer =
                    allocate(ctx.device, spec.label, spec.output_row_bytes(), table.rows);
                table.epoch += 1;
                grew = true;
            }
            // Transforms only matter to tables placed from them.
            let reads_transforms = spec.reads_transforms()
                || table
                    .layers
                    .iter()
                    .any(|(layer, _)| layer.reads_transforms());
            let placed_from = transforms.filter(|_| reads_transforms);
            let lit_by = lights.filter(|_| spec.reads_lights());
            let mut table_inputs = vec![owners, generations, hidden, placed_from, lit_by, source];
            table_inputs.extend(
                table
                    .layers
                    .iter()
                    .map(|(layer, _)| sources[layer.source as usize]),
            );
            let inputs_now = signature(&table_inputs, table.epoch);
            if !grew && table.last_inputs == Some(inputs_now) {
                continue;
            }
            table.last_inputs = Some(inputs_now);
            encoder.clear_buffer(&table.buffer, 0, None);
            // The table's own source, then each layer over it in order.
            let passes = std::iter::once((&table.spec, &table.uniforms)).chain(
                table
                    .layers
                    .iter()
                    .map(|(layer, uniforms)| (layer, uniforms)),
            );
            for (spec, uniforms) in passes {
                let source_row_bytes = spec.source.row_bytes();
                let source = sources[spec.source as usize];
                let rows = source.map_or(0, |s| {
                    (s.buffer.size() / source_row_bytes).min(u64::from(u32::MAX)) as u32
                });
                let placeable = !spec.reads_transforms() || transforms.is_some();
                let (Some(owners), Some(generations), Some(source), true) =
                    (owners, generations, source, placeable)
                else {
                    continue;
                };
                ctx.queue.write_buffer(
                    uniforms,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        rows: if spec.capacity.is_some() {
                            rows
                        } else {
                            rows.min(table.rows)
                        },
                        source_words: (source_row_bytes / 4) as u32,
                        output_words: spec.output_words,
                        flags: spec.flags,
                        capacity: spec.capacity.unwrap_or(0),
                        source_offset: spec.source_offset,
                        copy_words: spec.copy_words,
                        gate_word: spec.gate_word,
                        pool: CORONA_POOL_PARTICLES,
                        _pad: [0; 3],
                    }),
                );
                let (pipeline, workgroups) = match spec.capacity {
                    // One workgroup walks every row in order.
                    Some(_) => (&self.compact, 1),
                    None => (&self.pipeline, rows.div_ceil(WORKGROUP).max(1)),
                };
                let layout = pipeline.get_bind_group_layout(0);
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(spec.label),
                    layout: &layout,
                    entries: &[
                        entry(0, uniforms),
                        entry(1, &owners.buffer),
                        entry(2, &generations.buffer),
                        entry(3, hidden.map_or(&self.empty, |h| &h.buffer)),
                        entry(4, transforms.map_or(&self.empty, |h| &h.buffer)),
                        entry(5, &source.buffer),
                        entry(6, &table.buffer),
                        entry(7, lights.map_or(&self.empty, |h| &h.buffer)),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(spec.label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            table.content_generation += 1;
            recorded = true;
        }
        SceneDerivationOutput {
            buffers: self
                .tables
                .iter()
                .map(|table| (table.spec.key, table.handle()))
                .collect(),
            recorded,
        }
    }
}
