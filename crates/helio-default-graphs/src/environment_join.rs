//! The environment join: the frontend's authored fog volumes, post-process
//! volumes, camera post-process baselines, water volumes and foliage to the
//! rows the volumetric fog, post-process, water and foliage passes read, on
//! the GPU (Pulsar-Native#1035, Phase 4).
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
//! | `water_volumes` | `"water_volumes"`, packed into [`MAX_WATER_VOLUMES`] rows | as for volumes; the surface height follows the owner's Y |
//! | `foliage` | `"foliage_types"`, `"foliage_layers"`, `"foliage_wind"`, each packed | attached, enabled, owner current and visible, with a density |
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
    GENERATION_ROW_BYTES, HIDDEN_ROW_BYTES, OWNER_ROW_BYTES, TRANSFORM_ROW_BYTES,
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
/// Rows the water passes read (`helio_pass_water_sim::MAX_SIM_VOLUMES`):
/// placed water volumes beyond these are not drawn.
pub const MAX_WATER_VOLUMES: u32 = helio_pass_water_sim::MAX_SIM_VOLUMES;
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

const WORKGROUP: u32 = 64;
const SPATIAL: u32 = 1;
const GATE_HIDDEN: u32 = 2;
const SURFACE: u32 = 4;
const LAYER: u32 = 8;
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
}

impl Source {
    const ALL: [Self; 6] = [
        Self::GlobalFog,
        Self::LocalFog,
        Self::PostProcessVolumes,
        Self::CameraPostProcess,
        Self::WaterVolumes,
        Self::Foliage,
    ];

    fn key(self, keys: &EnvironmentJoinKeys) -> BufferKey {
        match self {
            Self::GlobalFog => keys.global_fog,
            Self::LocalFog => keys.local_fog,
            Self::PostProcessVolumes => keys.post_process_volumes,
            Self::CameraPostProcess => keys.camera_post_process,
            Self::WaterVolumes => keys.water_volumes,
            Self::Foliage => keys.foliage,
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

    fn reads_transforms(&self) -> bool {
        self.flags & (SPATIAL | LAYER) != 0
    }

    fn output_row_bytes(&self) -> u64 {
        u64::from(self.output_words) * 4
    }
}

/// One pass buffer and its derivation state.
struct Table {
    spec: Spec,
    uniforms: wgpu::Buffer,
    buffer: wgpu::Buffer,
    rows: u32,
    epoch: u64,
    content_generation: u64,
    last_inputs: Option<u64>,
}

impl Table {
    fn new(device: &wgpu::Device, spec: Spec) -> Self {
        let rows = spec.capacity.unwrap_or(1);
        Self {
            uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(spec.label),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
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
            Spec::new(
                "Environment Join Water Volumes",
                WATER_VOLUMES_KEY,
                Source::WaterVolumes,
                64,
                volume | SURFACE,
            )
            .packed(MAX_WATER_VOLUMES),
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
            // The foliage passes read the first wind row.
            Spec::new(
                "Environment Join Foliage Wind",
                FOLIAGE_WIND_KEY,
                Source::Foliage,
                12,
                GATE_HIDDEN,
            )
            .packed(1)
            .slice(28, 12)
            .gated_on(0),
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
            let placed_from = transforms.filter(|_| spec.reads_transforms());
            let inputs_now = signature(
                &[owners, generations, hidden, placed_from, source],
                table.epoch,
            );
            if !grew && table.last_inputs == Some(inputs_now) {
                continue;
            }
            table.last_inputs = Some(inputs_now);
            encoder.clear_buffer(&table.buffer, 0, None);
            let placeable = !spec.reads_transforms() || transforms.is_some();
            if let (Some(owners), Some(generations), Some(source), true) =
                (owners, generations, source, placeable)
            {
                ctx.queue.write_buffer(
                    &table.uniforms,
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
                        entry(0, &table.uniforms),
                        entry(1, &owners.buffer),
                        entry(2, &generations.buffer),
                        entry(3, hidden.map_or(&self.empty, |h| &h.buffer)),
                        entry(4, transforms.map_or(&self.empty, |h| &h.buffer)),
                        entry(5, &source.buffer),
                        entry(6, &table.buffer),
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
