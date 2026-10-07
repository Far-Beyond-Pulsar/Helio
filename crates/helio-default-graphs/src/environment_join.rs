//! The environment join: the frontend's authored fog volumes, post-process
//! volumes and camera post-process baselines to the rows the volumetric
//! fog and post-process passes read, on the GPU (Pulsar-Native#1035,
//! Phase 4).
//!
//! The same contract as [`crate::scene_join`]: each component instance
//! derives a source row in its own space (`helio_component`'s
//! `environment_rows`), keyed by the instance entity. Each frame one of its
//! inputs changed, the join copies every *placed* source row into the same
//! row of its pass buffer:
//!
//! | Source key | Published as | Placed when |
//! |---|---|---|
//! | `global_fog` | `"global_fog_media"` | attached, enabled, owner current and visible |
//! | `local_fog` | `"local_fog_media"` | as above; bounds = the owner-oriented box's world AABB |
//! | `post_process_volumes` | `"post_process_volumes"` | as above; bounds as above |
//! | `camera_post_process` | `"camera_postprocess"` | attached, enabled, owner current (visibility does not apply) |
//!
//! Every other row is zero, which each pass treats as inert. A volume's
//! bounds follow its owner's position, rotation and scale. The passes and
//! their fallbacks are unchanged.

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

pub const GLOBAL_FOG_KEY: BufferKey = BufferKey::of("global_fog_media");
pub const LOCAL_FOG_KEY: BufferKey = BufferKey::of("local_fog_media");
pub const POST_PROCESS_VOLUMES_KEY: BufferKey = BufferKey::of("post_process_volumes");
pub const CAMERA_POST_PROCESS_KEY: BufferKey = BufferKey::of("camera_postprocess");

const WORKGROUP: u32 = 64;
const SPATIAL: u32 = 1;
const GATE_HIDDEN: u32 = 2;

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
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    rows: u32,
    source_words: u32,
    output_words: u32,
    flags: u32,
}

/// One source kind and the pass buffer it becomes.
struct Table {
    label: &'static str,
    key: BufferKey,
    source_row_bytes: u64,
    output_row_bytes: u64,
    flags: u32,
    uniforms: wgpu::Buffer,
    buffer: wgpu::Buffer,
    rows: u32,
    epoch: u64,
    content_generation: u64,
    last_inputs: Option<u64>,
}

impl Table {
    fn new(
        device: &wgpu::Device,
        label: &'static str,
        key: BufferKey,
        source_row_bytes: u64,
        output_row_bytes: u64,
        flags: u32,
    ) -> Self {
        Self {
            label,
            key,
            source_row_bytes,
            output_row_bytes,
            flags,
            uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            buffer: allocate(device, label, output_row_bytes, 1),
            rows: 1,
            epoch: 0,
            content_generation: 0,
            last_inputs: None,
        }
    }

    fn handle(&self) -> BufferHandle {
        BufferHandle {
            buffer: self.buffer.clone(),
            epoch: self.epoch,
            row_bytes: self.output_row_bytes,
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
    empty: wgpu::Buffer,
    tables: [Table; 4],
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
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Environment Join"),
            layout: None,
            module: &module,
            entry_point: Some("cs_join_rows"),
            compilation_options: Default::default(),
            cache: None,
        });
        let volume = SPATIAL | GATE_HIDDEN;
        Self {
            keys,
            pipeline,
            empty: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Environment Join Empty Input"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            tables: [
                Table::new(
                    device,
                    "Environment Join Global Fog",
                    GLOBAL_FOG_KEY,
                    GLOBAL_FOG_SOURCE_ROW_BYTES,
                    16 * 4,
                    GATE_HIDDEN,
                ),
                Table::new(
                    device,
                    "Environment Join Local Fog",
                    LOCAL_FOG_KEY,
                    LOCAL_FOG_SOURCE_ROW_BYTES,
                    28 * 4,
                    volume,
                ),
                Table::new(
                    device,
                    "Environment Join Post-Process Volumes",
                    POST_PROCESS_VOLUMES_KEY,
                    POST_PROCESS_VOLUME_SOURCE_ROW_BYTES,
                    164 * 4,
                    volume,
                ),
                Table::new(
                    device,
                    "Environment Join Camera Post-Process",
                    CAMERA_POST_PROCESS_KEY,
                    CAMERA_POST_PROCESS_SOURCE_ROW_BYTES,
                    152 * 4,
                    0,
                ),
            ],
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
        let source_keys = [
            keys.global_fog,
            keys.local_fog,
            keys.post_process_volumes,
            keys.camera_post_process,
        ];
        let sources: Vec<Option<&BufferHandle>> = (0..4)
            .map(|i| {
                let row_bytes = self.tables[i].source_row_bytes;
                self.input(inputs, source_keys[i], row_bytes)
            })
            .collect();
        let layout = self.pipeline.get_bind_group_layout(0);
        let mut recorded = false;
        for (table, source) in self.tables.iter_mut().zip(sources) {
            let rows = source.map_or(0, |s| {
                (s.buffer.size() / table.source_row_bytes).min(u64::from(u32::MAX)) as u32
            });
            let mut grew = false;
            if rows > table.rows {
                table.rows = rows.next_power_of_two();
                table.buffer =
                    allocate(ctx.device, table.label, table.output_row_bytes, table.rows);
                table.epoch += 1;
                grew = true;
            }
            let inputs_now = signature(
                &[owners, generations, hidden, transforms, source],
                table.epoch,
            );
            if !grew && table.last_inputs == Some(inputs_now) {
                continue;
            }
            table.last_inputs = Some(inputs_now);
            encoder.clear_buffer(&table.buffer, 0, None);
            let placeable = table.flags & SPATIAL == 0 || transforms.is_some();
            if let (Some(owners), Some(generations), Some(source), true) =
                (owners, generations, source, placeable)
            {
                ctx.queue.write_buffer(
                    &table.uniforms,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        rows: rows.min(table.rows),
                        source_words: (table.source_row_bytes / 4) as u32,
                        output_words: (table.output_row_bytes / 4) as u32,
                        flags: table.flags,
                    }),
                );
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(table.label),
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
                    label: Some(table.label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(rows.div_ceil(WORKGROUP).max(1), 1, 1);
            }
            table.content_generation += 1;
            recorded = true;
        }
        SceneDerivationOutput {
            buffers: self
                .tables
                .iter()
                .map(|table| (table.key, table.handle()))
                .collect(),
            recorded,
        }
    }
}
