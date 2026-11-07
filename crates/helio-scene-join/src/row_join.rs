//! [`RowJoin`]: a kind's component rows placed into the rows its passes
//! read, on the GPU.
//!
//! Each component instance derives a source row in its own space, keyed by
//! the instance entity. Each frame one of a table's inputs changed, the join
//! copies every *placed* source row into the table's output buffer, at the
//! same row, or packed into the leading rows for a pass that reads a fixed
//! number of them (`RowTable::packed`). A row is placed when it is attached
//! (its owner row is current), enabled, and (with
//! [`placement::GATE_HIDDEN`]) its owner visible; a [`placement`] then moves
//! it into the frame's space. Every other output row is zero, which passes
//! treat as inert.
//!
//! A table that places nothing from the owners' transforms is re-derived
//! only when its own source, the owners, generations or visibility change,
//! so moving an unrelated object does not re-derive it.
//!
//! The kind that owns the rows describes its tables and builds its own
//! `RowJoin` (fog, post-process volumes, the lens, water, foliage, the sky,
//! decals, particle emitters, sprites); several joins run side by side as
//! separate scene derivations. The join itself knows only what every kind
//! shares: attachment, visibility, gates, slices, packing and the owner's
//! world box ([`placement::OWNER_AABB`]). Anything a kind's rows need beyond
//! that is the kind's own [`Placement`]: WGSL hooks the join calls for its
//! table (see `row_join.wgsl`), compiled into a pipeline of its own, so the
//! join names no kind and a new kind places its rows without changing it.

use crate::{
    entry, signature, BufferHandle, BufferKey, SceneBufferProjection, SceneDerivation,
    SceneDerivationContext, SceneDerivationOutput, GENERATION_ROW_BYTES, HIDDEN_ROW_BYTES,
    OWNER_BOUNDS_ROW_BYTES, OWNER_ROW_BYTES, TRANSFORM_ROW_BYTES,
};

/// What the join itself does with a table's placed rows (bits; see
/// `row_join.wgsl`'s `JoinUniforms`).
pub mod placement {
    /// The source starts with a local size `vec4`, the output with its
    /// world AABB (`min`, `max`; `w` 0) from the owner's transform.
    pub const OWNER_AABB: u32 = 1;
    /// A hidden owner turns the row off.
    pub const GATE_HIDDEN: u32 = 2;
    /// Set by the join for a [`super::Placement`] that reads transforms.
    pub(crate) const READS_TRANSFORMS: u32 = 4;
}

/// A kind's placement: WGSL defining any of the row join's placement hooks
/// (`placement_placed`, `placement_write_row`, `placement_source_header`,
/// `placement_write`, `placement_request`, `placement_range`; see
/// `row_join.wgsl` for what each does and what it may read). Hooks it leaves
/// out keep their defaults. The kind's crate owns it, next to the rows it
/// places.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// Names the placement's pipeline (one per distinct name in a join).
    pub name: &'static str,
    pub wgsl: &'static str,
    /// Places from the owner's transform (an owner without one is not
    /// placed).
    pub reads_transforms: bool,
    /// Writes positions, which follow the frame's world origin.
    pub follows_origin: bool,
    /// Reads `directional_light` ([`RowJoinKeys::directional_light`]).
    pub reads_lights: bool,
    /// Reads `owner_bounds` ([`RowJoinKeys::owner_bounds`]).
    pub reads_owner_bounds: bool,
}

impl Placement {
    /// A placement named `name` from `wgsl`, reading nothing but its rows.
    pub const fn new(name: &'static str, wgsl: &'static str) -> Self {
        Self {
            name,
            wgsl,
            reads_transforms: false,
            follows_origin: false,
            reads_lights: false,
            reads_owner_bounds: false,
        }
    }

    /// Places from the owner's transform, writing positions.
    pub const fn from_owner(self) -> Self {
        Self {
            reads_transforms: true,
            follows_origin: true,
            ..self
        }
    }

    /// Reads the owner's transform without writing positions (a rotation).
    pub const fn reads_owner(self) -> Self {
        Self {
            reads_transforms: true,
            ..self
        }
    }

    /// Reads the directional light.
    pub const fn lit(self) -> Self {
        Self {
            reads_lights: true,
            ..self
        }
    }

    /// Reads the bounds of its owner's content.
    pub const fn owner_bounded(self) -> Self {
        Self {
            reads_owner_bounds: true,
            ..self
        }
    }
}

/// The hooks a placement may leave out, with their defaults.
const DEFAULT_HOOKS: [(&str, &str); 6] = [
    (
        "fn placement_placed(",
        "fn placement_placed(row: u32) -> bool { return true; }",
    ),
    (
        "fn placement_write_row(",
        "fn placement_write_row(row: u32, slot: u32, output: u32) -> bool { return false; }",
    ),
    (
        "fn placement_source_header(",
        "fn placement_source_header() -> u32 { return 0u; }",
    ),
    (
        "fn placement_write(",
        "fn placement_write(row: u32, output: u32) {}",
    ),
    (
        "fn placement_request(",
        "fn placement_request(row: u32) -> u32 { return 0u; }",
    ),
    (
        "fn placement_range(",
        "fn placement_range(row: u32, slot: u32, offset: u32) {}",
    ),
];

/// The row join's shader with `placement`'s hooks (or none: every default).
pub fn placement_source(placement: Option<&Placement>) -> String {
    let wgsl = placement.map_or("", |placement| placement.wgsl);
    let mut source = crate::row_join_base();
    source.push('\n');
    source.push_str(wgsl);
    for (signature, default) in DEFAULT_HOOKS {
        if !wgsl.contains(signature) {
            source.push('\n');
            source.push_str(default);
        }
    }
    source.push('\n');
    source
}

use placement::*;

const WORKGROUP: u32 = 64;
const NO_GATE_WORD: u32 = u32::MAX;

/// Where the rows every table reads live.
#[derive(Clone, Copy, Debug)]
pub struct RowJoinKeys {
    pub owners: BufferKey,
    pub generations: BufferKey,
    pub hidden: BufferKey,
    pub transforms: BufferKey,
    /// The local bounding spheres of content attached to owners (keyed like
    /// the owners; whichever kind publishes them: the engine names the key),
    /// for a placement bounded by its owner's content
    /// ([`Placement::owner_bounded`]).
    pub owner_bounds: BufferKey,
    /// The directional light a placement takes ([`Placement::lit`]: its
    /// first `vec4`, the direction toward it and 1), named by the kind that
    /// places a sun (water: the lights' directional lights). `None`: no
    /// sun.
    pub directional_light: Option<BufferKey>,
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
    /// Particles the pool ranges are allocated from.
    pool: u32,
    _pad: [u32; 3],
    /// The frame's world origin as an f32 pair, subtracted from placed
    /// positions: every table is in the frame's space.
    origin_hi: [f32; 4],
    origin_lo: [f32; 4],
    /// The origin's X and Z modulo the table's phase period.
    phase: [f32; 4],
}

/// One output buffer and what it is made from.
pub struct RowTable {
    label: &'static str,
    key: BufferKey,
    source: BufferKey,
    source_row_bytes: u64,
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
    pool: u32,
    phase_period: f64,
    /// The kind's placement, if its rows need one.
    placement: Option<Placement>,
    /// Sources written over this one (see [`RowTable::over`]).
    layers: Vec<RowTable>,
}

impl RowTable {
    /// A table published as `key`, of `output_words`-word rows, from the
    /// `source_row_bytes`-byte rows at `source`, placed by `flags`
    /// ([`placement`]) and, if it has one, the kind's
    /// [`placement`](RowTable::placed_by).
    pub fn new(
        label: &'static str,
        key: BufferKey,
        source: BufferKey,
        source_row_bytes: u64,
        output_words: u32,
        flags: u32,
    ) -> Self {
        Self {
            label,
            key,
            source,
            source_row_bytes,
            output_words,
            flags,
            capacity: None,
            source_offset: 0,
            copy_words: 0,
            gate_word: NO_GATE_WORD,
            pool: 0,
            phase_period: 0.0,
            placement: None,
            layers: Vec::new(),
        }
    }

    /// The kind's own placement of the rows.
    pub fn placed_by(self, placement: Placement) -> Self {
        Self {
            placement: Some(placement),
            ..self
        }
    }

    /// Packed into `capacity` leading rows.
    pub fn packed(self, capacity: u32) -> Self {
        Self {
            capacity: Some(capacity),
            ..self
        }
    }

    /// Reads `copy_words` words from `source_offset` of each source row.
    pub fn slice(self, source_offset: u32, copy_words: u32) -> Self {
        Self {
            source_offset,
            copy_words,
            ..self
        }
    }

    /// A row is placed only when its word `gate_word` is non-zero.
    pub fn gated_on(self, gate_word: u32) -> Self {
        Self { gate_word, ..self }
    }

    /// The size of the pool the placement allocates from
    /// (`placement_request`, `placement_range`).
    pub fn pool(self, pool: u32) -> Self {
        Self { pool, ..self }
    }

    /// Rows carry the origin's X and Z modulo `period` (a world tiling's
    /// phase, exact however far the origin is).
    pub fn phase_period(self, phase_period: f64) -> Self {
        Self {
            phase_period,
            ..self
        }
    }

    /// Also write `layer`'s placed rows into this table, after this
    /// table's own and any earlier layer's: a packed table's leading rows
    /// then hold the last source that placed any, so later layers take
    /// precedence. `layer` keeps its own source, slice, gate and flags; its
    /// key and capacity are this table's.
    pub fn over(mut self, layer: RowTable) -> Self {
        self.layers.push(RowTable {
            key: self.key,
            capacity: self.capacity,
            output_words: self.output_words,
            ..layer
        });
        self
    }

    /// The key the table is published as.
    pub fn key(&self) -> BufferKey {
        self.key
    }

    fn reads_lights(&self) -> bool {
        self.placement
            .is_some_and(|placement| placement.reads_lights)
    }

    fn reads_owner_bounds(&self) -> bool {
        self.placement
            .is_some_and(|placement| placement.reads_owner_bounds)
    }

    /// Whether this table holds positions, which follow the frame's world
    /// origin.
    fn follows_origin(&self) -> bool {
        self.flags & OWNER_AABB != 0
            || self
                .placement
                .is_some_and(|placement| placement.follows_origin)
    }

    fn reads_transforms(&self) -> bool {
        self.flags & OWNER_AABB != 0
            || self
                .placement
                .is_some_and(|placement| placement.reads_transforms)
    }

    fn uniform_flags(&self) -> u32 {
        let reads = self
            .placement
            .is_some_and(|placement| placement.reads_transforms);
        self.flags | if reads { READS_TRANSFORMS } else { 0 }
    }

    fn output_row_bytes(&self) -> u64 {
        u64::from(self.output_words) * 4
    }

    fn phase(&self, origin: glam::DVec3) -> [f32; 4] {
        if self.phase_period <= 0.0 {
            return [0.0; 4];
        }
        [
            origin.x.rem_euclid(self.phase_period) as f32,
            origin.z.rem_euclid(self.phase_period) as f32,
            0.0,
            0.0,
        ]
    }
}

/// One output buffer and its derivation state.
struct Table {
    spec: RowTable,
    uniforms: wgpu::Buffer,
    /// Further sources written into the same rows after the table's own,
    /// in order, each over the last where it places a row (see
    /// [`RowTable::over`]); each with its own uniforms.
    layers: Vec<(RowTable, wgpu::Buffer)>,
    buffer: wgpu::Buffer,
    rows: u32,
    epoch: u64,
    content_generation: u64,
    last_inputs: Option<u64>,
}

impl Table {
    fn new(device: &wgpu::Device, mut spec: RowTable) -> Self {
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

/// A placement's pipelines: rows in place, rows packed.
struct Pipelines {
    rows: wgpu::ComputePipeline,
    compact: wgpu::ComputePipeline,
}

impl Pipelines {
    fn new(
        device: &wgpu::Device,
        name: &'static str,
        layout: &wgpu::PipelineLayout,
        placement: Option<&Placement>,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(placement_source(placement).into()),
        });
        let pipeline = |entry_point| {
            helio_core::shader::create_compute_pipeline(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        Self {
            rows: pipeline("cs_join_rows"),
            compact: pipeline("cs_compact_rows"),
        }
    }
}

/// A kind's tables, derived from its component rows every frame one of
/// their inputs changed (see the [module docs](self)).
pub struct RowJoin {
    name: &'static str,
    keys: RowJoinKeys,
    /// Per placement (by name; `None`: the join's own): the in-place and
    /// packing pipelines.
    pipelines: Vec<(Option<&'static str>, Pipelines)>,
    /// Every placement's bindings: one layout whatever a placement reads, so
    /// one bind group serves any of them.
    bind_layout: wgpu::BindGroupLayout,
    empty: wgpu::Buffer,
    tables: Vec<Table>,
    reported: Vec<BufferKey>,
}

impl RowJoin {
    /// A join named `name` (diagnostics) publishing `tables`.
    pub fn new(
        device: &wgpu::Device,
        name: &'static str,
        keys: RowJoinKeys,
        tables: Vec<RowTable>,
    ) -> Self {
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(name),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, read),
                entry(2, read),
                entry(3, read),
                entry(4, read),
                entry(5, read),
                entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(7, read),
                entry(8, read),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let mut pipelines: Vec<(Option<&'static str>, Pipelines)> = Vec::new();
        let placements = tables.iter().flat_map(|table| {
            std::iter::once(table.placement).chain(table.layers.iter().map(|layer| layer.placement))
        });
        for placement in placements {
            let key = placement.map(|placement| placement.name);
            if !pipelines.iter().any(|(name, _)| *name == key) {
                pipelines.push((
                    key,
                    Pipelines::new(device, name, &pipeline_layout, placement.as_ref()),
                ));
            }
        }
        Self {
            name,
            keys,
            pipelines,
            bind_layout,
            empty: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Row Join Empty Input"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            tables: tables
                .into_iter()
                .map(|spec| Table::new(device, spec))
                .collect(),
            reported: Vec::new(),
        }
    }

    fn input<'a>(
        reported: &mut Vec<BufferKey>,
        name: &str,
        inputs: &'a SceneBufferProjection,
        key: BufferKey,
        row_bytes: u64,
    ) -> Option<&'a BufferHandle> {
        let handle = inputs.get(key)?;
        if handle.row_bytes != 0 && handle.row_bytes != row_bytes {
            if !reported.contains(&key) {
                reported.push(key);
                log::error!(
                    "{name}: buffer {key:?} has {}-byte rows, expected {row_bytes}; \
                     the rows it feeds are left empty",
                    handle.row_bytes
                );
            }
            return None;
        }
        Some(handle)
    }
}

impl SceneDerivation for RowJoin {
    fn name(&self) -> &'static str {
        self.name
    }

    fn derive(
        &mut self,
        ctx: &SceneDerivationContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> SceneDerivationOutput {
        let keys = self.keys;
        let name = self.name;
        let inputs = ctx.inputs;
        let reported = &mut self.reported;
        let owners = Self::input(reported, name, inputs, keys.owners, OWNER_ROW_BYTES);
        let generations = Self::input(
            reported,
            name,
            inputs,
            keys.generations,
            GENERATION_ROW_BYTES,
        );
        let hidden = Self::input(reported, name, inputs, keys.hidden, HIDDEN_ROW_BYTES);
        let transforms = Self::input(reported, name, inputs, keys.transforms, TRANSFORM_ROW_BYTES);
        let lights = keys.directional_light.and_then(|key| inputs.get(key));
        let owner_bounds = Self::input(
            reported,
            name,
            inputs,
            keys.owner_bounds,
            OWNER_BOUNDS_ROW_BYTES,
        );
        let mut recorded = false;
        let (origin_hi, origin_lo) = crate::origin_pair(ctx.world_origin);
        let origin_key = crate::origin_bits(ctx.world_origin)
            .iter()
            .fold(0u64, |key, bits| key.rotate_left(21) ^ bits);
        for table in &mut self.tables {
            let spec = &table.spec;
            let source = Self::input(reported, name, inputs, spec.source, spec.source_row_bytes);
            let rows = source.map_or(0, |s| {
                (s.buffer.size() / spec.source_row_bytes).min(u64::from(u32::MAX)) as u32
            });
            let mut grew = false;
            if spec.capacity.is_none() && rows > table.rows {
                table.rows = rows.next_power_of_two();
                table.buffer =
                    allocate(ctx.device, spec.label, spec.output_row_bytes(), table.rows);
                table.epoch += 1;
                grew = true;
            }
            let reads_transforms = spec.reads_transforms()
                || table
                    .layers
                    .iter()
                    .any(|(layer, _)| layer.reads_transforms());
            let placed_from = transforms.filter(|_| reads_transforms);
            let lit_by = lights.filter(|_| {
                spec.reads_lights() || table.layers.iter().any(|(layer, _)| layer.reads_lights())
            });
            let bounded_by = owner_bounds.filter(|_| {
                spec.reads_owner_bounds()
                    || table
                        .layers
                        .iter()
                        .any(|(layer, _)| layer.reads_owner_bounds())
            });
            let mut table_inputs = vec![
                owners,
                generations,
                hidden,
                placed_from,
                lit_by,
                bounded_by,
                source,
            ];
            for (layer, _) in &table.layers {
                table_inputs.push(Self::input(
                    reported,
                    name,
                    inputs,
                    layer.source,
                    layer.source_row_bytes,
                ));
            }
            let follows = table.spec.follows_origin()
                || table.layers.iter().any(|(layer, _)| layer.follows_origin());
            let extra = if follows {
                table.epoch ^ origin_key.rotate_left(1)
            } else {
                table.epoch
            };
            let inputs_now = signature(&table_inputs, &[extra]);
            if !grew && table.last_inputs == Some(inputs_now) {
                continue;
            }
            table.last_inputs = Some(inputs_now);
            encoder.clear_buffer(&table.buffer, 0, None);
            let passes = std::iter::once((&table.spec, &table.uniforms)).chain(
                table
                    .layers
                    .iter()
                    .map(|(layer, uniforms)| (layer, uniforms)),
            );
            for (spec, uniforms) in passes {
                let source =
                    Self::input(reported, name, inputs, spec.source, spec.source_row_bytes);
                let rows = source.map_or(0, |s| {
                    (s.buffer.size() / spec.source_row_bytes).min(u64::from(u32::MAX)) as u32
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
                        source_words: (spec.source_row_bytes / 4) as u32,
                        output_words: spec.output_words,
                        flags: spec.uniform_flags(),
                        capacity: spec.capacity.unwrap_or(0),
                        source_offset: spec.source_offset,
                        copy_words: spec.copy_words,
                        gate_word: spec.gate_word,
                        pool: spec.pool,
                        _pad: [0; 3],
                        origin_hi,
                        origin_lo,
                        phase: spec.phase(ctx.world_origin),
                    }),
                );
                let key = spec.placement.map(|placement| placement.name);
                let pipelines = &self
                    .pipelines
                    .iter()
                    .find(|(name, _)| *name == key)
                    .expect("a pipeline per placement")
                    .1;
                let (pipeline, workgroups) = match spec.capacity {
                    Some(_) => (&pipelines.compact, 1),
                    None => (&pipelines.rows, rows.div_ceil(WORKGROUP).max(1)),
                };
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(spec.label),
                    layout: &self.bind_layout,
                    entries: &[
                        entry(0, uniforms),
                        entry(1, &owners.buffer),
                        entry(2, &generations.buffer),
                        entry(3, hidden.map_or(&self.empty, |h| &h.buffer)),
                        entry(4, transforms.map_or(&self.empty, |h| &h.buffer)),
                        entry(5, &source.buffer),
                        entry(6, &table.buffer),
                        entry(7, lights.map_or(&self.empty, |h| &h.buffer)),
                        entry(8, owner_bounds.map_or(&self.empty, |h| &h.buffer)),
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
            flushed: Vec::new(),
            recorded,
        }
    }
}
