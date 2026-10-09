//! The scene join: the frontend's authored component rows to the object,
//! material and light rows Helio's passes draw from, on the GPU
//! (Pulsar-Native#1035, Phase 2; `03-gpu-reflection-and-rendering.md`).
//!
//! A frontend whose components are separate entities linked to their owner
//! objects (a mesh instance, a light instance, each joined to its object's
//! transform and visibility) installs a [`SceneJoin`] as a scene derivation
//! ([`helio::RendererBuilder::with_scene_derivation`]). Each frame its input
//! rows changed, the join recomputes, for every instance:
//!
//! - **meshes**: one `StaticObjectComponent`-shaped row and one material row
//!   per section of every placed mesh instance, published as
//!   `"static_objects"`/`"materials"` for the object-batch, G-buffer, shadow
//!   and virtual-geometry passes;
//! - **lights**: one `GpuLight` row per placed light instance, positioned and
//!   oriented by its owner, published as `"scene_lights"`; and, in editor
//!   mode, a billboard per light as `"billboard_instances"`.
//!
//! "Placed" means: the instance is attached and enabled, the generation it
//! recorded for its owner is the owner's current one, the owner is not
//! hidden and has a transform. Nothing else is drawn, so a mesh added,
//! edited, disabled, re-enabled or removed through any write path appears
//! or disappears the frame its rows change, with no notification, armed
//! subscription or CPU-side projection involved. The join owns its outputs
//! (they are execution data, rebuilt from the inputs) and keeps no copy of
//! the scene.
//!
//! The frontend names its buffers in [`SceneJoinKeys`]; the expected row
//! layouts are the `*_ROW_BYTES` constants below and the structs in
//! `shaders/scene_join_*.wgsl`. An input whose rows do not have the
//! expected size is reported once and that half of the join is skipped
//! (nothing drawn) rather than read with the wrong layout.
//!
//! Requires up to 13 storage buffers in one compute stage (native backends
//! report far more; WebGPU's default of 8 does not suffice).

pub use helio_core::{
    BufferHandle, BufferKey, SceneBufferProjection, SceneDerivation, SceneDerivationContext,
    SceneDerivationOutput, ENTITY_GENERATIONS_KEY,
};

/// `ComponentOwner`: `owner_index`, `owner_generation`, `enabled` (u32 each).
pub const OWNER_ROW_BYTES: u64 = 12;
/// One `u32` generation per entity index.
pub const GENERATION_ROW_BYTES: u64 = 4;
/// An object's hidden flag (`u32`, zero = visible).
pub const HIDDEN_ROW_BYTES: u64 = 4;
/// An object's transform: position, YXZ Euler rotation in degrees, scale
/// (three `[f32; 3]`).
pub const TRANSFORM_ROW_BYTES: u64 = 36;
/// A var-len handle: `offset`, `count` (u32 each).
pub const HANDLE_ROW_BYTES: u64 = 8;
/// A mesh instance's local bounding sphere `[center.xyz, radius]`.
pub const MESH_BOUNDS_ROW_BYTES: u64 = 16;
/// A mesh instance's object-row flags (`u32`).
pub const MESH_FLAGS_ROW_BYTES: u64 = 4;
/// One mesh section: draw range, material class and graph hash, then a
/// `helio_mats::GpuMaterial`.
pub const MESH_SECTION_ROW_BYTES: u64 = 128;
/// A `GpuLight` in light space, bit 31 of its `_pad` word carrying `enabled`.
pub const LIGHT_SOURCE_ROW_BYTES: u64 = 128;

const OBJECT_ROW_BYTES: u64 = 236;
const MATERIAL_ROW_BYTES: u64 = 96;
const LIGHT_ROW_BYTES: u64 = 128;
const BILLBOARD_ROW_BYTES: u64 = 48;
const WORKGROUP: u32 = 64;

/// Where the frontend's authored rows live. See the module doc for what
/// each holds.
#[derive(Clone, Copy, Debug)]
pub struct SceneJoinKeys {
    /// Instance -> owner links, keyed by the instance entity.
    pub owners: BufferKey,
    /// Entity generations (`helio_core::ENTITY_GENERATIONS_KEY`).
    pub generations: BufferKey,
    /// Object hidden flags, keyed by the object entity.
    pub hidden: BufferKey,
    /// Object transforms, keyed by the object entity.
    pub transforms: BufferKey,
    /// Mesh instances' vertex ranges (handle table of the vertex pool).
    pub vertex_handles: BufferKey,
    /// Mesh instances' index ranges (handle table of the index pool).
    pub index_handles: BufferKey,
    pub mesh_bounds: BufferKey,
    pub mesh_flags: BufferKey,
    /// Mesh instances' section ranges (handle table of `mesh_sections`).
    pub section_handles: BufferKey,
    /// The section pool; one output object row per pool slot.
    pub mesh_sections: BufferKey,
    /// Light instances' light-space rows, keyed by the instance entity.
    pub light_sources: BufferKey,
}

pub const OBJECTS_KEY: BufferKey = BufferKey::of("static_objects");
pub const MATERIALS_KEY: BufferKey = BufferKey::of("materials");
pub const LIGHTS_KEY: BufferKey = BufferKey::of("scene_lights");
pub const BILLBOARDS_KEY: BufferKey = BufferKey::of("billboard_instances");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    rows: u32,
    second: u32,
    _pad: [u32; 2],
}

/// A buffer the join owns and publishes.
struct Output {
    label: &'static str,
    buffer: wgpu::Buffer,
    rows: u32,
    row_bytes: u64,
    epoch: u64,
    content_generation: u64,
}

impl Output {
    fn new(device: &wgpu::Device, label: &'static str, row_bytes: u64) -> Self {
        Self {
            label,
            buffer: Self::allocate(device, label, row_bytes, 1),
            rows: 1,
            row_bytes,
            epoch: 0,
            content_generation: 0,
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

    /// Hold at least `rows` rows (doubling), returning whether it grew.
    fn ensure(&mut self, device: &wgpu::Device, rows: u32) -> bool {
        if rows <= self.rows {
            return false;
        }
        let rows = rows.next_power_of_two();
        self.buffer = Self::allocate(device, self.label, self.row_bytes, rows);
        self.rows = rows;
        self.epoch += 1;
        true
    }

    fn handle(&self) -> BufferHandle {
        BufferHandle {
            buffer: self.buffer.clone(),
            epoch: self.epoch,
            row_bytes: self.row_bytes,
            content_generation: self.content_generation,
        }
    }
}

/// The scene join as a scene derivation. See the module doc.
pub struct SceneJoin {
    keys: SceneJoinKeys,
    billboards: bool,
    mesh_pipeline: wgpu::ComputePipeline,
    light_pipeline: wgpu::ComputePipeline,
    mesh_uniforms: wgpu::Buffer,
    light_uniforms: wgpu::Buffer,
    /// Bound in place of an optional input that is not registered yet.
    empty: wgpu::Buffer,
    objects: Output,
    materials: Output,
    lights: Output,
    billboard_rows: Output,
    last_mesh_inputs: Option<u64>,
    last_light_inputs: Option<u64>,
    reported: Vec<BufferKey>,
}

impl SceneJoin {
    /// `billboards`: also publish a billboard per placed light (editor
    /// light icons).
    pub fn new(device: &wgpu::Device, keys: SceneJoinKeys, billboards: bool) -> Self {
        let common = include_str!("../shaders/scene_join_common.wgsl");
        let pipeline = |label: &str, source: &str, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(format!("{common}\n{source}").into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let uniforms = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            keys,
            billboards,
            mesh_pipeline: pipeline(
                "Scene Join Meshes",
                include_str!("../shaders/scene_join_meshes.wgsl"),
                "cs_join_meshes",
            ),
            light_pipeline: pipeline(
                "Scene Join Lights",
                include_str!("../shaders/scene_join_lights.wgsl"),
                "cs_join_lights",
            ),
            mesh_uniforms: uniforms("Scene Join Mesh Uniforms"),
            light_uniforms: uniforms("Scene Join Light Uniforms"),
            empty: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Scene Join Empty Input"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            objects: Output::new(device, "Scene Join Objects", OBJECT_ROW_BYTES),
            materials: Output::new(device, "Scene Join Materials", MATERIAL_ROW_BYTES),
            lights: Output::new(device, "Scene Join Lights", LIGHT_ROW_BYTES),
            billboard_rows: Output::new(device, "Scene Join Billboards", BILLBOARD_ROW_BYTES),
            last_mesh_inputs: None,
            last_light_inputs: None,
            reported: Vec::new(),
        }
    }

    /// `key`'s buffer when it is registered with the expected row size.
    /// A size mismatch is reported once and treated as missing.
    fn input<'a>(
        &mut self,
        inputs: &'a helio_core::SceneBufferProjection,
        key: BufferKey,
        row_bytes: u64,
    ) -> Option<&'a BufferHandle> {
        let handle = inputs.get(key)?;
        if handle.row_bytes != 0 && handle.row_bytes != row_bytes {
            if !self.reported.contains(&key) {
                self.reported.push(key);
                log::error!(
                    "scene join: buffer {key:?} has {}-byte rows, expected {row_bytes}; \
                     the join that reads it is skipped",
                    handle.row_bytes
                );
            }
            return None;
        }
        Some(handle)
    }
}

fn rows_of(handle: &BufferHandle, row_bytes: u64) -> u32 {
    (handle.buffer.size() / row_bytes).min(u64::from(u32::MAX)) as u32
}

/// A value that moves whenever any of `handles` was reallocated, rewritten
/// or resized, or `extra` changed.
fn signature(handles: &[Option<&BufferHandle>], extra: &[u64]) -> u64 {
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
    for value in extra {
        mix(*value);
    }
    hash
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl SceneDerivation for SceneJoin {
    fn name(&self) -> &'static str {
        "SceneJoin"
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
        let vertex_handles = self.input(inputs, keys.vertex_handles, HANDLE_ROW_BYTES);
        let index_handles = self.input(inputs, keys.index_handles, HANDLE_ROW_BYTES);
        let mesh_bounds = self.input(inputs, keys.mesh_bounds, MESH_BOUNDS_ROW_BYTES);
        let mesh_flags = self.input(inputs, keys.mesh_flags, MESH_FLAGS_ROW_BYTES);
        let section_handles = self.input(inputs, keys.section_handles, HANDLE_ROW_BYTES);
        let sections = self.input(inputs, keys.mesh_sections, MESH_SECTION_ROW_BYTES);
        let sources = self.input(inputs, keys.light_sources, LIGHT_SOURCE_ROW_BYTES);
        let mut recorded = false;

        // ── Meshes ──────────────────────────────────────────────────────
        let instance_rows = section_handles.map_or(0, |h| rows_of(h, HANDLE_ROW_BYTES));
        let output_rows = sections.map_or(0, |h| rows_of(h, MESH_SECTION_ROW_BYTES));
        let grew = self.objects.ensure(ctx.device, output_rows)
            | self.materials.ensure(ctx.device, output_rows);
        let mesh_inputs = [
            owners,
            generations,
            hidden,
            transforms,
            vertex_handles,
            index_handles,
            mesh_bounds,
            mesh_flags,
            section_handles,
            sections,
        ];
        let mesh_signature = signature(&mesh_inputs, &[self.objects.epoch, self.materials.epoch]);
        if grew || self.last_mesh_inputs != Some(mesh_signature) {
            self.last_mesh_inputs = Some(mesh_signature);
            encoder.clear_buffer(&self.objects.buffer, 0, None);
            encoder.clear_buffer(&self.materials.buffer, 0, None);
            if let (
                Some(owners),
                Some(generations),
                Some(transforms),
                Some(vertex_handles),
                Some(index_handles),
                Some(section_handles),
                Some(sections),
            ) = (
                owners,
                generations,
                transforms,
                vertex_handles,
                index_handles,
                section_handles,
                sections,
            ) {
                ctx.queue.write_buffer(
                    &self.mesh_uniforms,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        rows: instance_rows,
                        second: self.objects.rows,
                        _pad: [0; 2],
                    }),
                );
                let layout = self.mesh_pipeline.get_bind_group_layout(0);
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Scene Join Meshes"),
                    layout: &layout,
                    entries: &[
                        entry(0, &self.mesh_uniforms),
                        entry(1, &owners.buffer),
                        entry(2, &generations.buffer),
                        entry(3, hidden.map_or(&self.empty, |h| &h.buffer)),
                        entry(4, &transforms.buffer),
                        entry(5, &vertex_handles.buffer),
                        entry(6, &index_handles.buffer),
                        entry(7, mesh_bounds.map_or(&self.empty, |h| &h.buffer)),
                        entry(8, mesh_flags.map_or(&self.empty, |h| &h.buffer)),
                        entry(9, &section_handles.buffer),
                        entry(10, &sections.buffer),
                        entry(11, &self.objects.buffer),
                        entry(12, &self.materials.buffer),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Scene Join Meshes"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.mesh_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(instance_rows.div_ceil(WORKGROUP).max(1), 1, 1);
            }
            self.objects.content_generation += 1;
            self.materials.content_generation += 1;
            recorded = true;
        }

        // ── Lights ──────────────────────────────────────────────────────
        let light_rows = sources.map_or(0, |h| rows_of(h, LIGHT_SOURCE_ROW_BYTES));
        let grew = self.lights.ensure(ctx.device, light_rows)
            | (self.billboards && self.billboard_rows.ensure(ctx.device, light_rows));
        let light_inputs = [owners, generations, hidden, transforms, sources];
        let light_signature = signature(
            &light_inputs,
            &[
                self.lights.epoch,
                self.billboard_rows.epoch,
                u64::from(self.billboards),
            ],
        );
        if grew || self.last_light_inputs != Some(light_signature) {
            self.last_light_inputs = Some(light_signature);
            encoder.clear_buffer(&self.lights.buffer, 0, None);
            if self.billboards {
                encoder.clear_buffer(&self.billboard_rows.buffer, 0, None);
            }
            if let (Some(owners), Some(generations), Some(transforms), Some(sources)) =
                (owners, generations, transforms, sources)
            {
                ctx.queue.write_buffer(
                    &self.light_uniforms,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        rows: light_rows.min(self.lights.rows),
                        second: u32::from(self.billboards),
                        _pad: [0; 2],
                    }),
                );
                let layout = self.light_pipeline.get_bind_group_layout(0);
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Scene Join Lights"),
                    layout: &layout,
                    entries: &[
                        entry(0, &self.light_uniforms),
                        entry(1, &owners.buffer),
                        entry(2, &generations.buffer),
                        entry(3, hidden.map_or(&self.empty, |h| &h.buffer)),
                        entry(4, &transforms.buffer),
                        entry(5, &sources.buffer),
                        entry(6, &self.lights.buffer),
                        entry(7, &self.billboard_rows.buffer),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Scene Join Lights"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.light_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(light_rows.div_ceil(WORKGROUP).max(1), 1, 1);
            }
            self.lights.content_generation += 1;
            self.billboard_rows.content_generation += 1;
            recorded = true;
        }

        let mut buffers = vec![
            (OBJECTS_KEY, self.objects.handle()),
            (MATERIALS_KEY, self.materials.handle()),
            (LIGHTS_KEY, self.lights.handle()),
        ];
        if self.billboards {
            buffers.push((BILLBOARDS_KEY, self.billboard_rows.handle()));
        }
        SceneDerivationOutput { buffers, recorded }
    }
}
