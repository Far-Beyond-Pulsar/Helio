//! Parented transforms and rigid frames on the GPU.
//!
//! The frontend's transform column holds each object's transform **relative
//! to its parent** (to the world for an object with none), next to a column
//! naming each object's parent and one naming the objects that are **rigid
//! frames** (and their slot in the coordinate-space table). This derivation
//! composes the chains and publishes two results per object:
//!
//! * the **world** transform, under the transform column's own key, so every
//!   later derivation (the scene join's lights, the environment join) and
//!   every pass that reads that key sees world-space rows without knowing
//!   hierarchies exist;
//! * the **placement**: composed only up to the nearest rigid frame above the
//!   object (not including it), under [`LOCAL_PLACEMENT_KEY`], with that
//!   frame's slot under [`OBJECT_SPACE_KEY`]. The scene join places meshes
//!   from these and tags them with their space, so the coordinate-space table
//!   moves a frame's whole contents with one matrix and a frame moving rewrites
//!   none of its members.
//!
//! # Precision
//!
//! World rows stay **world positions as f32 pairs**, like the frontend's rows
//! they replace (the frontend's f64): chains compose in pair arithmetic, so an
//! object far from the world's origin keeps its millimetres whichever of its
//! ancestors are far, and every consumer reads the rows as it reads a flat
//! scene's (the joins take the frame origin off pair-wise). Placement rows
//! inside a frame are relative to the frame and small; the frame's own (f64)
//! position is the coordinate-space table's business.
//!
//! # Cost
//!
//! Moving a parent is one row write. The first resolve (and any resolve after
//! a reallocation or a gap in SceneDB's flush records)
//! recomposes every row. Otherwise only rows whose own transform or any
//! ancestor's changed are recomposed: SceneDB's flushed-row records of the
//! transform, parent and frame columns seed a changed set, and each row looks
//! up its chain for a changed ancestor (reads only). The rows whose
//! *placement* changed are published as a GPU-built moved list in SceneDB's
//! flushed-row layout (`[count, rows..]`) with indirect dispatch arguments
//! ([`HIERARCHY_MOVED_ARGS_KEY`]), so `SceneJoin`'s moved-owner join works for
//! hierarchical scenes too, at a cost that follows what moved. A rigid frame
//! moving changes no member's placement, so it adds nothing to that list.
//!
//! A scene with no parented object never runs any of this and publishes
//! nothing, so flat scenes keep their fast paths untouched.
//!
//! Run it before the joins: `with_scene_derivation(HierarchyResolve)` first.

pub use helio_core::{
    BufferHandle, BufferKey, FlushedRows, SceneBufferProjection, SceneDerivation,
    SceneDerivationContext, SceneDerivationOutput,
};

use crate::scene_join::{join_source, TRANSFORM_ROW_BYTES};

/// An object's parent: its entity index plus one, `0` for none (`u32`).
pub const PARENT_ROW_BYTES: u64 = 4;
/// An object's frame slot, `0` when it is not a rigid frame (`u32`).
pub const FRAME_ROW_BYTES: u64 = 4;
/// Published with the moved list: `[x, y, z]` workgroup counts sized for it.
/// Its `content_generation` equals the placement buffer's, so a join can tell
/// the list belongs to the rows it is reading.
pub const HIERARCHY_MOVED_ARGS_KEY: BufferKey = BufferKey::of("hierarchy_moved_args");
/// Each object composed only up to its nearest rigid frame (48-byte transform
/// rows, like the world rows). Equal to the world rows for objects in no frame.
pub const LOCAL_PLACEMENT_KEY: BufferKey = BufferKey::of("Transform::placement");
/// Each object's space: the slot of its nearest rigid frame ancestor, `0` for
/// the world (`u32` rows). Its `content_generation` moves only when a parent
/// link or frame changed (or after a full resolve), never for a plain move.
pub const OBJECT_SPACE_KEY: BufferKey = BufferKey::of("object_space");
const WORKGROUP: u32 = 64;

/// Where the frontend's rows live.
#[derive(Clone, Copy, Debug)]
pub struct HierarchyResolveKeys {
    /// Parent-relative transforms, keyed by object entity. The world-space
    /// result is published under this same key.
    pub transforms: BufferKey,
    /// Parent links, keyed by object entity.
    pub parents: BufferKey,
    /// Frame slots, keyed by object entity. Optional at run time: without it
    /// there are no frames.
    pub frames: BufferKey,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    rows: u32,
    /// 1.0 (the shader's guard against reassociated floats).
    one: f32,
    _pad: [u32; 2],
}

/// How one column's flush record relates to the last one this resolve saw.
enum Column {
    /// Nothing flushed since.
    Quiet,
    /// Exactly one flush since: its rows are the changes.
    Changed(FlushedRows),
    /// Unknown: a gap, a reset, or no record. Everything must be redone.
    Unknown,
}

fn classify(last: Option<u64>, now: Option<&FlushedRows>) -> Column {
    match (last, now) {
        (Some(last), Some(now)) if now.sequence == last => Column::Quiet,
        (Some(last), Some(now)) if now.sequence == last + 1 => Column::Changed(now.clone()),
        _ => Column::Unknown,
    }
}

/// What the last resolve looked at, to tell what changed since.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Seen {
    rows: u32,
    transforms_epoch: u64,
    parents_epoch: u64,
    frames_epoch: u64,
    transforms_size: u64,
    parents_size: u64,
    frames_size: u64,
}

/// See the module doc.
pub struct HierarchyResolve {
    keys: HierarchyResolveKeys,
    /// resolve, mark transform, mark parent, mark frame, resolve moved, finish.
    pipelines: [wgpu::ComputePipeline; 6],
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    world: wgpu::Buffer,
    placement: wgpu::Buffer,
    spaces: wgpu::Buffer,
    changed: wgpu::Buffer,
    moved: wgpu::Buffer,
    args: wgpu::Buffer,
    /// Bound as the flushed list where a dispatch does not read one, and as
    /// the frame column when the frontend has none.
    empty: wgpu::Buffer,
    rows: u32,
    epoch: u64,
    content_generation: u64,
    spaces_generation: u64,
    /// Counts the moved lists published, so a consumer can tell consecutive ones.
    sequence: u64,
    seen: Option<Seen>,
    /// The last resolve recomposed every row (so its moved list is not a subset).
    last_full: bool,
    transforms_flush: Option<u64>,
    parents_flush: Option<u64>,
    frames_flush: Option<u64>,
    binds: Vec<(Vec<wgpu::Buffer>, wgpu::BindGroup)>,
}

fn rows_of(handle: &BufferHandle, row_bytes: u64) -> u32 {
    (handle.buffer.size() / row_bytes).min(u64::from(u32::MAX)) as u32
}

const RW: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

impl HierarchyResolve {
    pub fn new(device: &wgpu::Device, keys: HierarchyResolveKeys) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hierarchy Resolve"),
            source: wgpu::ShaderSource::Wgsl(
                join_source(include_str!("../shaders/hierarchy_resolve.wgsl")).into(),
            ),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hierarchy Resolve"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, true),
                storage(5, false),
                storage(6, false),
                storage(7, false),
                storage(8, true),
                storage(9, false),
                storage(10, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hierarchy Resolve"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            helio_core::shader::create_compute_pipeline(
                &device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        Self {
            keys,
            pipelines: [
                pipeline("cs_resolve"),
                pipeline("cs_mark_transform"),
                pipeline("cs_mark_parent"),
                pipeline("cs_mark_frame"),
                pipeline("cs_resolve_moved"),
                pipeline("cs_finish"),
            ],
            layout,
            uniforms: buffer(
                device,
                "Hierarchy Resolve Uniforms",
                std::mem::size_of::<Uniforms>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            world: buffer(device, "Hierarchy Resolve World", TRANSFORM_ROW_BYTES, RW),
            placement: buffer(
                device,
                "Hierarchy Resolve Placement",
                TRANSFORM_ROW_BYTES,
                RW,
            ),
            spaces: buffer(device, "Hierarchy Resolve Spaces", 4, RW),
            changed: buffer(device, "Hierarchy Resolve Changed", 16, RW),
            moved: buffer(device, "Hierarchy Resolve Moved", 16, RW),
            args: buffer(
                device,
                "Hierarchy Resolve Args",
                16,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            empty: buffer(
                device,
                "Hierarchy Resolve Empty",
                16,
                wgpu::BufferUsages::STORAGE,
            ),
            rows: 1,
            epoch: 0,
            content_generation: 0,
            spaces_generation: 0,
            sequence: 0,
            seen: None,
            last_full: true,
            transforms_flush: None,
            parents_flush: None,
            frames_flush: None,
            binds: Vec::new(),
        }
    }

    fn handle(
        &self,
        buffer: &wgpu::Buffer,
        row_bytes: u64,
        content_generation: u64,
    ) -> BufferHandle {
        BufferHandle {
            buffer: buffer.clone(),
            epoch: self.epoch,
            row_bytes,
            content_generation,
        }
    }

    /// The bind group for a dispatch over `inputs` that reads `flushed`.
    fn bind(
        &mut self,
        device: &wgpu::Device,
        inputs: [&wgpu::Buffer; 3],
        flushed: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let [local, parents, frames] = inputs;
        let bound = vec![
            local.clone(),
            parents.clone(),
            self.world.clone(),
            flushed.clone(),
            self.changed.clone(),
            self.moved.clone(),
            self.args.clone(),
            frames.clone(),
            self.placement.clone(),
            self.spaces.clone(),
        ];
        if let Some((_, group)) = self
            .binds
            .iter()
            .find(|(cached, _)| cached.iter().zip(&bound).all(|(a, b)| a == b))
        {
            return group.clone();
        }
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let mut entries = vec![entry(0, &self.uniforms)];
        entries.extend(
            bound
                .iter()
                .enumerate()
                .map(|(i, b)| entry(i as u32 + 1, b)),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hierarchy Resolve"),
            layout: &self.layout,
            entries: &entries,
        });
        if self.binds.len() >= 12 {
            self.binds.remove(0);
        }
        self.binds.push((bound, group.clone()));
        group
    }
}

impl SceneDerivation for HierarchyResolve {
    fn name(&self) -> &'static str {
        "HierarchyResolve"
    }

    fn derive(
        &mut self,
        ctx: &SceneDerivationContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> SceneDerivationOutput {
        let (Some(transforms), Some(parents)) = (
            ctx.inputs.get(self.keys.transforms),
            ctx.inputs.get(self.keys.parents),
        ) else {
            return SceneDerivationOutput::default();
        };
        let frames = ctx.inputs.get(self.keys.frames);
        if (transforms.row_bytes != 0 && transforms.row_bytes != TRANSFORM_ROW_BYTES)
            || (parents.row_bytes != 0 && parents.row_bytes != PARENT_ROW_BYTES)
            || frames.is_some_and(|f| f.row_bytes != 0 && f.row_bytes != FRAME_ROW_BYTES)
        {
            log::error!("hierarchy resolve: unexpected row sizes; transforms are not composed");
            return SceneDerivationOutput::default();
        }
        // The parent column has never been written: nothing is parented, the
        // transforms are already world-space, and nothing is republished.
        if parents.content_generation == 0 {
            return SceneDerivationOutput::default();
        }

        let rows = rows_of(transforms, TRANSFORM_ROW_BYTES);
        let mut reallocated = false;
        if rows > self.rows {
            self.rows = rows.next_power_of_two();
            let transform_bytes = TRANSFORM_ROW_BYTES * u64::from(self.rows);
            self.world = buffer(ctx.device, "Hierarchy Resolve World", transform_bytes, RW);
            self.placement = buffer(
                ctx.device,
                "Hierarchy Resolve Placement",
                transform_bytes,
                RW,
            );
            self.spaces = buffer(
                ctx.device,
                "Hierarchy Resolve Spaces",
                u64::from(self.rows) * 4,
                RW,
            );
            self.changed = buffer(
                ctx.device,
                "Hierarchy Resolve Changed",
                u64::from(self.rows) * 4,
                RW,
            );
            self.moved = buffer(
                ctx.device,
                "Hierarchy Resolve Moved",
                (u64::from(self.rows) + 1) * 4,
                RW,
            );
            self.epoch += 1;
            self.binds.clear();
            reallocated = true;
        }

        let seen = Seen {
            rows,
            transforms_epoch: transforms.epoch,
            parents_epoch: parents.epoch,
            frames_epoch: frames.map_or(0, |f| f.epoch),
            transforms_size: transforms.buffer.size(),
            parents_size: parents.buffer.size(),
            frames_size: frames.map_or(0, |f| f.buffer.size()),
        };
        let transforms_now = ctx.inputs.flushed_rows(self.keys.transforms).cloned();
        let parents_now = ctx.inputs.flushed_rows(self.keys.parents).cloned();
        let frames_now = frames.and_then(|_| ctx.inputs.flushed_rows(self.keys.frames).cloned());
        let t_column = classify(self.transforms_flush, transforms_now.as_ref());
        let p_column = classify(self.parents_flush, parents_now.as_ref());
        // A scene that never wrote the frame column has nothing to follow.
        let f_column = if frames.is_none() {
            Column::Quiet
        } else if frames_now.is_none() && self.frames_flush.is_none() {
            Column::Quiet
        } else {
            classify(self.frames_flush, frames_now.as_ref())
        };
        self.transforms_flush = transforms_now.as_ref().map(|r| r.sequence);
        self.parents_flush = parents_now.as_ref().map(|r| r.sequence);
        self.frames_flush = frames_now.as_ref().map(|r| r.sequence);

        let full = reallocated
            || self.seen != Some(seen)
            || matches!(t_column, Column::Unknown)
            || matches!(p_column, Column::Unknown)
            || matches!(f_column, Column::Unknown);
        self.seen = Some(seen);
        // A parent link or frame changed: objects may have changed space.
        let structural = full
            || matches!(p_column, Column::Changed(_))
            || matches!(f_column, Column::Changed(_));
        let lists: Vec<(usize, FlushedRows)> = [t_column, p_column, f_column]
            .into_iter()
            .enumerate()
            .filter_map(|(which, column)| match column {
                // pipelines[1 + which]: transform, parent, frame marks.
                Column::Changed(rows) => Some((1 + which, rows)),
                _ => None,
            })
            .collect();

        let mut output = SceneDerivationOutput::default();
        if full || !lists.is_empty() {
            helio_core::cpu_scope!("HierarchyResolve");
            ctx.queue.write_buffer(
                &self.uniforms,
                0,
                bytemuck::bytes_of(&Uniforms {
                    rows,
                    one: 1.0,
                    _pad: [0; 2],
                }),
            );
            // The moved list starts empty either way: a full resolve moves
            // everything, which a consumer learns from the empty list's gap.
            encoder.clear_buffer(&self.moved, 0, Some(4));
            let empty = self.empty.clone();
            let frame_buffer = frames.map_or(empty.clone(), |f| f.buffer.clone());
            let inputs = [&transforms.buffer, &parents.buffer, &frame_buffer];
            let main = self.bind(ctx.device, inputs, &empty);
            let dense = rows.div_ceil(WORKGROUP).max(1);
            let pipelines = self.pipelines.clone();
            if full {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Hierarchy Resolve"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines[0]);
                pass.set_bind_group(0, &main, &[]);
                pass.dispatch_workgroups(dense, 1, 1);
            } else {
                encoder.clear_buffer(&self.changed, 0, Some(u64::from(rows.max(1)) * 4));
                for (which, list) in &lists {
                    let group = self.bind(ctx.device, inputs, &list.buffer);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Hierarchy Resolve Mark"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipelines[*which]);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(list.count.div_ceil(WORKGROUP).max(1), 1, 1);
                }
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Hierarchy Resolve Moved"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines[4]);
                pass.set_bind_group(0, &main, &[]);
                pass.dispatch_workgroups(dense, 1, 1);
                drop(pass);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Hierarchy Resolve Finish"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines[5]);
                pass.set_bind_group(0, &main, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.last_full = full;
            self.content_generation += 1;
            if structural {
                self.spaces_generation += 1;
            }
            self.sequence += 1;
            output.recorded = true;
        }

        output.buffers.push((
            self.keys.transforms,
            self.handle(&self.world, TRANSFORM_ROW_BYTES, self.content_generation),
        ));
        output.buffers.push((
            LOCAL_PLACEMENT_KEY,
            self.handle(
                &self.placement,
                TRANSFORM_ROW_BYTES,
                self.content_generation,
            ),
        ));
        output.buffers.push((
            OBJECT_SPACE_KEY,
            self.handle(&self.spaces, 4, self.spaces_generation),
        ));
        // Say which placements moved, so a consumer follows the changes
        // instead of redoing every row. After a full resolve the list is empty
        // (count 0): it keeps the sequence continuous without claiming a subset.
        if self.content_generation > 0 {
            output.flushed.push((
                LOCAL_PLACEMENT_KEY,
                FlushedRows {
                    buffer: self.moved.clone(),
                    // The real count lives on the GPU (the list's first word);
                    // this is the most it can be, for sizing only.
                    count: rows.max(1),
                    sequence: self.sequence,
                },
            ));
            output.buffers.push((
                HIERARCHY_MOVED_ARGS_KEY,
                BufferHandle {
                    buffer: self.args.clone(),
                    epoch: self.epoch,
                    row_bytes: 0,
                    // Only a partial resolve built a list the consumer may use.
                    content_generation: if self.last_full {
                        0
                    } else {
                        self.content_generation
                    },
                },
            ));
        }
        output
    }
}
