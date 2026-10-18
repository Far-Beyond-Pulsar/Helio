//! Parented transforms on the GPU.
//!
//! The frontend's transform column holds each object's transform **relative
//! to its parent** (to the world for an object with none), next to a column
//! naming each object's parent. This derivation composes every object's
//! world transform from its chain and republishes it under the transform
//! column's own key, so every later derivation (the scene join, the
//! environment join) and every pass that reads that key sees world-space rows
//! without knowing hierarchies exist.
//!
//! Moving a parent is one row write; the descendants follow in the next
//! resolve, a parallel pass over the transform rows. A scene with no parented
//! object never runs it and publishes nothing, so flat scenes keep their
//! dirty-row fast paths (`SceneJoin`'s moved-owner join) untouched. Once
//! parents are in play, the transform buffer is replaced, which also drops its
//! flushed-row record (see [`SceneBufferProjection::insert`]): a parent's move
//! changes rows its flush never named.
//!
//! Run it before the joins: `with_scene_derivation(HierarchyResolve)` first.

pub use helio_core::{
    BufferHandle, BufferKey, SceneBufferProjection, SceneDerivation, SceneDerivationContext,
    SceneDerivationOutput,
};

use crate::scene_join::{join_source, TRANSFORM_ROW_BYTES};

/// An object's parent: its entity index plus one, `0` for none (`u32`).
pub const PARENT_ROW_BYTES: u64 = 4;
const WORKGROUP: u32 = 64;

/// Where the frontend's rows live.
#[derive(Clone, Copy, Debug)]
pub struct HierarchyResolveKeys {
    /// Parent-relative transforms, keyed by object entity. The world-space
    /// result is published under this same key.
    pub transforms: BufferKey,
    /// Parent links, keyed by object entity.
    pub parents: BufferKey,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    rows: u32,
    _pad: [u32; 3],
}

/// See the module doc.
pub struct HierarchyResolve {
    keys: HierarchyResolveKeys,
    pipeline: wgpu::ComputePipeline,
    uniforms: wgpu::Buffer,
    world: wgpu::Buffer,
    rows: u32,
    epoch: u64,
    content_generation: u64,
    /// Inputs at the last resolve; the world rows are rewritten only when one
    /// moved.
    last_inputs: Option<u64>,
    bind: Option<([wgpu::Buffer; 3], wgpu::BindGroup)>,
}

fn rows_of(handle: &BufferHandle, row_bytes: u64) -> u32 {
    (handle.buffer.size() / row_bytes).min(u64::from(u32::MAX)) as u32
}

impl HierarchyResolve {
    pub fn new(device: &wgpu::Device, keys: HierarchyResolveKeys) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hierarchy Resolve"),
            source: wgpu::ShaderSource::Wgsl(
                join_source(include_str!("../shaders/hierarchy_resolve.wgsl")).into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hierarchy Resolve"),
            layout: None,
            module: &module,
            entry_point: Some("cs_resolve"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            keys,
            pipeline,
            uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Hierarchy Resolve Uniforms"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            world: Self::allocate(device, 1),
            rows: 1,
            epoch: 0,
            content_generation: 0,
            last_inputs: None,
            bind: None,
        }
    }

    fn allocate(device: &wgpu::Device, rows: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hierarchy Resolve World Transforms"),
            size: TRANSFORM_ROW_BYTES * u64::from(rows.max(1)),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn handle(&self) -> BufferHandle {
        BufferHandle {
            buffer: self.world.clone(),
            epoch: self.epoch,
            row_bytes: TRANSFORM_ROW_BYTES,
            content_generation: self.content_generation,
        }
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
        if (transforms.row_bytes != 0 && transforms.row_bytes != TRANSFORM_ROW_BYTES)
            || (parents.row_bytes != 0 && parents.row_bytes != PARENT_ROW_BYTES)
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
        let grew = rows > self.rows;
        if grew {
            self.rows = rows.next_power_of_two();
            self.world = Self::allocate(ctx.device, self.rows);
            self.epoch += 1;
        }

        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in [
            transforms.epoch,
            transforms.content_generation,
            transforms.buffer.size(),
            parents.epoch,
            parents.content_generation,
            parents.buffer.size(),
            self.epoch,
        ] {
            hash ^= value;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        let mut recorded = false;
        if grew || self.last_inputs != Some(hash) {
            self.last_inputs = Some(hash);
            helio_core::cpu_scope!("HierarchyResolve");
            ctx.queue.write_buffer(
                &self.uniforms,
                0,
                bytemuck::bytes_of(&Uniforms {
                    rows,
                    _pad: [0; 3],
                }),
            );
            let bound = [
                transforms.buffer.clone(),
                parents.buffer.clone(),
                self.world.clone(),
            ];
            let reusable = self
                .bind
                .as_ref()
                .is_some_and(|(cached, _)| cached.iter().zip(&bound).all(|(a, b)| a == b));
            if !reusable {
                let layout = self.pipeline.get_bind_group_layout(0);
                let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Hierarchy Resolve"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniforms.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: bound[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: bound[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: bound[2].as_entire_binding(),
                        },
                    ],
                });
                self.bind = Some((bound, group));
            }
            let group = &self.bind.as_ref().expect("bound above").1;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hierarchy Resolve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(WORKGROUP).max(1), 1, 1);
            drop(pass);
            self.content_generation += 1;
            recorded = true;
        }
        SceneDerivationOutput {
            buffers: vec![(self.keys.transforms, self.handle())],
            recorded,
        }
    }
}
