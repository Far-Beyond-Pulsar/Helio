//! Depth prepass — writes depth buffer before main geometry pass.
//!
//! O(1) CPU: single `multi_draw_indexed_indirect` call regardless of scene size.
//!
//! # Vertex / Index Buffers
//!
//! This pass owns **no** mesh data.  The caller (render graph) must bind the
//! shared mesh vertex buffer (slot 0) and index buffer **before** this pass
//! executes, or the GPU draw will read from undefined memory.

use helio_core::{BufferKey, PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub struct DepthPrepassPass {
    depth_materials: helio_pass_gbuffer::depth_material::DepthMaterialBindings,
    /// The draw slots the vertex shader reads (see `helio_pass_gbuffer::DrawSlots`).
    draw_slots: helio_pass_gbuffer::DrawSlots,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<[wgpu::Buffer; 5]>,
    fallback_skeletal_storage: wgpu::Buffer,
}

impl DepthPrepassPass {
    /// Create the depth-prepass pipeline.
    ///
    /// * `depth_format` – format of the depth attachment (e.g. `Depth32Float`)
    pub fn new(device: &wgpu::Device, depth_format: wgpu::TextureFormat) -> Self {
        let depth_materials =
            helio_pass_gbuffer::depth_material::DepthMaterialBindings::new(device);
        let source = depth_materials.shader_source(include_str!("../shaders/depth_prepass.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cutout depth prepass"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DepthPrepass BGL"),
            entries: &[
                // binding 0: camera (storage, VERTEX)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: per-instance transforms (VERTEX, read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: compacted_indices (VERTEX, read-only storage) — per-group
                // surviving instance slots written by IndirectDispatchPass.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DepthPrepass PL"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&depth_materials.layout)],
            immediate_size: 0,
        });

        // Vertex layout matches the shared mesh vertex buffer (stride = 40 bytes).
        //   offset  0 — position       (Float32x3, location 0)
        //   offset 12 — bitangent_sign (Float32,   location 1) — skipped here
        //   offset 16 — tex_coords0   (Float32x2, location 2)
        //   offset 24 — tex_coords1   (Float32x2) — skipped here
        //   offset 32 — normal        (Uint32,    location 3) — skipped here
        //   offset 36 — tangent       (Uint32,    location 4) — skipped here
        let pipeline = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: Some("DepthPrepass Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[
                        Some(wgpu::VertexBufferLayout {
                            array_stride: 40, // PackedVertex: pos(12)+bitan(4)+uv0(8)+uv1(8)+normal(4)+tangent(4)
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 0,
                                    shader_location: 0,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x2,
                                    offset: 16,
                                    shader_location: 2,
                                },
                            ],
                        }),
                        Some(helio_pass_gbuffer::DRAW_SLOT_LAYOUT),
                    ],
                },
                // Depth-only: no fragment stage, no color outputs.
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(helio_core::depth::NEARER),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        let fallback_skeletal_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Empty skeletal storage"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            depth_materials,
            draw_slots: Default::default(),
            pipeline,
            bind_group_layout,
            bind_group: None,
            bind_group_key: None,
            fallback_skeletal_storage,
        }
    }
}

impl RenderPass for DepthPrepassPass {
    fn scene_buffers(&self) -> &'static [&'static str] {
        &[
            "builtin_mesh_index",
            "builtin_mesh_vertex",
            "skeletal_bone_palette",
            "skeletal_skin_vertices",
        ]
    }

    fn name(&self) -> &'static str {
        "DepthPrepass"
    }

    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        builder.read("material_textures");
        builder.read("object_batch");
        builder.read("culled_batch");
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            storage.retain_boxed_slice(Box::new([]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("DepthPrepass"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(helio_core::depth::CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.depth_materials.update(ctx);
        if self.depth_materials.group.is_none() {
            return Ok(());
        }
        let Some(batch) = ctx
            .registry
            .get::<helio_pass_gbuffer::ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new(
                "object_batch",
            ))
        else {
            return Ok(());
        };
        let Some(culled) = ctx
            .registry
            .get::<helio_pass_gbuffer::CulledBatchFrameData<'_>>(helio_core::ResourceKey::new(
                "culled_batch",
            ))
        else {
            return Ok(());
        };
        // O(1): single multi_draw_indexed_indirect — no CPU loop over draw calls.
        let (max_draws, draw_count) = batch.all_draws();
        if max_draws == 0 {
            return Ok(());
        }
        let vertices = ctx
            .scene_buffers
            .get(BufferKey::of("builtin_mesh_vertex"))
            .ok_or_else(|| {
                helio_core::Error::InvalidPassConfig(
                    "DepthPrepass requires builtin_mesh_vertex".to_string(),
                )
            })?;
        let indices = ctx
            .scene_buffers
            .get(BufferKey::of("builtin_mesh_index"))
            .ok_or_else(|| {
                helio_core::Error::InvalidPassConfig(
                    "DepthPrepass requires builtin_mesh_index".to_string(),
                )
            })?;
        let skin = ctx
            .scene_buffers
            .get(BufferKey::of("skeletal_skin_vertices"));
        let palette = ctx
            .scene_buffers
            .get(BufferKey::of("skeletal_bone_palette"));
        let skin_buffer = skin
            .map(|h| &h.buffer)
            .unwrap_or(&self.fallback_skeletal_storage);
        let palette_buffer = palette
            .map(|h| &h.buffer)
            .unwrap_or(&self.fallback_skeletal_storage);

        // Extract before the mutable encoder borrow.
        let key = [
            ctx.camera.clone(),
            batch.instances.clone(),
            culled.compacted_indices.clone(),
            skin_buffer.clone(),
            palette_buffer.clone(),
        ];
        if self.bind_group_key.as_ref() != Some(&key) {
            log::debug!("DepthPrepass: rebuilding bind group (buffers changed)");
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DepthPrepass BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: batch.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: culled.compacted_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: skin_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: palette_buffer.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }
        let indirect = culled.indirect;
        // The slot indexes `compacted_indices`.
        let slots = self
            .draw_slots
            .buffer(
                &ctx.device,
                helio_pass_gbuffer::DrawSlots::len_of(culled.compacted_indices.size(), 4),
            )
            .clone();

        let mut pass = ctx.render_cmds().unwrap();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.depth_materials.group.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
        pass.set_vertex_buffer(1, slots.slice(..));
        pass.set_index_buffer(indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
        helio_pass_gbuffer::multi_draw_indexed_indirect(
            &mut pass, indirect, 0, max_draws, draw_count,
        );
        Ok(())
    }
}
