//! Transparent geometry pass with SrcAlpha / OneMinusSrcAlpha blending,
//! read-only depth, and Radiant template support.
//!
//! Templates are composed with `transparent_base.wgsl` (shared in the `helio`
//! crate) and registered via `renderer.transparent_template_registry_mut()`.
//! The default template (class 0) uses ambient + normal shading.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use helio::radiant::{RadiantShaderCache, RadiantShaderKey};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use pulsar_scenedb::gpu::{world_mirror::DEFAULT_AUTO_REGISTER_CAPACITY, BufferKey};

/// Fixed capacity for the `"scene_lights"` SceneDB buffer, kept equal to
/// `helio_pass_forward_lit::MAX_LIGHTS`/`helio_pass_light_cull::MAX_LIGHTS`
/// by construction (all three are literally `DEFAULT_AUTO_REGISTER_CAPACITY`).
const MAX_LIGHTS: u32 = DEFAULT_AUTO_REGISTER_CAPACITY;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransparentGlobals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],
    rc_world_min: [f32; 4],
    rc_world_max: [f32; 4],
    csm_splits: [f32; 4],
    num_tiles_x: u32,
    num_tiles_y: u32,
    screen_width: f32,
    screen_height: f32,
    // 1 when `lights`/`transforms` share the same raw entity index
    // (SceneDB-direct); 0 when `lights` is `ctx.scene.lights`, a freshly
    // rebuilt dense array needing `light_entity_indices[light_idx]`. See
    // `transparent_base.wgsl`'s `light_entity_indices` binding doc.
    light_mode_direct_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct TransparentPass {
    pipelines: HashMap<RadiantShaderKey, wgpu::RenderPipeline>,
    shader_cache: RadiantShaderCache,
    /// This pass's own class-0 override (the transparent base — never
    /// synced from the scene).
    local_class0: helio::radiant::RadiantTemplate,
    /// User-registered custom templates (id >= 5), shared with the renderer
    /// and other passes — never deep-cloned (see `SharedTemplateRegistry`).
    shared_registry: Option<helio::radiant::SharedTemplateRegistry>,
    /// Key set as of the last sync, to detect content changes cheaply.
    last_shared_keys: Vec<u32>,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group_layout_0: wgpu::BindGroupLayout,
    /// Rebuilt per-frame in `execute()` — its `instances` source
    /// (`object_batch.instances`) is a `GrowableBuffer` that can reallocate
    /// across frames, so this can't be built once at construction time (see
    /// `bind_group_key`).
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize)>,
    bind_group_layout_1: wgpu::BindGroupLayout,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_1_key: Option<(usize, usize, usize, usize, usize)>,
    globals_buf: wgpu::Buffer,
    surface_format: wgpu::TextureFormat,
}

impl TransparentPass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transparent Globals"),
            size: std::mem::size_of::<TransparentGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Transparent BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        });

        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Transparent BGL 1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: light_entity_indices storage read -- parallel to `lights`
                // only in CPU-resolved mode, see `transparent_base.wgsl`'s
                // binding doc.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: SceneDB `Transform` storage read -- entity-indexed the same
                // way `lights` is, see `transparent_base.wgsl`'s binding doc.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
            label: Some("Transparent PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1)],
            immediate_size: 0,
        });

        // Just the transparent base shader at class 0 — NOT
        // RadiantTemplateRegistry::new(), which populates classes 0-4 with
        // gbuffer templates that have incompatible bind group layouts.
        let base_src = include_str!("../../../../helio/templates/transparent_base.wgsl");
        let resolved_src: &'static str = if base_src.contains("//!use pbr_eval") {
            let mut resolved =
                String::with_capacity(base_src.len() + libhelio::shader::PBR_EVAL.len());
            resolved.push_str(libhelio::shader::PBR_EVAL);
            resolved.push('\n');
            resolved.push_str(base_src);
            Box::leak(resolved.into_boxed_str())
        } else {
            base_src
        };
        let local_class0 = helio::radiant::RadiantTemplate {
            name: "transparent_base",
            wgsl_source: resolved_src,
        };

        Self {
            pipelines: HashMap::new(),
            shader_cache: RadiantShaderCache::new(),
            local_class0,
            shared_registry: None,
            last_shared_keys: Vec::new(),
            pipeline_layout,
            bind_group_layout_0: bgl_0,
            bind_group: None,
            bind_group_key: None,
            bind_group_layout_1: bgl_1,
            bind_group_1: None,
            bind_group_1_key: None,
            globals_buf,
            surface_format,
        }
    }
}

impl RenderPass for TransparentPass {
    fn name(&self) -> &'static str {
        "TransparentPass"
    }

    fn chain_transparent(&self) -> bool {
        true
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            "main_scene",
            "depth",
            "cluster_light_grid",
            "object_batch",
            "culled_batch",
        ]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("depth");
        builder.read("cluster_light_grid");
        builder.read("object_batch");
        builder.read("culled_batch");
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let num_tiles_x = ctx.width.div_ceil(16);
        let num_tiles_y = ctx.height.div_ceil(16);
        // Prefer the SceneDB-direct `"scene_lights"` buffer (fixed capacity
        // `MAX_LIGHTS`) when populated; else the `Renderer`-seeded
        // `LightsFrameData` bridge's `movable_light_count`, production's
        // actual light count today -- see `ForwardLitPass`'s identical
        // `light_mode_direct_index` doc for the full reasoning.
        let use_direct_index = ctx.scene_buffers.contains(BufferKey::of("scene_lights"));
        let light_count = if use_direct_index {
            MAX_LIGHTS
        } else {
            ctx.pass_resources
                .lights
                .get()
                .map(|l| l.movable_light_count)
                .unwrap_or(0)
        };
        ctx.queue.write_buffer(
            &self.globals_buf,
            0,
            bytemuck::bytes_of(&TransparentGlobals {
                frame: ctx.frame_num as u32,
                delta_time: 0.0,
                light_count,
                ambient_intensity: 0.6,
                ambient_color: [0.3, 0.35, 0.4, 1.0],
                rc_world_min: [0.0; 4],
                rc_world_max: [0.0; 4],
                csm_splits: [0.0; 4],
                num_tiles_x,
                num_tiles_y,
                screen_width: ctx.width as f32,
                screen_height: ctx.height as f32,
                light_mode_direct_index: use_direct_index as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );
        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::PassResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        let depth_view = resources.full_res_depth.get().unwrap_or(depth);
        Some(wgpu::RenderPassDescriptor {
            label: Some("Transparent"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
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
        let Some(batch) = ctx.resources.object_batch.get() else {
            return Ok(());
        };
        let Some(culled) = ctx.resources.culled_batch.get() else {
            return Ok(());
        };
        let draw_count = batch.draw_count;
        log::info!(
            "[TransparentPass] execute: draw_count={}, transparent_ranges={:?}",
            draw_count,
            batch.transparent_ranges
        );
        if draw_count == 0 {
            return Ok(());
        }

        let main_scene = ctx.resources.main_scene.read("Transparent");
        let ms = main_scene.as_ref().ok_or_else(|| {
            helio_core::Error::InvalidPassConfig("TransparentPass requires main_scene".to_string())
        })?;
        let Some(vertices_handle) = ctx
            .scene_buffers
            .get(BufferKey::of("builtin_mesh_vertex"))
        else {
            return Ok(());
        };
        let Some(indices_handle) = ctx
            .scene_buffers
            .get(BufferKey::of("builtin_mesh_index"))
        else {
            return Ok(());
        };
        let vertices = &vertices_handle.buffer;
        let indices = &indices_handle.buffer;

        // Rebuild bind group 1 (lights + transforms + cluster data) when
        // buffer pointers change. Prefer the SceneDB-direct `"scene_lights"`
        // buffer when populated; else `ctx.scene.lights` -- production's
        // real, actively-populated light source (see `ForwardLitPass`'s
        // identical `light_mode_direct_index` doc).
        let cluster = ctx.resources.cluster_light_grid.get();
        let lights_data = ctx.resources.lights.get();
        let lights_buf = ctx
            .scene_buffers
            .get(BufferKey::of("scene_lights"))
            .map(|handle| &handle.buffer)
            .unwrap_or_else(|| lights_data.map(|l| l.lights).unwrap_or(batch.instances));
        let lights_ptr = lights_buf as *const _ as usize;
        let light_entity_indices_ptr = lights_data
            .map(|l| l.light_entity_indices as *const _ as usize)
            .unwrap_or(0);
        let tile_lists_ptr = cluster
            .map(|c| c.tile_light_lists as *const _ as usize)
            .unwrap_or(0);
        let tile_counts_ptr = cluster
            .map(|c| c.tile_light_counts as *const _ as usize)
            .unwrap_or(0);
        let transforms_ptr = lights_data
            .and_then(|l| l.transforms)
            .map(|b| b as *const _ as usize)
            .unwrap_or(0);
        let bg1_key = (
            lights_ptr,
            light_entity_indices_ptr,
            tile_lists_ptr,
            tile_counts_ptr,
            transforms_ptr,
        );
        if self.bind_group_1_key != Some(bg1_key) {
            let fallback = batch.instances;
            let tile_lists = cluster.map(|c| c.tile_light_lists).unwrap_or(fallback);
            let tile_counts = cluster.map(|c| c.tile_light_counts).unwrap_or(fallback);
            // `light_count` is 0 whenever no real `Transform` buffer exists
            // yet, so this fallback is never actually dereferenced at a live
            // light's index in practice -- same reasoning as
            // `helio_pass_forward_lit`'s identical fallback.
            let transforms = lights_data.and_then(|l| l.transforms).unwrap_or(fallback);
            let light_entity_indices_buf = lights_data
                .map(|l| l.light_entity_indices)
                .unwrap_or(fallback);
            self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Transparent BG 1"),
                layout: &self.bind_group_layout_1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tile_lists.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tile_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: light_entity_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: transforms.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_1_key = Some(bg1_key);
        }

        // Rebuild bind group 0 (camera + globals + instances) when buffer
        // pointers change -- `batch.instances` is a `GrowableBuffer` that can
        // reallocate across frames as the scene grows, so this can't be
        // built once at construction time (mirrors `bind_group_1`'s pattern).
        let camera_ptr = ctx.camera as *const _ as usize;
        let instances_ptr = batch.instances as *const _ as usize;
        let bg0_key = (camera_ptr, instances_ptr);
        if self.bind_group_key != Some(bg0_key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Transparent BG 0"),
                layout: &self.bind_group_layout_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.globals_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: batch.instances.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(bg0_key);
        }

        let indirect = culled.indirect;
        let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        rp.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        rp.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        rp.set_vertex_buffer(0, vertices.slice(..));
        rp.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);

        let ranges = batch.transparent_ranges;
        if ranges.is_empty() {
            let pipeline = self.get_or_create_pipeline(
                &ctx.device,
                RadiantShaderKey {
                    template_id: 0,
                    graph_hash: 0,
                    feature_flags: 0,
                },
                "",
            );
            rp.set_pipeline(pipeline);
            #[cfg(not(target_arch = "wasm32"))]
            rp.multi_draw_indexed_indirect(indirect, 0, draw_count);
            #[cfg(target_arch = "wasm32")]
            for i in 0..draw_count {
                rp.draw_indexed_indirect(indirect, i as u64 * 20);
            }
        } else {
            for &(class, graph_hash, start, count) in ranges {
                if count == 0 {
                    continue;
                }
                let key = RadiantShaderKey {
                    template_id: class,
                    graph_hash,
                    feature_flags: 0,
                };
                let pipeline = self.get_or_create_pipeline(&ctx.device, key, "");
                rp.set_pipeline(pipeline);
                #[cfg(not(target_arch = "wasm32"))]
                rp.multi_draw_indexed_indirect(indirect, start as u64 * 20, count);
                #[cfg(target_arch = "wasm32")]
                for i in start..start + count {
                    rp.draw_indexed_indirect(indirect, i as u64 * 20);
                }
            }
        }
        Ok(())
    }
}

impl TransparentPass {
    fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        key: RadiantShaderKey,
        graph_wgsl: &str,
    ) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(&key) {
            // Ids >= 5 are user-registered custom templates that live in the
            // shared scene-wide registry; class 0 is this pass's own
            // transparent base. `shared_arc` is a fresh local Arc clone so
            // `guard`'s lifetime doesn't tie up `self`.
            let shared_arc = if key.template_id >= 5 {
                self.shared_registry.clone()
            } else {
                None
            };
            let guard = shared_arc.as_ref().map(|a| a.read().unwrap());
            let template = guard
                .as_ref()
                .and_then(|g| g.get(key.template_id))
                .unwrap_or_else(|| {
                    if key.template_id >= 5 {
                        log::debug!(
                            "[Transparent] template class {} not found, falling back to class 0",
                            key.template_id
                        );
                    }
                    &self.local_class0
                });
            let module = self.shader_cache.get_or_compile(
                device,
                key,
                template,
                graph_wgsl,
                libhelio::MaterialBindingConfig::for_device(device),
                "Transparent Shader",
            );
            let alpha_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            };
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Transparent Pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 40,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 24,
                                shader_location: 5,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 32,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 36,
                                shader_location: 4,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: Some(alpha_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            self.pipelines.insert(key, pipeline);
        }
        self.pipelines.get(&key).unwrap()
    }
}
