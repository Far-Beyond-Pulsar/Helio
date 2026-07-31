use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use helio::radiant::{RadiantShaderCache, RadiantShaderKey, RadiantTemplateRegistry};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroU32;

#[cfg(not(any(target_arch = "wasm32", target_os = "macos", target_os = "ios", target_os = "android")))]
const MAX_TEXTURES: usize = 256;
#[cfg(any(target_arch = "wasm32", target_os = "macos", target_os = "ios", target_os = "android"))]
const MAX_TEXTURES: usize = 16;

const TILE_SIZE: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ForwardLitGlobals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],
    num_tiles_x: u32,
    num_tiles_y: u32,
    screen_width: f32,
    screen_height: f32,
}

pub struct ForwardLitPass {
    pipelines: HashMap<RadiantShaderKey, wgpu::RenderPipeline>,
    shader_cache: RadiantShaderCache,
    template_registry: RadiantTemplateRegistry,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    bind_group_0: Option<wgpu::BindGroup>,
    bind_group_0_key: Option<(usize, usize, usize, usize, usize, usize)>,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_1_version: Option<u64>,
    globals_buf: wgpu::Buffer,
    surface_format: wgpu::TextureFormat,
    /// When true, renders from `material_class_ranges` (all opaque draws)
    /// instead of `forward_material_class_ranges` (only FLAG_FORWARD_SHADING).
    pub render_all_opaque: bool,
}

impl ForwardLitPass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ForwardLitGlobals"),
            size: std::mem::size_of::<ForwardLitGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ForwardLit BGL 0"),
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
                        visibility: wgpu::ShaderStages::FRAGMENT,
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
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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

        let bind_group_layout_1 = create_material_bgl(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ForwardLit PL"),
            bind_group_layouts: &[Some(&bind_group_layout_0), Some(&bind_group_layout_1)],
            immediate_size: 0,
        });

        let mut reg = RadiantTemplateRegistry::new_empty();
        let base_src_raw = include_str!("../shaders/forward_lit.wgsl");
        if base_src_raw.contains("//!use pbr_eval") {
            // Insert PBR_EVAL after the `enable` directive (must come before
            // any declarations in WGSL but after the enable line).
            let insert_pos = base_src_raw.find("enable ").and_then(|i| {
                base_src_raw[i..].find(';').map(|j| i + j + 1)
            }).unwrap_or(0);
            let mut resolved = String::with_capacity(
                base_src_raw.len() + libhelio::shader::PBR_EVAL.len(),
            );
            resolved.push_str(&base_src_raw[..insert_pos]);
            resolved.push('\n');
            resolved.push_str(libhelio::shader::PBR_EVAL);
            resolved.push_str(&base_src_raw[insert_pos..]);
            let leaked: &'static str = Box::leak(resolved.into_boxed_str());
            reg.override_class(0, "forward_lit", leaked);
        } else {
            reg.override_class(0, "forward_lit", base_src_raw);
        };

        Self {
            pipelines: HashMap::new(),
            shader_cache: RadiantShaderCache::new(),
            template_registry: reg,
            pipeline_layout,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_0: None,
            bind_group_0_key: None,
            bind_group_1: None,
            bind_group_1_version: None,
            globals_buf,
            surface_format,
            render_all_opaque: false,
        }
    }

    pub fn template_registry_mut(&mut self) -> &mut RadiantTemplateRegistry {
        &mut self.template_registry
    }

    fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        key: RadiantShaderKey,
        graph_wgsl: &str,
        write_depth: bool,
    ) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(&key) {
            let template = match self.template_registry.get(key.template_id) {
                Some(t) => t,
                None => {
                    log::debug!("[ForwardLit] template class {} not found, falling back to class 0", key.template_id);
                    self.template_registry.get(0).expect("Default forward-lit template missing")
                }
            };
            let module = self.shader_cache.get_or_compile(
                device, key, template, graph_wgsl, MAX_TEXTURES, "ForwardLit Shader",
            );
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ForwardLit Pipeline"),
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
                        blend: if write_depth {
                            None
                        } else {
                            Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::Zero,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent::OVER,
                            })
                        },
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(write_depth),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
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

impl RenderPass for ForwardLitPass {
    fn name(&self) -> &'static str {
        "ForwardLit"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["main_scene", "depth", "pre_aa", "cluster_light_grid"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("depth");
        builder.read("cluster_light_grid");
        builder.write_color_raw("pre_aa", self.surface_format, ResourceSize::MatchSurface);
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let pre_aa_view = resources.pre_aa.read("ForwardLit")?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: pre_aa_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("ForwardLit"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: if self.render_all_opaque { wgpu::LoadOp::Clear(1.0) } else { wgpu::LoadOp::Load },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let (ambient_color, ambient_intensity) =
            if let Some(ref ms) = ctx.frame_resources.main_scene.get().as_ref() {
                (ms.ambient_color, ms.ambient_intensity)
            } else {
                ([0.1, 0.1, 0.15], 0.1)
            };

        let num_tiles_x = ctx.width.div_ceil(TILE_SIZE);
        let num_tiles_y = ctx.height.div_ceil(TILE_SIZE);

        let globals = ForwardLitGlobals {
            frame: ctx.frame_num as u32,
            delta_time: ctx.delta_time,
            light_count: ctx.scene.lights.len() as u32,
            ambient_intensity,
            ambient_color: [ambient_color[0], ambient_color[1], ambient_color[2], 1.0],
            num_tiles_x,
            num_tiles_y,
            screen_width: ctx.width as f32,
            screen_height: ctx.height as f32,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let draw_count = ctx.scene.draw_count;
        let main_scene = ctx.resources.main_scene;

        if draw_count == 0 || main_scene.is_none() {
            return Ok(());
        }
        let ms = main_scene.read("ForwardLit").unwrap();

        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let instances_ptr = ctx.scene.instances as *const _ as usize;
        let compacted_indices_ptr = ctx.scene.compacted_indices_2 as *const _ as usize;
        let lights_ptr = ctx.scene.lights as *const _ as usize;

        let cluster = ctx.resources.cluster_light_grid.get();
        let tile_lists_ptr = cluster
            .map(|c| c.tile_light_lists as *const _ as usize)
            .unwrap_or(0);
        let tile_counts_ptr = cluster
            .map(|c| c.tile_light_counts as *const _ as usize)
            .unwrap_or(0);

        let bg0_key = (
            camera_ptr,
            instances_ptr,
            compacted_indices_ptr,
            lights_ptr,
            tile_lists_ptr,
            tile_counts_ptr,
        );
        if self.bind_group_0_key != Some(bg0_key) {
            let cluster_ref = ctx.resources.cluster_light_grid.get();
            let fallback_buf = ctx.scene.instances; // fallback buffer for tile lists when cluster is absent
            let tile_lists = cluster_ref
                .map(|c| c.tile_light_lists)
                .unwrap_or(fallback_buf);
            let tile_counts = cluster_ref
                .map(|c| c.tile_light_counts)
                .unwrap_or(fallback_buf);

            log::debug!("ForwardLit: rebuilding bind group 0 (buffer pointers changed)");
            self.bind_group_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ForwardLit BG 0"),
                layout: &self.bind_group_layout_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.globals_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.scene.compacted_indices_2.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.scene.lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: tile_lists.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: tile_counts.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_0_key = Some(bg0_key);
        }

        let needs_rebuild = self.bind_group_1_version != Some(ms.material_textures.version)
            || self.bind_group_1.is_none();
        if needs_rebuild {
            log::debug!("ForwardLit: rebuilding bind group 1 (material textures version changed)");
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.scene.materials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ms.material_textures.material_textures.as_entire_binding(),
                },
            ];
            #[cfg(not(target_arch = "wasm32"))]
            {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureViewArray(
                        ms.material_textures.texture_views,
                    ),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::SamplerArray(
                        ms.material_textures.samplers,
                    ),
                });
            }
            #[cfg(target_arch = "wasm32")]
            {
                for (index, view) in ms.material_textures.texture_views.iter().enumerate() {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2 + index as u32,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                }
                for (index, sampler) in ms.material_textures.samplers.iter().enumerate() {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2 + MAX_TEXTURES as u32 + index as u32,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    });
                }
            }
            self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ForwardLit BG 1"),
                layout: &self.bind_group_layout_1,
                entries: &entries,
            }));
            self.bind_group_1_version = Some(ms.material_textures.version);
        }

        let indirect = ctx.scene.indirect;
        let pass = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        pass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, ms.mesh_buffers.vertices.slice(..));
        pass.set_index_buffer(
            ms.mesh_buffers.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );

        if let Some(reg_any) = ctx.scene.template_registry.as_ref() {
            if let Some(reg) = reg_any.downcast_ref::<helio::radiant::RadiantTemplateRegistry>() {
                let old_keys = self.template_registry.keys();
                for (id, tpl) in reg.iter() {
                    if *id >= 5 {
                        self.template_registry.register(*id, tpl.clone());
                    }
                }
                let new_keys = self.template_registry.keys();
                if old_keys != new_keys {
                    self.pipelines.clear();
                    self.shader_cache = helio::radiant::RadiantShaderCache::new();
                }
            }
        }

        let ranges = if self.render_all_opaque {
            ctx.scene.material_class_ranges
        } else {
            ctx.scene.forward_material_class_ranges
        };
        if ranges.is_empty() {
            let key = RadiantShaderKey {
                template_id: 0,
                graph_hash: 0,
                feature_flags: if self.render_all_opaque { 1 } else { 0 },
            };
            let pipeline = self.get_or_create_pipeline(&ctx.device, key, "", self.render_all_opaque);
            pass.set_pipeline(pipeline);
            #[cfg(not(target_arch = "wasm32"))]
            pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
            #[cfg(target_arch = "wasm32")]
            for i in 0..draw_count {
                pass.draw_indexed_indirect(indirect, i as u64 * 20);
            }
        } else {
            for &(class, graph_hash, start, count) in ranges {
                if count == 0 {
                    continue;
                }
                let mut key = RadiantShaderKey {
                    template_id: class,
                    graph_hash,
                    feature_flags: 0,
                };
                if self.render_all_opaque {
                    key.feature_flags |= 1;
                }
                let graph_wgsl = ctx
                    .scene
                    .graph_wgsl_snippets
                    .get(&graph_hash)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let pipeline = self.get_or_create_pipeline(&ctx.device, key, graph_wgsl, self.render_all_opaque);
                pass.set_pipeline(pipeline);
                #[cfg(not(target_arch = "wasm32"))]
                pass.multi_draw_indexed_indirect(indirect, start as u64 * 20, count);
                #[cfg(target_arch = "wasm32")]
                for i in start..start + count {
                    pass.draw_indexed_indirect(indirect, i as u64 * 20);
                }
            }
        }
        Ok(())
    }
}

fn create_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    #[cfg(not(target_arch = "wasm32"))]
    let texture_array_count =
        NonZeroU32::new(MAX_TEXTURES as u32).expect("non-zero texture table size");

    let mut entries = vec![
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
    ];
    #[cfg(not(target_arch = "wasm32"))]
    {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: Some(texture_array_count),
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: Some(texture_array_count),
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        for index in 0..MAX_TEXTURES {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + index as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        for index in 0..MAX_TEXTURES {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + MAX_TEXTURES as u32 + index as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
    }

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ForwardLit BGL 1"),
        entries: &entries,
    })
}
