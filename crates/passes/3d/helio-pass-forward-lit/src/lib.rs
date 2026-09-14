use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use helio::radiant::{RadiantShaderCache, RadiantShaderKey};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

mod components;
pub use components::{LightComponent, MAX_LIGHTS};
use pulsar_scenedb::gpu::BufferKey;

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
    /// 1 when `lights[i]`/`transforms[i]` share the same raw entity index
    /// (the SceneDB-direct `"scene_lights"` path) and can be indexed
    /// directly; 0 when `lights` is instead `ctx.scene.lights` -- a
    /// freshly-rebuilt-every-frame dense array (`Renderer::submit_light_
    /// frame`, driven by `engine_backend`'s per-frame SceneDB resolve, the
    /// path production actually uses today) whose entry `i` came from
    /// whatever entity `light_entity_indices[i]` names, not entity `i`
    /// itself. See `light_entity_indices`'s binding doc in the shader.
    light_mode_direct_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct ForwardLitPass {
    material_binding: libhelio::MaterialBindingConfig,
    pipelines: HashMap<RadiantShaderKey, wgpu::RenderPipeline>,
    shader_cache: RadiantShaderCache,
    /// This pass's own class-0 override (never synced from the scene).
    local_class0: helio::radiant::RadiantTemplate,
    /// User-registered custom templates (id >= 5), shared with the renderer
    /// and other passes — never deep-cloned (see `SharedTemplateRegistry`).
    shared_registry: Option<helio::radiant::SharedTemplateRegistry>,
    /// Key set as of the last sync, to detect content changes cheaply.
    last_shared_keys: Vec<u32>,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    bind_group_0: Option<wgpu::BindGroup>,
    bind_group_0_key: Option<(usize, usize, usize, usize, usize, usize, usize, usize)>,
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
        let material_binding = libhelio::MaterialBindingConfig::for_device(device);
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
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

        let bind_group_layout_1 = create_material_bgl(device, material_binding);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ForwardLit PL"),
            bind_group_layouts: &[Some(&bind_group_layout_0), Some(&bind_group_layout_1)],
            immediate_size: 0,
        });

        let base_src_raw = include_str!("../shaders/forward_lit.wgsl");
        let wgsl_source: &'static str = if base_src_raw.contains("//!use pbr_eval") {
            // Insert PBR_EVAL after the `enable` directive (must come before
            // any declarations in WGSL but after the enable line).
            let insert_pos = base_src_raw
                .find("enable ")
                .and_then(|i| base_src_raw[i..].find(';').map(|j| i + j + 1))
                .unwrap_or(0);
            let mut resolved =
                String::with_capacity(base_src_raw.len() + libhelio::shader::PBR_EVAL.len());
            resolved.push_str(&base_src_raw[..insert_pos]);
            resolved.push('\n');
            resolved.push_str(libhelio::shader::PBR_EVAL);
            resolved.push_str(&base_src_raw[insert_pos..]);
            Box::leak(resolved.into_boxed_str())
        } else {
            base_src_raw
        };
        let local_class0 = helio::radiant::RadiantTemplate {
            name: "forward_lit",
            wgsl_source,
        };

        Self {
            material_binding,
            pipelines: HashMap::new(),
            shader_cache: RadiantShaderCache::new(),
            local_class0,
            shared_registry: None,
            last_shared_keys: Vec::new(),
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

    fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        key: RadiantShaderKey,
        graph_wgsl: &str,
        write_depth: bool,
    ) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(&key) {
            // Ids >= 5 are user-registered custom templates that live in the
            // shared scene-wide registry; anything else (including a miss)
            // falls back to this pass's own class-0 override. `shared_arc`
            // is a fresh local Arc clone so `guard`'s lifetime doesn't tie
            // up `self`.
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
                            "[ForwardLit] template class {} not found, falling back to class 0",
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
                self.material_binding,
                "ForwardLit Shader",
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
        &[
            "material_textures",
            "render_environment",
            "depth",
            "pre_aa",
            "cluster_light_grid",
            "object_batch",
            "culled_batch",
        ]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("depth");
        builder.read("cluster_light_grid");
        builder.read("object_batch");
        builder.read("culled_batch");
        builder.write_color_raw("pre_aa", self.surface_format, ResourceSize::MatchSurface);
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::PassResources<'a>) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::PassResources<'a>,
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
                    load: if self.render_all_opaque {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
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
            if let Some(ref environment) = ctx.pass_resources.render_environment.get().as_ref() {
                (environment.ambient_color, environment.ambient_intensity)
            } else {
                ([0.1, 0.1, 0.15], 0.1)
            };

        let num_tiles_x = ctx.width.div_ceil(TILE_SIZE);
        let num_tiles_y = ctx.height.div_ceil(TILE_SIZE);

        // Prefer the SceneDB-direct `"scene_lights"` buffer (fixed capacity
        // `MAX_LIGHTS`, no per-frame CPU query -- see `LightComponent`'s
        // module doc) when something has actually inserted one. Nothing in
        // `engine_backend`/`helio_component` does today -- production's real
        // light source is `libhelio::LightsFrameData` (the `Renderer`-seeded
        // `light_count`/`lights` bridge -- see that struct's own doc), so
        // that CPU-resolved count is the fallback, not a legacy dead end.
        let use_direct_index = ctx.scene_buffers.contains(BufferKey::of("scene_lights"));
        let light_count = if use_direct_index { MAX_LIGHTS } else { 0 };

        let globals = ForwardLitGlobals {
            frame: ctx.frame_num as u32,
            delta_time: ctx.delta_time,
            light_count,
            ambient_intensity,
            ambient_color: [ambient_color[0], ambient_color[1], ambient_color[2], 1.0],
            num_tiles_x,
            num_tiles_y,
            screen_width: ctx.width as f32,
            screen_height: ctx.height as f32,
            light_mode_direct_index: use_direct_index as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let Some(batch) = ctx.resources.object_batch.get() else {
            return Ok(());
        };
        let Some(culled) = ctx.resources.culled_batch.get() else {
            return Ok(());
        };
        let draw_count = batch.draw_count;

        if draw_count == 0 {
            return Ok(());
        }
        let Some(material_textures) = ctx.resources.material_textures.read("ForwardLit") else {
            return Ok(());
        };
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

        // Material rows are SceneDB component data.  Resolve the column by
        // key so the renderer never becomes the material authority again.
        let materials_handle = ctx.scene_buffers.get(BufferKey::of("materials"));
        let materials_buf = materials_handle
            .map(|handle| &handle.buffer)
            .unwrap_or(batch.instances);
        let materials_epoch = materials_handle.map(|handle| handle.epoch).unwrap_or(0);

        let lights_buf = ctx
            .scene_buffers
            .get(BufferKey::of("scene_lights"))
            .map(|handle| &handle.buffer)
            .unwrap_or(batch.instances);

        let camera_ptr = ctx.camera as *const _ as usize;
        let instances_ptr = batch.instances as *const _ as usize;
        let compacted_indices_ptr = culled.compacted_indices as *const _ as usize;
        let lights_ptr = lights_buf as *const _ as usize;
        let light_entity_indices_ptr = 0;
        // `None` (mirror not attached / no entity has a Transform yet) folds
        // to 0, same as the `cluster` map-or-0 below -- distinct from any
        // real buffer's address, so it still forces a rebind the moment a
        // real Transform buffer shows up.
        let transforms_ptr = 0;

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
            light_entity_indices_ptr,
            transforms_ptr,
        );
        if self.bind_group_0_key != Some(bg0_key) {
            let cluster_ref = ctx.resources.cluster_light_grid.get();
            let fallback_buf = batch.instances; // fallback buffer for tile lists when cluster is absent
            let tile_lists = cluster_ref
                .map(|c| c.tile_light_lists)
                .unwrap_or(fallback_buf);
            let tile_counts = cluster_ref
                .map(|c| c.tile_light_counts)
                .unwrap_or(fallback_buf);
            // Same fallback idea as `tile_lists`/`tile_counts` above: before
            // `Scene::rebind_transform_buffer` has ever been called (e.g.
            // the very first frame), bind *some* valid buffer so bind-group
            // creation can't fail -- `light_count` is 0 whenever no real
            // `Transform` buffer exists yet, so this fallback is never
            // actually dereferenced at a live light's index in practice.
            let transforms = fallback_buf;
            let light_entity_indices_buf = fallback_buf;

            log::debug!("ForwardLit: rebuilding bind group 0 (buffer pointers changed)");
            self.bind_group_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ForwardLit BG 0"),
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
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: culled.compacted_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: lights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: tile_lists.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: tile_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: light_entity_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: transforms.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_0_key = Some(bg0_key);
        }

        let needs_rebuild = self.bind_group_1_version != Some(
            material_textures.version ^ materials_epoch,
        )
            || self.bind_group_1.is_none();
        if needs_rebuild {
            log::debug!("ForwardLit: rebuilding bind group 1 (material textures version changed)");
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: materials_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: material_textures.material_textures.as_entire_binding(),
                },
            ];
            self.material_binding.append_bind_group_entries(
                &mut entries,
                2,
                material_textures.texture_views,
                material_textures.samplers,
            );
            self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ForwardLit BG 1"),
                layout: &self.bind_group_layout_1,
                entries: &entries,
            }));
            self.bind_group_1_version = Some(material_textures.version ^ materials_epoch);
        }

        let indirect = culled.indirect;
        let pass = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        pass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);

        let ranges = if self.render_all_opaque {
            batch.opaque_ranges
        } else {
            batch.forward_ranges
        };
        if ranges.is_empty() {
            let key = RadiantShaderKey {
                template_id: 0,
                graph_hash: 0,
                feature_flags: if self.render_all_opaque { 1 } else { 0 },
            };
            let pipeline =
                self.get_or_create_pipeline(&ctx.device, key, "", self.render_all_opaque);
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
                let pipeline = self.get_or_create_pipeline(
                    &ctx.device,
                    key,
                    "",
                    self.render_all_opaque,
                );
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

fn create_material_bgl(
    device: &wgpu::Device,
    material_binding: libhelio::MaterialBindingConfig,
) -> wgpu::BindGroupLayout {
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
    material_binding.append_layout_entries(&mut entries, 2, wgpu::ShaderStages::FRAGMENT);

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ForwardLit BGL 1"),
        entries: &entries,
    })
}
