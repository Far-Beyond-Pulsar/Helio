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
    bind_group: wgpu::BindGroup,
    bind_group_layout_1: wgpu::BindGroupLayout,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_1_key: Option<(usize, usize, usize)>,
    globals_buf: wgpu::Buffer,
    surface_format: wgpu::TextureFormat,
}

impl TransparentPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        instances_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
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
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Transparent BG 0"),
            layout: &bgl_0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: instances_buf.as_entire_binding() },
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
            let mut resolved = String::with_capacity(
                base_src.len() + libhelio::shader::PBR_EVAL.len(),
            );
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
            bind_group,
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
        &["main_scene", "depth", "cluster_light_grid"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("depth");
        builder.read("cluster_light_grid");
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let num_tiles_x = ctx.width.div_ceil(16);
        let num_tiles_y = ctx.height.div_ceil(16);
        ctx.queue.write_buffer(
            &self.globals_buf,
            0,
            bytemuck::bytes_of(&TransparentGlobals {
                frame: ctx.frame_num as u32,
                delta_time: 0.0,
                light_count: ctx.scene.movable_light_count,
                ambient_intensity: 0.6,
                ambient_color: [0.3, 0.35, 0.4, 1.0],
                rc_world_min: [0.0; 4],
                rc_world_max: [0.0; 4],
                csm_splits: [0.0; 4],
                num_tiles_x,
                num_tiles_y,
                screen_width: ctx.width as f32,
                screen_height: ctx.height as f32,
            }),
        );
        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })]));
        let depth_view = resources.full_res_depth.get().unwrap_or(depth);
        Some(wgpu::RenderPassDescriptor {
            label: Some("Transparent"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let draw_count = ctx.scene.draw_count;
        log::info!("[TransparentPass] execute: draw_count={}, transparent_ranges={:?}",
            draw_count, ctx.scene.transparent_material_class_ranges);
        if draw_count == 0 { return Ok(()); }

        // Sync transparent templates from GpuScene (merge into existing registry,
        // keeping the transparent base at class 0).
        if let Some(reg_any) = ctx.scene.transparent_template_registry.as_ref() {
            if let Some(shared) = reg_any.downcast_ref::<helio::radiant::SharedTemplateRegistry>() {
                // Only custom templates (id >= 5) apply here — class 0 is
                // always the transparent base and must not be overwritten.
                let new_keys: Vec<u32> = shared
                    .read()
                    .unwrap()
                    .keys()
                    .into_iter()
                    .filter(|id| *id >= 5)
                    .collect();
                if self.last_shared_keys != new_keys {
                    self.pipelines.clear();
                    self.shader_cache = helio::radiant::RadiantShaderCache::new();
                    self.last_shared_keys = new_keys;
                }
                self.shared_registry = Some(std::sync::Arc::clone(shared));
            }
        }

        let main_scene = ctx.resources.main_scene.read("Transparent");
        let ms = main_scene.as_ref().ok_or_else(|| {
            helio_core::Error::InvalidPassConfig("TransparentPass requires main_scene".to_string())
        })?;

        // Rebuild bind group 1 (lights + cluster data) when buffer pointers change
        let cluster = ctx.resources.cluster_light_grid.get();
        let lights_ptr = ctx.scene.lights as *const _ as usize;
        let tile_lists_ptr = cluster.map(|c| c.tile_light_lists as *const _ as usize).unwrap_or(0);
        let tile_counts_ptr = cluster.map(|c| c.tile_light_counts as *const _ as usize).unwrap_or(0);
        let bg1_key = (lights_ptr, tile_lists_ptr, tile_counts_ptr);
        if self.bind_group_1_key != Some(bg1_key) {
            let fallback = ctx.scene.instances;
            let tile_lists = cluster.map(|c| c.tile_light_lists).unwrap_or(fallback);
            let tile_counts = cluster.map(|c| c.tile_light_counts).unwrap_or(fallback);
            self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Transparent BG 1"),
                layout: &self.bind_group_layout_1,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.lights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: tile_lists.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: tile_counts.as_entire_binding() },
                ],
            }));
            self.bind_group_1_key = Some(bg1_key);
        }

        let indirect = ctx.scene.indirect;
        let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        rp.set_bind_group(0, &self.bind_group, &[]);
        rp.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        rp.set_vertex_buffer(0, ms.mesh_buffers.vertices.slice(..));
        rp.set_index_buffer(ms.mesh_buffers.indices.slice(..), wgpu::IndexFormat::Uint32);

        let ranges = ctx.scene.transparent_material_class_ranges;
        if ranges.is_empty() {
            let pipeline = self.get_or_create_pipeline(&ctx.device, RadiantShaderKey { template_id: 0, graph_hash: 0, feature_flags: 0 }, "");
            rp.set_pipeline(pipeline);
            #[cfg(not(target_arch = "wasm32"))]
            rp.multi_draw_indexed_indirect(indirect, 0, draw_count);
            #[cfg(target_arch = "wasm32")]
            for i in 0..draw_count { rp.draw_indexed_indirect(indirect, i as u64 * 20); }
        } else {
            for &(class, graph_hash, start, count) in ranges {
                if count == 0 { continue; }
                let key = RadiantShaderKey { template_id: class, graph_hash, feature_flags: 0 };
                let graph_wgsl = ctx.scene.graph_wgsl_snippets.get(&graph_hash).map(|s| s.as_str()).unwrap_or("");
                let pipeline = self.get_or_create_pipeline(&ctx.device, key, graph_wgsl);
                rp.set_pipeline(pipeline);
                #[cfg(not(target_arch = "wasm32"))]
                rp.multi_draw_indexed_indirect(indirect, start as u64 * 20, count);
                #[cfg(target_arch = "wasm32")]
                for i in start..start + count { rp.draw_indexed_indirect(indirect, i as u64 * 20); }
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
            let template = guard.as_ref().and_then(|g| g.get(key.template_id)).unwrap_or_else(|| {
                if key.template_id >= 5 {
                    log::debug!("[Transparent] template class {} not found, falling back to class 0", key.template_id);
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
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 12, shader_location: 1 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 5 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 32, shader_location: 3 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 36, shader_location: 4 },
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
