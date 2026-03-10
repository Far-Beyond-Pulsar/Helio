//! SdfFeature — implements the Feature trait for SDF rendering
//!
//! Creates GPU resources (3D volume texture, edit buffer, compute + render pipelines)
//! in `register()` and uploads per-frame data in `prepare()`.

use crate::features::{Feature, FeatureContext, PrepareContext, ShaderDefine};
use crate::Result;
use super::edit_list::{SdfEditList, SdfEdit, GpuSdfEdit, MAX_EDITS};
use super::uniforms::SdfGridParams;
use super::passes::{SdfEvaluateDensePass, SdfRayMarchPass};
use std::collections::HashMap;
use std::sync::Arc;

/// Default grid resolution (128^3 = 4 MB at R16Float)
const DEFAULT_GRID_DIM: u32 = 128;

/// SDF rendering feature
///
/// # Usage
/// ```ignore
/// let sdf = SdfFeature::new()
///     .with_grid_dim(128)
///     .with_volume_bounds([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
/// ```
pub struct SdfFeature {
    enabled: bool,
    grid_dim: u32,
    volume_min: [f32; 3],
    volume_max: [f32; 3],

    // CPU-side state
    edit_list: SdfEditList,
    last_uploaded_gen: u64,

    // GPU resources (populated in register())
    volume_texture: Option<Arc<wgpu::Texture>>,
    volume_view: Option<Arc<wgpu::TextureView>>,
    edit_buffer: Option<Arc<wgpu::Buffer>>,
    params_buffer: Option<Arc<wgpu::Buffer>>,
    sampler: Option<Arc<wgpu::Sampler>>,

    // Compute pipeline + bind group for evaluation
    eval_pipeline: Option<Arc<wgpu::ComputePipeline>>,
    eval_bind_group: Option<Arc<wgpu::BindGroup>>,
    eval_bg_layout: Option<wgpu::BindGroupLayout>,

    // Render pipeline + bind group for ray marching
    march_pipeline: Option<Arc<wgpu::RenderPipeline>>,
    march_bind_group: Option<Arc<wgpu::BindGroup>>,
}

impl SdfFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            grid_dim: DEFAULT_GRID_DIM,
            volume_min: [-10.0, -10.0, -10.0],
            volume_max: [10.0, 10.0, 10.0],
            edit_list: SdfEditList::new(),
            last_uploaded_gen: u64::MAX,
            volume_texture: None,
            volume_view: None,
            edit_buffer: None,
            params_buffer: None,
            sampler: None,
            eval_pipeline: None,
            eval_bind_group: None,
            eval_bg_layout: None,
            march_pipeline: None,
            march_bind_group: None,
        }
    }

    pub fn with_grid_dim(mut self, dim: u32) -> Self {
        self.grid_dim = dim;
        self
    }

    pub fn with_volume_bounds(mut self, min: [f32; 3], max: [f32; 3]) -> Self {
        self.volume_min = min;
        self.volume_max = max;
        self
    }

    /// Add an SDF edit to the scene.
    /// Changes take effect on the next frame.
    pub fn add_edit(&mut self, edit: SdfEdit) {
        self.edit_list.add(edit);
    }

    /// Remove an SDF edit by index.
    pub fn remove_edit(&mut self, index: usize) {
        self.edit_list.remove(index);
    }

    /// Replace an SDF edit at the given index.
    pub fn set_edit(&mut self, index: usize, edit: SdfEdit) {
        self.edit_list.set(index, edit);
    }

    /// Clear all SDF edits.
    pub fn clear_edits(&mut self) {
        self.edit_list.clear();
    }

    /// Access the edit list.
    pub fn edit_list(&self) -> &SdfEditList {
        &self.edit_list
    }

    /// Mutable access to the edit list.
    pub fn edit_list_mut(&mut self) -> &mut SdfEditList {
        &mut self.edit_list
    }

    /// Rebuild the evaluation bind group (needed after params/edit buffer changes)
    fn rebuild_eval_bind_group(&mut self, device: &wgpu::Device) {
        let layout = self.eval_bg_layout.as_ref().expect("eval BG layout must be set");
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Evaluate Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.edit_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        self.volume_view.as_ref().unwrap(),
                    ),
                },
            ],
        });
        self.eval_bind_group = Some(Arc::new(bg));
    }
}

impl Feature for SdfFeature {
    fn name(&self) -> &str { "sdf" }

    fn register(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        let device = ctx.device;
        let dim = self.grid_dim;

        log::info!("SDF Feature: registering with {}^3 grid", dim);

        // ── 3D Volume Texture ──────────────────────────────────────────────────
        let volume_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SDF Volume"),
            size: wgpu::Extent3d {
                width: dim,
                height: dim,
                depth_or_array_layers: dim,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let volume_view = Arc::new(volume_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        }));

        // ── Edit Storage Buffer ─────────────────────────────────────────────────
        let edit_buffer_size = (MAX_EDITS * std::mem::size_of::<GpuSdfEdit>()) as u64;
        let edit_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Edit Buffer"),
            size: edit_buffer_size.max(64), // wgpu requires non-zero
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // ── Params Uniform Buffer ──────────────────────────────────────────────
        let params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Params Uniform"),
            size: std::mem::size_of::<SdfGridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // ── Sampler (trilinear, clamp to edge) ─────────────────────────────────
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SDF Volume Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        }));

        // ── Compute Pipeline (SDF evaluation) ──────────────────────────────────
        let eval_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Evaluate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sdf/sdf_evaluate_dense.wgsl").into(),
            ),
        });
        let eval_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("SDF Evaluate Pipeline"),
                layout: None, // auto-derive from shader
                module: &eval_shader,
                entry_point: Some("cs_evaluate"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        let eval_bg_layout = eval_pipeline.get_bind_group_layout(0);

        let eval_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Evaluate Bind Group"),
            layout: &eval_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&volume_view),
                },
            ],
        }));

        // ── Render Pipeline (SDF ray march) ────────────────────────────────────
        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Ray March Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sdf/sdf_ray_march.wgsl").into(),
            ),
        });

        // Create the SDF-specific bind group layout (group 1)
        let march_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Ray March BG Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Use the global bind group layout for group 0
        let global_layout = &ctx.resources.bind_group_layouts.global;
        let march_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SDF Ray March Pipeline Layout"),
            bind_group_layouts: &[
                Some(global_layout.as_ref()),
                Some(&march_bg_layout),
            ],
            immediate_size: 0,
        });

        let march_pipeline = Arc::new(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("SDF Ray March Pipeline"),
                layout: Some(&march_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &march_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &march_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        let march_bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Ray March Bind Group"),
            layout: &march_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        }));

        // ── Register passes on the render graph ────────────────────────────────
        ctx.graph.add_pass(SdfEvaluateDensePass::new(
            eval_pipeline.clone(),
            eval_bind_group.clone(),
            dim,
        ));
        ctx.graph.add_pass(SdfRayMarchPass::new(
            march_pipeline.clone(),
            march_bind_group.clone(),
        ));

        // Store resources
        self.volume_texture = Some(volume_texture);
        self.volume_view = Some(volume_view);
        self.edit_buffer = Some(edit_buffer);
        self.params_buffer = Some(params_buffer);
        self.sampler = Some(sampler);
        self.eval_pipeline = Some(eval_pipeline);
        self.eval_bind_group = Some(eval_bind_group);
        self.eval_bg_layout = Some(eval_bg_layout);
        self.march_pipeline = Some(march_pipeline);
        self.march_bind_group = Some(march_bind_group);

        log::info!("SDF Feature: registered successfully");
        Ok(())
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        if !self.enabled { return Ok(()); }

        let gen = self.edit_list.generation();
        let needs_upload = gen != self.last_uploaded_gen;

        if needs_upload {
            // Upload edit list to GPU
            let gpu_edits = self.edit_list.flush_gpu_data();
            if !gpu_edits.is_empty() {
                ctx.queue.write_buffer(
                    self.edit_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&gpu_edits),
                );
            }

            // Upload grid params
            let params = SdfGridParams::new(
                self.volume_min,
                self.volume_max,
                self.grid_dim,
                self.edit_list.len() as u32,
            );
            ctx.queue.write_buffer(
                self.params_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&params),
            );

            self.last_uploaded_gen = gen;
        }

        Ok(())
    }

    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        let mut defines = HashMap::new();
        defines.insert("ENABLE_SDF".into(), ShaderDefine::Bool(self.enabled));
        defines
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn cleanup(&mut self, _device: &wgpu::Device) {
        self.volume_texture = None;
        self.volume_view = None;
        self.edit_buffer = None;
        self.params_buffer = None;
        self.sampler = None;
        self.eval_pipeline = None;
        self.eval_bind_group = None;
        self.eval_bg_layout = None;
        self.march_pipeline = None;
        self.march_bind_group = None;
    }
}
