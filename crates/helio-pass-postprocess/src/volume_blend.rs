//! Post-process volume blending, as a standalone pass.
//!
//! Runs `cs_volume_blend` (in `postprocess.wgsl`) to blend the active post-process
//! volumes against the camera defaults, then copies the result over the shared
//! post-process uniform buffer.
//!
//! # Why this is not inside PostProcessPass
//!
//! It used to be. But the blended uniforms are the frame's post-process config, and
//! `PostProcessPass` runs near the end of the graph — so anything *else* that reads
//! the config runs before the blend and sees the unblended camera defaults instead.
//! `VolumetricFogPass` is exactly that: it reads the fog block early, at internal
//! resolution, so with the blend still buried in `PostProcessPass` fog would ignore
//! every post-process volume in the scene, silently.
//!
//! Scheduling this ahead of the first consumer gives every reader the same values.

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, RenderPass, Result as HelioResult};

pub struct PostProcessVolumeBlendPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// cs_volume_blend writes here; execute() then copies it over the uniform buffer.
    blend_output_buf: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize, usize)>,
}

impl PostProcessVolumeBlendPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // Same source as PostProcessPass, and it now opts into the prelude, so it
        // has to be resolved the same way or the shared symbols are missing.
        let shader = helio_core::shader::module(
            device,
            "PostProcess Volume Blend Shader",
            include_str!("../shaders/postprocess.wgsl"),
        );

        let cv = wgpu::ShaderStages::COMPUTE;
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cv,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cv,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // Matches PostProcessPass's blend_bgl: postprocess (b0), camera (b1),
        // pp_volumes (b15), blend_output (b16).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Volume Blend BGL"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                storage_entry(15, true),
                storage_entry(16, false),
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Volume Blend PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PostProcess Volume Blend"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_volume_blend"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blend_output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Blend Output"),
            size: std::mem::size_of::<libhelio::GpuPostProcessUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            blend_output_buf,
            bind_group: None,
            bind_group_key: None,
        }
    }
}

impl RenderPass for PostProcessVolumeBlendPass {
    fn name(&self) -> &'static str {
        "PostProcessVolumeBlendPass"
    }

    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn chain_transparent(&self) -> bool {
        // execute() only touches ctx.compute_encoder_ptr.
        true
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // No volumes: the camera defaults the renderer already uploaded are the
        // final config, so there is nothing to blend and nothing to copy.
        if ctx.resources.pp_volume_count == 0 {
            return Ok(());
        }

        let Some(postprocess_buf) = ctx.resources.postprocess_uniforms.get() else {
            return Ok(());
        };
        let Some(pp_volumes_buf) = ctx.resources.pp_volumes.get() else {
            return Ok(());
        };
        let camera_buf = ctx.scene.camera;

        let key = (
            postprocess_buf as *const _ as usize,
            camera_buf as *const _ as usize,
            pp_volumes_buf as *const _ as usize,
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PostProcess Volume Blend BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: postprocess_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 15, resource: pp_volumes_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 16, resource: self.blend_output_buf.as_entire_binding() },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let Some(bind_group) = self.bind_group.as_ref() else {
            return Ok(());
        };

        let ce = ctx.compute_encoder_ptr;
        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Volume Blend"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        unsafe { &mut *ce }.copy_buffer_to_buffer(
            &self.blend_output_buf,
            0,
            postprocess_buf,
            0,
            std::mem::size_of::<libhelio::GpuPostProcessUniforms>() as u64,
        );

        Ok(())
    }
}
