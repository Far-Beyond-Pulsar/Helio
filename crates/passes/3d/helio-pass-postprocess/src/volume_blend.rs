//! Pass-owned SceneDB settings resolution, including empty scenes.
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, RenderPass, ResourceKey, Result as HelioResult};
use pulsar_scenedb::gpu::BufferKey;
use wgpu::util::DeviceExt;

pub struct PostProcessVolumeBlendPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    defaults: wgpu::Buffer,
    blend_output_buf: wgpu::Buffer,
    resolved: wgpu::Buffer,
    fallback_pp_volumes: wgpu::Buffer,
    fallback_cameras: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<[wgpu::Buffer; 3]>,
}
impl PostProcessVolumeBlendPass {
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_defaults(device, &crate::PostProcessSettings::default())
    }
    pub fn with_defaults(device: &wgpu::Device, settings: &crate::PostProcessSettings) -> Self {
        let shader = helio_core::shader::module(device, "PostProcess Resolver", include_str!("../shaders/postprocess.wgsl"));
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Resolver BGL"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(15, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(16, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(20, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Resolver PL"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PostProcess Resolver"), layout: Some(&layout), module: &shader,
            entry_point: Some("cs_volume_blend"), compilation_options: Default::default(), cache: None,
        });
        let defaults = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PostProcess Defaults"), contents: bytemuck::bytes_of(&settings.to_gpu()), usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        let size = std::mem::size_of::<crate::GpuPostProcessUniforms>() as u64;
        Self {
            pipeline, bgl, defaults,
            blend_output_buf: buffer("PostProcess Resolve Storage", size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            resolved: buffer("PostProcess Resolved Uniforms", size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC),
            fallback_pp_volumes: buffer("PostProcess Empty Volumes", std::mem::size_of::<crate::GpuPostProcessVolume>() as u64, wgpu::BufferUsages::STORAGE),
            fallback_cameras: buffer("PostProcess Empty Cameras", std::mem::size_of::<crate::CameraPostProcessComponent>() as u64, wgpu::BufferUsages::STORAGE),
            bind_group: None, bind_group_key: None,
        }
    }
    /// GPU-derived settings, valid after the resolver dispatch and copy.
    pub fn resolved_uniforms(&self) -> &wgpu::Buffer { &self.resolved }
}
impl RenderPass for PostProcessVolumeBlendPass {
    fn name(&self) -> &'static str { "PostProcessVolumeBlendPass" }
    fn declare_resources(&self, builder: &mut ResourceBuilder) { builder.write_buffer("postprocess_uniforms"); }
    fn writes(&self) -> &'static [&'static str] { &["postprocess_uniforms"] }
    fn render_pass_descriptor<'a>(&'a self, _: &'a wgpu::TextureView, _: &'a wgpu::TextureView, _: &'a helio_core::ResourceRegistry<'a>) -> Option<wgpu::RenderPassDescriptor<'a>> { None }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let volumes = ctx.scene_buffers.get(BufferKey::of("post_process_volumes")).map(|h| &h.buffer).unwrap_or(&self.fallback_pp_volumes);
        let cameras = ctx.scene_buffers.get(BufferKey::of("camera_postprocess")).map(|h| &h.buffer).unwrap_or(&self.fallback_cameras);
        let key = [ctx.camera.clone(), volumes.clone(), cameras.clone()];
        if self.bind_group_key.as_ref() != Some(&key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PostProcess Resolver BG"), layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.defaults.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ctx.camera.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 15, resource: volumes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 16, resource: self.blend_output_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 20, resource: cameras.as_entire_binding() },
                ],
            }));
            self.bind_group_key = Some(key);
        }
        // Record with the fog consumers to preserve producer/copy/consumer order.
        let encoder = unsafe { &mut *ctx.encoder_ptr };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("PostProcess Resolve"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.blend_output_buf, 0, &self.resolved, 0, std::mem::size_of::<crate::GpuPostProcessUniforms>() as u64);
        Ok(())
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let buffer: &'a wgpu::Buffer = unsafe { std::mem::transmute(&self.resolved) };
        frame.write(ResourceKey::new("postprocess_uniforms"), buffer, self.name());
    }
}
