use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub struct PlanarReflectionPass;

impl PlanarReflectionPass {
    pub fn new(
        _device: &wgpu::Device,
        _camera_buf: &wgpu::Buffer,
        _surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self
    }
}

impl RenderPass for PlanarReflectionPass {
    fn name(&self) -> &'static str {
        "PlanarReflection"
    }

    fn declare_resources(&self, _builder: &mut ResourceBuilder) {}

    fn reads(&self) -> &'static [&'static str] {
        &[]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}
}
