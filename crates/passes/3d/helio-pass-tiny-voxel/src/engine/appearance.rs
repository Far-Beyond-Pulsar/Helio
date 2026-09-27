//! Experimental spatial reconstruction of already-visible, already-lit samples.
//! It changes neither hits nor source data and cannot recover missed coverage.
use helio_core::{PassContext, RenderPass, ResourceKey, ResourceRegistry, Result};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

pub(super) struct Inputs {
    pub frame: u64,
    pub size: [u32; 2],
    pub hits: wgpu::Buffer,
    pub params: wgpu::Buffer,
}

/// Pair one terrain producer with its lighting resolve. Only same-frame inputs
/// are accepted, including after source removal, resize and graph rebuilding.
/// This experiment supports one tiny terrain producer per graph: the receiver
/// marker identifies the backend, not an individual instance.
#[derive(Clone, Default)]
pub struct Source(Arc<Mutex<Option<Inputs>>>);
impl Source {
    pub(super) fn publish(&self, inputs: Option<Inputs>) {
        *self.0.lock().unwrap() = inputs;
    }
}

pub struct AppearancePass {
    source: Source,
    pipeline: wgpu::ComputePipeline,
    output: wgpu::Texture,
    view: wgpu::TextureView,
    dummy_params: wgpu::Buffer,
    dummy_hits: wgpu::Buffer,
    size: [u32; 2],
    format: wgpu::TextureFormat,
}
impl AppearancePass {
    pub fn new(
        device: &wgpu::Device,
        source: Source,
        size: [u32; 2],
        format: wgpu::TextureFormat,
    ) -> Self {
        let storage_format = match format {
            wgpu::TextureFormat::Rgba16Float => "rgba16float",
            wgpu::TextureFormat::Rgba8Unorm => "rgba8unorm",
            _ => panic!("unsupported experimental appearance format: {format:?}"),
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visible voxel lighting reconstruction"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    crate::SHADER,
                    include_str!("appearance.wgsl").replace("rgba16float", storage_format)
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("voxel appearance"),
            layout: None,
            module: &shader,
            entry_point: Some("appearance"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = Self::target(device, size, format);
        let view = output.create_view(&Default::default());
        let dummy = |label, bytes: usize| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &vec![0u8; bytes],
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
            })
        };
        Self {
            source,
            pipeline,
            output,
            view,
            size,
            format,
            dummy_params: dummy(
                "inactive voxel appearance parameters",
                std::mem::size_of::<crate::Params>(),
            ),
            dummy_hits: dummy("inactive voxel appearance hit", 32),
        }
    }

    fn target(device: &wgpu::Device, size: [u32; 2], format: wgpu::TextureFormat) -> wgpu::Texture {
        // Own this output: publishing it as pre_aa must not let a pool alias
        // its storage with the input or a later intermediate in the graph.
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel appearance HDR"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    pub fn allocation_bytes(&self) -> u64 {
        u64::from(self.size[0])
            * u64::from(self.size[1])
            * u64::from(self.format.block_copy_size(None).unwrap())
            + self.dummy_params.size()
            + self.dummy_hits.size()
    }
}
impl RenderPass for AppearancePass {
    fn name(&self) -> &'static str {
        "TinyVoxelAppearance"
    }
    fn reads(&self) -> &'static [&'static str] {
        &["pre_aa", "gbuffer_lightmap_uv"]
    }
    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }
    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        for name in self.reads() {
            builder.read(name);
        }
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let input = ctx
            .registry
            .texture_view(ResourceKey::new("pre_aa"))
            .ok_or_else(|| helio_core::Error::ResourceNotFound("pre_aa".into()))?;
        let receiver = ctx
            .registry
            .texture_view(ResourceKey::new("gbuffer_lightmap_uv"))
            .ok_or_else(|| helio_core::Error::ResourceNotFound("gbuffer_lightmap_uv".into()))?;
        let source = self.source.0.lock().unwrap();
        let current = source
            .as_ref()
            .filter(|s| s.frame == ctx.frame_num && s.size == self.size);
        let (params, hits) = current.map_or((&self.dummy_params, &self.dummy_hits), |s| {
            (&s.params, &s.hits)
        });
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel appearance current inputs"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: hits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 32,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 33,
                    resource: wgpu::BindingResource::TextureView(receiver),
                },
                wgpu::BindGroupEntry {
                    binding: 34,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
            ],
        });
        let descriptor = wgpu::ComputePassDescriptor::default();
        let mut pass = ctx.begin_graphics_compute_pass(&descriptor);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        Ok(())
    }
    fn publish<'a>(&self, frame: &mut ResourceRegistry<'a>) {
        frame.route_named_texture("pre_aa", &self.view, self.name());
    }
    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.size = [width, height];
        self.output = Self::target(device, self.size, self.format);
        self.view = self.output.create_view(&Default::default());
    }
}
