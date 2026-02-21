use blade_graphics as gpu;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GIMode {
    None,
    Realtime,
    Baked,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GIParams {
    pub num_rays: u32,
    pub max_bounces: u32,
    pub intensity: f32,
    pub _pad: u32,
}

impl Default for GIParams {
    fn default() -> Self {
        Self {
            num_rays: 1,
            max_bounces: 2,
            intensity: 1.0,
            _pad: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LightData {
    pub position: [f32; 3],
    pub light_type: u32,
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub range: f32,
}

pub struct GlobalIllumination {
    mode: GIMode,
    gi_pipeline: Option<gpu::ComputePipeline>,
    gi_buffer: Option<gpu::Texture>,
    gi_view: Option<gpu::TextureView>,
    params: GIParams,
    context: Arc<gpu::Context>,
}

impl GlobalIllumination {
    pub fn new(
        context: Arc<gpu::Context>,
        mode: GIMode,
        width: u32,
        height: u32,
    ) -> Self {
        let (gi_pipeline, gi_buffer, gi_view) = if mode == GIMode::Realtime {
            Self::create_rt_gi(context.clone(), width, height)
        } else {
            (None, None, None)
        };
        
        Self {
            mode,
            gi_pipeline,
            gi_buffer,
            gi_view,
            params: GIParams::default(),
            context,
        }
    }
    
    fn create_rt_gi(
        context: Arc<gpu::Context>,
        width: u32,
        height: u32,
    ) -> (Option<gpu::ComputePipeline>, Option<gpu::Texture>, Option<gpu::TextureView>) {
        let capabilities = context.capabilities();
        if !capabilities.ray_query.contains(gpu::ShaderVisibility::COMPUTE) {
            log::warn!("Ray tracing not supported, falling back to no GI");
            return (None, None, None);
        }
        
        let shader_source = match std::fs::read_to_string("shaders/gi_raytracing.wgsl") {
            Ok(src) => src,
            Err(_) => {
                log::warn!("GI shader not found, skipping GI setup");
                return (None, None, None);
            }
        };
        
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &shader_source,
        });
        
        #[derive(blade_macros::ShaderData)]
        struct GIData {
            params: GIParams,
            acc_struct: gpu::AccelerationStructure,
            output: gpu::TextureView,
        }
        
        let layout = <GIData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "gi_rt",
            data_layouts: &[&layout],
            compute: shader.at("compute_gi"),
        });
        
        let gi_buffer = context.create_texture(gpu::TextureDesc {
            name: "gi_buffer",
            format: gpu::TextureFormat::Rgba16Float,
            size: gpu::Extent { width, height, depth: 1 },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::STORAGE,
            sample_count: 1,
            external: None,
        });
        
        let gi_view = context.create_texture_view(
            gi_buffer,
            gpu::TextureViewDesc {
                name: "gi_view",
                format: gpu::TextureFormat::Rgba16Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        
        (Some(pipeline), Some(gi_buffer), Some(gi_view))
    }
    
    pub fn mode(&self) -> GIMode {
        self.mode
    }
    
    pub fn set_mode(&mut self, mode: GIMode, width: u32, height: u32) {
        if self.mode == mode {
            return;
        }
        
        self.cleanup();
        
        self.mode = mode;
        if mode == GIMode::Realtime {
            let (pipeline, buffer, view) = Self::create_rt_gi(
                self.context.clone(),
                width,
                height,
            );
            self.gi_pipeline = pipeline;
            self.gi_buffer = buffer;
            self.gi_view = view;
        }
    }
    
    pub fn set_params(&mut self, params: GIParams) {
        self.params = params;
    }
    
    pub fn params(&self) -> GIParams {
        self.params
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.mode != GIMode::Realtime {
            return;
        }
        
        if let Some(view) = self.gi_view {
            self.context.destroy_texture_view(view);
        }
        if let Some(buffer) = self.gi_buffer {
            self.context.destroy_texture(buffer);
        }
        
        let gi_buffer = self.context.create_texture(gpu::TextureDesc {
            name: "gi_buffer",
            format: gpu::TextureFormat::Rgba16Float,
            size: gpu::Extent { width, height, depth: 1 },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::STORAGE,
            sample_count: 1,
            external: None,
        });
        
        let gi_view = self.context.create_texture_view(
            gi_buffer,
            gpu::TextureViewDesc {
                name: "gi_view",
                format: gpu::TextureFormat::Rgba16Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        
        self.gi_buffer = Some(gi_buffer);
        self.gi_view = Some(gi_view);
    }
    
    pub fn gi_view(&self) -> Option<gpu::TextureView> {
        self.gi_view
    }
    
    fn cleanup(&mut self) {
        if let Some(view) = self.gi_view.take() {
            self.context.destroy_texture_view(view);
        }
        if let Some(buffer) = self.gi_buffer.take() {
            self.context.destroy_texture(buffer);
        }
    }
}

impl Drop for GlobalIllumination {
    fn drop(&mut self) {
        self.cleanup();
    }
}
