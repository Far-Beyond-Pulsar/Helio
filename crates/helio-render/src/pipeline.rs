use blade_graphics as gpu;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PipelineFlags: u32 {
        const ALPHA_BLEND = 1 << 0;
        const DEPTH_TEST = 1 << 1;
        const DEPTH_WRITE = 1 << 2;
        const CULL_BACK = 1 << 3;
        const CULL_FRONT = 1 << 4;
        const WIREFRAME = 1 << 5;
        const CONSERVATIVE_RASTER = 1 << 6;
    }
}

pub struct PipelineDesc {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub flags: PipelineFlags,
}

pub struct Pipeline {
    pub flags: PipelineFlags,
}

impl Pipeline {
    pub fn new(_context: &gpu::Context, desc: &PipelineDesc) -> Self {
        Self {
            flags: desc.flags,
        }
    }
}

pub struct ComputePipeline {
    pub shader: String,
}

impl ComputePipeline {
    pub fn new(_context: &gpu::Context, shader: &str) -> Self {
        Self {
            shader: shader.to_string(),
        }
    }
}
