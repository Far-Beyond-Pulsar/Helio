use blade_graphics as gpu;
use helio_core::*;
use parking_lot::RwLock;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, HelioError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPath {
    Forward,
    Deferred,
    ForwardPlus,
    VisibilityBuffer,
}

#[derive(Debug, Clone)]
pub struct RendererConfig {
    pub render_path: RenderPath,
    pub enable_msaa: bool,
    pub msaa_samples: u32,
    pub enable_taa: bool,
    pub enable_hdr: bool,
    pub enable_depth_prepass: bool,
    pub enable_async_compute: bool,
    pub shadow_resolution: u32,
    pub max_lights: u32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            render_path: RenderPath::Deferred,
            enable_msaa: true,
            msaa_samples: 4,
            enable_taa: true,
            enable_hdr: true,
            enable_depth_prepass: true,
            enable_async_compute: true,
            shadow_resolution: 2048,
            max_lights: 1024,
        }
    }
}

pub struct Renderer {
    #[allow(dead_code)]
    context: Arc<RenderContext>,
    #[allow(dead_code)]
    config: RwLock<RendererConfig>,
    #[allow(dead_code)]
    frame_graph: RwLock<FrameGraph>,
    
    // Frame data
    frame_count: u64,
}

impl Renderer {
    pub fn new(context: Arc<RenderContext>, config: RendererConfig) -> Self {
        Self {
            context,
            config: RwLock::new(config),
            frame_graph: RwLock::new(FrameGraph::new()),
            frame_count: 0,
        }
    }
    
    pub fn initialize(&mut self, _width: u32, _height: u32) -> Result<()> {
        Ok(())
    }
    
    pub fn render(&mut self, _scene: &Scene, _viewport: &Viewport) -> Result<()> {
        self.frame_count += 1;
        Ok(())
    }
    
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.initialize(width, height)
    }
    
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}
