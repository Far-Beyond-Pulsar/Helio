/// Per-resource debug info for the debug overlay.
#[derive(Clone)]
pub struct DebugResourceInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format_name: String,
    pub size_kb: u64,
    pub alias: String,
    /// True → texture is only accessed within a subpass chain; content stays
    /// in tile memory and `StoreOp::Discard` prevents a VRAM write-back.
    pub chain_local: bool,
    /// Index of the pass that first writes this resource.
    pub first_write_pass: usize,
    /// Index of the last pass that reads it.
    pub last_read_pass: usize,
}

/// Per-pass debug info for the debug overlay.
#[derive(Clone)]
pub struct DebugPassInfo {
    pub index: usize,
    pub name: String,
    pub kind: String, // "C" or "R"
    pub writes: Vec<String>,
    pub chain_marker: String,
}

/// All debug data for a single frame.
#[derive(Clone, Default)]
pub struct FrameDebugData {
    pub resources: Vec<DebugResourceInfo>,
    pub total_vram_kb: u64,
    pub passes: Vec<DebugPassInfo>,
    pub subpass_chains: Vec<String>,
    pub frame_count: u64,
    pub delta_time: f32,
}

pub(crate) fn format_bpp(fmt: wgpu::TextureFormat) -> u32 {
    use wgpu::TextureFormat::*;
    match fmt {
        R8Unorm | R8Snorm | R8Uint | R8Sint => 8,
        R16Unorm | R16Snorm | R16Uint | R16Sint | R16Float | Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint => 16,
        R32Uint | R32Sint | R32Float | Rg16Unorm | Rg16Snorm | Rg16Uint | Rg16Sint | Rg16Float | Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Rgba8Sint | Bgra8UnormSrgb => 32,
        Rg32Uint | Rg32Sint | Rg32Float | Rgba16Unorm | Rgba16Snorm | Rgba16Uint | Rgba16Sint | Rgba16Float => 64,
        Rgba32Uint | Rgba32Sint | Rgba32Float => 128,
        Depth32Float => 32,
        _ => 32,
    }
}

pub(crate) fn format_name(fmt: wgpu::TextureFormat) -> &'static str {
    use wgpu::TextureFormat::*;
    match fmt {
        Rgba16Float => "Rgba16Float",
        Rgba8Unorm => "Rgba8Unorm",
        Rgba8UnormSrgb => "Rgba8UnormSrgb",
        Bgra8UnormSrgb => "Bgra8UnormSrgb",
        R32Float => "R32Float",
        R16Float => "R16Float",
        R8Unorm => "R8Unorm",
        Rg16Float => "Rg16Float",
        Depth32Float => "Depth32Float",
        _ => "Other",
    }
}

// Re-exports for the public API surface of the graph module.
pub use super::execution::RenderGraph;
