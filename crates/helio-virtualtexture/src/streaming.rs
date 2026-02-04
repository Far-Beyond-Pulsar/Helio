pub struct VirtualTexture {
    pub page_size: u32,
    pub cache_size: u32,
    pub max_anisotropy: u32,
    pub mip_bias: f32,
}

impl Default for VirtualTexture {
    fn default() -> Self {
        Self {
            page_size: 256,
            cache_size: 2048,
            max_anisotropy: 16,
            mip_bias: 0.0,
        }
    }
}

pub struct StreamingVirtualTexture {
    pub enabled: bool,
    pub streaming_pool_size: u64,
    pub priority_threshold: f32,
}

impl Default for StreamingVirtualTexture {
    fn default() -> Self {
        Self {
            enabled: true,
            streaming_pool_size: 512 * 1024 * 1024, // 512 MB
            priority_threshold: 0.5,
        }
    }
}
