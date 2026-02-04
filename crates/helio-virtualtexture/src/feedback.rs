pub struct FeedbackBuffer {
    pub resolution: u32,
    pub mip_bias: f32,
}

impl Default for FeedbackBuffer {
    fn default() -> Self {
        Self {
            resolution: 1024,
            mip_bias: 0.0,
        }
    }
}
