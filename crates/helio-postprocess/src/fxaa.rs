pub struct FXAA {
    pub enabled: bool,
    pub quality_preset: FXAAQuality,
    pub subpixel_quality: f32,
    pub edge_threshold: f32,
    pub edge_threshold_min: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FXAAQuality {
    Low,
    Medium,
    High,
    Ultra,
}

impl Default for FXAA {
    fn default() -> Self {
        Self {
            enabled: false,
            quality_preset: FXAAQuality::High,
            subpixel_quality: 0.75,
            edge_threshold: 0.166,
            edge_threshold_min: 0.0833,
        }
    }
}
