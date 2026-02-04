pub struct SMAA {
    pub enabled: bool,
    pub quality_preset: SMAAQuality,
    pub edge_detection: EdgeDetection,
    pub threshold: f32,
    pub max_search_steps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SMAAQuality {
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDetection {
    Luma,
    Color,
    Depth,
}

impl Default for SMAA {
    fn default() -> Self {
        Self {
            enabled: false,
            quality_preset: SMAAQuality::High,
            edge_detection: EdgeDetection::Luma,
            threshold: 0.1,
            max_search_steps: 16,
        }
    }
}
