pub struct MotionBlur {
    pub enabled: bool,
    pub quality: MotionBlurQuality,
    pub shutter_angle: f32,
    pub sample_count: u32,
    pub max_velocity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionBlurQuality {
    Low,
    Medium,
    High,
    Ultra,
}

impl Default for MotionBlur {
    fn default() -> Self {
        Self {
            enabled: false,
            quality: MotionBlurQuality::High,
            shutter_angle: 180.0,
            sample_count: 8,
            max_velocity: 64.0,
        }
    }
}
