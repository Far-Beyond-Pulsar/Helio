pub struct SSAO {
    pub enabled: bool,
    pub radius: f32,
    pub intensity: f32,
    pub bias: f32,
    pub samples: u32,
    pub sample_count: u32,
    pub blur_passes: u32,
}

impl Default for SSAO {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 0.5,
            intensity: 1.0,
            bias: 0.025,
            samples: 16,
            sample_count: 16,
            blur_passes: 2,
        }
    }
}

pub struct HBAO {
    pub enabled: bool,
    pub radius: f32,
    pub intensity: f32,
    pub angle_bias: f32,
    pub directions: u32,
    pub steps: u32,
}

impl Default for HBAO {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 0.8,
            intensity: 1.5,
            angle_bias: 0.1,
            directions: 8,
            steps: 4,
        }
    }
}

pub struct GTAO {
    pub enabled: bool,
    pub radius: f32,
    pub intensity: f32,
    pub thickness: f32,
    pub falloff_range: f32,
    pub slice_count: u32,
    pub steps_per_slice: u32,
}

impl Default for GTAO {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 1.0,
            intensity: 1.0,
            thickness: 0.5,
            falloff_range: 0.615,
            slice_count: 4,
            steps_per_slice: 4,
        }
    }
}
