pub struct PathTracer {
    pub max_bounces: u32,
    pub samples_per_pixel: u32,
    pub russian_roulette_depth: u32,
    pub enable_nee: bool, // Next Event Estimation
    pub enable_mis: bool, // Multiple Importance Sampling
}

impl Default for PathTracer {
    fn default() -> Self {
        Self {
            max_bounces: 8,
            samples_per_pixel: 64,
            russian_roulette_depth: 3,
            enable_nee: true,
            enable_mis: true,
        }
    }
}

pub struct ReSTIR {
    pub enabled: bool,
    pub spatial_reuse: bool,
    pub temporal_reuse: bool,
    pub max_history_length: u32,
}

impl Default for ReSTIR {
    fn default() -> Self {
        Self {
            enabled: false,
            spatial_reuse: true,
            temporal_reuse: true,
            max_history_length: 20,
        }
    }
}
