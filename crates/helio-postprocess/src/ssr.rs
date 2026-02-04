pub struct SSR {
    pub enabled: bool,
    pub max_steps: u32,
    pub binary_search_steps: u32,
    pub max_distance: f32,
    pub stride: f32,
    pub thickness: f32,
    pub jitter: f32,
    pub fade_start: f32,
    pub fade_end: f32,
}

impl Default for SSR {
    fn default() -> Self {
        Self {
            enabled: true,
            max_steps: 64,
            binary_search_steps: 8,
            max_distance: 100.0,
            stride: 1.0,
            thickness: 0.5,
            jitter: 0.5,
            fade_start: 0.7,
            fade_end: 1.0,
        }
    }
}
