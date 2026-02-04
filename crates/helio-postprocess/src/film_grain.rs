pub struct FilmGrain {
    pub enabled: bool,
    pub intensity: f32,
    pub response: f32,
    pub size: f32,
}

impl Default for FilmGrain {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.2,
            response: 0.8,
            size: 1.0,
        }
    }
}
