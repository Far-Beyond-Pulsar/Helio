pub struct ChromaticAberration {
    pub enabled: bool,
    pub intensity: f32,
    pub samples: u32,
}

impl Default for ChromaticAberration {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.5,
            samples: 3,
        }
    }
}
