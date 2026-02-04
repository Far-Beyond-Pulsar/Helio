// terrain system implementation
pub struct terrainSystem {
    enabled: bool,
}

impl terrainSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for terrainSystem {
    fn default() -> Self {
        Self::new()
    }
}
