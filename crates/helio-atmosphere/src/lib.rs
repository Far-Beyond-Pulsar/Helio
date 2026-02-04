// atmosphere system implementation
pub struct atmosphereSystem {
    enabled: bool,
}

impl atmosphereSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for atmosphereSystem {
    fn default() -> Self {
        Self::new()
    }
}
