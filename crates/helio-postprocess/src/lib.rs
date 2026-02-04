// postprocess system implementation
pub struct postprocessSystem {
    enabled: bool,
}

impl postprocessSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for postprocessSystem {
    fn default() -> Self {
        Self::new()
    }
}
