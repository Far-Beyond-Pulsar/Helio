// culling system implementation
pub struct cullingSystem {
    enabled: bool,
}

impl cullingSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for cullingSystem {
    fn default() -> Self {
        Self::new()
    }
}
