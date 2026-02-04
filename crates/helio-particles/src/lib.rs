// particles system implementation
pub struct particlesSystem {
    enabled: bool,
}

impl particlesSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for particlesSystem {
    fn default() -> Self {
        Self::new()
    }
}
