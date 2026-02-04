// material system implementation
pub struct materialSystem {
    enabled: bool,
}

impl materialSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for materialSystem {
    fn default() -> Self {
        Self::new()
    }
}
