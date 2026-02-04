// animation system implementation
pub struct animationSystem {
    enabled: bool,
}

impl animationSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for animationSystem {
    fn default() -> Self {
        Self::new()
    }
}
