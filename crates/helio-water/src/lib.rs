// water system implementation
pub struct waterSystem {
    enabled: bool,
}

impl waterSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for waterSystem {
    fn default() -> Self {
        Self::new()
    }
}
