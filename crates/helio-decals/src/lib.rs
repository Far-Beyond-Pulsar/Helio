// decals system implementation
pub struct decalsSystem {
    enabled: bool,
}

impl decalsSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for decalsSystem {
    fn default() -> Self {
        Self::new()
    }
}
