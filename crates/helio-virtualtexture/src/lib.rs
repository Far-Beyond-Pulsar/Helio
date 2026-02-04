// virtualtexture system implementation
pub struct virtualtextureSystem {
    enabled: bool,
}

impl virtualtextureSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for virtualtextureSystem {
    fn default() -> Self {
        Self::new()
    }
}
