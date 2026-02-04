// ui system implementation
pub struct uiSystem {
    enabled: bool,
}

impl uiSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for uiSystem {
    fn default() -> Self {
        Self::new()
    }
}
