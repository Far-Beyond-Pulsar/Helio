pub struct FrameGraph {
    passes: Vec<String>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
        }
    }

    pub fn add_pass(&mut self, name: impl Into<String>) {
        self.passes.push(name.into());
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}
