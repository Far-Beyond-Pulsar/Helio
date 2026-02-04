pub struct RenderPass {
    pub name: String,
}

impl RenderPass {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
        }
    }
}
