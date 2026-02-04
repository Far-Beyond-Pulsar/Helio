pub struct BlendNode {
    pub weight: f32,
    pub animation_index: usize,
}

pub struct BlendTree {
    pub nodes: Vec<BlendNode>,
}

impl BlendTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }
}

impl Default for BlendTree {
    fn default() -> Self {
        Self::new()
    }
}
