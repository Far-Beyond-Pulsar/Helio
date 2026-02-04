use glam::{Vec2, Vec4};

pub struct TextRenderer {
    pub font: Option<u32>,
    pub font_size: f32,
    pub color: Vec4,
}

pub struct TextElement {
    pub text: String,
    pub position: Vec2,
    pub font_size: f32,
    pub color: Vec4,
}

impl TextRenderer {
    pub fn new() -> Self {
        Self {
            font: None,
            font_size: 16.0,
            color: Vec4::ONE,
        }
    }
}

impl Default for TextRenderer {
    fn default() -> Self {
        Self::new()
    }
}
