use glam::{Vec2, Vec4};

pub struct Button {
    pub position: Vec2,
    pub size: Vec2,
    pub label: String,
    pub color: Vec4,
    pub hover_color: Vec4,
}

pub struct Slider {
    pub position: Vec2,
    pub size: Vec2,
    pub value: f32,
    pub min: f32,
    pub max: f32,
}

pub struct Panel {
    pub position: Vec2,
    pub size: Vec2,
    pub background_color: Vec4,
    pub children: Vec<usize>,
}
