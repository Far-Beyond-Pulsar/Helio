use glam::{Vec2, Vec4};

pub struct UICanvas {
    pub size: Vec2,
    pub scale_mode: ScaleMode,
    pub elements: Vec<UIElement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    ConstantPixelSize,
    ScaleWithScreenSize,
    ConstantPhysicalSize,
}

pub struct UIElement {
    pub position: Vec2,
    pub size: Vec2,
    pub color: Vec4,
    pub visible: bool,
}

impl UICanvas {
    pub fn new(size: Vec2) -> Self {
        Self {
            size,
            scale_mode: ScaleMode::ScaleWithScreenSize,
            elements: Vec::new(),
        }
    }
}
