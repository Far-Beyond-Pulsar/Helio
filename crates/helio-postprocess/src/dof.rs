pub struct DepthOfField {
    pub enabled: bool,
    pub method: DOFMethod,
    pub focus_distance: f32,
    pub aperture: f32,
    pub focal_length: f32,
    pub bokeh_shape: BokehShape,
    pub bokeh_rotation: f32,
    pub max_blur_size: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DOFMethod {
    Gaussian,
    BokehDOF,
    CircularDOF,
    DiaphragmDOF,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BokehShape {
    Circle,
    Hexagon,
    Octagon,
}

impl Default for DepthOfField {
    fn default() -> Self {
        Self {
            enabled: false,
            method: DOFMethod::BokehDOF,
            focus_distance: 10.0,
            aperture: 2.8,
            focal_length: 50.0,
            bokeh_shape: BokehShape::Hexagon,
            bokeh_rotation: 0.0,
            max_blur_size: 20.0,
        }
    }
}
