//! Wire contract with other passes. The lens is resolved once per frame on
//! the GPU; consumers read the result instead of re-matching rows.

/// Graph storage buffer holding one [`ResolvedLens`], written every frame.
pub const RESOLVED_KEY: &str = "lens_resolved";

/// The view's lens after row matching, autofocus and sanitising, as
/// `cs_resolve` writes it. With no enabled row for the view it holds the
/// neutral 50 mm f/2.8 defaults of [`ResolvedLens::default`] and `enabled`
/// is 0, so optics consumers (the flare iris, field falloff) always have a
/// lens to read.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResolvedLens {
    pub f_number: f32,
    pub focal_length_mm: f32,
    pub sensor_width_mm: f32,
    /// 0 (circular) or 3..=16.
    pub aperture_blades: u32,
    /// Radians.
    pub aperture_rotation: f32,
    /// 1 when an enabled row matched the view; 0 passes the image through.
    pub enabled: u32,
    pub quality: u32,
    pub t_stop: f32,
    pub sensor_height_mm: f32,
    /// After autofocus easing.
    pub focus_distance_m: f32,
    pub max_coc_px: f32,
    pub field_curvature: f32,
    pub tilt_angle: f32,
    pub tilt_axis: f32,
    pub element_first: u32,
    pub element_count: u32,
}

const _: () = assert!(std::mem::size_of::<ResolvedLens>() == 64);

impl Default for ResolvedLens {
    fn default() -> Self {
        Self {
            f_number: 2.8,
            focal_length_mm: 50.0,
            sensor_width_mm: 36.0,
            aperture_blades: 6,
            aperture_rotation: 0.0,
            enabled: 0,
            quality: 0,
            t_stop: 2.8,
            sensor_height_mm: 24.0,
            focus_distance_m: 1.0e6,
            max_coc_px: 32.0,
            field_curvature: 0.0,
            tilt_angle: 0.0,
            tilt_axis: 0.0,
            element_first: 0,
            element_count: 0,
        }
    }
}
