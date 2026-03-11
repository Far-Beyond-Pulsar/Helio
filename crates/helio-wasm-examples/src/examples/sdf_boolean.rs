use helio_render_v2::features::{SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp};
use glam::Mat4;
use crate::sdf_common::SdfUpdater;

pub struct BooleanUpdater;

impl SdfUpdater for BooleanUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Cube,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.5, 1.0, 0.0)),
            params: SdfShapeParams::cube(1.0, 1.0, 1.0),
            blend_radius: 0.2,
        });
    }

    fn update(&mut self, sdf: &mut SdfFeature, time: f32) {
        let op = if time.sin() > 0.0 { BooleanOp::Union } else { BooleanOp::Subtraction };
        sdf.set_edit(1, SdfEdit {
            shape: SdfShapeType::Cube,
            op,
            transform: Mat4::from_translation(glam::Vec3::new(0.5, 1.0, 0.0)),
            params: SdfShapeParams::cube(1.0, 1.0, 1.0),
            blend_radius: 0.2,
        });
    }
}
