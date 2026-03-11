use helio_render_v2::features::{SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp};
use glam::Mat4;
use crate::sdf_common::SdfUpdater;

pub struct MorphUpdater;

impl SdfUpdater for MorphUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });
    }

    fn update(&mut self, sdf: &mut SdfFeature, time: f32) {
        if time.sin() > 0.0 {
            sdf.set_edit(0, SdfEdit {
                shape: SdfShapeType::Sphere,
                op: BooleanOp::Union,
                transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
                params: SdfShapeParams::sphere(1.0),
                blend_radius: 0.0,
            });
        } else {
            sdf.set_edit(0, SdfEdit {
                shape: SdfShapeType::Cube,
                op: BooleanOp::Union,
                transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
                params: SdfShapeParams::cube(1.0, 1.0, 1.0),
                blend_radius: 0.0,
            });
        }
    }
}
