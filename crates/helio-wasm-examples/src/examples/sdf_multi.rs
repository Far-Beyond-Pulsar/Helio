use helio_render_v2::features::{SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp};
use glam::Mat4;
use crate::sdf_common::SdfUpdater;

pub struct MultiUpdater;

impl SdfUpdater for MultiUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(2.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(0.5),
            blend_radius: 0.0,
        });
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(-2.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(0.5),
            blend_radius: 0.0,
        });
    }

    fn update(&mut self, sdf: &mut SdfFeature, time: f32) {
        let r   = 2.0_f32;
        let ang = time;
        let p0  = glam::Vec3::new(r * ang.cos(), 1.0, r * ang.sin());
        let p1  = glam::Vec3::new(-r * ang.cos(), 1.0, -r * ang.sin());
        sdf.set_edit(0, SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(p0),
            params: SdfShapeParams::sphere(0.5),
            blend_radius: 0.0,
        });
        let radius = 0.5 + (time * 3.0).sin() * 0.2;
        sdf.set_edit(1, SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(p1),
            params: SdfShapeParams::sphere(radius),
            blend_radius: 0.0,
        });
    }
}
