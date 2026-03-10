//! Demo 7: multiple moving objects
//!
//! Two spheres orbit each other; the second one also pulses its radius.

mod demo_portal;

mod sdf_demos_common;
use sdf_demos_common::{run_demo, SdfUpdater};
use helio_render_v2::features::{
    SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp,
};
use glam::Mat4;

struct MultiUpdater;
impl Default for MultiUpdater { fn default() -> Self { Self } }
impl SdfUpdater for MultiUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        // two spheres initially at opposite ends
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
        let r = 2.0;
        let ang = time;
        let p0 = [r * ang.cos(), 1.0, r * ang.sin()];
        let p1 = [-r * ang.cos(), 1.0, -r * ang.sin()];
        sdf.set_edit(0, SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(p0[0], p0[1], p0[2])),
            params: SdfShapeParams::sphere(0.5),
            blend_radius: 0.0,
        });
        let radius = 0.5 + (time * 3.0).sin() * 0.2;
        sdf.set_edit(1, SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(p1[0], p1[1], p1[2])),
            params: SdfShapeParams::sphere(radius),
            blend_radius: 0.0,
        });
    }
}

fn main() {
    run_demo("SDF multiple moving objects", MultiUpdater::default());
}