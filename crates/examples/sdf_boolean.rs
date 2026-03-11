//! Demo 4: boolean operation toggle
//!
//! Two primitives are present; the second switchs between union and

mod demo_portal;

mod sdf_demos_common;
use sdf_demos_common::{run_demo, SdfUpdater};
use helio_render_v2::features::{
    SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp,
};
use glam::Mat4;

struct BooleanUpdater;
impl Default for BooleanUpdater { fn default() -> Self { Self } }
impl SdfUpdater for BooleanUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        // first object: stationary sphere at origin
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });
        // second object: box offset slightly
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Box,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.5, 1.0, 0.0)),
            params: SdfShapeParams::box([1.0, 1.0, 1.0]),
            blend_radius: 0.2,
        });
    }
    fn update(&mut self, sdf: &mut SdfFeature, time: f32) {
        let op = if time.sin() > 0.0 { BooleanOp::Union } else { BooleanOp::Difference };
        // update second edit only
        let transform = Mat4::from_translation(glam::Vec3::new(0.5, 1.0, 0.0));
        sdf.set_edit(1, SdfEdit {
            shape: SdfShapeType::Box,
            op,
            transform,
            params: SdfShapeParams::box([1.0, 1.0, 1.0]),
            blend_radius: 0.2,
        });
    }
}

fn main() {
    run_demo("SDF boolean toggle", BooleanUpdater::default());
}