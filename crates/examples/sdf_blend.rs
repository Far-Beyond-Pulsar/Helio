//! Demo 5: blending shapes
//!
//! Two overlapping primitives whose blend radius oscillates, producing a

mod demo_portal;
//! smooth transition between them.

mod sdf_demos_common;
use sdf_demos_common::{run_demo, SdfUpdater};
use helio_render_v2::features::{
    SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp,
};
use glam::Mat4;

struct BlendUpdater;
impl Default for BlendUpdater { fn default() -> Self { Self } }
impl SdfUpdater for BlendUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        // sphere at center
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(-0.5, 1.0, 0.0)),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });
        // box overlapping
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Box,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.5, 1.0, 0.0)),
            params: SdfShapeParams::box([1.0, 1.0, 1.0]),
            blend_radius: 0.0,
        });
    }
    fn update(&mut self, sdf: &mut SdfFeature, time: f32) {
        let blend = (time.sin() * 0.5 + 0.5) * 0.75; // oscillate between 0 & 0.75
        let transform = Mat4::from_translation(glam::Vec3::new(0.5, 1.0, 0.0));
        sdf.set_edit(1, SdfEdit {
            shape: SdfShapeType::Box,
            op: BooleanOp::Union,
            transform,
            params: SdfShapeParams::box([1.0, 1.0, 1.0]),
            blend_radius: blend,
        });
    }
}

fn main() {
    run_demo("SDF blend shapes", BlendUpdater::default());
}