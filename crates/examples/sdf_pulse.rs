//! Demo 2: pulsing radius
//!
//! The same single sphere from the move demo, but its radius oscillates over


mod demo_portal;

mod sdf_demos_common;
use sdf_demos_common::{run_demo, SdfUpdater};
use helio_render_v2::features::{SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp};
use glam::Mat4;

struct PulseUpdater;
impl Default for PulseUpdater { fn default() -> Self { Self } }
impl SdfUpdater for PulseUpdater {
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
        let rad = 0.5 + (time * 2.0).sin() * 0.3;
        sdf.set_edit(0, SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(rad),
            blend_radius: 0.0,
        });
    }
}
fn main() {
    run_demo("SDF pulsing sphere", PulseUpdater::default());
}