//! Demo 3: morphing shape
//!
//! The edit alternates between sphere and box every half‑cycle, showing a

mod demo_portal;
//! basic morphological transition by replacing the edit each frame.

mod sdf_demos_common;
use sdf_demos_common::{run_demo, SdfUpdater};
use helio_render_v2::features::{
    SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp,
};
use glam::Mat4;

struct MorphUpdater;
impl Default for MorphUpdater { fn default() -> Self { Self } }
impl SdfUpdater for MorphUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        // start with sphere
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });
    }
    fn update(&mut self, sdf: &mut SdfFeature, time: f32) {
        let t = time.sin();
        if t > 0.0 {
            sdf.set_edit(0, SdfEdit {
                shape: SdfShapeType::Sphere,
                op: BooleanOp::Union,
                transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
                params: SdfShapeParams::sphere(1.0),
                blend_radius: 0.0,
            });
        } else {
            sdf.set_edit(0, SdfEdit {
                shape: SdfShapeType::Box,
                op: BooleanOp::Union,
                transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
                params: SdfShapeParams::box([1.0, 1.0, 1.0]),
                blend_radius: 0.0,
            });
        }
    }
}

fn main() {
    run_demo("SDF morphing shape", MorphUpdater::default());
}