//! Demo 1: moving sphere (wrapper)
//!
//! This binary uses `sdf_demos_common` and supplies an updater that drifts

mod demo_portal;
// a single sphere back and forth along the X axis.

mod sdf_demos_common;
use sdf_demos_common::{run_demo, SdfUpdater};
use helio_render_v2::features::{SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp};
use glam::Mat4;

struct MoveUpdater;

impl Default for MoveUpdater { fn default() -> Self { Self } }

impl SdfUpdater for MoveUpdater {
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
        let pos = [time.sin() * 2.0, 1.0, 0.0];
        sdf.set_edit(0, SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(pos[0], pos[1], pos[2])),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });
    }
}

fn main() {
    run_demo("SDF moving sphere", MoveUpdater::default());
}
