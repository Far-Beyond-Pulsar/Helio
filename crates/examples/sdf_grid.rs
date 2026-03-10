//! Demo 6: grid of spheres waving
//!
//! Creates a 5x5 grid of spheres.  Each sphere bobs up and down in a

mod demo_portal;

mod sdf_demos_common;
use sdf_demos_common::{run_demo, SdfUpdater};
use helio_render_v2::features::{
    SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp,
};
use glam::Mat4;

const GRID: i32 = 5;

struct GridUpdater;
impl Default for GridUpdater { fn default() -> Self { Self } }
impl SdfUpdater for GridUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        for x in 0..GRID {
            for z in 0..GRID {
                let px = (x as f32 - (GRID as f32 - 1.0) / 2.0) * 1.5;
                let pz = (z as f32 - (GRID as f32 - 1.0) / 2.0) * 1.5;
                sdf.add_edit(SdfEdit {
                    shape: SdfShapeType::Sphere,
                    op: BooleanOp::Union,
                    transform: Mat4::from_translation(glam::Vec3::new(px, 1.0, pz)),
                    params: SdfShapeParams::sphere(0.5),
                    blend_radius: 0.0,
                });
            }
        }
    }
    fn update(&mut self, sdf: &mut SdfFeature, time: f32) {
        let mut idx = 0;
        for x in 0..GRID {
            for z in 0..GRID {
                let px = (x as f32 - (GRID as f32 - 1.0) / 2.0) * 1.5;
                let pz = (z as f32 - (GRID as f32 - 1.0) / 2.0) * 1.5;
                let y = 1.0 + ((time + px).sin()) * 0.5;
                sdf.set_edit(idx as usize, SdfEdit {
                    shape: SdfShapeType::Sphere,
                    op: BooleanOp::Union,
                    transform: Mat4::from_translation(glam::Vec3::new(px, y, pz)),
                    params: SdfShapeParams::sphere(0.5),
                    blend_radius: 0.0,
                });
                idx += 1;
            }
        }
    }
}

fn main() {
    run_demo("SDF grid wave", GridUpdater::default());
}