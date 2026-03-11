const GRID: i32 = 5;

use helio_render_v2::features::{SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp};
use glam::Mat4;
use crate::sdf_common::SdfUpdater;

pub struct GridUpdater;

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
        let mut idx = 0usize;
        for x in 0..GRID {
            for z in 0..GRID {
                let px = (x as f32 - (GRID as f32 - 1.0) / 2.0) * 1.5;
                let pz = (z as f32 - (GRID as f32 - 1.0) / 2.0) * 1.5;
                let y  = 1.0 + (time + px).sin() * 0.5;
                sdf.set_edit(idx, SdfEdit {
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
