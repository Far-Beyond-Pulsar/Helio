use helio_render_v2::features::{SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp, TerrainConfig};
use glam::Mat4;
use crate::sdf_common::SdfUpdater;

pub struct TerrainUpdater;

impl SdfUpdater for TerrainUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        sdf.set_terrain(Some(TerrainConfig::rolling()));
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });
    }

    fn update(&mut self, sdf: &mut SdfFeature, time: f32) {
        let mut cfg = TerrainConfig::rolling();
        cfg.amplitude = 4.0 + time.sin() * 2.0;
        sdf.set_terrain(Some(cfg));

        sdf.set_edit(0, SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: Mat4::from_translation(glam::Vec3::new(
                time.cos() * 2.0, 1.0, time.sin() * 2.0,
            )),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });
    }
}
