//! Demo 8: animated terrain
//!
//! Demonstrates the `with_terrain` API.  The terrain amplitude pulses and a

mod demo_portal;

mod sdf_demos_common;
use sdf_demos_common::{run_demo, SdfUpdater};
use helio_render_v2::features::{
    SdfFeature, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp,
    TerrainConfig,
};
use glam::Mat4;

struct TerrainUpdater;
impl Default for TerrainUpdater { fn default() -> Self { Self } }
impl SdfUpdater for TerrainUpdater {
    fn init(&mut self, sdf: &mut SdfFeature) {
        // the feature is already created; set the terrain config instead of
        // consuming the feature with `with_terrain`.
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
        // animate terrain amplitude
        let mut cfg = TerrainConfig::rolling();
        cfg.amplitude = 4.0 + time.sin() * 2.0;
        sdf.set_terrain(Some(cfg));

        // move sphere in a circle
        let pos = [time.cos() * 2.0, 1.0, time.sin() * 2.0];
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
    run_demo("SDF animated terrain", TerrainUpdater::default());
}