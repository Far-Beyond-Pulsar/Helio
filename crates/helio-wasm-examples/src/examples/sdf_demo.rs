//! Standalone SDF demo — terrain + orbiting sphere.
//! Implements WasmScene directly (not via SdfUpdater).

use helio_render_v2::Renderer;
use helio_render_v2::features::{
    FeatureRegistryBuilder,
    LightingFeature, BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance,
    RadianceCascadesFeature,
    SdfFeature, SdfMode, SdfEdit, SdfShapeType, SdfShapeParams, BooleanOp,
    TerrainConfig,
};

use crate::harness::{WasmScene, load_sprite};

const RC_WORLD_MIN: [f32; 3] = [-3.5, -0.3, -3.5];
const RC_WORLD_MAX: [f32; 3] = [3.5, 5.0, 3.5];

pub struct SdfDemo;

impl WasmScene for SdfDemo {
    fn configure_features(&mut self, builder: FeatureRegistryBuilder) -> FeatureRegistryBuilder {
        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();

        let mut sdf = SdfFeature::new()
            .with_mode(SdfMode::ClipMap)
            .with_grid_dim(128)
            .with_volume_bounds([-3.0, -1.0, -3.0], [3.0, 3.0, 3.0])
            .with_terrain(TerrainConfig::rolling());
        sdf.add_edit(SdfEdit {
            shape: SdfShapeType::Sphere,
            op: BooleanOp::Union,
            transform: glam::Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            params: SdfShapeParams::sphere(1.0),
            blend_radius: 0.0,
        });

        builder
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.4).with_threshold(1.2))
            .with_feature(ShadowsFeature::new().with_atlas_size(1024).with_max_lights(4))
            .with_feature(
                BillboardsFeature::new()
                    .with_sprite(sprite_rgba, sprite_w, sprite_h)
                    .with_max_instances(5000),
            )
            .with_feature(
                RadianceCascadesFeature::new().with_world_bounds(RC_WORLD_MIN, RC_WORLD_MAX),
            )
            .with_feature(sdf)
    }

    fn setup_scene(&mut self, renderer: &mut Renderer) {
        let cube1  = renderer.create_mesh_cube([ 0.0, 0.5,  0.0], 0.5);
        let cube2  = renderer.create_mesh_cube([-2.0, 0.4, -1.0], 0.4);
        let cube3  = renderer.create_mesh_cube([ 2.0, 0.3,  0.5], 0.3);
        let ground = renderer.create_mesh_plane([0.0, 0.0, 0.0], 5.0);
        renderer.add_object(&cube1,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&cube2,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&cube3,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&ground, None, glam::Mat4::IDENTITY);

        let p0 = [0.0f32, 2.2, 0.0];
        let p1 = [-3.5f32, 2.0, -1.5];
        let p2 = [3.5f32, 1.5, 1.5];
        renderer.add_light(helio_render_v2::SceneLight::point(p0, [1.0, 0.55, 0.15], 6.0, 5.0));
        renderer.add_light(helio_render_v2::SceneLight::point(p1, [0.25, 0.5, 1.0], 5.0, 6.0));
        renderer.add_light(helio_render_v2::SceneLight::point(p2, [1.0, 0.3, 0.5], 5.0, 6.0));

        renderer.add_billboard(BillboardInstance::new(p0, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0]));
        renderer.add_billboard(BillboardInstance::new(p1, [0.35, 0.35]).with_color([0.25, 0.5, 1.0, 1.0]));
        renderer.add_billboard(BillboardInstance::new(p2, [0.35, 0.35]).with_color([1.0, 0.3, 0.5, 1.0]));
    }

    fn update_scene(&mut self, renderer: &mut Renderer, time: f32) {
        if let Some(sdf) = renderer.get_feature_mut::<SdfFeature>("sdf") {
            let mut cfg = TerrainConfig::rolling();
            cfg.amplitude = 4.0 + time.sin() * 2.0;
            sdf.set_terrain(Some(cfg));

            sdf.set_edit(0, SdfEdit {
                shape: SdfShapeType::Sphere,
                op: BooleanOp::Union,
                transform: glam::Mat4::from_translation(glam::Vec3::new(
                    time.cos() * 2.0, 1.0, time.sin() * 2.0,
                )),
                params: SdfShapeParams::sphere(1.0),
                blend_radius: 0.0,
            });
        }
    }
}
