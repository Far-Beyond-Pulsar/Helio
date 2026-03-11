//! Basic render_v2 demo — cubes, point lights, radiance cascades, bloom.
//! Implements WasmScene directly.

use helio_render_v2::Renderer;
use helio_render_v2::features::{
    FeatureRegistryBuilder,
    LightingFeature, BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance,
    RadianceCascadesFeature,
};

use crate::harness::{WasmScene, load_sprite};

const RC_WORLD_MIN: [f32; 3] = [-3.5, -0.3, -3.5];
const RC_WORLD_MAX: [f32; 3] = [3.5, 5.0, 3.5];

pub struct RenderV2Basic {
    // light p0 bobs, keep its id so we can update it each frame
    light_p0_id: Option<helio_render_v2::LightId>,
}

impl RenderV2Basic {
    pub fn new() -> Self { Self { light_p0_id: None } }
}

impl WasmScene for RenderV2Basic {
    fn configure_features(&mut self, builder: FeatureRegistryBuilder) -> FeatureRegistryBuilder {
        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
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
        self.light_p0_id = Some(
            renderer.add_light(helio_render_v2::SceneLight::point(p0, [1.0, 0.55, 0.15], 6.0, 5.0))
        );
        renderer.add_light(helio_render_v2::SceneLight::point(p1, [0.25, 0.5, 1.0], 5.0, 6.0));
        renderer.add_light(helio_render_v2::SceneLight::point(p2, [1.0, 0.3, 0.5], 5.0, 6.0));

        renderer.add_billboard(BillboardInstance::new(p0, [0.35, 0.35]).with_color([1.0, 0.55, 0.15, 1.0]));
        renderer.add_billboard(BillboardInstance::new(p1, [0.35, 0.35]).with_color([0.25, 0.5, 1.0, 1.0]));
        renderer.add_billboard(BillboardInstance::new(p2, [0.35, 0.35]).with_color([1.0, 0.3, 0.5, 1.0]));
    }

    fn update_scene(&mut self, renderer: &mut Renderer, time: f32) {
        // animate p0 bobbing
        if let Some(id) = self.light_p0_id {
            let y = 2.2 + time.sin() * 0.8;
            renderer.update_light(id, helio_render_v2::SceneLight::point(
                [0.0, y, 0.0], [1.0, 0.55, 0.15], 6.0, 5.0,
            ));
        }
    }
}
