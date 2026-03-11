//! Sky atmosphere demo — dynamic sun, volumetric clouds, skylight.
//! Implements WasmScene directly.

use helio_render_v2::{Renderer, SceneLight, SkyAtmosphere, VolumetricClouds, Skylight};
use helio_render_v2::features::{
    FeatureRegistryBuilder,
    LightingFeature, BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance,
    RadianceCascadesFeature,
};

use crate::harness::{WasmScene, load_sprite};

const RC_WORLD_MIN: [f32; 3] = [-10.0, -0.3, -10.0];
const RC_WORLD_MAX: [f32; 3] = [10.0, 8.0, 10.0];

pub struct RenderV2Sky {
    sun_light_id: Option<helio_render_v2::LightId>,
}

impl RenderV2Sky {
    pub fn new() -> Self { Self { sun_light_id: None } }
}

impl WasmScene for RenderV2Sky {
    fn configure_features(&mut self, builder: FeatureRegistryBuilder) -> FeatureRegistryBuilder {
        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
        builder
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.3).with_threshold(1.2))
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
        let ground = renderer.create_mesh_plane([0.0, 0.0, 0.0], 20.0);
        let roof   = renderer.create_mesh_rect3d([0.0, 2.85, 0.0], [4.5, 0.15, 4.5]);
        renderer.add_object(&cube1,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&cube2,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&cube3,  None, glam::Mat4::IDENTITY);
        renderer.add_object(&ground, None, glam::Mat4::IDENTITY);
        renderer.add_object(&roof,   None, glam::Mat4::IDENTITY);

        // initial sun angle = 1.0 (afternoon-ish)
        let sun_angle = 1.0_f32;
        let sun_dir   = glam::Vec3::new(sun_angle.cos() * 0.3, sun_angle.sin(), 0.5).normalize();
        let light_dir = [-sun_dir.x, -sun_dir.y, -sun_dir.z];
        let elev      = sun_dir.y.clamp(-1.0, 1.0);
        let lux       = (elev * 3.0).clamp(0.0, 1.0);
        self.sun_light_id = Some(
            renderer.add_light(SceneLight::directional(light_dir, [1.0, 0.85, 0.7], (lux * 0.35).max(0.01)))
        );
        renderer.add_light(SceneLight::point([ 0.0, 2.5,  0.0], [1.0, 0.85, 0.6], 4.0, 8.0));
        renderer.add_light(SceneLight::point([-2.5, 2.0, -1.5], [0.4, 0.6,  1.0], 3.5, 7.0));
        renderer.add_light(SceneLight::point([ 2.5, 1.8,  1.5], [1.0, 0.3,  0.3], 3.0, 6.0));

        renderer.set_sky_atmosphere(Some(
            SkyAtmosphere::new()
                .with_sun_intensity(22.0)
                .with_exposure(4.0)
                .with_mie_g(0.76)
                .with_clouds(
                    VolumetricClouds::new()
                        .with_coverage(0.30)
                        .with_density(0.7)
                        .with_layer(800.0, 1800.0)
                        .with_wind([1.0, 0.0], 0.08),
                ),
        ));
        renderer.set_skylight(Some(Skylight::new().with_intensity(0.08).with_tint([1.0, 1.0, 1.0])));

        renderer.add_billboard(BillboardInstance::new([ 0.0, 2.5,  0.0], [0.35, 0.35]).with_color([1.0, 0.85, 0.6, 1.0]));
        renderer.add_billboard(BillboardInstance::new([-2.5, 2.0, -1.5], [0.35, 0.35]).with_color([0.4, 0.6, 1.0, 1.0]));
        renderer.add_billboard(BillboardInstance::new([ 2.5, 1.8,  1.5], [0.35, 0.35]).with_color([1.0, 0.3, 0.3, 1.0]));
    }

    fn update_scene(&mut self, renderer: &mut Renderer, time: f32) {
        if let Some(id) = self.sun_light_id {
            // sun slowly orbits over time
            let sun_angle = 1.0 + time * 0.05;
            let sun_dir   = glam::Vec3::new(sun_angle.cos() * 0.3, sun_angle.sin(), 0.5).normalize();
            let light_dir = [-sun_dir.x, -sun_dir.y, -sun_dir.z];
            let elev      = sun_dir.y.clamp(-1.0, 1.0);
            let lux       = (elev * 3.0).clamp(0.0, 1.0);
            let color = [
                1.0_f32.min(1.0 + (1.0 - elev) * 0.3),
                (0.85 + elev * 0.15).clamp(0.0, 1.0),
                (0.7 + elev * 0.3).clamp(0.0, 1.0),
            ];
            renderer.update_light(id, SceneLight::directional(light_dir, color, (lux * 0.35).max(0.01)));
        }
    }
}
