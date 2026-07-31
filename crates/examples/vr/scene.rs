//! Scene setup for the VR demo: three cubes, a ground plane, three point
//! lights and a sky, all authored around the world origin (OpenXR's stage
//! origin maps 1:1 onto it at head height, so content should sit within arm's
//! reach of `(0, 1.6, 0)`).

use glam::{Mat4, Vec3};
use helio::Renderer;

use crate::v3_demo_common::{cube_mesh, insert_object, make_material, plane_mesh, point_light};

pub fn build(renderer: &mut Renderer) {
    let mat = renderer
        .scene_mut()
        .insert_material(make_material([0.7, 0.7, 0.72, 1.0], 0.7, 0.0, [0.0, 0.0, 0.0], 0.0));

    let cube1 = renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.5)))
        .as_mesh()
        .unwrap();
    let cube2 = renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.4)))
        .as_mesh()
        .unwrap();
    let cube3 = renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.3)))
        .as_mesh()
        .unwrap();
    let ground = renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 4.0)))
        .as_mesh()
        .unwrap();

    // Ground is at y=0; cubes float just above it so they read clearly in VR.
    let _ = insert_object(renderer, cube1, mat, Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)), 0.5);
    let _ = insert_object(renderer, cube2, mat, Mat4::from_translation(Vec3::new(-1.2, 0.4, -0.8)), 0.4);
    let _ = insert_object(renderer, cube3, mat, Mat4::from_translation(Vec3::new(1.2, 0.3, 0.6)), 0.3);
    let _ = insert_object(renderer, ground, mat, Mat4::IDENTITY, 4.0);

    renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::light(point_light([0.0, 2.2, 0.0], [1.0, 0.55, 0.15], 6.0, 5.0)));
    renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::light(point_light([-3.5, 2.0, -1.5], [0.25, 0.5, 1.0], 5.0, 6.0)));
    renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::light(point_light([3.5, 1.5, 1.5], [1.0, 0.3, 0.5], 5.0, 6.0)));

    renderer.scene_mut().insert_actor(helio::SceneActor::sky(
        helio::SkyActor::new().with_sky_color([0.15, 0.25, 0.45]),
    ));
}
