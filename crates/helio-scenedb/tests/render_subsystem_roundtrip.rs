//! End-to-end round trip through [`HelioRenderSubsystem`]: spawn a
//! `StaticMeshComponent`-bearing entity as plain `World` writes, run
//! `scene_db.step()` + `subsystem.apply_to(&mut scene)` (the documented
//! per-frame call sequence -- see `subsystem.rs`'s module doc), and verify
//! Helio's `Scene` actually gained the object -- then move it (proving the
//! O(1) `update_object_transform` path, not a remove+reinsert), then
//! despawn it and verify it's gone. Also covers the `user_tag` `+1`
//! packing offset for the very first entity a fresh `World` ever spawns
//! (`Entity::new(0, 0).bits() == 0`, and Helio's `user_tag == 0` means
//! "untagged" -- an unoffset tag would make this exact entity's object
//! unreachable via `Scene::object_by_tag`).

#[path = "support/mod.rs"]
mod support;

use std::sync::Arc;

use glam::{Mat4, Vec3};
use helio::{GroupMask, Scene};
use helio_scenedb::{HelioRenderSubsystem, RenderBounds, RenderFlags, RenderTransform, StaticMeshComponent};
use libhelio::material::GpuMaterial;
use libhelio::GpuLight;
use pulsar_scenedb::gpu::{GpuMirrorHandle, SceneGpuStore};
use pulsar_scenedb::{SceneDb, SharedChangeTracker};

fn triangle_mesh() -> helio::MeshUpload {
    helio::MeshUpload {
        vertices: vec![Default::default(); 3],
        indices: vec![0, 1, 2],
    }
}

/// Common setup: a real `Scene`, a `SceneDb` with an attached GPU mirror
/// (for `StaticMeshComponent`'s inline material) and an attached change
/// tracker (for `HelioRenderSubsystem`'s `Delta` feed), and one mesh
/// registered. Mesh registration itself is NOT part of this integration --
/// see the crate-level integration plan's "stays with Helio" list.
fn setup() -> (SceneDb, Scene, helio::MeshId) {
    let ctx = support::test_context();
    let device = Arc::clone(ctx.device());
    let queue = Arc::clone(ctx.queue());

    let store = Arc::new(SceneGpuStore::new(&ctx, support::scene_cfg()));
    let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(&queue));
    let mut scene_db = SceneDb::new_with_gpu_mirror(mirror);
    scene_db.world.attach_change_tracker(SharedChangeTracker::new());
    scene_db.register_subsystem(HelioRenderSubsystem::new());

    let mut scene = Scene::new(device, queue);
    let mesh_id = scene.insert_mesh(triangle_mesh());

    (scene_db, scene, mesh_id)
}

fn run_frame(scene_db: &mut SceneDb, scene: &mut Scene) {
    scene_db.step();
    scene_db.subsystem_mut::<HelioRenderSubsystem>().unwrap().apply_to(scene);
    scene.flush();
}

#[test]
fn spawn_move_despawn_round_trips_through_helio_scene() {
    let (mut scene_db, mut scene, mesh_id) = setup();

    let object = scene_db.world.spawn_bundle((
        StaticMeshComponent::new(
            mesh_id,
            GpuMaterial { base_color: [1.0, 0.0, 0.0, 1.0], ..bytemuck::Zeroable::zeroed() },
        ),
        RenderTransform(Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))),
        RenderBounds([1.0, 2.0, 3.0, 1.0]),
        RenderFlags { flags: 0, movability: Some(libhelio::Movability::Movable), groups: GroupMask::NONE },
    ));

    run_frame(&mut scene_db, &mut scene);

    let tag = HelioRenderSubsystem::tag_for(object);
    let obj_id = scene.object_by_tag(tag).expect("object must exist in Scene after the first frame");
    assert_eq!(
        scene.get_object_transform(obj_id).unwrap(),
        Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
    );

    // -- Move it: this must take the O(1) update_object_transform path,
    // not a remove+reinsert -- same ObjectId before and after.
    scene_db.world.get_mut::<RenderTransform>(object).unwrap().0 =
        Mat4::from_translation(Vec3::new(9.0, 9.0, 9.0));
    run_frame(&mut scene_db, &mut scene);

    let obj_id_after_move = scene.object_by_tag(tag).expect("object must still exist");
    assert_eq!(obj_id_after_move, obj_id, "moving must not change the ObjectId (no rebuild for a transform-only change)");
    assert_eq!(
        scene.get_object_transform(obj_id_after_move).unwrap(),
        Mat4::from_translation(Vec3::new(9.0, 9.0, 9.0)),
    );

    // -- Despawn: the object must disappear from Scene too.
    scene_db.world.despawn(object);
    run_frame(&mut scene_db, &mut scene);

    assert!(
        scene.object_by_tag(tag).is_none(),
        "despawning the SceneDB entity must remove the Helio object"
    );
}

#[test]
fn the_very_first_entity_spawned_in_a_fresh_world_still_resolves_via_user_tag() {
    // Entity::new(0, 0).bits() == 0, and Helio documents user_tag == 0 as
    // its "untagged" sentinel -- this is exactly the entity that would
    // break without HelioRenderSubsystem::tag_for's `+1` offset.
    let (mut scene_db, mut scene, mesh_id) = setup();

    let object = scene_db.world.spawn_bundle((
        StaticMeshComponent::new(mesh_id, bytemuck::Zeroable::zeroed()),
        RenderTransform(Mat4::IDENTITY),
        RenderBounds([0.0, 0.0, 0.0, 1.0]),
        RenderFlags::default(),
    ));
    assert_eq!(object.bits(), 0, "test assumption: this must be the zero entity");

    run_frame(&mut scene_db, &mut scene);

    assert!(
        scene.object_by_tag(HelioRenderSubsystem::tag_for(object)).is_some(),
        "the zero-index entity's object must still be reachable via object_by_tag"
    );
}

#[test]
fn light_spawn_update_despawn_round_trips_through_helio_scene() {
    let (mut scene_db, mut scene, _mesh_id) = setup();

    let light = scene_db.world.spawn();
    scene_db.world.insert(
        light,
        helio_scenedb::LightComponent::new(GpuLight {
            color_intensity: [1.0, 1.0, 1.0, 500.0],
            ..bytemuck::Zeroable::zeroed()
        }),
    );

    run_frame(&mut scene_db, &mut scene);

    let tag = HelioRenderSubsystem::tag_for(light);
    let light_id = scene.light_by_tag(tag).expect("light must exist in Scene after the first frame");
    assert_eq!(scene.get_light(light_id).unwrap().color_intensity[3], 500.0);

    // -- Update: same LightId before and after (update_light, not remove+reinsert).
    scene_db.world.get_mut::<helio_scenedb::LightComponent>(light).unwrap().light.color_intensity[3] = 900.0;
    run_frame(&mut scene_db, &mut scene);

    let light_id_after = scene.light_by_tag(tag).expect("light must still exist");
    assert_eq!(light_id_after, light_id, "updating a light must not change its LightId");
    assert_eq!(scene.get_light(light_id_after).unwrap().color_intensity[3], 900.0);

    // -- Despawn: the light must disappear from Scene too.
    scene_db.world.despawn(light);
    run_frame(&mut scene_db, &mut scene);

    assert!(scene.light_by_tag(tag).is_none(), "despawning the SceneDB entity must remove the Helio light");
}
