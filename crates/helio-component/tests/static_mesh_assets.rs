//! Asset behavior of `StaticMeshComponent` (Pulsar-Native#1035 acceptance,
//! #1081): missing and corrupt assets, reload by reassigning the path,
//! rapid replacement, shared asset reuse, and a property-only edit that
//! neither reloads the asset nor re-uploads the geometry.
//!
//! A mesh asset is loaded synchronously inside the write that names it
//! (the boundary decoder, or the `mesh_asset` property hook), so there is
//! no in-flight load to cancel, outlive its component or complete stale.
//!
//! One test, because the project path the loader resolves against is a
//! process global. Needs a GPU adapter for the upload checks (lavapipe
//! works); skips them without one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use helio_component::components::{ObjectMovability, StaticMeshComponent};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::World;

fn primitive(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../../assets/meshes/primitives")
        .join(name)
}

fn context() -> Option<EngineGpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
    Some(EngineGpuContext::new(Arc::new(device), Arc::new(queue)))
}

fn set_mesh_asset(world: &mut World, entity: pulsar_scenedb::Entity, path: &str) {
    pulsar_world_registry::set_world_component_property(
        "StaticMeshComponent",
        world,
        entity,
        "mesh_asset",
        Box::new(helio_component::components::MeshAssetPath::new(path)),
    )
    .unwrap_or_else(|_| panic!("mesh_asset write to {path} refused"));
}

fn bytes(vertices: &[helio::PackedVertex]) -> &[u8] {
    bytemuck::cast_slice(vertices)
}

fn vertex_count(world: &World, entity: pulsar_scenedb::Entity) -> usize {
    world
        .get::<StaticMeshComponent>(entity)
        .unwrap()
        .vertices
        .len()
}

#[test]
fn mesh_assets_load_replace_and_survive_bad_input() {
    let project = tempfile::tempdir().unwrap();
    let meshes = project.path().join("meshes");
    std::fs::create_dir_all(&meshes).unwrap();
    std::fs::copy(primitive("SM_Cube.fbx"), meshes.join("cube.fbx")).unwrap();
    std::fs::copy(primitive("SM_Sphere.fbx"), meshes.join("sphere.fbx")).unwrap();
    std::fs::copy(primitive("SM_Torus.fbx"), meshes.join("torus.fbx")).unwrap();
    std::fs::write(meshes.join("corrupt.fbx"), b"not a mesh at all").unwrap();
    engine_state::EngineContext::new().set_global();
    engine_state::set_project_path(project.path().display().to_string());

    let cube = StaticMeshComponent::for_mesh_asset("meshes/cube.fbx");
    let sphere = StaticMeshComponent::for_mesh_asset("meshes/sphere.fbx");
    assert!(!cube.vertices.is_empty() && !sphere.vertices.is_empty());
    assert_ne!(
        cube.vertices.len(),
        sphere.vertices.len(),
        "distinct fixtures"
    );

    // ── Missing and corrupt assets: an empty, invisible mesh, no panic ────
    for path in ["meshes/missing.fbx", "meshes/corrupt.fbx"] {
        let mesh = StaticMeshComponent::for_mesh_asset(path);
        assert!(
            mesh.vertices.is_empty() && mesh.indices.is_empty(),
            "{path}"
        );
        assert_eq!(
            mesh.bounds_local,
            [0.0, 0.0, 0.0, 0.5],
            "{path}: the no-mesh bounds"
        );
        // The boundary decoder accepts it too: the path is kept, for a later fix.
        let decoded = pulsar_world_registry::decode_world_component_value(
            "StaticMeshComponent",
            &serde_json::json!({ "mesh_asset": path }),
        )
        .expect("registered")
        .expect("a bad asset is not a decode error");
        let decoded = decoded.downcast::<StaticMeshComponent>().unwrap();
        assert_eq!(decoded.mesh_asset.as_str(), path);
        assert!(decoded.vertices.is_empty());
    }

    let mut world = World::new();
    let ctx = context();
    let store = ctx.as_ref().map(|ctx| {
        let store = Arc::new(SceneGpuStore::new(
            ctx,
            SceneGpuConfig {
                classes: vec![],
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        ));
        world.attach_gpu_mirror(GpuMirrorHandle::new(
            Arc::clone(&store),
            Arc::clone(ctx.queue()),
        ));
        store
    });
    let flush = |world: &World| {
        ctx.as_ref()
            .map(|ctx| world.flush_gpu_mirror(ctx.queue()).unwrap())
    };

    // ── Rapid replacement: the last write wins ────────────────────────────
    let entity = world.spawn();
    world.insert(
        entity,
        StaticMeshComponent::for_mesh_asset("meshes/cube.fbx"),
    );
    for path in [
        "meshes/sphere.fbx",
        "meshes/missing.fbx",
        "meshes/torus.fbx",
        "meshes/sphere.fbx",
    ] {
        set_mesh_asset(&mut world, entity, path);
    }
    assert_eq!(vertex_count(&world, entity), sphere.vertices.len());
    assert_eq!(
        bytes(&world.get::<StaticMeshComponent>(entity).unwrap().vertices),
        bytes(&sphere.vertices)
    );
    flush(&world);

    // ── Shared reuse: two components of one asset share one GPU allocation ─
    let other = world.spawn();
    world.insert(
        other,
        StaticMeshComponent::for_mesh_asset("meshes/sphere.fbx"),
    );
    flush(&world);
    if let Some(store) = &store {
        let handle =
            |e: pulsar_scenedb::Entity| StaticMeshComponent::vertices_gpu_handle(store, e.index());
        assert_eq!(
            handle(entity),
            handle(other),
            "one asset, one geometry allocation"
        );
    }

    // ── A property-only edit: no reload, no geometry upload ───────────────
    std::fs::remove_file(meshes.join("sphere.fbx")).unwrap();
    pulsar_world_registry::set_world_component_property(
        "StaticMeshComponent",
        &mut world,
        entity,
        "movability",
        Box::new(ObjectMovability::Movable),
    )
    .unwrap();
    assert_eq!(
        vertex_count(&world, entity),
        sphere.vertices.len(),
        "the edit did not reload the (now deleted) asset"
    );
    if let Some(stats) = flush(&world) {
        let geometry_bytes = std::mem::size_of_val(sphere.vertices.as_slice()) as u64;
        assert!(
            stats.bytes < geometry_bytes / 10,
            "a movability edit uploaded {} bytes; the geometry is {geometry_bytes}",
            stats.bytes
        );
    }

    // ── Reload: the file changes, the path is assigned again ──────────────
    std::fs::copy(primitive("SM_Torus.fbx"), meshes.join("sphere.fbx")).unwrap();
    set_mesh_asset(&mut world, entity, "meshes/sphere.fbx");
    let torus = StaticMeshComponent::for_mesh_asset("meshes/torus.fbx");
    assert_eq!(
        bytes(&world.get::<StaticMeshComponent>(entity).unwrap().vertices),
        bytes(&torus.vertices),
        "reassigning the path loads the file's new content"
    );
}
