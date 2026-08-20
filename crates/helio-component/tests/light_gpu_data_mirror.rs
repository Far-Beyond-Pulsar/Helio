//! Proves `LightGpuData` (Pulsar-Native#561: light's render data belongs in
//! SceneDB the same way `StaticMeshComponent::vertices`/`indices` do)
//! actually GPU-mirrors through `World` -- the fixed-size counterpart to
//! `static_mesh_component_gpu_mirror.rs`'s var-len proof. Uses the
//! zero-manual-steps auto-registration path (SceneDB issue #41, see
//! `pulsar_scenedb`'s own `gpu_zero_manual_steps.rs`): no
//! `register_gpu_columns_growable` call anywhere in this file, matching
//! how `HelioRenderer` never calls it for `LightGpuData` either --
//! `World::insert` alone is what gets a `#[gpu]`-mirrored component onto
//! the GPU.

use helio::GpuLight;
use helio_component::components::{LightGpuData, LightGpuRow};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::{GpuColumnSet, World};
use std::sync::Arc;

fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("light-gpu-data-mirror-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

#[test]
fn light_gpu_data_lands_on_the_gpu_through_plain_world_insert_alone() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let light = GpuLight {
        color_intensity: [0.25, 0.5, 0.75, 42.0],
        light_type: 2,
        ..Default::default()
    };
    // The ONLY GPU-relevant call: a plain insert. No register_gpu_columns_
    // growable, no manual flush call -- `World::insert`'s auto-registration
    // dispatch and `flush_gpu_mirror` below are the entire setup.
    world.insert(entity, LightGpuData { row: light.into() });
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let field_id = LightGpuData::gpu_columns()[0].field_token.id();
    let handle = store
        .resolve_buffer_handle(store.buffer_key_for(field_id).expect("insert must auto-register the buffer"))
        .expect("resolvable");
    let got: LightGpuRow = pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, entity.index());

    assert_eq!(got.0.color_intensity, [0.25, 0.5, 0.75, 42.0], "must be real GPU-resident data, not just the CPU World row");
    assert_eq!(got.0.light_type, 2);

    // The CPU-side World row must also read back correctly, same guarantee
    // static mesh's own mirror test makes for its non-#[gpu] field.
    let stored = world.get::<LightGpuData>(entity).expect("component must be readable back");
    assert_eq!(stored.row.0.color_intensity, [0.25, 0.5, 0.75, 42.0]);
}

#[test]
fn re_insert_with_different_values_overwrites_the_same_row() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    world.insert(entity, LightGpuData { row: GpuLight { light_type: 0, ..Default::default() }.into() });
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    // Re-insert with a different value -- e.g. the user changing the light
    // type in the properties panel, which re-triggers hydrate.
    world.insert(entity, LightGpuData { row: GpuLight { light_type: 1, ..Default::default() }.into() });
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let field_id = LightGpuData::gpu_columns()[0].field_token.id();
    let handle = store.resolve_buffer_handle(store.buffer_key_for(field_id).unwrap()).expect("resolvable");
    let got: LightGpuRow = pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, entity.index());
    assert_eq!(got.0.light_type, 1, "the GPU row must reflect the latest insert, not the first one");
}
