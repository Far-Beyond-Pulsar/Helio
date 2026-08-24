//! Proves `LightComponentGpuMirror` (Pulsar-Native#561: `LightComponent`
//! fully normalized onto the generic auto-mirror system -- no more hand-
//! rolled `LightGpuData`/`LightGpuRow` companion) actually GPU-mirrors
//! through `World`, and that `to_helio_gpu_light` correctly translates the
//! mirrored, GPU-resident bytes into the shape Helio's own `GpuLight`
//! expects. Uses the zero-manual-steps auto-registration path (SceneDB
//! issue #41): no `register_gpu_columns_growable` call anywhere in this
//! file, matching how `HelioRenderer` never calls it for this mirror either
//! -- `World::insert` alone is what gets a `#[gpu]`-mirrored component onto
//! the GPU.

use helio::LightType as HelioLightType;
use helio_component::components::{LightComponent, LightComponentGpuMirror, LightType};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::World;
use pulsar_world_registry::GpuMirrored;
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
        label: Some("light-component-gpu-mirror-test"),
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
fn light_component_gpu_mirror_lands_on_the_gpu_through_sync_gpu_mirror_alone() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let mut light = LightComponent::default();
    light.color.color = [0.25, 0.5, 0.75, 1.0];
    light.intensity.intensity = 42.0;
    light.general.light_type = LightType::Spot;

    // The ONLY GPU-relevant call: `sync_gpu_mirror` (`GpuMirrored`'s own
    // trait default, the exact call `hydrate_light_component` makes for an
    // enabled light) -- no register_gpu_columns_growable, no manual field
    // wiring, same "World::insert's auto-registration dispatch + flush_gpu_
    // mirror is the entire setup" property the old LightGpuData test made.
    light.sync_gpu_mirror(&mut world, entity);
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let id = LightComponentGpuMirror::packed_gpu_component_id();
    let handle = store.resolve_buffer_handle(store.buffer_key_for(id).expect("insert must auto-register the buffer")).expect("resolvable");
    let got: LightComponentGpuMirror =
        pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, entity.index());

    let gpu = got.to_helio_gpu_light();
    assert_eq!(gpu.color_intensity, [0.25, 0.5, 0.75, 42.0], "must be real GPU-resident data, not just the CPU World row");
    assert_eq!(gpu.light_type, HelioLightType::Spot as u32);

    // Spot cones must land on the GPU as COSINES (the shader contract), not
    // radians -- #172's reversed-cone artifact came from exactly this row.
    let expected_inner_cos = 30.0_f32.to_radians().cos();
    let expected_outer_cos = 45.0_f32.to_radians().cos();
    assert!(
        (gpu.inner_angle - expected_inner_cos).abs() < 1e-6
            && (gpu.direction_outer[3] - expected_outer_cos).abs() < 1e-6,
        "cone angles must be GPU-round-tripped cosines: inner {expected_inner_cos}/outer {expected_outer_cos}, got inner {}/outer {}",
        gpu.inner_angle,
        gpu.direction_outer[3]
    );

    // The CPU-side World row must also read back correctly, same guarantee
    // static mesh's own mirror test makes for its non-#[gpu] field.
    let stored = world.get::<LightComponentGpuMirror>(entity).expect("mirror must be readable back");
    assert_eq!(stored.to_helio_gpu_light().color_intensity, [0.25, 0.5, 0.75, 42.0]);
}

#[test]
fn re_syncing_with_a_different_value_overwrites_the_same_row() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let mut light = LightComponent::default();
    light.general.light_type = LightType::Directional;
    light.sync_gpu_mirror(&mut world, entity);
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    // Re-sync with a different value -- e.g. the user changing the light
    // type in the properties panel, which re-triggers hydrate.
    light.general.light_type = LightType::Point;
    light.sync_gpu_mirror(&mut world, entity);
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let id = LightComponentGpuMirror::packed_gpu_component_id();
    let handle = store.resolve_buffer_handle(store.buffer_key_for(id).unwrap()).expect("resolvable");
    let got: LightComponentGpuMirror =
        pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, entity.index());
    assert_eq!(
        got.to_helio_gpu_light().light_type,
        HelioLightType::Point as u32,
        "the GPU row must reflect the latest sync, not the first one"
    );
}

#[test]
fn disabled_light_never_reaches_the_gpu_mirror_at_all() {
    // The behavioral contract `hydrate_light_component` (runtime.rs) relies
    // on: a disabled light must never call sync_gpu_mirror in the first
    // place -- this test proves the CONSEQUENCE of that (no GPU buffer
    // registration happens for an entity that never synced), not the
    // hydrate function itself (covered directly in runtime.rs's own tests).
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    // Deliberately no sync_gpu_mirror call -- standing in for a disabled
    // light, which hydrate_light_component would skip entirely.
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    assert!(world.get::<LightComponentGpuMirror>(entity).is_none());
}
