//! Proves `LightComponentGpuMirror`'s GPU row follows the authored
//! `LightComponent` through SceneDB's own write path: inserting, editing,
//! disabling and removing the light are the only calls made, and the row is
//! never a component anyone inserts or refreshes. Also checks that
//! `to_helio_gpu_light` translates the GPU-resident bytes into the shape
//! Helio's `GpuLight` expects. No `register_gpu_columns_growable` call: the
//! authored component's dispatch auto-registers the companion buffer.
//!
//! Needs a GPU adapter (any Vulkan device, including lavapipe).

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

fn mirrored_row(ctx: &EngineGpuContext, store: &SceneGpuStore, entity: pulsar_scenedb::Entity) -> LightComponentGpuMirror {
    let id = LightComponentGpuMirror::packed_gpu_component_id();
    let handle = store
        .resolve_buffer_handle(store.buffer_key_for(id).expect("the light insert must auto-register the buffer"))
        .expect("resolvable");
    pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, entity.index())
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

#[test]
fn inserting_a_light_writes_its_gpu_row() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let mut light = LightComponent::default();
    light.color.color = [0.25, 0.5, 0.75, 1.0];
    light.intensity.intensity = 42.0;
    light.general.light_type = LightType::Spot;
    let expected = light.to_gpu_mirror().to_helio_gpu_light();

    // The only call: an ordinary insert of the authored component.
    world.insert(entity, light);
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let got = mirrored_row(&ctx, &store, entity);
    assert_eq!(got.general.enabled.0, 1);
    let gpu = got.to_helio_gpu_light();
    assert_eq!(gpu.color_intensity, expected.color_intensity, "must be real GPU-resident data");
    assert_eq!(&gpu.color_intensity[..3], &[0.25, 0.5, 0.75]);
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

    // The companion is GPU-only: nothing inserted it as a component.
    assert!(world.get::<LightComponentGpuMirror>(entity).is_none());
}

#[test]
fn edits_disable_and_removal_reach_the_same_row() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let mut light = LightComponent::default();
    light.general.light_type = LightType::Directional;
    world.insert(entity, light);
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    // A live edit through the write guard -- what the properties panel and
    // scripts do -- with no refresh call after it.
    world.get_mut::<LightComponent>(entity).unwrap().general.light_type = LightType::Point;
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    assert_eq!(
        mirrored_row(&ctx, &store, entity).to_helio_gpu_light().light_type,
        HelioLightType::Point as u32,
        "the GPU row must reflect the latest write"
    );

    // Disabling keeps the row and marks it absent.
    world.get_mut::<LightComponent>(entity).unwrap().general.enabled = false;
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    assert_eq!(mirrored_row(&ctx, &store, entity).general.enabled.0, 0);

    // Removal clears it.
    world.remove::<LightComponent>(entity);
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    let cleared = mirrored_row(&ctx, &store, entity);
    assert_eq!(cleared.general.enabled.0, 0);
    assert_eq!(cleared.to_helio_gpu_light().color_intensity, [0.0; 4]);
}

#[test]
fn a_light_present_before_the_mirror_attaches_is_replayed() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));

    let mut world = World::new();
    let entity = world.spawn();
    let mut light = LightComponent::default();
    light.intensity.intensity = 7.0;
    let expected = light.to_gpu_mirror().to_helio_gpu_light().color_intensity;
    world.insert(entity, light);

    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    assert_eq!(mirrored_row(&ctx, &store, entity).to_helio_gpu_light().color_intensity, expected);
}
