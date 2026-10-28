//! The default graph follows the scene: in a mesh room with no water,
//! emitters, billboards, decals, portals or sprites, and no debug mode, those
//! passes are dormant (no prepare, no execute, no content resources). Each
//! joins when its content appears: a debug line, a debug view, a decal row
//! in a column the host registered up front. A resize keeps every pass.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, RendererBuilder, RendererConfig,
};
use helio_default_passes::build_default_graph_external_with_context;
use helio_pass_decal::DecalComponent;
use helio_pass_gbuffer::{MaterialComponent, MeshComponent, StaticObjectComponent};

/// Passes that need scene content or a debug mode the room does not have.
const CONTENT_PASSES: &[&str] = &[
    "WaterSim",
    "Corona",
    "Billboard",
    "DecalApply",
    "PortalCull",
    "PortalMask",
    "PortalInstance",
    "PortalEditorOverlay",
    "SpriteCull",
    "SpriteBatch",
    "BufferView",
    "DebugDraw",
    "PerfOverlay",
    "PerfOverlay Color Analyzer",
    "PerfOverlay Cost Analyzer",
];

fn scene_db(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> pulsar_scenedb::SceneDb {
    let mut scene_db = pulsar_scenedb::SceneDb::new();
    let ctx = pulsar_scenedb::gpu::EngineGpuContext::new(device.clone(), queue.clone());
    let mut store = pulsar_scenedb::gpu::SceneGpuStore::new(
        &ctx,
        pulsar_scenedb::gpu::SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    MeshComponent::register_gpu_columns_growable(&mut store, 64, device);
    MaterialComponent::register_gpu_columns_growable(&mut store, 64, device);
    StaticObjectComponent::register_gpu_columns_growable(&mut store, 64, device);
    // Registered before any decal exists, as editors do: the column's
    // presence alone must not wake the decal pass.
    DecalComponent::register_gpu_columns_growable(&mut store, 64, device);
    scene_db
        .world
        .attach_gpu_mirror(pulsar_scenedb::gpu::GpuMirrorHandle::new(
            Arc::new(store),
            queue.clone(),
        ));
    scene_db
}

/// A floor slab.
fn floor(world: &mut pulsar_scenedb::World) {
    let half = Vec3::new(5.0, 0.25, 5.0);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (a, b) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        vertices.push(helio::PackedVertex::from_components(
            [a * half.x, half.y, b * half.z],
            [0.0, 1.0, 0.0],
            [0.0; 2],
            [1.0, 0.0, 0.0],
            1.0,
        ));
    }
    indices.extend_from_slice(&[0, 2, 1, 0, 3, 2]);
    let mesh = world.spawn();
    world.insert(mesh, MeshComponent { vertices, indices });
    let material = world.spawn();
    let row = MaterialComponent::new([0.7, 0.7, 0.7, 1.0], 0.8, 0.0, [0.0; 3], 0.0);
    world.insert(material, row);
    let store = world.gpu_mirror().cloned().unwrap();
    let vertex_range = MeshComponent::vertices_gpu_handle(store.store(), mesh.index()).unwrap();
    let index_range = MeshComponent::indices_gpu_handle(store.store(), mesh.index()).unwrap();
    let object = StaticObjectComponent::new(
        mesh.index(),
        mesh.generation().wrapping_add(1),
        material.index(),
        material.generation().wrapping_add(1),
        Mat4::IDENTITY,
        [0.0, 0.0, 0.0, half.length()],
        index_range.count,
        index_range.offset,
        vertex_range.offset as i32,
        row.material_class,
        0,
        0,
    );
    let entity = world.spawn();
    world.insert(entity, object);
}

#[test]
fn passes_join_with_their_content_and_resize_keeps_them() {
    let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })) else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: scene-driven graph");
        return;
    };
    let features = required_wgpu_features(helio::usable_adapter_features(&adapter));
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        required_limits: required_wgpu_limits(adapter.limits()),
        experimental_features: helio::required_experimental_features(features),
        ..Default::default()
    }))
    .unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut scene = scene_db(&device, &queue);
    floor(&mut scene.world);
    let mut config = RendererConfig::new(64, 48, wgpu::TextureFormat::Rgba8UnormSrgb);
    config.enable_foliage = false;
    let mut renderer = RendererBuilder::new(config, scene.world.gpu_mirror().cloned().unwrap())
        .with_pass_build_context(Box::new(build_default_graph_external_with_context))
        .with_external_device()
        .build(
            device.clone(),
            queue.clone(),
            config.width,
            config.height,
            config.surface_format,
        );
    let camera = Camera::perspective_look_at(
        Vec3::new(0.0, 3.0, 6.0),
        Vec3::ZERO,
        Vec3::Y,
        1.0,
        64.0 / 48.0,
        0.1,
        100.0,
    );
    let target_for = |width: u32, height: u32| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: config.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let mut view = target_for(64, 48);
    let frames = |renderer: &mut helio::Renderer,
                  scene: &mut pulsar_scenedb::SceneDb,
                  view: &wgpu::TextureView,
                  count: u32| {
        for _ in 0..count {
            scene.world.flush_gpu_mirror(&queue);
            renderer.render(&camera, view).unwrap();
            // Row readbacks complete between frames.
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }
    };
    let has = |renderer: &helio::Renderer, name: &str| renderer.graph_members().contains(&name);

    frames(&mut renderer, &mut scene, &view, 4);
    for name in CONTENT_PASSES {
        assert!(
            !has(&renderer, name),
            "`{name}` runs in a scene without its content: {:?}",
            renderer.graph_members()
        );
    }
    assert!(has(&renderer, "GBuffer") && has(&renderer, "PostProcess"));

    // A debug line wakes the user debug draw.
    renderer.debug_line(Vec3::ZERO, Vec3::Y, [1.0, 0.0, 0.0, 1.0]);
    frames(&mut renderer, &mut scene, &view, 1);
    assert!(has(&renderer, "DebugDraw"));

    // A debug view wakes the buffer view; leaving it puts it back to sleep.
    renderer.set_debug_mode(helio_pass_buffer_view::DEPTH_VIEW);
    frames(&mut renderer, &mut scene, &view, 1);
    assert!(has(&renderer, "BufferView"));
    renderer.set_debug_mode(0);
    frames(&mut renderer, &mut scene, &view, 1);
    assert!(!has(&renderer, "BufferView"));

    // A decal row in the up-front column: the decal pass joins once the row
    // has been read back (a frame or two).
    let decal = scene.world.spawn();
    scene.world.insert(
        decal,
        DecalComponent {
            transform: Mat4::from_scale(Vec3::splat(2.0)).to_cols_array(),
            color: [1.0, 0.0, 0.0, 1.0],
            ..bytemuck::Zeroable::zeroed()
        },
    );
    frames(&mut renderer, &mut scene, &view, 4);
    assert!(
        has(&renderer, "DecalApply"),
        "{:?}",
        renderer.graph_members()
    );

    // A resize resizes in place: the same pass objects, at the new size.
    let before = renderer
        .find_pass::<helio_pass_deferred_light::DeferredLightPass>()
        .map(|pass| pass as *const _ as usize);
    renderer.set_render_size(80, 60);
    view = target_for(80, 60);
    frames(&mut renderer, &mut scene, &view, 2);
    let after = renderer
        .find_pass::<helio_pass_deferred_light::DeferredLightPass>()
        .map(|pass| pass as *const _ as usize);
    assert!(before.is_some());
    assert_eq!(before, after, "a resize kept the passes");
    assert!(renderer.last_resize_ms() > 0.0);

    let error = pollster::block_on(errors.pop());
    assert!(error.is_none(), "{error:?}");
}
