//! On-device smoke test for sublevels + portals (docs/portals_and_sublevels.md).
//!
//! Built directly on `limited_native.rs`'s headless-device harness. Registers
//! one active sublevel and one active portal pair, renders through the real
//! default graph with `enable_sublevels`/`enable_portals` on, and asserts:
//! - zero wgpu validation errors (the pipelines/bind groups/resource pool
//!   sizing are all actually consistent, not just "compiles"),
//! - the target actually received non-background pixels (something drew),
//! - moving the sublevel is O(1) — touches no instance data,
//! - a render with both flags off is unaffected (the zero-overhead contract).

use std::sync::{Arc, Mutex};

use glam::{Mat4, Vec2, Vec3};
use helio::{
    Camera, DebugCameraUniform, DebugDrawState, GroupId, GroupMask, MeshUpload, ObjectDescriptor,
    PackedVertex, PortalDescriptor, PortalPose, Renderer, RendererConfig, Scene, SublevelDescriptor,
};
use helio_default_graphs::build_default_graph_external;
use helio_pass_proxy_composite::ProxyCompositePass;
use helio_pass_secondary_gbuffer::SecondaryGBufferPass;

#[test]
fn sublevel_and_portal_render_without_validation_errors() {
    pollster::block_on(async {
        let Some((device, queue, adapter)) = request_device().await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: portals_and_sublevels smoke");
            return;
        };
        let _ = &adapter;
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut scene = Scene::new(Arc::clone(&device), Arc::clone(&queue));

        let mesh_id = scene.insert_mesh(triangle_mesh());
        let material_id = scene.insert_material(flat_material([0.8, 0.2, 0.2, 1.0]));

        // An ordinary object, always visible through the main camera.
        scene
            .insert_object(ObjectDescriptor {
                mesh: mesh_id,
                material: material_id,
                transform: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                bounds: [0.0, 0.0, 0.0, 2.0],
                flags: libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE,
                groups: GroupMask::NONE,
                movability: None,
                user_tag: 0,
            })
            .expect("insert plain object");

        // A sublevel: one member object with a *local* (unplaced) transform.
        let sublevel_group = GroupId::new(20);
        let sublevel_material = scene.insert_material(flat_material([0.2, 0.8, 0.3, 1.0]));
        scene
            .insert_object(ObjectDescriptor {
                mesh: mesh_id,
                material: sublevel_material,
                transform: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
                bounds: [0.0, 0.0, 0.0, 2.0],
                flags: libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE,
                groups: GroupMask::from(sublevel_group),
                movability: Some(libhelio::Movability::Movable),
                user_tag: 0,
            })
            .expect("insert sublevel member");
        let sublevel = scene.add_sublevel(SublevelDescriptor {
            group: sublevel_group,
            placement: Mat4::from_translation(Vec3::new(20.0, 0.0, 0.0)),
            user_tag: 0,
        });

        // A portal pair, both surfaces in front of the main camera so the
        // portal is actually visible (and allocates a camera slot) this frame.
        let portal_a = PortalPose::from_look_at(
            Vec3::new(-3.0, 0.0, -5.0),
            Vec3::new(-3.0, 0.0, -4.0),
            Vec3::Y,
        );
        let portal_b = PortalPose::from_look_at(
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 1.0),
            Vec3::Y,
        );
        scene.add_portal(PortalDescriptor {
            a: portal_a,
            b: portal_b,
            half_extent: Vec2::new(2.0, 2.0),
            user_tag: 0,
        });

        scene.flush();

        let mut config = RendererConfig::new(128, 128, wgpu::TextureFormat::Rgba8Unorm);
        config.enable_portals = true;
        config.enable_sublevels = true;

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));
        let debug_camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Smoke Debug Camera"),
            size: core::mem::size_of::<DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Smoke Cull Stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let graph = build_default_graph_external(
            &device,
            &queue,
            &scene,
            config,
            Arc::clone(&debug_state),
            &debug_camera,
            &cull_stats,
            None,
        );

        #[allow(deprecated)]
        let mut renderer = Renderer::new_with_external_device(
            Arc::clone(&device),
            Arc::clone(&queue),
            config.surface_format,
            config.width,
            config.height,
            config.render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera,
            cull_stats,
        );

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Smoke Render Target"),
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());

        let camera = Camera::perspective_look_at(
            Vec3::new(0.0, 1.0, 6.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0_f32.to_radians(),
            1.0,
            0.1,
            500.0,
        );

        renderer.render(&camera, &target_view).expect("sublevel+portal frame must render");

        // O(1) sublevel motion: moving it must not touch any instance data —
        // exercised here as "doesn't panic / renders again cleanly", the
        // stronger dirty-range assertion lives in the unit tests
        // (helio-secondary-core, `helio` scene tests) that check
        // `move_sublevel` writes only the placement field.
        renderer
            .scene_mut()
            .move_sublevel(sublevel, Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)))
            .expect("move active sublevel");
        renderer.render(&camera, &target_view).expect("frame after moving the sublevel must render");

        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let validation_error = validation_scope.pop().await;
        assert!(
            validation_error.is_none(),
            "sublevel+portal default graph validation failed: {validation_error:?}"
        );

        let pixels = read_back_rgba(&device, &queue, &target, config.width, config.height).await;
        assert!(
            pixels.chunks(4).any(|p| p != [0, 0, 0, 0]),
            "expected at least one non-transparent-black pixel; the scene has visible \
             geometry directly in front of the camera, so an all-zero readback means \
             nothing drew"
        );
    });
}

#[test]
fn zero_overhead_when_disabled() {
    pollster::block_on(async {
        let Some((device, queue, _adapter)) = request_device().await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: zero-overhead smoke");
            return;
        };
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut scene = Scene::new(Arc::clone(&device), Arc::clone(&queue));
        let mesh_id = scene.insert_mesh(triangle_mesh());
        let material_id = scene.insert_material(flat_material([0.5, 0.5, 0.5, 1.0]));
        scene
            .insert_object(ObjectDescriptor {
                mesh: mesh_id,
                material: material_id,
                transform: Mat4::IDENTITY,
                bounds: [0.0, 0.0, 0.0, 2.0],
                flags: libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE,
                groups: GroupMask::NONE,
                movability: None,
                user_tag: 0,
            })
            .expect("insert object");
        scene.flush();

        // No sublevels/portals registered *and* both config flags off — the
        // default graph must never construct SecondaryGBufferPass/
        // ProxyCompositePass at all (helio-default-graphs::add_geometry_passes).
        let config = RendererConfig::new(64, 64, wgpu::TextureFormat::Rgba8Unorm);
        assert!(!config.enable_portals && !config.enable_sublevels);

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));
        let debug_camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Zero-Overhead Debug Camera"),
            size: core::mem::size_of::<DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Zero-Overhead Cull Stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut graph = build_default_graph_external(
            &device,
            &queue,
            &scene,
            config,
            Arc::clone(&debug_state),
            &debug_camera,
            &cull_stats,
            None,
        );
        assert!(
            graph.iter_passes_mut::<SecondaryGBufferPass>().next().is_none(),
            "the default graph must not construct SecondaryGBufferPass when both \
             enable_portals and enable_sublevels are false"
        );
        assert!(
            graph.iter_passes_mut::<ProxyCompositePass>().next().is_none(),
            "the default graph must not construct ProxyCompositePass when both \
             enable_portals and enable_sublevels are false"
        );

        #[allow(deprecated)]
        let mut renderer = Renderer::new_with_external_device(
            Arc::clone(&device),
            Arc::clone(&queue),
            config.surface_format,
            config.width,
            config.height,
            config.render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera,
            cull_stats,
        );
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Zero-Overhead Target"),
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let camera = Camera::perspective_look_at(
            Vec3::new(0.0, 1.0, 3.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0_f32.to_radians(),
            1.0,
            0.1,
            100.0,
        );
        renderer
            .render(&camera, &target.create_view(&Default::default()))
            .expect("plain scene must still render with both flags off");

        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let validation_error = validation_scope.pop().await;
        assert!(validation_error.is_none(), "zero-overhead render validation failed: {validation_error:?}");
    });
}

/// Depth-2 portal recursion, CPU-side only (no rendering): a portal-1 eye
/// positioned so portal-2's surface falls inside its frustum must allocate a
/// *second* secondary view whose `parent_index` points at the first — the
/// mechanism `ProxyCompositePass` relies on to composite portal-2's content
/// into portal-1's own off-screen G-buffer before portal-1 itself composites
/// into the main frame (docs/portals_and_sublevels.md §7.3).
#[test]
fn portal_recursion_allocates_a_nested_view_with_correct_parent() {
    pollster::block_on(async {
        let Some((device, queue, _adapter)) = request_device().await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: portal recursion");
            return;
        };
        let mut scene = Scene::new(Arc::new(device), Arc::new(queue));

        // Portal 1: at the origin, facing +Z; destination 50 units down +X,
        // same orientation (pure-translation pair_map).
        let a1 = PortalPose::from_look_at(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), Vec3::Y);
        let b1 = PortalPose::from_look_at(Vec3::new(50.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 1.0), Vec3::Y);
        scene.add_portal(PortalDescriptor { a: a1, b: b1, half_extent: Vec2::new(5.0, 5.0), user_tag: 0 });

        // Portal 2's entry surface sits where portal-1's eye (which renders
        // from ~(50,0,-5) looking down +Z, mirroring the main camera's own
        // offset-by-the-pair_map pose below) will see it: 10 units further
        // along that same view direction, dead centre.
        let a2 = PortalPose::from_look_at(Vec3::new(50.0, 0.0, 5.0), Vec3::new(50.0, 0.0, 6.0), Vec3::Y);
        let b2 = PortalPose::from_look_at(Vec3::new(200.0, 0.0, 0.0), Vec3::new(200.0, 0.0, 1.0), Vec3::Y);
        scene.add_portal(PortalDescriptor { a: a2, b: b2, half_extent: Vec2::new(5.0, 5.0), user_tag: 0 });

        // Main camera looking straight through portal 1 (wide FOV — this
        // test only needs the geometry to be *comfortably* inside frame, not
        // pixel-exact).
        let camera = Camera::perspective_look_at(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::Y,
            90.0_f32.to_radians(),
            800.0 / 600.0,
            0.1,
            1000.0,
        );
        scene.update_camera(camera);
        let main_uniforms = *scene.gpu_scene().camera.data();
        scene.refresh_secondary_views(&main_uniforms, [800, 600]);

        let count = scene.secondary_view_count();
        let parents = scene.secondary_view_parent_indices();
        assert!(
            count >= 2,
            "expected portal-1 (depth 1) + portal-2 (depth 2, seen through portal-1's eye) \
             to both allocate a view; got {count} view(s), parents={parents:?}"
        );
        assert!(
            parents.iter().any(|&p| p != helio_secondary_core::NO_PARENT),
            "expected at least one nested (parent_index != NO_PARENT) view; got {parents:?}"
        );
        // The depth-1 view (portal 1, seen by the main camera) must itself
        // have no parent — it composites into the main frame.
        assert!(
            parents.iter().any(|&p| p == helio_secondary_core::NO_PARENT),
            "expected at least one depth-1 (parent_index == NO_PARENT) view; got {parents:?}"
        );
    });
}

fn triangle_mesh() -> MeshUpload {
    let v = |p: [f32; 3]| PackedVertex::from_components(p, [0.0, 0.0, 1.0], [0.0, 0.0], [1.0, 0.0, 0.0], 1.0);
    MeshUpload {
        vertices: vec![
            v([-1.0, -1.0, 0.0]),
            v([1.0, -1.0, 0.0]),
            v([0.0, 1.0, 0.0]),
        ],
        indices: vec![0, 1, 2],
    }
}

fn flat_material(base_color: [f32; 4]) -> libhelio::GpuMaterial {
    libhelio::GpuMaterial {
        base_color,
        emissive: [0.0, 0.0, 0.0, 0.0],
        roughness_metallic: [0.6, 0.0, 1.5, 1.0],
        tex_base_color: u32::MAX,
        tex_normal: u32::MAX,
        tex_roughness: u32::MAX,
        tex_emissive: u32::MAX,
        tex_occlusion: u32::MAX,
        workflow: 0,
        flags: 0,
        material_class: 0,
        class_params: [0.0; 4],
    }
}

async fn read_back_rgba(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
    let bytes_per_row = (width * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Smoke Readback"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Smoke Readback Encoder") });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(height) },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = futures_channel_oneshot();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map_async callback never fired").expect("buffer map failed");

    let data = slice.get_mapped_range().expect("buffer slice should be mapped after map_async completed");
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * bytes_per_row) as usize;
        out.extend_from_slice(&data[start..start + (width * 4) as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}

/// Minimal std-only oneshot channel (avoids pulling in an async-channel dep
/// just for this one readback).
fn futures_channel_oneshot<T>() -> (std::sync::mpsc::Sender<T>, std::sync::mpsc::Receiver<T>) {
    std::sync::mpsc::channel()
}

async fn request_device() -> Option<(wgpu::Device, wgpu::Queue, wgpu::Adapter)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut adapter = None;
    for force_fallback_adapter in [false, true] {
        if let Ok(a) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            adapter = Some(a);
            break;
        }
    }
    let adapter = adapter?;
    if !adapter.features().contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
        return None;
    }
    let limits = helio::required_wgpu_limits(adapter.limits());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Portals/Sublevels Smoke Device"),
            required_features: wgpu::Features::INDIRECT_FIRST_INSTANCE,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            ..Default::default()
        })
        .await
        .ok()?;
    Some((device, queue, adapter))
}
