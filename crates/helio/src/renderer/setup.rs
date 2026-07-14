use std::sync::{Arc, Mutex};

use web_time::Instant;

use helio_pass_debug::DebugCameraUniform;
use helio_pass_debug_overlay::DebugOverlayState;

use crate::scene::Scene;

use super::config::RendererConfig;
use super::debug::DebugDrawState;
use super::graph::{build_default_graph, build_default_graph_external, create_depth_resources};
use super::renderer_impl::{GraphKind, Renderer};

impl Renderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, config: RendererConfig) -> Self {
        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));

        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debug_overlay_shared = DebugOverlayState::new();
        let cull_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut graph = build_default_graph(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buffer,
            &cull_stats_buffer,
            Some(&debug_overlay_shared),
        );

        let (depth_texture, depth_view) =
            create_depth_resources(&device, config.internal_width(), config.internal_height());

        let (full_res_depth_texture, full_res_depth_view) = if config.render_scale < 1.0 {
            let (t, v) = create_depth_resources(&device, config.width, config.height);
            (Some(t), Some(v))
        } else {
            (None, None)
        };

        let water_volumes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Volumes Buffer"),
            size: 256 * 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let water_hitboxes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Hitboxes Buffer"),
            size: 256 * 80,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            graph,
            graph_kind: GraphKind::Default,
            scene,
            depth_texture,
            depth_view,
            output_width: config.width,
            output_height: config.height,
            render_scale: config.render_scale,
            full_res_depth_texture,
            full_res_depth_view,
            surface_format: config.surface_format,
            debug_camera_buffer,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            shadow_quality: config.shadow_quality,
            shadow_atlas_size: config.shadow_atlas_size,
            debug_mode: config.debug_mode,
            debug_depth_test: true,
            editor_mode: false,
            custom_graph_builder: None,
            custom_graph_config: None,
            perf_overlay_mode: config.perf_overlay_mode,
            debug_state,
            debug_overlay_shared,
            billboard_instances: Vec::new(),
            billboard_scratch: Vec::new(),
            billboard_dirty: true,
            billboard_cached_light_count: usize::MAX,
            billboard_cached_light_gen: u64::MAX,
            billboard_cached_editor_hidden: false,
            billboard_cached_corona_gen: u64::MAX,
            billboard_generation: 0,

            corona_emitters: Vec::new(),
            corona_emitter_generation: 0,

            water_volumes_buffer,
            water_hitboxes_buffer,
            last_render_time: Instant::now(),
            delta_time: 0.0,
            cull_stats: [0; 8],
            graph_time_ms: 0.0,
            frame_times: vec![0.0; 200],
            frame_times_cursor: 0,
            clear_target_next_frame: true,
            owns_device: true,
            pending_resize: Some((config.width, config.height)),
            gizmo_camera: None,
            gizmo_viewport_height: 0.0,
            cull_stats_buffer,
        }
    }

    pub fn new_with_external_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RendererConfig,
    ) -> Self {
        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);

        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));

        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debug_overlay_shared = DebugOverlayState::new();
        let cull_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut graph = build_default_graph_external(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buffer,
            &cull_stats_buffer,
            Some(&debug_overlay_shared),
        );

        let (depth_texture, depth_view) =
            create_depth_resources(&device, config.internal_width(), config.internal_height());

        let (full_res_depth_texture, full_res_depth_view) = if config.render_scale < 1.0 {
            let (t, v) = create_depth_resources(&device, config.width, config.height);
            (Some(t), Some(v))
        } else {
            (None, None)
        };

        let water_volumes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Volumes Buffer"),
            size: 256 * 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let water_hitboxes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Hitboxes Buffer"),
            size: 256 * 80,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            graph,
            graph_kind: GraphKind::Default,
            scene,
            depth_texture,
            depth_view,
            output_width: config.width,
            output_height: config.height,
            render_scale: config.render_scale,
            full_res_depth_texture,
            full_res_depth_view,
            surface_format: config.surface_format,
            debug_camera_buffer,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            shadow_quality: config.shadow_quality,
            shadow_atlas_size: config.shadow_atlas_size,
            debug_mode: config.debug_mode,
            debug_depth_test: true,
            editor_mode: false,
            custom_graph_builder: None,
            custom_graph_config: None,
            perf_overlay_mode: config.perf_overlay_mode,
            debug_state,
            debug_overlay_shared,
            billboard_instances: Vec::new(),
            billboard_scratch: Vec::new(),
            billboard_dirty: true,
            billboard_cached_light_count: usize::MAX,
            billboard_cached_light_gen: u64::MAX,
            billboard_cached_editor_hidden: false,
            billboard_cached_corona_gen: u64::MAX,
            billboard_generation: 0,

            corona_emitters: Vec::new(),
            corona_emitter_generation: 0,

            water_volumes_buffer,
            water_hitboxes_buffer,
            last_render_time: Instant::now(),
            delta_time: 0.0,
            cull_stats: [0; 8],
            graph_time_ms: 0.0,
            frame_times: vec![0.0; 200],
            frame_times_cursor: 0,
            gizmo_camera: None,
            gizmo_viewport_height: 0.0,
            owns_device: false,
            pending_resize: Some((config.width, config.height)),
            clear_target_next_frame: true,
            cull_stats_buffer,
        }
    }
}
