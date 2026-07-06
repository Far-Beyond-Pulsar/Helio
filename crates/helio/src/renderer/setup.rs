use std::sync::{Arc, Mutex};

#[cfg(target_arch = "wasm32")]
use web_time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use crate::scene::Scene;
use helio_core::RenderGraph;

use super::config::RendererConfig;
use super::debug::DebugDrawState;
use super::renderer_impl::{Renderer, DebugCameraUniform, HALTON_JITTER};

impl Renderer {
    pub(crate) fn compute_jitter_matrices(width: u32, height: u32) -> [glam::Mat4; 16] {
        let mut matrices = [glam::Mat4::IDENTITY; 16];
        for (i, raw) in HALTON_JITTER.iter().enumerate() {
            let jx = ((raw[0] - 0.5) * 2.0) / (width as f32);
            let jy = ((raw[1] - 0.5) * 2.0) / (height as f32);
            matrices[i] = glam::Mat4::from_translation(glam::Vec3::new(jx, jy, 0.0));
        }
        matrices
    }

    pub(crate) fn create_depth_resources(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Helio Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        render_scale: f32,
        config: RendererConfig,
        mut scene: Scene,
        graph: RenderGraph,
        debug_state: Arc<Mutex<DebugDrawState>>,
        debug_camera_buffer: wgpu::Buffer,
        cull_stats_buffer: wgpu::Buffer,
    ) -> Self {
        scene.set_render_size(width, height);

        #[cfg(target_arch = "wasm32")]
        if !device.features().contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
            log::error!(
                "helio: INDIRECT_FIRST_INSTANCE (WebGPU indirect-first-instance) is not \
                 available on this device. Only the first object in every scene will render. \
                 Please use a browser that supports the indirect-first-instance WebGPU feature \
                 (Chrome 113+, Firefox 122+, Safari 17+)."
            );
        }

        let internal_w = config.internal_width();
        let internal_h = config.internal_height();

        let (depth_texture, depth_view) = Self::create_depth_resources(
            &device,
            internal_w,
            internal_h,
        );

        let (full_res_depth_texture, full_res_depth_view) = if render_scale < 1.0 {
            let (t, v) = Self::create_depth_resources(&device, width, height);
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

        let jitter_matrices = Self::compute_jitter_matrices(internal_w, internal_h);

        let cull_stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CullStats Staging"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            graph,
            scene,
            depth_texture,
            depth_view,
            output_width: width,
            output_height: height,
            render_scale,
            full_res_depth_texture,
            full_res_depth_view,
            surface_format,
            debug_camera_buffer,
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
            clear_color: [0.02, 0.02, 0.03, 1.0],
            gi_config: config.gi_config,
            shadow_quality: config.shadow_quality,
            debug_mode: config.debug_mode,
            debug_depth_test: true,
            editor_mode: false,
            debug_state,
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
            cull_stats_staging,
            cull_stats: [0; 8],
            graph_time_ms: 0.0,
            frame_times: vec![0.0; 200],
            frame_times_cursor: 0,
            jitter_matrices,
            jitter_cache_width: internal_w,
            jitter_cache_height: internal_h,
            #[cfg(feature = "bake")]
            bake_pending: None,
            #[cfg(feature = "bake")]
            baked_data: None,
            clear_target_next_frame: true,
            owns_device: true,
            pending_resize: Some((width, height)),
            gizmo_camera: None,
            gizmo_viewport_height: 0.0,
            cull_stats_buffer,
            graph_rebuilder: None,
            debug_overlay_shared: helio_pass_debug_overlay::DebugOverlayState::new(),
        }
    }

    pub fn new_with_external_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        render_scale: f32,
        config: RendererConfig,
        mut scene: Scene,
        graph: RenderGraph,
        debug_state: Arc<Mutex<DebugDrawState>>,
        debug_camera_buffer: wgpu::Buffer,
        cull_stats_buffer: wgpu::Buffer,
    ) -> Self {
        let mut renderer = Self::new(
            device,
            queue,
            surface_format,
            width,
            height,
            render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera_buffer,
            cull_stats_buffer,
        );
        renderer.owns_device = false;
        renderer
    }
}
