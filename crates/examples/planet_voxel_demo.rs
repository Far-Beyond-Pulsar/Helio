//! Earth-radius camera-local validation for Helio's production planetary voxel path.
//!
//! Controls:
//!   Mouse click        - capture cursor / look
//!   W/A/S/D            - move through canonical planet space
//!   Space/Left Shift   - move up/down
//!   Escape             - release cursor / exit

use glam::{EulerRot, Quat, Vec3};
use helio::{
    Camera, DebugDrawState, RenderGraph, Renderer, RendererConfig, Scene,
    required_experimental_features, required_wgpu_features, required_wgpu_limits,
};
use helio_pass_fxaa::FxaaPass;
use helio_pass_planetary_voxel::{
    ExtractionFixture, ExtractionFixtureKind, PlanetarySurfaceUpload, PlanetaryVoxelRenderConfig,
    PlanetaryVoxelRenderPass, TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT,
    TransvoxelTransitionFaceFixture,
};
use helio_planet_voxel_core::{
    CellWord, LOD0_CELL_SIZE_METERS, PAGE_EDGE, PAGE_EDGE_CELLS, PageKey, PageUpload,
    PlanetFrameUniform, PlanetId, PlanetPageKey, PlanetPosition, SourceGeneration, TransitionFace,
    VisiblePage, VisiblePageSet,
};
use std::{collections::HashSet, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

const EARTH_RADIUS_METERS: f64 = 6_371_000.0;
const LOOK_SENSITIVITY: f32 = 0.002;
const MOVE_SPEED_METERS_PER_SECOND: f64 = 1.5;
const INITIAL_YAW: f32 = -std::f32::consts::FRAC_PI_2;
const INITIAL_PITCH: f32 = -0.55;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { state: None };
    event_loop
        .run_app(&mut app)
        .expect("planet demo event loop");
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    renderer: Renderer,
    planet: PlanetId,
    canonical_camera_m: [f64; 3],
    spawn_camera_m: [f64; 3],
    frame_index: u64,
    yaw: f32,
    pitch: f32,
    keys: HashSet<KeyCode>,
    mouse_delta: (f32, f32),
    cursor_grabbed: bool,
    last_frame: Instant,
    last_title_update: Instant,
}

impl AppState {
    fn reset_transient_input(&mut self) {
        self.keys.clear();
        self.mouse_delta = (0.0, 0.0);
        self.cursor_grabbed = false;
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
        self.last_frame = Instant::now();
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.reset_transient_input();
        if width == 0 || height == 0 {
            return;
        }
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: self.alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
        self.renderer.set_render_size(width, height);
        self.renderer
            .find_pass_mut::<PlanetaryVoxelRenderPass>()
            .expect("planetary pass")
            .residency_mut()
            .resize(width, height);
    }

    fn orientation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }

    fn update(&mut self, dt: f64) {
        self.yaw -= self.mouse_delta.0 * LOOK_SENSITIVITY;
        self.pitch = (self.pitch - self.mouse_delta.1 * LOOK_SENSITIVITY).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);
        let orientation = self.orientation();
        advance_camera(&mut self.canonical_camera_m, &self.keys, orientation, dt);
    }

    fn camera(&self, width: u32, height: u32) -> Camera {
        let orientation = self.orientation();
        Camera::perspective_look_at(
            Vec3::ZERO,
            orientation * -Vec3::Z,
            orientation * Vec3::Y,
            std::f32::consts::FRAC_PI_3,
            width as f32 / height.max(1) as f32,
            0.01,
            2_000.0,
        )
    }

    fn update_planet_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
        let camera = PlanetPosition::from_meters(self.canonical_camera_m)
            .expect("bounded camera input remains canonical");
        let frame = PlanetFrameUniform::from_camera(self.planet, camera, self.frame_index);
        self.renderer
            .find_pass_mut::<PlanetaryVoxelRenderPass>()
            .expect("planetary pass")
            .set_planet_frame(&self.queue, frame)
            .expect("camera-local planet frame");
    }

    fn update_title(&mut self) {
        if self.last_title_update.elapsed().as_millis() < 250 {
            return;
        }
        self.last_title_update = Instant::now();
        let device = self.device.clone();
        let queue = self.queue.clone();
        let (render, residency, diagnostics) = {
            let pass = self
                .renderer
                .find_pass_mut::<PlanetaryVoxelRenderPass>()
                .expect("planetary pass");
            let diagnostics = pass.poll_diagnostics(&device, &queue);
            (pass.counters(), pass.residency().counters(), diagnostics)
        };
        let lods = diagnostics
            .resident_lods
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let source_generations = match (
            diagnostics.source_generation_min,
            diagnostics.source_generation_max,
        ) {
            (Some(minimum), Some(maximum)) => format!(
                "{}:{}-{}:{}",
                minimum.planet, minimum.page, maximum.planet, maximum.page
            ),
            _ => "-".into(),
        };
        let publication_generations = match (
            diagnostics.publication_generation_min,
            diagnostics.publication_generation_max,
        ) {
            (Some(minimum), Some(maximum)) => format!("{minimum}-{maximum}"),
            _ => "-".into(),
        };
        let camera_delta = [
            self.canonical_camera_m[0] - self.spawn_camera_m[0],
            self.canonical_camera_m[1] - self.spawn_camera_m[1],
            self.canonical_camera_m[2] - self.spawn_camera_m[2],
        ];
        self.window.set_title(&format!(
            "Helio Planet Voxels | cam[{:+.2},{:+.2},{:+.2}]m look[{:+.2},{:+.2}] focus{} grab{} keys{} | R={EARTH_RADIUS_METERS:.0}m 10cm | pages {} lod[{lods}] | src {source_generations} pub {publication_generations} | gpu jobs {}/{} reject s{} o{} i{} | regular V{} I{} D{} | seam V{} I{} D{} | queued {} bp{} rb{}",
            camera_delta[0],
            camera_delta[1],
            camera_delta[2],
            self.yaw,
            self.pitch,
            u8::from(self.window.has_focus()),
            u8::from(self.cursor_grabbed),
            self.keys.len(),
            residency.resident_pages,
            diagnostics.gpu_published_jobs,
            diagnostics.gpu_submitted_jobs,
            diagnostics.gpu_stale_rejections,
            diagnostics.gpu_overflow_rejections,
            diagnostics.gpu_incomplete_rejections,
            diagnostics.regular_vertices,
            diagnostics.regular_indices,
            diagnostics.visible_regular_draws,
            diagnostics.transition_vertices,
            diagnostics.transition_indices,
            diagnostics.visible_transition_draws,
            render.queued_surfaces,
            render.pending_backpressure + u64::from(residency.backpressure_events),
            diagnostics.readback_failures,
        ));
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        event_loop.listen_device_events(DeviceEvents::WhenFocused);
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio Planet Voxels")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
                )
                .expect("planet demo window"),
        );
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("planet demo surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("planet demo GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Helio Planet Voxel Demo Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("planet demo device");
        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("planet demo uncaptured GPU error: {error}");
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let size = window.inner_size();
        let capabilities = surface.get_capabilities(&adapter);
        let surface_format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(capabilities.formats[0]);
        let alpha_mode = capabilities.alpha_modes[0];
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let renderer_config =
            RendererConfig::new(size.width, size.height, surface_format).with_render_scale(1.0);
        let scene = Scene::new(device.clone(), queue.clone());
        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Demo Debug Camera"),
            size: core::mem::size_of::<helio::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Demo Cull Stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let planet_pass = PlanetaryVoxelRenderPass::new(
            &device,
            &queue,
            surface_format,
            PlanetaryVoxelRenderConfig::validation_demo(),
        )
        .expect("bounded planetary render pass");
        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(planet_pass));
        graph.add_pass(Box::new(FxaaPass::new(&device, surface_format)));
        graph.lock(size.width, size.height);
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            renderer_config.surface_format,
            renderer_config.width,
            renderer_config.height,
            renderer_config.render_scale,
            renderer_config,
            scene,
            graph,
            debug_state,
            debug_camera_buffer,
            cull_stats_buffer,
        );
        renderer.set_jitter_enabled(false);

        let planet = PlanetId(*b"HELIO-EARTH-DEMO");
        let (canonical_camera_m, pages) = build_earth_radius_patch(planet);
        {
            let pass = renderer
                .find_pass_mut::<PlanetaryVoxelRenderPass>()
                .expect("planetary pass");
            let camera = PlanetPosition::from_meters(canonical_camera_m).unwrap();
            pass.set_planet_frame(&queue, PlanetFrameUniform::from_camera(planet, camera, 1))
                .unwrap();
            pass.apply_upload_batch(
                &device,
                &queue,
                pages.iter().map(|page| page.page_upload.clone()).collect(),
            )
            .unwrap();
            pass.apply_visible_set(
                &queue,
                VisiblePageSet {
                    frame_index: 1,
                    pages: pages
                        .iter()
                        .map(|page| VisiblePage {
                            key: page.page_upload.key,
                            generation: page.page_upload.generation,
                            transition_mask: page.surface.transition_mask,
                        })
                        .collect(),
                },
            )
            .unwrap();
            for page in pages {
                pass.queue_surface(page.surface).unwrap();
            }
        }

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            alpha_mode,
            renderer,
            planet,
            canonical_camera_m,
            spawn_camera_m: canonical_camera_m,
            frame_index: 1,
            yaw: INITIAL_YAW,
            pitch: INITIAL_PITCH,
            keys: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            cursor_grabbed: false,
            last_frame: Instant::now(),
            last_title_update: Instant::now(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::Focused(false) => {
                state.reset_transient_input();
            }
            WindowEvent::Focused(true) => state.last_frame = Instant::now(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if state.cursor_grabbed {
                    state.reset_transient_input();
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: key_state,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } => match key_state {
                ElementState::Pressed => {
                    state.keys.insert(key);
                }
                ElementState::Released => {
                    state.keys.remove(&key);
                }
            },
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !state.cursor_grabbed => {
                state.window.focus_window();
                let grabbed = state
                    .window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                    .is_ok();
                if grabbed {
                    state.cursor_grabbed = true;
                    state.window.set_cursor_visible(false);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f64().min(0.05);
                state.last_frame = now;
                state.update(dt);
                state.update_planet_frame();
                state.update_title();
                let size = state.window.inner_size();
                if size.width == 0 || size.height == 0 {
                    return;
                }
                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    _ => return,
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(error) = state
                    .renderer
                    .render(&state.camera(size.width, size.height), &view)
                {
                    log::error!("planet render error: {error:?}");
                }
                state.queue.present(output);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta: (x, y) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += x as f32;
                state.mouse_delta.1 += y as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn advance_camera(position_m: &mut [f64; 3], keys: &HashSet<KeyCode>, orientation: Quat, dt: f64) {
    let look_forward = orientation * -Vec3::Z;
    let look_right = orientation * Vec3::X;
    let forward = Vec3::new(look_forward.x, 0.0, look_forward.z).normalize_or_zero();
    let right = Vec3::new(look_right.x, 0.0, look_right.z).normalize_or_zero();
    let mut direction = Vec3::ZERO;
    if keys.contains(&KeyCode::KeyW) {
        direction += forward;
    }
    if keys.contains(&KeyCode::KeyS) {
        direction -= forward;
    }
    if keys.contains(&KeyCode::KeyA) {
        direction -= right;
    }
    if keys.contains(&KeyCode::KeyD) {
        direction += right;
    }
    if keys.contains(&KeyCode::Space) {
        direction += Vec3::Y;
    }
    if keys.contains(&KeyCode::ShiftLeft) {
        direction -= Vec3::Y;
    }
    let step = direction.normalize_or_zero() * (MOVE_SPEED_METERS_PER_SECOND * dt) as f32;
    for axis in 0..3 {
        position_m[axis] += f64::from(step[axis]);
    }
}

struct DemoPage {
    page_upload: PageUpload,
    surface: PlanetarySurfaceUpload,
}

fn build_earth_radius_patch(planet: PlanetId) -> ([f64; 3], Vec<DemoPage>) {
    let radius_cell = (EARTH_RADIUS_METERS / LOD0_CELL_SIZE_METERS) as i64;
    let coarse_page_x = radius_cell.div_euclid(PAGE_EDGE_CELLS * 2);
    let coarse = PageKey::new(1, [coarse_page_x, -1, -1]);
    let fine_x = coarse_page_x * 2 + 2;
    let pages = [
        (coarse, TransitionFace::PositiveX.bit()),
        (PageKey::new(0, [fine_x, -2, -2]), 0),
        (PageKey::new(0, [fine_x, -1, -2]), 0),
        (PageKey::new(0, [fine_x, -2, -1]), 0),
        (PageKey::new(0, [fine_x, -1, -1]), 0),
    ];
    let demo_pages = pages
        .into_iter()
        .enumerate()
        .map(|(index, (page, transition_mask))| {
            let fixture = ExtractionFixture::new(ExtractionFixtureKind::Plane, page).unwrap();
            let generation = SourceGeneration::new(1, index as u64 + 1);
            let key = PlanetPageKey::new(planet, page);
            let cells = inner_page_cells(&fixture);
            let transition_face_slabs = if page.lod == 0 {
                vec![CellWord::AIR; TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT].into_boxed_slice()
            } else {
                TransitionFace::ALL
                    .into_iter()
                    .flat_map(|face| {
                        TransvoxelTransitionFaceFixture::new(
                            ExtractionFixtureKind::Plane,
                            page,
                            face,
                        )
                        .unwrap()
                        .slab_samples()
                        .into_vec()
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            };
            DemoPage {
                page_upload: PageUpload::new(key, generation, cells).unwrap(),
                surface: PlanetarySurfaceUpload {
                    key,
                    generation,
                    halo_samples: fixture.samples().to_vec().into_boxed_slice(),
                    transition_face_slabs,
                    transition_mask,
                    dirty_microbricks: fixture.metrics().active_microbrick_mask,
                },
            }
        })
        .collect();
    let coarse_min = coarse.lod0_cell_min().unwrap();
    let canonical_camera_m = [
        (coarse_min[0] + 54) as f64 * LOD0_CELL_SIZE_METERS,
        15.0 * LOD0_CELL_SIZE_METERS,
        (coarse_min[2] + 32) as f64 * LOD0_CELL_SIZE_METERS,
    ];
    (canonical_camera_m, demo_pages)
}

fn inner_page_cells(fixture: &ExtractionFixture) -> Vec<CellWord> {
    let mut cells = Vec::with_capacity(PAGE_EDGE * PAGE_EDGE * PAGE_EDGE);
    for z in 0..PAGE_EDGE as i32 {
        for y in 0..PAGE_EDGE as i32 {
            for x in 0..PAGE_EDGE as i32 {
                cells.push(fixture.sample([x, y, z]).unwrap());
            }
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_camera_ray_intersects_the_fine_patch() {
        let planet = PlanetId(*b"HELIO-EARTH-DEMO");
        let (camera, pages) = build_earth_radius_patch(planet);
        let fine = pages
            .iter()
            .find(|page| {
                page.page_upload.key.page.lod == 0 && page.page_upload.key.page.page_xyz[1] == -1
            })
            .expect("validation patch includes a surface-bearing fine page")
            .page_upload
            .key
            .page;
        let fine_min = fine.lod0_cell_min().unwrap();
        let fine_min_x = fine_min[0] as f64 * LOD0_CELL_SIZE_METERS;
        let fine_max_x = fine_min_x + PAGE_EDGE_CELLS as f64 * LOD0_CELL_SIZE_METERS;
        let forward = Quat::from_euler(EulerRot::YXZ, INITIAL_YAW, INITIAL_PITCH, 0.0) * -Vec3::Z;
        assert!(forward.x > 0.0 && forward.y < 0.0);
        let surface_y = -LOD0_CELL_SIZE_METERS;
        let distance = (surface_y - camera[1]) / f64::from(forward.y);
        let intersection_x = camera[0] + f64::from(forward.x) * distance;
        assert!(
            (fine_min_x..=fine_max_x).contains(&intersection_x),
            "camera ray intersects x={intersection_x}, outside fine patch {fine_min_x}..={fine_max_x}"
        );

        let planet_camera = PlanetPosition::from_meters(camera).unwrap();
        let frame = PlanetFrameUniform::from_camera(planet, planet_camera, 1);
        let metadata = helio_planet_voxel_core::GpuPageMeta::new(
            fine,
            frame.frame_origin_lod0_cell(),
            0,
            1,
            0,
        )
        .unwrap();
        let local_surface = [16.0, 31.0, 16.0];
        let world = metadata.camera_local_position_m(frame, local_surface);
        let orientation = Quat::from_euler(EulerRot::YXZ, INITIAL_YAW, INITIAL_PITCH, 0.0);
        let camera_uniform = Camera::perspective_look_at(
            Vec3::ZERO,
            orientation * -Vec3::Z,
            orientation * Vec3::Y,
            std::f32::consts::FRAC_PI_3,
            1280.0 / 720.0,
            0.01,
            2_000.0,
        );
        let clip = camera_uniform.proj * camera_uniform.view * Vec3::from_array(world).extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(
            clip.w > 0.0
                && ndc.x.abs() <= 1.0
                && ndc.y.abs() <= 1.0
                && (0.0..=1.0).contains(&ndc.z),
            "surface vertex world={world:?} projects to clip={clip:?}, ndc={ndc:?}"
        );
    }

    #[test]
    fn movement_is_horizontal_and_uses_explicit_vertical_keys() {
        let orientation = Quat::from_euler(EulerRot::YXZ, INITIAL_YAW, INITIAL_PITCH, 0.0);
        let mut position = [0.0; 3];
        let mut keys = HashSet::from([KeyCode::KeyW]);
        advance_camera(&mut position, &keys, orientation, 1.0);
        assert!((position[0] - MOVE_SPEED_METERS_PER_SECOND).abs() < 1.0e-5);
        assert_eq!(position[1], 0.0);
        assert!(position[2].abs() < 1.0e-5);

        position = [0.0; 3];
        keys = HashSet::from([KeyCode::Space]);
        advance_camera(&mut position, &keys, orientation, 1.0);
        assert_eq!(position, [0.0, MOVE_SPEED_METERS_PER_SECOND, 0.0]);
    }
}
