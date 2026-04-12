//! Editor Demo — Helio v3
//!
//! Demonstrates the editor API: object selection via ray-picking and
//! transform gizmo overlay (translate / rotate / scale).
//!
//! # Controls
//!
//! | Input | Action |
//! |-------|--------|
//! | Right-click | Grab cursor for free-fly camera |
//! | Escape | Release cursor / deselect / exit |
//! | WASD | Fly forward/left/back/right (cursor grabbed) |
//! | Space / Shift-L | Fly up / down (cursor grabbed) |
//! | Left-click | Pick object under cursor (cursor *not* grabbed) |
//! | G | Translate gizmo |
//! | R | Rotate gizmo |
//! | S | Scale gizmo |
//! | Tab | Toggle editor grid |
//!
//! **Selected object** is highlighted with a yellow wireframe sphere and
//! the active transform gizmo drawn on top:
//!
//! * **Translate** — red/green/blue arrows (+X/+Y/+Z)
//! * **Rotate**    — red/green/blue rings (YZ / XZ / XY planes)
//! * **Scale**     — red/green/blue axes with cube end-caps

mod v3_demo_common;

use helio::{
    Camera, EditorState, GizmoMode, Movability, Renderer, RendererConfig,
    SceneActor, required_wgpu_features, required_wgpu_limits,
};
use v3_demo_common::{
    box_mesh, cube_mesh, insert_object_with_movability, make_material, plane_mesh, point_light,
    sphere_mesh,
};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("run");
}

// ─────────────────────────────────────────────────────────────────────────────
// App scaffold
// ─────────────────────────────────────────────────────────────────────────────

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,

    // ── Camera ────────────────────────────────────────────────────────────────
    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
    /// Last known cursor position in logical pixels (for ray picking).
    cursor_pos: (f32, f32),

    // ── Editor ───────────────────────────────────────────────────────────────
    editor: EditorState,
    /// Whether the grid overlay is visible.
    grid_enabled: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        // ── Window & wgpu setup ───────────────────────────────────────────────
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio — Editor Demo  (G=Translate  R=Rotate  S=Scale)")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits: required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
        ))
        .expect("device");
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU] {:?}", e);
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let sz = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: sz.width,
                height: sz.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        // ── Renderer ──────────────────────────────────────────────────────────
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(sz.width, sz.height, format),
        );
        renderer.set_editor_mode(true);
        renderer.set_clear_color([0.08, 0.09, 0.12, 1.0]);
        renderer.set_ambient([0.12, 0.14, 0.18], 0.25);

        // ── Scene geometry ────────────────────────────────────────────────────
        // Materials
        let mat_floor = renderer
            .scene_mut()
            .insert_material(make_material([0.55, 0.55, 0.55, 1.0], 0.8, 0.0, [0.0; 3], 0.0));
        let mat_red = renderer
            .scene_mut()
            .insert_material(make_material([0.9, 0.15, 0.15, 1.0], 0.5, 0.0, [0.0; 3], 0.0));
        let mat_green = renderer
            .scene_mut()
            .insert_material(make_material([0.15, 0.85, 0.25, 1.0], 0.5, 0.0, [0.0; 3], 0.0));
        let mat_blue = renderer
            .scene_mut()
            .insert_material(make_material([0.15, 0.35, 0.95, 1.0], 0.5, 0.0, [0.0; 3], 0.0));
        let mat_gold = renderer
            .scene_mut()
            .insert_material(make_material([1.0, 0.76, 0.1, 1.0], 0.3, 0.8, [0.0; 3], 0.0));
        let mat_sphere = renderer
            .scene_mut()
            .insert_material(make_material([0.8, 0.5, 0.9, 1.0], 0.35, 0.15, [0.0; 3], 0.0));

        // Meshes
        let floor_mesh = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(plane_mesh([0.0; 3], 8.0)))
            .as_mesh()
            .unwrap();
        let box_a = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(box_mesh([0.0; 3], [0.55, 0.55, 0.55])))
            .as_mesh()
            .unwrap();
        let box_b = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(box_mesh([0.0; 3], [0.4, 0.75, 0.4])))
            .as_mesh()
            .unwrap();
        let box_c = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(box_mesh([0.0; 3], [0.6, 0.35, 0.6])))
            .as_mesh()
            .unwrap();
        let cube_gold = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(cube_mesh([0.0; 3], 0.45)))
            .as_mesh()
            .unwrap();
        let sphere_a = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(sphere_mesh([0.0; 3], 0.65)))
            .as_mesh()
            .unwrap();

        // Floor (Static — not selectable for transform, but still visible)
        let _ = insert_object_with_movability(
            &mut renderer,
            floor_mesh,
            mat_floor,
            glam::Mat4::IDENTITY,
            8.5,
            None, // Static
        );

        // Pickable / movable objects
        insert_object_with_movability(
            &mut renderer,
            box_a,
            mat_red,
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, 0.55, 0.5)),
            1.0,
            Some(Movability::Movable),
        )
        .expect("red box");

        insert_object_with_movability(
            &mut renderer,
            box_b,
            mat_green,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.75, -1.0)),
            1.0,
            Some(Movability::Movable),
        )
        .expect("green box");

        insert_object_with_movability(
            &mut renderer,
            box_c,
            mat_blue,
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.35, 0.5)),
            0.85,
            Some(Movability::Movable),
        )
        .expect("blue box");

        insert_object_with_movability(
            &mut renderer,
            cube_gold,
            mat_gold,
            glam::Mat4::from_rotation_y(0.6) * glam::Mat4::from_translation(glam::Vec3::new(0.5, 0.45, 2.5)),
            0.75,
            Some(Movability::Movable),
        )
        .expect("gold cube");

        insert_object_with_movability(
            &mut renderer,
            sphere_a,
            mat_sphere,
            glam::Mat4::from_translation(glam::Vec3::new(-1.0, 0.65, -2.5)),
            0.85,
            Some(Movability::Movable),
        )
        .expect("sphere");

        // ── Lights ────────────────────────────────────────────────────────────
        renderer
            .scene_mut()
            .insert_actor(SceneActor::light(point_light(
                [0.0, 4.5, 2.0],
                [1.0, 0.85, 0.7],
                14.0,
                12.0,
            )));
        renderer
            .scene_mut()
            .insert_actor(SceneActor::light(point_light(
                [-4.0, 3.0, -3.0],
                [0.4, 0.55, 1.0],
                8.0,
                9.0,
            )));
        renderer
            .scene_mut()
            .insert_actor(SceneActor::light(point_light(
                [4.0, 2.5, -2.0],
                [1.0, 0.4, 0.3],
                6.0,
                8.0,
            )));

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 4.0, 9.5),
            cam_yaw: 0.0,
            cam_pitch: -0.35,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            cursor_pos: (640.0, 360.0),
            editor: EditorState::new(),
            grid_enabled: true,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(sz) if sz.width > 0 && sz.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: sz.width,
                        height: sz.height,
                        present_mode: wgpu::PresentMode::AutoVsync,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    },
                );
                state.renderer.set_render_size(sz.width, sz.height);
            }

            WindowEvent::CursorMoved { position, .. } => {
                state.cursor_pos = (position.x as f32, position.y as f32);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ks,
                        ..
                    },
                ..
            } => {
                match ks {
                    ElementState::Pressed => {
                        state.keys.insert(code);
                        match code {
                            KeyCode::Escape => {
                                if state.editor.selected().is_some() {
                                    state.editor.deselect();
                                } else if state.cursor_grabbed {
                                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                                    state.window.set_cursor_visible(true);
                                    state.cursor_grabbed = false;
                                } else {
                                    event_loop.exit();
                                }
                            }
                            KeyCode::KeyG => state.editor.set_gizmo_mode(GizmoMode::Translate),
                            KeyCode::KeyR => state.editor.set_gizmo_mode(GizmoMode::Rotate),
                            KeyCode::KeyS if !state.cursor_grabbed => {
                                state.editor.set_gizmo_mode(GizmoMode::Scale)
                            }
                            KeyCode::Tab => {
                                state.grid_enabled = !state.grid_enabled;
                                state.renderer.set_editor_mode(state.grid_enabled);
                            }
                            _ => {}
                        }
                    }
                    ElementState::Released => {
                        state.keys.remove(&code);
                    }
                }
            }

            // Right-click → grab cursor for flying.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                if !state.cursor_grabbed {
                    if state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok()
                    {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }

            // Left-click → pick object (only when cursor is free).
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    let sz = state.window.inner_size();
                    let width = sz.width as f32;
                    let height = sz.height as f32;

                    // Rebuild the camera to get the current view-proj.
                    let (sy, cy) = state.cam_yaw.sin_cos();
                    let (sp, cp) = state.cam_pitch.sin_cos();
                    let fwd = glam::Vec3::new(sy * cp, sp, -cy * cp);
                    let aspect = width / height.max(1.0);
                    let proj = glam::Mat4::perspective_rh(
                        std::f32::consts::FRAC_PI_4,
                        aspect,
                        0.1,
                        500.0,
                    );
                    let view = glam::Mat4::look_at_rh(
                        state.cam_pos,
                        state.cam_pos + fwd,
                        glam::Vec3::Y,
                    );
                    let vp_inv = (proj * view).inverse();

                    let (ray_o, ray_d) = EditorState::ray_from_screen(
                        state.cursor_pos.0,
                        state.cursor_pos.1,
                        width,
                        height,
                        vp_inv,
                    );
                    state.editor.pick(state.renderer.scene(), ray_o, ray_d);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.update_camera(dt);
                state.render();
                state.window.request_redraw();
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(s) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if s.cursor_grabbed {
                s.mouse_delta.0 += delta.0 as f32;
                s.mouse_delta.1 += delta.1 as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-frame update & render
// ─────────────────────────────────────────────────────────────────────────────

impl AppState {
    fn update_camera(&mut self, dt: f32) {
        const LOOK: f32 = 0.0025;
        const MOVE: f32 = 6.0;

        if self.cursor_grabbed {
            self.cam_yaw += self.mouse_delta.0 * LOOK;
            self.cam_pitch -= self.mouse_delta.1 * LOOK;
            self.cam_pitch = self
                .cam_pitch
                .clamp(-std::f32::consts::FRAC_PI_2 * 0.99, std::f32::consts::FRAC_PI_2 * 0.99);
        }
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let fwd = glam::Vec3::new(sy, 0.0, -cy);
        let right = glam::Vec3::new(cy, 0.0, sy);

        let mut vel = glam::Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) { vel += fwd; }
        if self.keys.contains(&KeyCode::KeyS) { vel -= fwd; }
        if self.keys.contains(&KeyCode::KeyD) { vel += right; }
        if self.keys.contains(&KeyCode::KeyA) { vel -= right; }
        if self.keys.contains(&KeyCode::Space) { vel += glam::Vec3::Y; }
        if self.keys.contains(&KeyCode::ShiftLeft) { vel -= glam::Vec3::Y; }
        if vel.length_squared() > 0.0 {
            self.cam_pos += vel.normalize() * MOVE * dt;
        }
    }

    fn render(&mut self) {
        let sz = self.window.inner_size();
        let aspect = sz.width as f32 / sz.height.max(1) as f32;

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd = glam::Vec3::new(sy * cp, sp, -cy * cp);

        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + fwd,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            500.0,
        );

        // ── Gizmo overlay (uses debug draw, cleared each frame) ───────────────
        self.renderer.debug_clear();
        self.editor.draw_gizmos(&mut self.renderer);

        // Draw a small cross at the screen centre when cursor is free (pick aid).
        if !self.cursor_grabbed {
            let hit_point = self.cam_pos + fwd * 0.1;
            let r = 0.004;
            self.renderer.debug_line(
                (hit_point - glam::Vec3::X * r).to_array(),
                (hit_point + glam::Vec3::X * r).to_array(),
                [1.0, 1.0, 1.0, 0.7],
            );
            self.renderer.debug_line(
                (hit_point - glam::Vec3::Y * r).to_array(),
                (hit_point + glam::Vec3::Y * r).to_array(),
                [1.0, 1.0, 1.0, 0.7],
            );
        }

        // ── Present ───────────────────────────────────────────────────────────
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => { log::warn!("surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("render: {:?}", e);
        }
        output.present();
    }
}
