//! Editor Demo — Helio v3
//!
//! Demonstrates the editor API: object selection via BVH ray-picking and
//! transform gizmo overlay (translate / rotate / scale).
//!
//! # Controls
//!
//! | Input           | Action                                        |
//! |-----------------|-----------------------------------------------|
//! | **Right-click hold** | Capture cursor for free-fly camera       |
//! | **Right-click release** | Release cursor for object picking      |
//! | WASD            | Fly forward / left / back / right (hold RMB)  |
//! | Space / L-Shift | Fly up / down (hold RMB)                      |
//! | **Left-click**  | Pick object under cursor (cursor free)        |
//! | G               | Switch to **Translate** gizmo (cursor free)   |
//! | R               | Switch to **Rotate** gizmo (cursor free)      |
//! | S               | Switch to **Scale** gizmo (cursor free)       |
//! | Ctrl+D          | **Duplicate** selected object                 |
//! | Delete          | **Delete** selected object                    |
//! | Tab             | Toggle editor grid                            |
//! | **F11**         | Toggle fullscreen                             |
//! | **Alt+Enter**   | Toggle fullscreen                             |
//! | Escape          | Deselect → exit                               |
//!
//! Picking uses a two-phase BVH ray caster (broad-phase AABB + per-mesh
//! Möller-Trumbore triangle intersection) — no sphere approximations.


mod v3_demo_common;

use helio::{
    Camera, EditorState, GizmoMode, Movability, Renderer, RendererConfig,
    SceneActor, ScenePicker, required_wgpu_features, required_wgpu_limits,
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
    /// True while the right mouse button is held — cursor is grabbed and camera
    /// is in fly mode.  Released to allow free-cursor object picking.
    right_mouse_held: bool,
    mouse_delta: (f32, f32),
    /// Last known cursor position in logical pixels (only valid when cursor is free).
    cursor_pos: (f32, f32),

    // ── Editor ───────────────────────────────────────────────────────────────
    editor: EditorState,
    /// BVH-accelerated ray picker; rebuilt once after scene construction.
    picker: ScenePicker,
    /// Whether the grid overlay is visible.
    grid_enabled: bool,
    /// True while the window is in borderless-fullscreen mode.
    is_fullscreen: bool,
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
        // ScenePicker is built alongside the scene so mesh BVHs are registered
        // with the same upload data consumed by the renderer.
        let mut picker = ScenePicker::new();

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

        // Meshes — clone each upload so we can register it with the picker
        // (the scene takes ownership of the original, picker keeps the clone).
        let floor_upload = plane_mesh([0.0; 3], 8.0);
        let floor_mesh = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(floor_upload.clone()))
            .as_mesh()
            .unwrap();
        picker.register_mesh(floor_mesh, &floor_upload);

        let box_a_upload = box_mesh([0.0; 3], [0.55, 0.55, 0.55]);
        let box_a = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(box_a_upload.clone()))
            .as_mesh()
            .unwrap();
        picker.register_mesh(box_a, &box_a_upload);

        let box_b_upload = box_mesh([0.0; 3], [0.4, 0.75, 0.4]);
        let box_b = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(box_b_upload.clone()))
            .as_mesh()
            .unwrap();
        picker.register_mesh(box_b, &box_b_upload);

        let box_c_upload = box_mesh([0.0; 3], [0.6, 0.35, 0.6]);
        let box_c = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(box_c_upload.clone()))
            .as_mesh()
            .unwrap();
        picker.register_mesh(box_c, &box_c_upload);

        let cube_gold_upload = cube_mesh([0.0; 3], 0.45);
        let cube_gold = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(cube_gold_upload.clone()))
            .as_mesh()
            .unwrap();
        picker.register_mesh(cube_gold, &cube_gold_upload);

        let sphere_a_upload = sphere_mesh([0.0; 3], 0.65);
        let sphere_a = renderer
            .scene_mut()
            .insert_actor(SceneActor::mesh(sphere_a_upload.clone()))
            .as_mesh()
            .unwrap();
        picker.register_mesh(sphere_a, &sphere_a_upload);

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

        // Sync the picker with all just-inserted objects.
        picker.rebuild_instances(renderer.scene());

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
            right_mouse_held: false,
            mouse_delta: (0.0, 0.0),
            cursor_pos: (640.0, 360.0),
            editor: EditorState::new(),
            picker,
            grid_enabled: true,
            is_fullscreen: false,
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
                // Update hover highlight and, if dragging, apply drag.
                if !state.right_mouse_held {
                    let (ray_o, ray_d) = state.build_ray();
                    state.editor.update_hover(ray_o, ray_d, state.renderer.scene());
                    if state.editor.is_dragging() {
                        state.editor.update_drag(ray_o, ray_d, state.renderer.scene_mut());
                    }
                }
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
                            // ── Fullscreen toggle ────────────────────────────
                            KeyCode::F11 => state.toggle_fullscreen(),
                            KeyCode::Enter | KeyCode::NumpadEnter
                                if state.keys.contains(&KeyCode::AltLeft)
                                    || state.keys.contains(&KeyCode::AltRight) =>
                            {
                                state.toggle_fullscreen();
                            }
                            // ─────────────────────────────────────────────────
                            KeyCode::Escape => {
                                if state.editor.selected().is_some() {
                                    state.editor.deselect();
                                } else {
                                    event_loop.exit();
                                }
                            }
                            KeyCode::Delete if !state.right_mouse_held => {
                                if state.editor.delete_selected(state.renderer.scene_mut()) {
                                    state.picker.rebuild_instances(state.renderer.scene());
                                }
                            }
                            // Gizmo keys only fire when the cursor is free
                            // (not flying the camera) to avoid clashes with WASD.
                            KeyCode::KeyG if !state.right_mouse_held => {
                                state.editor.set_gizmo_mode(GizmoMode::Translate)
                            }
                            KeyCode::KeyR if !state.right_mouse_held => {
                                state.editor.set_gizmo_mode(GizmoMode::Rotate)
                            }
                            KeyCode::KeyS if !state.right_mouse_held => {
                                state.editor.set_gizmo_mode(GizmoMode::Scale)
                            }
                            KeyCode::KeyD
                                if !state.right_mouse_held
                                && (state.keys.contains(&KeyCode::ControlLeft)
                                    || state.keys.contains(&KeyCode::ControlRight)) => {
                                if let Some(_new_id) = state.editor.duplicate_selected(&mut state.renderer) {
                                    state.picker.rebuild_instances(state.renderer.scene());
                                }
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

            // Right-click PRESS → grab cursor and enter fly mode.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                if !state.right_mouse_held {
                    let _ = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked));
                    state.window.set_cursor_visible(false);
                    state.right_mouse_held = true;
                }
            }

            // Right-click RELEASE → restore cursor for object picking.
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Right,
                ..
            } => {
                let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                state.window.set_cursor_visible(true);
                state.right_mouse_held = false;
            }

            // Left-click → try to drag a gizmo handle, else pick an object.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.right_mouse_held {
                    let (ray_o, ray_d) = state.build_ray();

                    // Gizmo drag has priority over scene picking.
                    if !state.editor.try_start_drag(ray_o, ray_d, state.renderer.scene()) {
                        // BVH ray cast — exact triangle intersection.
                        state.picker.rebuild_instances(state.renderer.scene());
                        if let Some(hit) = state.picker.cast_ray(ray_o, ray_d) {
                            state.editor.select(hit.object_id);
                        } else {
                            state.editor.deselect();
                        }
                    }
                }
            }

            // Left-click RELEASE → finish any active gizmo drag.
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                state.editor.end_drag();
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
            if s.right_mouse_held {
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
    /// Compute a world-space ray from the current cursor position.
    fn build_ray(&self) -> (glam::Vec3, glam::Vec3) {
        let sz     = self.window.inner_size();
        let width  = sz.width as f32;
        let height = sz.height as f32;
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd    = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let aspect = width / height.max(1.0);
        let proj   = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 500.0);
        let view   = glam::Mat4::look_at_rh(self.cam_pos, self.cam_pos + fwd, glam::Vec3::Y);
        let vp_inv = (proj * view).inverse();
        EditorState::ray_from_screen(self.cursor_pos.0, self.cursor_pos.1, width, height, vp_inv)
    }

    fn update_camera(&mut self, dt: f32) {
        const LOOK: f32 = 0.0025;
        const MOVE: f32 = 6.0;

        // Only rotate and move when the right mouse button is held.
        if self.right_mouse_held {
            self.cam_yaw += self.mouse_delta.0 * LOOK;
            self.cam_pitch -= self.mouse_delta.1 * LOOK;
            self.cam_pitch = self
                .cam_pitch
                .clamp(-std::f32::consts::FRAC_PI_2 * 0.99, std::f32::consts::FRAC_PI_2 * 0.99);

            let (sy, cy) = self.cam_yaw.sin_cos();
            let fwd = glam::Vec3::new(sy, 0.0, -cy);
            let right = glam::Vec3::new(cy, 0.0, sy);

            let mut vel = glam::Vec3::ZERO;
            if self.keys.contains(&KeyCode::KeyW) { vel += fwd; }
            if self.keys.contains(&KeyCode::KeyS) { vel -= fwd; }
            if self.keys.contains(&KeyCode::KeyD) { vel += right; }
            if self.keys.contains(&KeyCode::KeyA) { vel -= right; }
            if self.keys.contains(&KeyCode::Space)     { vel += glam::Vec3::Y; }
            if self.keys.contains(&KeyCode::ShiftLeft) { vel -= glam::Vec3::Y; }
            if vel.length_squared() > 0.0 {
                self.cam_pos += vel.normalize() * MOVE * dt;
            }
        }
        self.mouse_delta = (0.0, 0.0);
    }

    /// Toggle between borderless fullscreen and windowed mode.
    ///
    /// On Windows this also calls `request_exclusive_fullscreen` so DXGI can
    /// use a direct hardware flip, bypassing DWM composition.
    fn toggle_fullscreen(&mut self) {
        use winit::window::Fullscreen;
        if self.is_fullscreen {
            self.window.set_fullscreen(None);
            self.is_fullscreen = false;
        } else {
            // Borderless fullscreen — covers the current monitor without a
            // mode switch, avoiding the flicker of exclusive fullscreen while
            // still allowing DXGI hardware flips (see request_exclusive_fullscreen).
            let monitor = self.window.current_monitor();
            self.window.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
            // Lift DXGI's window-association locks so the driver can flip directly.
            #[cfg(target_os = "windows")]
            {
                use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
                if let Ok(handle) = self.window.window_handle() {
                    if let RawWindowHandle::Win32(h) = handle.as_raw() {
                        let hwnd = h.hwnd.get() as *mut std::ffi::c_void;
                        // SAFETY: hwnd is valid for the lifetime of this window.
                        unsafe { self.renderer.request_exclusive_fullscreen(hwnd); }
                    }
                }
            }
            self.is_fullscreen = true;
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

        // Draw a small crosshair at the screen centre when cursor is free (pick aid).
        if !self.right_mouse_held {
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
