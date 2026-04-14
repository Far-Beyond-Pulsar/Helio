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
use helio_asset_compat::{load_scene_bytes_with_config, upload_sectioned_scene, LoadConfig};
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

const CRATES_FBX: &[u8] = include_bytes!("../../models/source/container with textures.fbx");

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
    /// Fly-mode movement speed in units/second. Scroll wheel adjusts this.
    cam_speed: f32,

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

        // ── Load Crates.fbx ───────────────────────────────────────────────────
        const CRATES_TARGET: glam::Vec3 = glam::Vec3::new(3.5, 0.0, -2.0);

        // Debug sphere — bright orange marker so we can confirm the target
        // location is actually in view.  Remove once placement is verified.
        {
            let dbg_upload = sphere_mesh(CRATES_TARGET.to_array(), 0.35);
            let dbg_mesh = renderer
                .scene_mut()
                .insert_actor(SceneActor::mesh(dbg_upload.clone()))
                .as_mesh()
                .unwrap();
            picker.register_mesh(dbg_mesh, &dbg_upload);
            let mat_dbg = renderer.scene_mut().insert_material(make_material(
                [1.0, 0.4, 0.05, 1.0], // bright orange
                0.3,
                0.0,
                [0.8, 0.3, 0.0], // emissive so it's visible even without lighting
                0.6,
            ));
            insert_object_with_movability(
                &mut renderer,
                dbg_mesh,
                mat_dbg,
                glam::Mat4::IDENTITY,
                0.5,
                Some(Movability::Movable),
            )
            .ok();
        }

        {
            let crates_base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("models/source");
            match load_scene_bytes_with_config(
                CRATES_FBX,
                "fbx",
                Some(crates_base.as_path()),
                LoadConfig::default()
                    .with_uv_flip(false)
                    .with_merge_meshes(true)
                    .with_import_scale(glam::Vec3::splat(1.0 / 20.0)),
            ) {
                Ok(scene) => {
                    match upload_sectioned_scene(&mut renderer, &scene) {
                        Ok((multi_mesh_id, section_mat_ids)) => {
                            let sm = scene.sectioned_mesh.as_ref().unwrap();
                            // Node transforms are already baked into the shared vertex buffer,
                            // so the mesh lives at IDENTITY in its own space.  Compute a
                            // local-space AABB and then translate it to CRATES_TARGET.
                            let mut bb_min = glam::Vec3::splat(f32::INFINITY);
                            let mut bb_max = glam::Vec3::splat(f32::NEG_INFINITY);
                            for v in &sm.vertices {
                                let p = glam::Vec3::from(v.position);
                                bb_min = bb_min.min(p);
                                bb_max = bb_max.max(p);
                            }
                            let local_center = (bb_min + bb_max) * 0.5;
                            let radius = ((bb_max - bb_min) * 0.5).length().max(0.5);
                            let placement =
                                glam::Mat4::from_translation(CRATES_TARGET - local_center);
                            let world_center = placement.transform_point3(local_center);
                            eprintln!(
                                "[editor_demo] sectioned mesh: {} sections, {} verts, r={radius:.2}, world_center={world_center:.2?}",
                                sm.sections.len(),
                                sm.vertices.len()
                            );
                            match renderer.scene_mut().insert_sectioned_object(
                                multi_mesh_id,
                                &section_mat_ids,
                                placement,
                                [world_center.x, world_center.y, world_center.z, radius],
                                Some(Movability::Movable),
                            ) {
                                Ok(_) => {
                                    eprintln!("[editor_demo] sectioned mesh inserted ok");
                                    // Register each section's geometry with the picker so that
                                    // ray-cast hits land on the mesh instead of passing through it.
                                    if let Some(section_ids) = renderer
                                        .scene()
                                        .sectioned_section_mesh_ids(multi_mesh_id)
                                    {
                                        let section_ids: Vec<_> = section_ids.to_vec();
                                        for (section_mesh_id, sec) in
                                            section_ids.iter().zip(sm.sections.iter())
                                        {
                                            picker.register_mesh(
                                                *section_mesh_id,
                                                &helio::MeshUpload {
                                                    vertices: sm.vertices.clone(),
                                                    indices: sec.indices.clone(),
                                                },
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("[editor_demo] sectioned mesh INSERT FAILED: {e:?}")
                                }
                            }
                        }
                        Err(e) => eprintln!("[editor_demo] upload_sectioned_scene ERR: {e}"),
                    }
                }
                Err(e) => eprintln!("[editor_demo] Failed to load Crates.fbx: {e}"),
            }
        }

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
            cam_speed: 6.0,
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
                            // Spawn a powerful blue point light at the camera position.
                            KeyCode::KeyL if !state.right_mouse_held => {
                                let pos = state.cam_pos.to_array();
                                state.renderer.scene_mut().insert_actor(
                                    SceneActor::light(point_light(pos, [0.2, 0.5, 1.0], 500.0, 150.0))
                                );
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
                        if let Some(hit) = state.picker.cast_ray(state.renderer.scene(), ray_o, ray_d) {
                            state.editor.select(hit.actor_id);
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

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(state) = self.state.as_mut() {
                    let lines = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                    };
                    // Each scroll notch multiplies/divides speed by ~1.15.
                    state.cam_speed = (state.cam_speed * 1.15_f32.powf(lines))
                        .clamp(0.5, 500.0);
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
                self.cam_pos += vel.normalize() * self.cam_speed * dt;
            }
        }
        self.mouse_delta = (0.0, 0.0);
    }

    /// Toggle between borderless fullscreen and true DXGI exclusive fullscreen.
    ///
    /// On Windows, entering fullscreen:
    ///   1. Moves the window to borderless fullscreen (no mode change = no flicker).
    ///   2. Calls `SetFullscreenState(TRUE)` on the DXGI swap chain, which
    ///      transitions the display into true hardware-exclusive mode and
    ///      completely bypasses DWM composition.
    ///
    /// On exit, `SetFullscreenState(FALSE)` is called first (DXGI requirement)
    /// before the window returns to windowed mode.
    fn toggle_fullscreen(&mut self) {
        use winit::window::Fullscreen;
        if self.is_fullscreen {
            // DXGI requires SetFullscreenState(FALSE) before we give the window
            // back to the desktop compositor.
            #[cfg(target_os = "windows")]
            unsafe { self.renderer.exit_exclusive_fullscreen(&self.surface); }
            self.window.set_fullscreen(None);
            self.is_fullscreen = false;
        } else {
            // First: move to borderless fullscreen so the window covers the full
            // monitor. DXGI's SetFullscreenState works best when the client area
            // already matches the display resolution.
            let monitor = self.window.current_monitor();
            self.window.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
            // Second: enter hardware-exclusive mode via DXGI.
            #[cfg(target_os = "windows")]
            {
                use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
                if let Ok(handle) = self.window.window_handle() {
                    if let RawWindowHandle::Win32(h) = handle.as_raw() {
                        let hwnd = h.hwnd.get() as *mut std::ffi::c_void;
                        // SAFETY: hwnd is valid for the lifetime of this window;
                        // surface is configured and associated with this hwnd.
                        unsafe {
                            self.renderer.request_exclusive_fullscreen(&self.surface, hwnd);
                        }
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
