//! WASM twin of `voxel/mesh_demo` — a procedurally generated voxel world
//! rendered as real triangles through a **custom render graph**
//! (`VoxelMeshPass` → `FxaaPass`), the same pipeline the native demo builds.
//!
//! This is the reference example for `HelioWasmApp::build_graph`: the default
//! deferred graph has no voxel pass, so the demo assembles its own.
//!
//! Controls:
//!   Click        — grab cursor / look
//!   WASD         — fly · Space/Shift — up/down
//!   Left click   — add a voxel sphere in front of the camera
//!   X            — carve a voxel sphere in front of the camera
//!   1–4          — select material (grass / dirt / stone / ore)
//!   R            — regenerate the world with a new seed
//!   Escape       — release cursor

use std::sync::Arc;

use glam::{EulerRot, Quat, Vec3};
use helio::{
    Camera, DebugDrawState, GpuLight, LightType, Movability, RenderGraph, Renderer, RendererConfig,
    Scene, SceneActor, VoxelMode, VoxelTerrain, VoxelVolumeDescriptor, VOXEL_TERRAIN_GRID_DIM,
};
use helio_pass_fxaa::FxaaPass;
use helio_pass_voxel_mesh::VoxelMeshPass;
use helio_voxel_core::GpuVoxelMaterial;
use helio_wasm::{HelioWasmApp, InputState, KeyCode};

const LOOK_SENS: f32 = 0.002;
const FLY_SPEED: f32 = 10.0;
const DRAG: f32 = 6.0;
// The GPU-side voxel volume is a dense grid fixed by the engine; `VOXEL_SIZE`
// just scales that grid into world units.
const VOXEL_SIZE: f32 = 0.75;
const ROOT_EXTENT: f32 = (VOXEL_TERRAIN_GRID_DIM as f32) * VOXEL_SIZE;

pub struct Demo {
    queue: Arc<wgpu::Queue>,
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
    world: VoxelTerrain,
    world_seed: u32,
    current_material: u8,
    prev_regen: bool,
    prev_carve: bool,
}

/// Converts a world-space position into the voxel volume's grid coordinates.
/// The volume is centered on the world origin, voxels scaled by `VOXEL_SIZE`.
fn world_to_grid(pos: Vec3) -> [f32; 3] {
    let half = VOXEL_TERRAIN_GRID_DIM as f32 / 2.0;
    [
        pos.x / VOXEL_SIZE + half,
        pos.y / VOXEL_SIZE + half,
        pos.z / VOXEL_SIZE + half,
    ]
}

impl Demo {
    /// Push the whole CPU voxel world into the mesh pass's buffers and mark
    /// every touched brick dirty so the surface-extract compute re-meshes it.
    fn upload_all(renderer: &mut Renderer, queue: &Arc<wgpu::Queue>, world: &mut VoxelTerrain) {
        if let Some(pass) = renderer.find_pass_mut::<VoxelMeshPass>() {
            let (meta_buf, data_buf) = (pass.brick_meta_buf().clone(), pass.voxel_data_buf().clone());
            let touched = world.upload_all_mesh(queue, &meta_buf, &data_buf, VOXEL_SIZE);
            for (brick_idx, origin) in touched {
                pass.mark_dirty(brick_idx, 0, origin, VOXEL_SIZE);
            }
        }
    }

    /// Add or subtract a voxel sphere 5 m in front of the camera.
    fn place_edit(&mut self, renderer: &mut Renderer, add: bool) {
        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let forward = orientation * -Vec3::Z;
        let center_grid = world_to_grid(self.cam_pos + forward * 5.0);
        let radius_grid = 2.0 / VOXEL_SIZE;

        if let Some(range) =
            self.world
                .paint_sphere(center_grid, radius_grid, self.current_material, add)
        {
            if let Some(pass) = renderer.find_pass_mut::<VoxelMeshPass>() {
                let (meta_buf, data_buf) =
                    (pass.brick_meta_buf().clone(), pass.voxel_data_buf().clone());
                let touched =
                    self.world
                        .upload_range_mesh(&self.queue, &meta_buf, &data_buf, VOXEL_SIZE, range);
                for (brick_idx, origin) in touched {
                    pass.mark_dirty(brick_idx, 0, origin, VOXEL_SIZE);
                }
            }
        }
    }
}

impl HelioWasmApp for Demo {
    fn title() -> &'static str {
        "Helio — Voxel Mesh Demo"
    }

    // The custom graph is FXAA-only (no TAA upscale), so color and depth must
    // share the full canvas resolution.
    fn render_scale() -> f32 {
        1.0
    }

    fn build_graph(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        _scene: &Scene,
        config: RendererConfig,
        _debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
        _debug_camera_buf: &wgpu::Buffer,
        _cull_stats_buf: &wgpu::Buffer,
    ) -> Option<RenderGraph> {
        // VoxelMeshPass rasterizes real triangles into "pre_aa"; FxaaPass reads
        // "pre_aa" and blits to the swapchain, doubling as anti-aliasing and the
        // final present. Mirrors the native voxel demo's graph exactly.
        let mut graph = RenderGraph::new(device, queue);
        graph.add_pass(Box::new(VoxelMeshPass::new(device, queue, config.surface_format)));
        graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));
        graph.lock(config.width, config.height);
        Some(graph)
    }

    fn init(
        renderer: &mut Renderer,
        _device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        _w: u32,
        _h: u32,
    ) -> Self {
        {
            let scene = renderer.scene_mut();

            // Material palette (index 0 is air / unused).
            let _ = scene.insert_voxel_volume(VoxelVolumeDescriptor {
                voxel_size: VOXEL_SIZE,
                root_extent: ROOT_EXTENT,
                local_to_world: glam::Mat4::IDENTITY,
                movability: Some(Movability::Stationary),
                mode: Some(VoxelMode::Auto),
                material_palette: vec![
                    GpuVoxelMaterial { color: [0.0, 0.0, 0.0], roughness: 1.0, metalness: 0.0, emissive: 0.0, _pad: [0; 2] }, // air
                    GpuVoxelMaterial { color: [0.3, 0.7, 0.25], roughness: 0.8, metalness: 0.0, emissive: 0.0, _pad: [0; 2] }, // grass
                    GpuVoxelMaterial { color: [0.45, 0.3, 0.15], roughness: 0.9, metalness: 0.0, emissive: 0.0, _pad: [0; 2] }, // dirt
                    GpuVoxelMaterial { color: [0.5, 0.5, 0.52], roughness: 0.85, metalness: 0.0, emissive: 0.0, _pad: [0; 2] }, // stone
                    GpuVoxelMaterial { color: [0.9, 0.75, 0.2], roughness: 0.4, metalness: 0.8, emissive: 0.0, _pad: [0; 2] }, // ore
                ],
            });

            // VoxelMeshPass sums the scene lights buffer directly, the same
            // infrastructure the default deferred lighting pass reads. This
            // custom graph has no ambient fill, so the lights are turned up and
            // a low sky-fill from below keeps shadowed faces readable.
            scene.insert_actor(SceneActor::light(GpuLight {
                position_range: [0.0, 0.0, 0.0, f32::MAX],
                direction_outer: [0.35, -0.8, 0.25, 0.0],
                color_intensity: [1.0, 0.96, 0.88, 6.0],
                shadow_index: u32::MAX,
                light_type: LightType::Directional as u32,
                inner_angle: 0.0,
                _pad: 0,
            }));
            scene.insert_actor(SceneActor::light(GpuLight {
                position_range: [0.0, 0.0, 0.0, f32::MAX],
                direction_outer: [-0.4, -0.3, -0.6, 0.0],
                color_intensity: [0.55, 0.65, 0.85, 2.5],
                shadow_index: u32::MAX,
                light_type: LightType::Directional as u32,
                inner_angle: 0.0,
                _pad: 0,
            }));
            // Upward sky-fill so downward-facing faces aren't pitch black.
            scene.insert_actor(SceneActor::light(GpuLight {
                position_range: [0.0, 0.0, 0.0, f32::MAX],
                direction_outer: [0.1, 0.9, 0.2, 0.0],
                color_intensity: [0.35, 0.4, 0.5, 1.5],
                shadow_index: u32::MAX,
                light_type: LightType::Directional as u32,
                inner_angle: 0.0,
                _pad: 0,
            }));
        }

        // In case VoxelMeshPass honors the scene ambient term, lift it too.
        renderer.set_ambient([0.5, 0.55, 0.6], 0.35);

        // The custom graph has no TAA to resolve the renderer's subpixel jitter,
        // so without disabling it the image shimmers frame to frame.
        renderer.set_jitter_enabled(false);

        let world_seed = 1;
        let mut world = VoxelTerrain::empty();
        world.generate(world_seed);
        Self::upload_all(renderer, &queue, &mut world);

        Self {
            queue,
            // Spawn above the terrain looking down so we don't start inside solid ground.
            cam_pos: Vec3::new(0.0, 30.0, 45.0),
            yaw: 0.0,
            pitch: -0.5,
            velocity: Vec3::ZERO,
            world,
            world_seed,
            current_material: 1,
            prev_regen: false,
            prev_carve: false,
        }
    }

    fn update(
        &mut self,
        renderer: &mut Renderer,
        dt: f32,
        _elapsed: f32,
        input: &InputState,
    ) -> Camera {
        // ── Look ──────────────────────────────────────────────────────────
        let (dx, dy) = input.mouse_delta;
        self.yaw -= dx * LOOK_SENS;
        self.pitch = (self.pitch - dy * LOOK_SENS).clamp(-1.5, 1.5);

        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let forward = orientation * -Vec3::Z;
        let right = orientation * Vec3::X;

        // ── Move ──────────────────────────────────────────────────────────
        let mut accel = Vec3::ZERO;
        if input.keys.contains(&KeyCode::KeyW) { accel += forward; }
        if input.keys.contains(&KeyCode::KeyS) { accel -= forward; }
        if input.keys.contains(&KeyCode::KeyA) { accel -= right; }
        if input.keys.contains(&KeyCode::KeyD) { accel += right; }
        if input.keys.contains(&KeyCode::Space) { accel += Vec3::Y; }
        if input.keys.contains(&KeyCode::ShiftLeft) { accel -= Vec3::Y; }
        if accel.length_squared() > 0.0 {
            accel = accel.normalize();
        }
        self.velocity += accel * FLY_SPEED * dt;
        self.velocity /= 1.0 + DRAG * dt;
        self.cam_pos += self.velocity * dt;

        // ── Material selection ────────────────────────────────────────────
        if input.keys.contains(&KeyCode::Digit1) { self.current_material = 1; }
        if input.keys.contains(&KeyCode::Digit2) { self.current_material = 2; }
        if input.keys.contains(&KeyCode::Digit3) { self.current_material = 3; }
        if input.keys.contains(&KeyCode::Digit4) { self.current_material = 4; }

        // ── Regenerate (R, edge-triggered) ────────────────────────────────
        let regen = input.keys.contains(&KeyCode::KeyR);
        if regen && !self.prev_regen {
            self.world_seed = self
                .world_seed
                .wrapping_add(1)
                .wrapping_mul(2654435761)
                .wrapping_add(1);
            let mut world = std::mem::replace(&mut self.world, VoxelTerrain::empty());
            world.generate(self.world_seed);
            Self::upload_all(renderer, &self.queue.clone(), &mut world);
            self.world = world;
        }
        self.prev_regen = regen;

        // ── Edits ─────────────────────────────────────────────────────────
        if input.mouse_left_just_pressed && input.cursor_grabbed {
            self.place_edit(renderer, true);
        }
        let carve = input.keys.contains(&KeyCode::KeyX);
        if carve && !self.prev_carve {
            self.place_edit(renderer, false);
        }
        self.prev_carve = carve;

        // ── Camera ────────────────────────────────────────────────────────
        let target = self.cam_pos + forward;
        let up = orientation * Vec3::Y;
        Camera::perspective_look_at(
            self.cam_pos,
            target,
            up,
            std::f32::consts::FRAC_PI_4,
            input.aspect_ratio(),
            0.01,
            2000.0,
        )
    }
}
