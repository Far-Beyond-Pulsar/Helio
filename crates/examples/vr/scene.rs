//! A long indoor showcase hallway for the VR demo.
//!
//! The corridor runs down −Z from the stage origin, divided into bays. Each bay
//! demonstrates one thing the renderer does — so walking its length is a tour
//! of the engine: PBR materials, spot lights, lens flare + volumetric fog,
//! water simulation, GPU particles, virtual geometry, emissive/HDR colour,
//! voxel meshes and post-process colour grading.
//!
//! # Why it is built the way it is
//!
//! **Everything is bounded tightly, per bay.** Culling is driven by one world-space
//! bounding *sphere* per object, and a sphere fits a long thin thing badly: a single
//! corridor-length wall would have a bounding sphere enclosing the whole level, cull
//! nothing useful, and be easy to get wrong in the direction that deletes visible
//! geometry. So the shell is built as short per-bay segments with correct radii — which
//! is also what real level geometry looks like.
//!
//! **Radii are half-diagonals, not half-extents.** `insert_object` takes a bounding
//! *radius*. Passing a box's half-extent leaves its corners outside the sphere, and the
//! segment vanishes while a corner is still plainly on screen.
//!
//! **Content is at human scale around the stage origin.** OpenXR's stage origin maps 1:1
//! onto the world origin at floor level, so the player starts at `(0, 0, 0)` facing −Z
//! with their eyes near y = 1.6. A 3 m ceiling reads correctly in a headset; 2.2 m feels
//! like a crawlspace.
//!
//! **It is a forward-opaque graph.** `main.rs` builds `build_forward_opaque_graph`, so the
//! hallway only shows features that graph actually renders. Decals, foliage and virtual
//! geometry are deferred-graph passes — VG in particular crashes the pass (it expects a
//! G-buffer to write into), so it is deliberately absent; the shipping container from
//! `editor_demo` is loaded as a classic mesh instead (bay 5), parked beside an instanced
//! crate stack.

use glam::{Mat4, Quat, Vec3};
use helio::{
    GpuLight, LightId, LightType, MaterialId, MeshId, MeshUpload, ObjectId, Renderer, VoxelMode,
    VoxelTerrain, VoxelVolumeDescriptor, VOXEL_TERRAIN_GRID_DIM,
};
use helio_asset_compat::{load_scene_bytes_with_config, upload_scene_materials, LoadConfig};
use helio_pass_voxel_mesh::VoxelMeshPass;
use helio_pass_water_sim::WaterSimPass;
use helio_voxel_core::GpuVoxelMaterial;
use libhelio::{
    CoronaEmitterDescriptor, PostProcessSettings, PostProcessVolumeDescriptor,
};

use crate::v3_demo_common::{
    box_mesh, cube_mesh, insert_object, make_material, point_light, sphere_mesh, spot_light,
};

/// Interior half-width of the corridor, in metres.
pub const HALL_HALF_WIDTH: f32 = 2.4;
/// Interior height.
pub const HALL_HEIGHT: f32 = 3.0;
/// Length of one bay along −Z.
pub const BAY_LENGTH: f32 = 8.0;
/// Number of bays; the corridor spans z = 0 to z = -BAY_COUNT * BAY_LENGTH.
pub const BAY_COUNT: usize = 9;

/// Objects the demo animates each frame.
///
/// Returned rather than kept in a global so `main.rs` owns the animation state and this
/// module stays a pure builder. The controller cubes are here too: `main.rs` reparents
/// them to the OpenXR grip poses every frame.
pub struct Animated {
    /// Rotating cubes, with the centre each rotates about (materials bay).
    pub spinners: Vec<(ObjectId, Vec3)>,
    /// Vertically bobbing orbs, with their rest positions (materials bay).
    pub bobbers: Vec<(ObjectId, Vec3)>,
    /// Per-bay accent lights, pulsed in sympathy with the emissive strips.
    pub pulse_lights: Vec<(LightId, Vec3, [f32; 3], f32)>,
    /// Cubes attached to the player's left and right controllers.
    pub hand_cubes: [ObjectId; 2],
    /// Water-bay orb: (object, rest position, pool x/z centre). It dips into the pool
    /// and splashes (`WaterSimPass::add_drop`) on every impact.
    pub water_orb: Option<(ObjectId, Vec3, [f32; 2])>,
    /// Hue-cycling accent lights (emissive / light-count bay).
    pub colour_lights: Vec<(LightId, Vec3, [f32; 3], f32)>,
    /// Corona emitter, re-uploaded each frame with an orbiting position.
    pub corona: Option<CoronaAnim>,
}

/// An orbiting corona emitter, re-uploaded every frame.
pub struct CoronaAnim {
    /// Base emitter; `position` is overwritten each frame from `centre`/`radius`.
    pub emitter: CoronaEmitterDescriptor,
    /// Orbit centre in world space.
    pub centre: Vec3,
    /// Orbit radius in the XZ plane.
    pub radius: f32,
    /// Orbit speed in radians/second.
    pub speed: f32,
}

fn bay_centre_z(index: usize) -> f32 {
    -(index as f32 + 0.5) * BAY_LENGTH
}

fn insert_box_mesh(renderer: &mut Renderer, half: Vec3) -> helio::MeshId {
    renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::mesh(box_mesh(
            [0.0, 0.0, 0.0],
            [half.x, half.y, half.z],
        )))
        .as_mesh()
        .unwrap()
}

/// Insert an object at a world-space position, ignoring errors.
fn place(renderer: &mut Renderer, mesh: MeshId, material: MaterialId, pos: Vec3, radius: f32) {
    let _ = insert_object(renderer, mesh, material, Mat4::from_translation(pos), radius);
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as u32) % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

// ── Shared scene resources ────────────────────────────────────────────────────

struct Mats {
    dark_trim: MaterialId,
    mirror: MaterialId,
    gold: MaterialId,
    copper: MaterialId,
    chalk: MaterialId,
    glossy_red: MaterialId,
    emissive_cyan: MaterialId,
    emissive_warm: MaterialId,
    emissive_white: MaterialId,
    steel: MaterialId,
}

struct Meshes {
    cube: MeshId,
    sphere: MeshId,
    plinth: MeshId,
    panel: MeshId,
}

// ── Per-bay exhibits ──────────────────────────────────────────────────────────

/// Bay 0 — the PBR material space: a rotating metal cube, a bobbing orb, an
/// emissive strip paired with a real light. Everything else is built from the
/// same `make_material` parameters, just spread across roughness/metallic.
fn bay_materials(renderer: &mut Renderer, z: f32, meshes: &Meshes, mats: &Mats, anim: &mut Animated) {
    let plinth_left = Vec3::new(-1.4, 0.45, z);
    place(renderer, meshes.plinth, mats.dark_trim, plinth_left, 0.6);
    let spin_centre = plinth_left + Vec3::new(0.0, 0.75, 0.0);
    if let Ok(id) = insert_object(
        renderer,
        meshes.cube,
        mats.glossy_red,
        Mat4::from_translation(spin_centre),
        // Half-diagonal of a 0.28 half-extent cube: 0.28 * sqrt(3).
        0.28 * 1.7320508,
    ) {
        anim.spinners.push((id, spin_centre));
    }

    let plinth_right = Vec3::new(1.4, 0.45, z);
    place(renderer, meshes.plinth, mats.dark_trim, plinth_right, 0.6);
    let orb_rest = plinth_right + Vec3::new(0.0, 0.85, 0.0);
    if let Ok(id) = insert_object(renderer, meshes.sphere, mats.gold, Mat4::from_translation(orb_rest), 0.3) {
        anim.bobbers.push((id, orb_rest));
    }

    for side in [-1.0_f32, 1.0] {
        place(
            renderer,
            meshes.panel,
            mats.emissive_cyan,
            Vec3::new(side * (HALL_HALF_WIDTH - 0.08), HALL_HEIGHT - 0.7, z),
            1.0,
        );
    }

    // Material shelf: one sphere per material, spanning the roughness/metallic
    // space from mirror to chalk, so the whole range reads at a glance.
    let shelf = [mats.mirror, mats.gold, mats.copper, mats.chalk, mats.glossy_red];
    for (i, material) in shelf.into_iter().enumerate() {
        let x = -1.9 + i as f32 * 0.6;
        place(renderer, meshes.sphere, material, Vec3::new(x, 0.3, z + 3.2), 0.3);
    }
    let position = Vec3::new(0.0, HALL_HEIGHT - 0.5, z);
    let colour = [0.25, 0.85, 1.0];
    let intensity = 7.5;
    let light = renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::light(point_light(position.into(), colour, intensity, BAY_LENGTH)))
        .as_light()
        .unwrap();
    anim.pulse_lights.push((light, position, colour, intensity));
}

/// Bay 1 — light forms: downward fluorescent spot cones and warm wall sconces,
/// each with a visible emissive fixture, demonstrating `LightType::Spot` and
/// per-light shadows.
fn bay_spotlights(renderer: &mut Renderer, z: f32, meshes: &Meshes, mats: &Mats) {
    for side in [-1.0_f32, 1.0] {
        let x = side * 1.2;
        renderer.scene_mut().insert_actor(helio::SceneActor::light(spot_light(
            [x, HALL_HEIGHT - 0.05, z],
            [0.0, -1.0, 0.0],
            [0.9, 0.95, 1.0],
            5.0,
            6.5,
            1.22,
            1.48,
        )));
        place(
            renderer,
            meshes.panel,
            mats.emissive_warm,
            Vec3::new(x, HALL_HEIGHT - 0.16, z),
            0.7,
        );

        renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light(
            [side * (HALL_HALF_WIDTH - 0.15), 1.6, z],
            [1.0, 0.65, 0.3],
            2.2,
            4.5,
        )));
        let sconce = insert_box_mesh(renderer, Vec3::new(0.06, 0.12, 0.25));
        place(
            renderer,
            sconce,
            mats.dark_trim,
            Vec3::new(side * (HALL_HALF_WIDTH - 0.06), 1.6, z),
            0.3,
        );
    }
}

/// Bay 2 — a bright flare light (ghost lens flare) plus a fog volume lit by a
/// god-ray overhead light, so the corridor fills with visible volumetric shafts.
fn bay_flare_fog(renderer: &mut Renderer, z: f32, meshes: &Meshes, mats: &Mats) {
    renderer.scene_mut().insert_actor(helio::SceneActor::light(GpuLight {
        position_range: [0.0, 1.6, z, 9.0],
        direction_outer: [0.0, -1.0, 0.0, 0.0],
        color_intensity: [1.0, 0.85, 0.5, 12.0],
        shadow_index: u32::MAX,
        light_type: LightType::Point as u32,
        inner_angle: 0.0,
        _pad: 0,
        flare_enabled: 1,
        flare_type: 1,
        flare_intensity: 0.3,
        flare_scale: 1.0,
        flare_tint_r: 1.0,
        flare_tint_g: 0.7,
        flare_tint_b: 0.35,
        ..Default::default()
    }));

    let mut shaft = point_light([0.0, HALL_HEIGHT - 0.15, z], [0.7, 0.8, 1.0], 7.0, 9.0);
    shaft.god_rays_enabled = 1;
    renderer.scene_mut().insert_actor(helio::SceneActor::light(shaft));

    renderer.scene_mut().insert_actor(helio::SceneActor::post_process_volume(
        PostProcessVolumeDescriptor {
            bounds_min: [-HALL_HALF_WIDTH, 0.0, z - BAY_LENGTH * 0.5],
            bounds_max: [HALL_HALF_WIDTH, HALL_HEIGHT, z + BAY_LENGTH * 0.5],
            priority: 10.0,
            blend_radius: 1.5,
            blend_weight: 1.0,
            unbound: false,
            settings: PostProcessSettings {
                fog_enabled: true,
                fog_density: 0.08,
                fog_color: [0.7, 0.76, 0.92],
                fog_scattering_anisotropy: 0.6,
                ..PostProcessSettings::default()
            },
        },
    ));

    place(renderer, meshes.cube, mats.emissive_warm, Vec3::new(0.0, 1.6, z), 0.4);
}

/// Bay 3 — water simulation: a raised pool with a sphere that bobs in and out
/// of the surface, splashing `WaterSimPass` ripples on every dip.
fn bay_water(renderer: &mut Renderer, z: f32, meshes: &Meshes, mats: &Mats, anim: &mut Animated) {
    let pool_centre = Vec3::new(-1.5, 0.45, z);
    let pool_half = Vec3::new(0.95, 0.45, 3.4);
    let pool_mesh = insert_box_mesh(renderer, pool_half);
    place(renderer, pool_mesh, mats.dark_trim, pool_centre, pool_half.length());

    // The water surface sits exactly on the pedestal top (y = 0.9).
    renderer.scene_mut().insert_actor(helio::SceneActor::water_volume(
        helio::WaterVolumeDescriptor {
            bounds_min: [pool_centre.x - pool_half.x + 0.05, 0.9, pool_centre.z - pool_half.z + 0.05],
            bounds_max: [pool_centre.x + pool_half.x - 0.05, 1.8, pool_centre.z + pool_half.z - 0.05],
            surface_height: 0.0,
            wave_amplitude: 0.15,
            wave_frequency: 0.5,
            wave_speed: 6.0,
            wave_direction: [0.6, 0.3],
            wave_steepness: 0.5,
            water_color: [0.03, 0.25, 0.4],
            extinction: [0.2, 0.08, 0.05],
            foam_threshold: 0.5,
            foam_amount: 0.6,
            reflection_strength: 0.9,
            refraction_strength: 1.0,
            fresnel_power: 5.0,
            caustics_enabled: true,
            caustics_intensity: 1.5,
            caustics_scale: 6.0,
            caustics_speed: 0.0,
            fog_density: 0.0,
            god_rays_intensity: 0.3,
            ssr_enabled: true,
            ssr_steps: 32,
            ssr_step_size: 0.05,
            ssr_thickness: 0.02,
            ior: 1.333,
            fresnel_min: 0.1,
            density: 0.03,
            shadow_rim: 1.0,
            shadow_hitbox: 0.0,
            shadow_ao: 1.0,
            sun_direction: [0.0, 1.0, 0.0],
            wave_spring: 1.2,
            wave_damping: 0.98,
            wind_direction: [0.6, 0.4],
            wind_strength: 1.5,
            wave_scale: 0.4,
        },
    ));
    if let Some(sim) = renderer.find_pass_mut::<WaterSimPass>() {
        sim.set_wind([0.6, 0.4], 1.5);
        sim.set_wave_scale(0.4);
    }

    let orb_rest = pool_centre + Vec3::new(0.0, 0.55, 0.0);
    if let Ok(id) = insert_object(
        renderer,
        meshes.sphere,
        mats.glossy_red,
        Mat4::from_translation(orb_rest),
        0.3,
    ) {
        anim.water_orb = Some((id, orb_rest, [pool_centre.x, pool_centre.z]));
    }
}

/// Bay 4 — GPU particles: a corona ember fountain, with the emitter re-uploaded
/// every frame so the source drifts on a slow orbit.
fn bay_corona(renderer: &mut Renderer, z: f32, meshes: &Meshes, mats: &Mats, anim: &mut Animated) {
    place(renderer, meshes.plinth, mats.emissive_warm, Vec3::new(0.0, 0.05, z), 0.8);

    let emitter = CoronaEmitterDescriptor {
        max_particles: 65_536,
        emit_rate: 2400.0,
        lifetime: 2.2,
        lifetime_variation: 0.8,
        start_size: [0.6, 0.6],
        end_size: [0.03, 0.03],
        start_color: [1.0, 0.5, 0.2, 1.0],
        end_color: [0.8, 0.1, 0.05, 0.0],
        velocity: [0.0, 3.2, 0.0],
        velocity_variation: [1.6, 1.2, 1.6],
        gravity: -1.2,
        shape: libhelio::CoronaEmitterShape::Point,
        texture_index: -1,
        position: [0.0, 0.35, z],
    };
    renderer.set_corona_emitters(&[emitter.to_gpu()]);

    anim.corona = Some(CoronaAnim {
        emitter,
        centre: Vec3::new(0.0, 0.35, z),
        radius: 0.5,
        speed: 0.7,
    });

    renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light(
        [0.0, 2.4, z],
        [1.0, 0.5, 0.2],
        3.0,
        5.0,
    )));
}

/// Bay 5 — instancing: a crate stack and tile floor of identical objects that
/// the renderer auto-batches into a handful of instanced draws, a condensed
/// one_million_cubes. The left side is kept clear for the shipping container
/// parked there by `build`.
fn bay_instancing(renderer: &mut Renderer, z: f32, mats: &Mats) {
    let crate_half = Vec3::new(0.3, 0.3, 0.3);
    let crate_mesh = insert_box_mesh(renderer, crate_half);
    for x in 0..3 {
        for d in 0..2 {
            for y in 0..4 {
                let pos = Vec3::new(
                    1.35 + x as f32 * 0.62,
                    0.3 + y as f32 * 0.62,
                    z + d as f32 * 0.62,
                );
                let material = if (x + d + y) % 2 == 0 {
                    mats.gold
                } else {
                    mats.chalk
                };
                place(renderer, crate_mesh, material, pos, crate_half.length());
            }
        }
    }

    let tile_mesh = insert_box_mesh(renderer, Vec3::new(0.12, 0.05, 0.12));
    for x in 0..5 {
        for d in 0..4 {
            place(
                renderer,
                tile_mesh,
                mats.steel,
                Vec3::new(0.5 + x as f32 * 0.5, 0.05, z - 3.0 + d as f32 * 0.5),
                0.17,
            );
        }
    }
}

/// Load the shipping-container FBX (the same asset `editor_demo` uses) as a
/// *classic* mesh so it renders through the forward-opaque graph — the meshlet
/// path VirtualGeometry uses is deferred-only and would not draw here.
///
/// Returns `(mesh, material, local_centre, local_size)` so the caller can place
/// the container on the floor regardless of the FBX's origin. `None` if the
/// asset cannot be loaded, so a missing model never crashes the demo.
fn load_container(renderer: &mut Renderer) -> Option<(MeshId, MaterialId, Vec3, Vec3)> {
    const CONTAINER_FBX: &[u8] =
        include_bytes!("../../../models/source/container with textures.fbx");
    let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models/source");

    let scene = match load_scene_bytes_with_config(
        CONTAINER_FBX,
        "fbx",
        Some(base_dir.as_path()),
        LoadConfig::default()
            .with_uv_flip(false)
            .with_merge_meshes(true)
            .with_import_scale(glam::Vec3::splat(1.0 / 200.0)),
    ) {
        Ok(scene) => scene,
        Err(e) => {
            log::warn!("[hallway] shipping container FBX failed to load: {e}");
            return None;
        }
    };

    let mat_ids = upload_scene_materials(renderer, &scene).unwrap_or_default();
    let sm = match scene.sectioned_mesh {
        Some(sm) => sm,
        None => {
            log::warn!("[hallway] shipping container FBX produced no sectioned mesh");
            return None;
        }
    };
    let Some(section) = sm.sections.iter().find(|s| !s.indices.is_empty()) else {
        log::warn!("[hallway] shipping container FBX has no index data");
        return None;
    };

    let mut bb_min = Vec3::splat(f32::INFINITY);
    let mut bb_max = Vec3::splat(f32::NEG_INFINITY);
    for v in &sm.vertices {
        let p = Vec3::from(v.position);
        bb_min = bb_min.min(p);
        bb_max = bb_max.max(p);
    }
    let local_centre = (bb_min + bb_max) * 0.5;

    let fallback = renderer
        .scene_mut()
        .insert_material(make_material([0.5, 0.5, 0.5, 1.0], 0.8, 0.0, [0.0; 3], 0.0));
    let material = section
        .material_index
        .and_then(|i| mat_ids.get(i))
        .copied()
        .unwrap_or(fallback);

    let mesh = renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::mesh(MeshUpload {
            vertices: sm.vertices.clone(),
            indices: section.indices.clone(),
        }))
        .as_mesh()
        .unwrap();

    Some((mesh, material, local_centre, bb_max - bb_min))
}

/// Bay 6 — emissive/HDR colour targets plus a grid of hue-cycling lights, a
/// condensed take on the HDR/colour-grading and light-benchmark demos.
fn bay_emissive_colour(renderer: &mut Renderer, z: f32, meshes: &Meshes, mats: &Mats, anim: &mut Animated) {
    let targets = [
        (make_material([1.0, 0.1, 0.1, 1.0], 0.3, 0.0, [10.0, 0.5, 0.5], 10.0), -1.4),
        (make_material([0.1, 1.0, 0.1, 1.0], 0.3, 0.0, [0.5, 10.0, 0.5], 10.0), 0.0),
        (make_material([0.1, 0.1, 1.0, 1.0], 0.3, 0.0, [0.5, 0.5, 10.0], 10.0), 1.4),
    ];
    for (gpu, x) in targets {
        let mat = renderer.scene_mut().insert_material(gpu);
        place(renderer, meshes.cube, mat, Vec3::new(x, 0.6, z), 0.6);
    }

    for i in 0..6 {
        let pos = Vec3::new(-1.7 + (i % 3) as f32 * 1.7, 2.2, z + ((i / 3) as f32 - 0.5) * 5.0);
        let colour = hsv_to_rgb(i as f32 / 6.0, 0.8, 1.0);
        let base = 3.0;
        let light = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(point_light(pos.into(), colour, base, 6.0)))
            .as_light()
            .unwrap();
        anim.colour_lights.push((light, pos, colour, base));
    }
    place(renderer, meshes.plinth, mats.emissive_white, Vec3::new(0.0, 0.05, z), 0.8);
}

/// Bay 7 — a voxel sculpture rendered as real triangles through `VoxelMeshPass`.
fn bay_voxel(renderer: &mut Renderer, z: f32) {
    const VOXEL_SIZE: f32 = 0.22;

    let _ = renderer.scene_mut().insert_voxel_volume(VoxelVolumeDescriptor {
        voxel_size: VOXEL_SIZE,
        root_extent: VOXEL_TERRAIN_GRID_DIM as f32 * VOXEL_SIZE,
        local_to_world: Mat4::from_translation(Vec3::new(0.0, 0.0, z)),
        movability: Some(helio::Movability::Stationary),
        mode: Some(VoxelMode::Auto),
        material_palette: vec![
            GpuVoxelMaterial { color: [0.0, 0.0, 0.0], roughness: 1.0, metalness: 0.0, emissive: 0.0, _pad: [0; 2] },
            GpuVoxelMaterial { color: [0.85, 0.7, 0.25], roughness: 0.5, metalness: 0.6, emissive: 0.0, _pad: [0; 2] },
            GpuVoxelMaterial { color: [0.25, 0.55, 0.85], roughness: 0.7, metalness: 0.2, emissive: 0.0, _pad: [0; 2] },
            GpuVoxelMaterial { color: [0.85, 0.3, 0.3], roughness: 0.6, metalness: 0.1, emissive: 0.2, _pad: [0; 2] },
        ],
    });

    // Sculpt an abstract piece around the volume's local origin. Grid coordinates:
    // local = (grid - GRID_DIM/2) * voxel_size, so the origin sits at grid centre and
    // the sculpture rises from the floor at the bay centre.
    let half = VOXEL_TERRAIN_GRID_DIM as f32 / 2.0;
    let mut world = VoxelTerrain::empty();
    world.paint_sphere([half, half, half], 6.0, 1, true);
    world.paint_sphere([half, half + 6.0, half], 4.5, 2, true);
    world.paint_sphere([half, half + 11.0, half], 3.0, 3, true);
    world.paint_sphere([half, half, half + 7.0], 3.5, 2, true);
    world.paint_sphere([half, half + 4.0, half + 7.0], 2.5, 3, true);

    let queue = renderer.queue().clone();
    let pass = renderer
        .find_pass_mut::<VoxelMeshPass>()
        .expect("VoxelMeshPass present in the forward-opaque graph");
    let (meta_buf, data_buf) = (pass.brick_meta_buf().clone(), pass.voxel_data_buf().clone());
    let touched = world.upload_all_mesh(&queue, &meta_buf, &data_buf, VOXEL_SIZE);
    for (brick_idx, origin, occupied) in touched {
        pass.mark_dirty(brick_idx, 0, origin, VOXEL_SIZE, occupied);
    }
}

/// Bay 8 — post-process colour grading: a warm vignette + saturation + bloom
/// volume over the whole bay, anchored on a blindingly bright emissive sun.
fn bay_colour_grade(renderer: &mut Renderer, z: f32, meshes: &Meshes) {
    renderer.scene_mut().insert_actor(helio::SceneActor::post_process_volume(
        PostProcessVolumeDescriptor {
            bounds_min: [-HALL_HALF_WIDTH, 0.0, z - BAY_LENGTH * 0.5],
            bounds_max: [HALL_HALF_WIDTH, HALL_HEIGHT, z + BAY_LENGTH * 0.5],
            priority: 10.0,
            blend_radius: 2.0,
            blend_weight: 1.0,
            unbound: false,
            settings: PostProcessSettings {
                vignette_intensity: 0.5,
                vignette_smoothness: 2.0,
                vignette_roundness: 1.2,
                vignette_color: [1.0, 0.6, 0.2],
                vignette_enabled: true,
                color_saturation: [1.2, 1.05, 0.85],
                color_contrast: [1.05, 1.05, 1.05],
                bloom_intensity: 0.7,
                bloom_threshold: 1.2,
                bloom_knee: 0.5,
                bloom_enabled: true,
                bloom_tint: [1.0, 0.9, 0.7],
                ..PostProcessSettings::default()
            },
        },
    ));

    let sun_mat = renderer
        .scene_mut()
        .insert_material(make_material([1.0, 0.9, 0.7, 1.0], 0.2, 0.0, [50.0, 45.0, 35.0], 50.0));
    place(renderer, meshes.cube, sun_mat, Vec3::new(0.0, 1.4, z), 0.6);
    renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light(
        [0.0, 1.4, z],
        [1.0, 0.9, 0.7],
        12.0,
        8.0,
    )));
}

// ── Build ─────────────────────────────────────────────────────────────────────

pub fn build(renderer: &mut Renderer) -> Animated {
    // ── Materials ────────────────────────────────────────────────────────────
    // Deliberately spread across the roughness/metallic space: a showcase that is all
    // mid-roughness dielectric demonstrates almost nothing about the BRDF. The mirror and
    // the chalk are the two ends; everything else sits between them.
    let mut mat = |c: [f32; 4], rough: f32, metal: f32, em: [f32; 3], strength: f32| {
        renderer
            .scene_mut()
            .insert_material(make_material(c, rough, metal, em, strength))
    };

    let concrete = mat([0.38, 0.38, 0.40, 1.0], 0.92, 0.0, [0.0; 3], 0.0);
    let dark_trim = mat([0.09, 0.10, 0.12, 1.0], 0.55, 0.0, [0.0; 3], 0.0);
    let mirror = mat([0.95, 0.96, 0.98, 1.0], 0.04, 1.0, [0.0; 3], 0.0);
    let gold = mat([1.0, 0.78, 0.34, 1.0], 0.28, 1.0, [0.0; 3], 0.0);
    let copper = mat([0.95, 0.55, 0.42, 1.0], 0.16, 1.0, [0.0; 3], 0.0);
    let chalk = mat([0.86, 0.84, 0.80, 1.0], 1.0, 0.0, [0.0; 3], 0.0);
    let glossy_red = mat([0.65, 0.06, 0.08, 1.0], 0.12, 0.0, [0.0; 3], 0.0);
    let emissive_cyan = mat([0.02, 0.05, 0.06, 1.0], 0.6, 0.0, [0.1, 0.9, 1.0], 6.0);
    let emissive_warm = mat([0.06, 0.04, 0.02, 1.0], 0.6, 0.0, [1.0, 0.62, 0.22], 7.0);
    let emissive_white = mat([0.05, 0.05, 0.05, 1.0], 0.6, 0.0, [0.7, 0.7, 0.7], 3.0);
    let steel = mat([0.55, 0.57, 0.60, 1.0], 0.35, 0.75, [0.0; 3], 0.0);

    let mats = Mats {
        dark_trim,
        mirror,
        gold,
        copper,
        chalk,
        glossy_red,
        emissive_cyan,
        emissive_warm,
        emissive_white,
        steel,
    };

    // ── Shell ────────────────────────────────────────────────────────────────
    let floor_half = Vec3::new(HALL_HALF_WIDTH, 0.1, BAY_LENGTH * 0.5);
    let wall_half = Vec3::new(0.1, HALL_HEIGHT * 0.5, BAY_LENGTH * 0.5);

    let floor_mesh = insert_box_mesh(renderer, floor_half);
    let wall_mesh = insert_box_mesh(renderer, wall_half);

    for bay in 0..BAY_COUNT {
        let z = bay_centre_z(bay);
        place(renderer, floor_mesh, concrete, Vec3::new(0.0, -0.1, z), floor_half.length());
        place(renderer, floor_mesh, dark_trim, Vec3::new(0.0, HALL_HEIGHT + 0.1, z), floor_half.length());
        for side in [-1.0_f32, 1.0] {
            place(
                renderer,
                wall_mesh,
                if bay % 2 == 0 { concrete } else { dark_trim },
                Vec3::new(side * (HALL_HALF_WIDTH + 0.1), HALL_HEIGHT * 0.5, z),
                wall_half.length(),
            );
        }
    }

    // Mirrored end cap, so the corridor terminates in geometry rather than in the void
    // and the far end shows the whole hall back at you.
    let end_half = Vec3::new(HALL_HALF_WIDTH + 0.2, HALL_HEIGHT * 0.5, 0.15);
    let end_mesh = insert_box_mesh(renderer, end_half);
    place(
        renderer,
        end_mesh,
        mirror,
        Vec3::new(0.0, HALL_HEIGHT * 0.5, bay_centre_z(BAY_COUNT - 1) - BAY_LENGTH * 0.5),
        end_half.length(),
    );

    // ── Shared exhibit meshes ────────────────────────────────────────────────
    let meshes = Meshes {
        cube: renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.28)))
            .as_mesh()
            .unwrap(),
        sphere: renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(sphere_mesh([0.0, 0.0, 0.0], 0.3)))
            .as_mesh()
            .unwrap(),
        plinth: insert_box_mesh(renderer, Vec3::new(0.35, 0.45, 0.35)),
        panel: insert_box_mesh(renderer, Vec3::new(0.06, 0.5, 1.6)),
    };

    // ── Controller cubes ─────────────────────────────────────────────────────
    // Small bright cubes `main.rs` reparents to the OpenXR grip poses each frame.
    let hand_mesh = insert_box_mesh(renderer, Vec3::new(0.05, 0.05, 0.05));
    let hand_mat = renderer
        .scene_mut()
        .insert_material(make_material([0.05, 0.05, 0.06, 1.0], 0.4, 0.0, [0.2, 1.0, 0.9], 8.0));
    let mut hand_cubes = [ObjectId::from_raw(0, 0); 2];
    for (i, side) in [1.0_f32, -1.0].into_iter().enumerate() {
        let start = Vec3::new(side * 0.2, 1.4, -0.4);
        if let Ok(id) = insert_object(renderer, hand_mesh, hand_mat, Mat4::from_translation(start), 0.15) {
            hand_cubes[i] = id;
        }
    }

    // ── Per-bay exhibits ─────────────────────────────────────────────────────
    let mut anim = Animated {
        spinners: Vec::new(),
        bobbers: Vec::new(),
        pulse_lights: Vec::new(),
        hand_cubes,
        water_orb: None,
        colour_lights: Vec::new(),
        corona: None,
    };

    for bay in 0..BAY_COUNT {
        let z = bay_centre_z(bay);
        match bay {
            0 => bay_materials(renderer, z, &meshes, &mats, &mut anim),
            1 => bay_spotlights(renderer, z, &meshes, &mats),
            2 => bay_flare_fog(renderer, z, &meshes, &mats),
            3 => bay_water(renderer, z, &meshes, &mats, &mut anim),
            4 => bay_corona(renderer, z, &meshes, &mats, &mut anim),
            5 => bay_instancing(renderer, z, &mats),
            6 => bay_emissive_colour(renderer, z, &meshes, &mats, &mut anim),
            7 => bay_voxel(renderer, z),
            8 => bay_colour_grade(renderer, z, &meshes),
            _ => unreachable!(),
        }
    }

    // ── Shipping container ──────────────────────────────────────────────────
    // Parked along the left wall of the instancing bay, long axis down the
    // corridor. Rotating about Y maps the FBX's long (X) axis onto the
    // corridor's Z axis; the height axis is unchanged so the bottom sits on
    // the floor.
    if let Some((container_mesh, container_mat, local_centre, local_size)) =
        load_container(renderer)
    {
        let z = bay_centre_z(5);
        let centre = Vec3::new(
            -HALL_HALF_WIDTH + local_size.z * 0.5 + 0.03,
            local_size.y * 0.5,
            z,
        );
        let transform = Mat4::from_translation(centre)
            * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2)
            * Mat4::from_translation(-local_centre);
        let radius = (local_size * 0.5).length().max(0.5);
        let _ = insert_object(renderer, container_mesh, container_mat, transform, radius);
    }

    // Cool fill at the entrance, so the first bay is not lit solely by its own accent —
    // otherwise the whole corridor reads as one colour from the doorway.
    renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::light(point_light(
            [0.0, HALL_HEIGHT - 0.6, 1.5],
            [0.6, 0.7, 1.0],
            6.0,
            8.0,
        )));

    // Indoors, but the sky still drives ambient — and `SkyPass` is what establishes the
    // colour target each frame, so its absence is what made geometry smear over itself.
    // See `Renderer::rebuild_graph_if_sky_changed`.
    renderer.scene_mut().insert_actor(helio::SceneActor::sky(
        helio::SkyActor::new().with_sky_color([0.05, 0.07, 0.11]),
    ));

    anim
}

/// Advance the animated exhibits. Called once per frame with the scene time.
pub fn animate(renderer: &mut Renderer, animated: &mut Animated, time: f32) {
    for (index, (id, centre)) in animated.spinners.iter().enumerate() {
        // Staggered rates: a corridor rotating in unison reads as one mechanism rather
        // than as separate exhibits.
        let rate = 0.6 + index as f32 * 0.17;
        let transform = Mat4::from_translation(*centre)
            * Mat4::from_quat(Quat::from_euler(
                glam::EulerRot::YXZ,
                time * rate,
                time * rate * 0.6,
                0.0,
            ));
        let _ = renderer.scene_mut().update_object_transform(*id, transform);
    }

    for (index, (id, rest)) in animated.bobbers.iter().enumerate() {
        let offset = (time * 1.1 + index as f32 * 0.8).sin() * 0.18;
        let _ = renderer
            .scene_mut()
            .update_object_transform(*id, Mat4::from_translation(*rest + Vec3::Y * offset));
    }

    for (index, (id, position, colour, base)) in animated.pulse_lights.iter().enumerate() {
        // Shallow pulse — deep flicker in a headset is unpleasant at best and a migraine
        // trigger at worst, so this stays well inside a gentle band.
        let pulse = 0.85 + 0.15 * (time * 0.9 + index as f32 * 1.3).sin();
        let _ = renderer.scene_mut().update_light(
            *id,
            point_light((*position).into(), *colour, base * pulse, BAY_LENGTH),
        );
    }

    // Water orb: bob up and down, splashing ripples when it pierces the surface.
    if let Some((id, rest, pool_xz)) = &mut animated.water_orb {
        let y = rest.y + (time * 1.4).sin() * 0.35;
        let _ = renderer
            .scene_mut()
            .update_object_transform(*id, Mat4::from_translation(Vec3::new(rest.x, y, rest.z)));
        if y < 0.9 {
            if let Some(sim) = renderer.find_pass_mut::<WaterSimPass>() {
                sim.add_drop(pool_xz[0], pool_xz[1], 0.5, 0.9);
            }
        }
    }

    // Hue-cycle the emissive-bay lights, like the light-benchmark's colour sweep.
    let count = animated.colour_lights.len().max(1);
    for (index, (id, position, _colour, base)) in animated.colour_lights.iter().enumerate() {
        let hue = (time * 0.4 + index as f32 / count as f32) % 1.0;
        let colour = hsv_to_rgb(hue, 0.8, 1.0);
        let _ = renderer
            .scene_mut()
            .update_light(*id, point_light((*position).into(), colour, *base, 6.0));
    }

    // Corona emitter: drift the source on a slow orbit so the particles visibly follow.
    if let Some(c) = &mut animated.corona {
        let a = time * c.speed;
        c.emitter.position = [
            c.centre.x + a.cos() * c.radius,
            c.centre.y + (a * 2.0).sin() * 0.35,
            c.centre.z + a.sin() * c.radius,
        ];
        renderer.set_corona_emitters(&[c.emitter.to_gpu()]);
    }
}
