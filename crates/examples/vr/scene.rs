//! A long indoor showcase hallway for the VR demo.
//!
//! The corridor runs down −Z from the stage origin, divided into bays. Each bay
//! demonstrates one thing the renderer does — reflective metals, rough dielectrics,
//! emissive panels, coloured lights, animated geometry — so walking its length is a tour
//! of the material and lighting model.
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

use glam::{Mat4, Quat, Vec3};
use helio::{LightId, ObjectId, Renderer};

use crate::v3_demo_common::{box_mesh, cube_mesh, insert_object, make_material, point_light, sphere_mesh};

/// Interior half-width of the corridor, in metres.
pub const HALL_HALF_WIDTH: f32 = 2.4;
/// Interior height.
pub const HALL_HEIGHT: f32 = 3.0;
/// Length of one bay along −Z.
pub const BAY_LENGTH: f32 = 8.0;
/// Number of bays; the corridor spans z = 0 to z = -BAY_COUNT * BAY_LENGTH.
pub const BAY_COUNT: usize = 6;

/// Objects the demo animates each frame.
///
/// Returned rather than kept in a global so `main.rs` owns the animation state and this
/// module stays a pure builder.
pub struct Animated {
    /// Rotating cubes, with the centre each rotates about.
    pub spinners: Vec<(ObjectId, Vec3)>,
    /// Vertically bobbing orbs, with their rest positions.
    pub bobbers: Vec<(ObjectId, Vec3)>,
    /// Per-bay lights, pulsed in sympathy with the emissive strips.
    pub pulse_lights: Vec<(LightId, Vec3, [f32; 3], f32)>,
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

    let accents = [glossy_red, gold, copper, mirror, chalk, emissive_cyan];

    // ── Shell ────────────────────────────────────────────────────────────────
    let floor_half = Vec3::new(HALL_HALF_WIDTH, 0.1, BAY_LENGTH * 0.5);
    let wall_half = Vec3::new(0.1, HALL_HEIGHT * 0.5, BAY_LENGTH * 0.5);

    let floor_mesh = insert_box_mesh(renderer, floor_half);
    let wall_mesh = insert_box_mesh(renderer, wall_half);

    for bay in 0..BAY_COUNT {
        let z = bay_centre_z(bay);
        let _ = insert_object(
            renderer,
            floor_mesh,
            concrete,
            Mat4::from_translation(Vec3::new(0.0, -0.1, z)),
            floor_half.length(),
        );
        let _ = insert_object(
            renderer,
            floor_mesh,
            dark_trim,
            Mat4::from_translation(Vec3::new(0.0, HALL_HEIGHT + 0.1, z)),
            floor_half.length(),
        );
        for side in [-1.0_f32, 1.0] {
            let _ = insert_object(
                renderer,
                wall_mesh,
                if bay % 2 == 0 { concrete } else { dark_trim },
                Mat4::from_translation(Vec3::new(
                    side * (HALL_HALF_WIDTH + 0.1),
                    HALL_HEIGHT * 0.5,
                    z,
                )),
                wall_half.length(),
            );
        }
    }

    // Mirrored end cap, so the corridor terminates in geometry rather than in the void
    // and the far end shows the whole hall back at you.
    let end_half = Vec3::new(HALL_HALF_WIDTH + 0.2, HALL_HEIGHT * 0.5, 0.15);
    let end_mesh = insert_box_mesh(renderer, end_half);
    let _ = insert_object(
        renderer,
        end_mesh,
        mirror,
        Mat4::from_translation(Vec3::new(
            0.0,
            HALL_HEIGHT * 0.5,
            bay_centre_z(BAY_COUNT - 1) - BAY_LENGTH * 0.5,
        )),
        end_half.length(),
    );

    // ── Exhibits ─────────────────────────────────────────────────────────────
    let plinth_half = Vec3::new(0.35, 0.45, 0.35);
    let plinth_mesh = insert_box_mesh(renderer, plinth_half);
    let panel_half = Vec3::new(0.06, 0.5, 1.6);
    let panel_mesh = insert_box_mesh(renderer, panel_half);
    let spinner_mesh = renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.28)))
        .as_mesh()
        .unwrap();
    let orb_mesh = renderer
        .scene_mut()
        .insert_actor(helio::SceneActor::mesh(sphere_mesh([0.0, 0.0, 0.0], 0.3)))
        .as_mesh()
        .unwrap();

    let mut spinners = Vec::new();
    let mut bobbers = Vec::new();
    let mut pulse_lights = Vec::new();

    for bay in 0..BAY_COUNT {
        let z = bay_centre_z(bay);
        let accent = accents[bay % accents.len()];

        // Left plinth: a cube in this bay's accent material, rotating so you see the
        // highlight travel across its faces — which is the point of showing a metal.
        let plinth = Vec3::new(-1.4, 0.45, z);
        let _ = insert_object(
            renderer,
            plinth_mesh,
            dark_trim,
            Mat4::from_translation(plinth),
            plinth_half.length(),
        );
        let spin_centre = plinth + Vec3::new(0.0, 0.75, 0.0);
        if let Ok(id) = insert_object(
            renderer,
            spinner_mesh,
            accent,
            Mat4::from_translation(spin_centre),
            // Half-diagonal of a 0.28 half-extent cube: 0.28 * sqrt(3).
            0.28 * 1.7320508,
        ) {
            spinners.push((id, spin_centre));
        }

        // Right plinth: an orb alternating mirror and gold, so the two reflection
        // behaviours sit side by side down the corridor and stay comparable.
        let orb_plinth = Vec3::new(1.4, 0.45, z);
        let _ = insert_object(
            renderer,
            plinth_mesh,
            dark_trim,
            Mat4::from_translation(orb_plinth),
            plinth_half.length(),
        );
        let orb_rest = orb_plinth + Vec3::new(0.0, 0.85, 0.0);
        if let Ok(id) = insert_object(
            renderer,
            orb_mesh,
            if bay % 2 == 0 { mirror } else { gold },
            Mat4::from_translation(orb_rest),
            0.3,
        ) {
            bobbers.push((id, orb_rest));
        }

        // Emissive strips, each paired with a real light. An emissive surface contributes
        // nothing to anything else — without the paired light the strip glows while
        // lighting nothing around it, which reads as a rendering bug rather than a
        // stylistic choice.
        for side in [-1.0_f32, 1.0] {
            let _ = insert_object(
                renderer,
                panel_mesh,
                if bay % 2 == 0 { emissive_cyan } else { emissive_warm },
                Mat4::from_translation(Vec3::new(
                    side * (HALL_HALF_WIDTH - 0.08),
                    HALL_HEIGHT - 0.7,
                    z,
                )),
                panel_half.length(),
            );
        }

        let colour = if bay % 2 == 0 {
            [0.25, 0.85, 1.0]
        } else {
            [1.0, 0.6, 0.22]
        };
        let position = Vec3::new(0.0, HALL_HEIGHT - 0.5, z);
        let intensity = 7.5;
        let light = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(point_light(
                position.into(),
                colour,
                intensity,
                BAY_LENGTH,
            )))
            .as_light()
            .unwrap();
        pulse_lights.push((light, position, colour, intensity));
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

    Animated {
        spinners,
        bobbers,
        pulse_lights,
    }
}

/// Advance the animated exhibits. Called once per frame.
pub fn animate(renderer: &mut Renderer, animated: &Animated, time: f32) {
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
}
