//! Lets the scene vocabulary's `Transform` methods honour `Movability`.
//!
//! `pulsar_scene_model` knows nothing about renderer promises. It runs every
//! registered [`MotionGate`] before a script (or anything else going
//! through those methods) moves an object, and this is Helio's: an object
//! authored `Static` or `Stationary` is never moved at runtime, because
//! passes cache what they derive from its transform (see
//! [`helio::Movability`]). An object with no `Movability` is not restricted.

use helio::Movability;
use pulsar_scene_model::motion::MotionGate;

pulsar_reflection::inventory::submit! {
    MotionGate {
        name: "Movability",
        check: |world, entity| match world.get::<Movability>(entity) {
            Some(movability) if !movability.can_move() => {
                Err(format!("it is {movability:?}; mark it Movable to move it at runtime"))
            }
            _ => Ok(()),
        },
    }
}
