//! Play-mode validation for `Movability` (Pulsar-Native#836): warn when an
//! object authored `Static` or `Stationary` has its `Transform` written.
//!
//! Passes cache what they derive from such an object (shadow pages, ray
//! tracing instances, uploads), so a runtime move is a bug in whatever caused
//! it. [`MotionGate`](pulsar_scene_model::motion::MotionGate) refuses the moves
//! scripts make; this catches the rest (physics, editor tools, native code),
//! which the gate never sees. The editor is allowed to move anything, so the
//! play-mode loop owns a watch and the editor loop does not.

use std::collections::HashSet;

use helio::Movability;
use pulsar_scene_model::Transform;
use pulsar_scenedb::{ChangeRead, ComponentChangeKind, Entity, World};

/// The movability that forbids moving `object`, if any: `Movability` is
/// projected onto the component instances (meshes, lights) that author it,
/// and an object is fixed when any of its instances is.
fn fixed_movability(world: &World, object: Entity) -> Option<Movability> {
    std::iter::once(object)
        .chain(pulsar_scene_model::attachments::instances(world, object))
        .filter_map(|entity| world.get::<Movability>(entity).copied())
        .find(|movability| !movability.can_move())
}

/// Reads `Transform` changes from SceneDB's change journal (its own cursor, so
/// no other reader is affected) and reports each fixed object that moved, once.
pub struct StaticMoveWatch {
    cursor: pulsar_scenedb::ChangeCursor,
    reported: HashSet<Entity>,
    scratch: Vec<pulsar_scenedb::ComponentChange>,
}

impl StaticMoveWatch {
    /// Start watching from now: moves made before this call are not reported.
    pub fn new(world: &World) -> Self {
        Self {
            cursor: world.open_change_cursor::<Transform>(),
            reported: HashSet::new(),
            scratch: Vec::new(),
        }
    }

    /// Check every `Transform` write since the last poll. Returns the fixed
    /// objects that moved for the first time (each is also logged as a
    /// warning); an object is reported once per watch, not once per frame.
    pub fn poll(&mut self, world: &World) -> Vec<Entity> {
        self.scratch.clear();
        if world.read_changes(&mut self.cursor, &mut self.scratch) == ChangeRead::Overflowed {
            // Entries were evicted unread; nothing to attribute. Carry on from
            // the newest entry rather than guess.
            return Vec::new();
        }
        let mut moved = Vec::new();
        for change in &self.scratch {
            if change.kind != ComponentChangeKind::Mutated {
                continue;
            }
            let Some(movability) = fixed_movability(world, change.entity) else {
                continue;
            };
            if !self.reported.insert(change.entity) {
                continue;
            }
            tracing::warn!(
                entity = ?change.entity,
                ?movability,
                "a {movability:?} object was moved at runtime; passes may have cached its transform \
                 (mark it Movable if it is meant to move)"
            );
            moved.push(change.entity);
        }
        // A despawned entity can be reused; forget it so a later occupant is judged fresh.
        self.reported.retain(|entity| world.is_alive(*entity));
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(world: &mut World, movability: Movability) -> Entity {
        let e = world.spawn();
        world.insert(e, Transform::default());
        world.insert(e, movability);
        e
    }

    fn nudge(world: &mut World, e: Entity) {
        world.get_mut::<Transform>(e).unwrap().position[0] += 1.0;
    }

    #[test]
    fn fixed_objects_are_reported_once_and_movable_ones_never() {
        let mut world = World::new();
        let fixed = object(&mut world, Movability::Static);
        let stationary = object(&mut world, Movability::Stationary);
        let movable = object(&mut world, Movability::Movable);
        let mut watch = StaticMoveWatch::new(&world);

        assert!(watch.poll(&world).is_empty(), "nothing moved yet");

        nudge(&mut world, fixed);
        nudge(&mut world, stationary);
        nudge(&mut world, movable);
        let mut moved = watch.poll(&world);
        moved.sort_by_key(|e| e.index());
        assert_eq!(moved, vec![fixed, stationary]);

        nudge(&mut world, fixed);
        assert!(watch.poll(&world).is_empty(), "already reported");
    }

    #[test]
    fn reads_and_objects_without_movability_are_ignored() {
        let mut world = World::new();
        let fixed = object(&mut world, Movability::Static);
        let unmarked = world.spawn();
        world.insert(unmarked, Transform::default());
        let mut watch = StaticMoveWatch::new(&world);

        let _ = world.get_mut::<Transform>(fixed).unwrap().position; // borrow, no write
        nudge(&mut world, unmarked);
        assert!(watch.poll(&world).is_empty());
    }

    #[test]
    fn an_object_is_fixed_through_its_component_instances() {
        let mut world = World::new();
        let object = world.spawn();
        world.insert(object, Transform::default());
        let mesh = pulsar_scene_model::attachments::spawn_instance(
            &mut world,
            object,
            pulsar_scene_model::NewInstance::new("StaticMeshComponent"),
        )
        .unwrap();
        world.insert(mesh, Movability::Static);
        let mut watch = StaticMoveWatch::new(&world);

        nudge(&mut world, object);
        assert_eq!(watch.poll(&world), vec![object]);
    }
}
