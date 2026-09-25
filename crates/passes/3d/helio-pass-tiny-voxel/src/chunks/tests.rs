use super::*;
use crate::world::{self, Edit};

fn oracle(w: &World, c: [i32; 3]) -> u32 {
    w.edits
        .iter()
        .rev()
        .find(|e| e.contains(c))
        .map_or_else(|| world::base_material(c), |e| e.material)
}
fn cells(k: [i32; 3]) -> impl Iterator<Item = [i32; 3]> {
    (0..CELLS as i32).map(move |i| {
        [
            k[0] * SIDE + i % SIDE,
            k[1] * SIDE + (i / SIDE) % SIDE,
            k[2] * SIDE + i / (SIDE * SIDE),
        ]
    })
}

#[test]
fn reuse_local_edits_eviction_and_world_snapshots_keep_canonical_materials() {
    let mut w = World::default();
    let surface = w
        .raycast(w.ground_spawn(-0.03, -256.03, 3.0), -glam::DVec3::Y, 10.0)
        .unwrap()
        .0;
    let k = key(surface);
    let other = [k[0] + 150, k[1], k[2] + 93];
    for c in cells(k).chain(cells(other)) {
        assert_eq!(w.material(c), oracle(&w, c));
    }
    let generated = w.chunks.stats().generated;
    for c in cells(k) {
        assert_eq!(w.material(c), oracle(&w, c));
    }
    assert_eq!(
        w.chunks.stats().generated,
        generated,
        "unchanged chunks must be reused"
    );
    let original = w.clone();
    w.edits.push(Edit {
        cell: surface,
        radius: 0.2,
        material: 0,
    });
    w.rebuild_edits();
    assert!(!w.chunks.contains(k));
    assert!(
        w.chunks.contains(other),
        "remote chunks must survive a local edit"
    );
    assert_eq!(original.material(surface), oracle(&original, surface));
    for c in cells(k) {
        assert_eq!(w.material(c), oracle(&w, c));
    }
    w.edits.push(Edit {
        cell: surface,
        radius: 0.05,
        material: 3,
    });
    w.rebuild_edits();
    assert_eq!(w.material(surface), 3);
    w.edits.pop();
    w.rebuild_edits();
    assert_eq!(w.material(surface), 0);
    w.edits.clear();
    w.rebuild_edits();
    assert_eq!(w.material(surface), oracle(&original, surface));
    let collisions: Vec<_> = (1..1_000_000)
        .map(|x| [k[0] + x, k[1], k[2]])
        .filter(|&candidate| slot(candidate) == slot(k))
        .take(WAYS)
        .collect();
    assert_eq!(collisions.len(), WAYS);
    for collision in collisions {
        w.chunks.material(&w, collision.map(|v| v * SIDE));
    }
    assert!(!w.chunks.contains(k));
    assert_eq!(
        w.material(surface),
        oracle(&w, surface),
        "eviction must regenerate the same world"
    );
    assert!(w.chunks.stats().resident <= CAPACITY);
}
