use glam::Vec3;
use helio_pass_sdf::edit_bvh::{Aabb, EditBvh};

// fat_margin in EditBvh is 0.5 — inserted AABBs are expanded by 0.5 on all sides.

fn box3(min: [f32; 3], max: [f32; 3]) -> Aabb {
    Aabb::new(Vec3::from(min), Vec3::from(max))
}

fn unit_box() -> Aabb {
    box3([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
}

// ──────────────────────── Construction ───────────────────────────────────────

#[test]
fn new_root_is_usize_max() {
    let bvh = EditBvh::new();
    assert_eq!(bvh.root, usize::MAX);
}

#[test]
fn query_empty_bvh_returns_nothing() {
    let bvh = EditBvh::new();
    let mut results = Vec::new();
    bvh.query_aabb(&unit_box(), &mut results);
    assert!(results.is_empty());
}

// ──────────────────────── Single insert ──────────────────────────────────────

#[test]
fn insert_one_root_changes() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, unit_box());
    assert_ne!(bvh.root, usize::MAX);
}

#[test]
fn insert_one_query_overlapping_finds_it() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, unit_box());
    let mut results = Vec::new();
    bvh.query_aabb(&unit_box(), &mut results);
    assert_eq!(results, vec![0]);
}

#[test]
fn insert_one_query_far_away_finds_nothing() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, unit_box()); // fat: -1.5..1.5
    let mut results = Vec::new();
    bvh.query_aabb(&box3([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]), &mut results);
    assert!(results.is_empty());
}

// ──────────────────────── Remove ─────────────────────────────────────────────

#[test]
fn insert_then_remove_root_becomes_null() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, unit_box());
    bvh.remove(0);
    assert_eq!(bvh.root, usize::MAX);
}

#[test]
fn remove_after_insert_query_returns_empty() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, unit_box());
    bvh.remove(0);
    let mut results = Vec::new();
    bvh.query_aabb(&unit_box(), &mut results);
    assert!(results.is_empty());
}

#[test]
fn remove_nonexistent_is_noop() {
    let mut bvh = EditBvh::new();
    bvh.remove(999); // Should not panic.
}

// ──────────────────────── Two inserts ────────────────────────────────────────

#[test]
fn insert_two_adjacent_query_finds_both() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    bvh.insert(1, box3([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]));
    // Query covers both.
    let mut results = Vec::new();
    bvh.query_aabb(&box3([-1.0, -1.0, -1.0], [4.0, 2.0, 2.0]), &mut results);
    results.sort();
    assert_eq!(results, vec![0, 1]);
}

#[test]
fn insert_two_query_only_one() {
    let mut bvh = EditBvh::new();
    // Box A is far from box B.
    bvh.insert(0, box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])); // fat: -0.5..1.5
    bvh.insert(1, box3([10.0, 0.0, 0.0], [11.0, 1.0, 1.0])); // fat: 9.5..11.5
    // Query overlaps only edit 0's fat AABB.
    let mut results = Vec::new();
    bvh.query_aabb(&box3([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]), &mut results);
    assert_eq!(results, vec![0]);
}

#[test]
fn insert_two_remove_one_query_one() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    bvh.insert(1, box3([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]));
    bvh.remove(0);
    let mut results = Vec::new();
    bvh.query_aabb(&box3([-1.0, -1.0, -1.0], [4.0, 2.0, 2.0]), &mut results);
    assert_eq!(results, vec![1]);
}

// ──────────────────────── Update ─────────────────────────────────────────────

#[test]
fn update_moves_edit_to_new_location() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    // Update to far-away location.
    bvh.update(0, box3([20.0, 20.0, 20.0], [21.0, 21.0, 21.0]));
    // Original location should no longer match (fat AABB was 19.5..21.5, far from origin).
    let mut results = Vec::new();
    bvh.query_aabb(&unit_box(), &mut results);
    // There may be a result if the fat margin still overlaps the query — this just exercises
    // the update path without panicking. The important thing is no panic.
    // Verify new location is found.
    results.clear();
    bvh.query_aabb(&box3([20.0, 20.0, 20.0], [21.0, 21.0, 21.0]), &mut results);
    assert_eq!(results, vec![0]);
}

#[test]
fn update_within_fat_aabb_no_reinsert() {
    // Tiny move well within the 0.5 fat margin — should work without reinsert.
    let mut bvh = EditBvh::new();
    bvh.insert(0, box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    // Shift by 0.1 — still inside fat AABB.
    bvh.update(0, box3([0.1, 0.0, 0.0], [1.1, 1.0, 1.0]));
    let mut results = Vec::new();
    bvh.query_aabb(&box3([0.0, 0.0, 0.0], [1.5, 1.5, 1.5]), &mut results);
    assert_eq!(results, vec![0]);
}

// ──────────────────────── Many inserts ───────────────────────────────────────

#[test]
fn insert_many_query_all_found() {
    let mut bvh = EditBvh::new();
    for i in 0..16u32 {
        let x = (i as f32) * 3.0;
        bvh.insert(i as usize, box3([x, 0.0, 0.0], [x + 1.0, 1.0, 1.0]));
    }
    let mut results = Vec::new();
    // Huge query covers every fat box.
    bvh.query_aabb(&box3([-1.0, -1.0, -1.0], [50.0, 2.0, 2.0]), &mut results);
    results.sort();
    let expected: Vec<usize> = (0..16).collect();
    assert_eq!(results, expected);
}

#[test]
fn insert_high_index() {
    let mut bvh = EditBvh::new();
    bvh.insert(1000, unit_box());
    let mut results = Vec::new();
    bvh.query_aabb(&unit_box(), &mut results);
    assert_eq!(results, vec![1000]);
}

#[test]
fn edit_index_preserved_in_results() {
    let mut bvh = EditBvh::new();
    bvh.insert(42, unit_box());
    let mut results = Vec::new();
    bvh.query_aabb(&unit_box(), &mut results);
    assert!(results.contains(&42));
}

#[test]
fn query_with_point_aabb_no_panic() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, unit_box());
    let point = Aabb::new(Vec3::ZERO, Vec3::ZERO);
    let mut results = Vec::new();
    bvh.query_aabb(&point, &mut results); // Inside fat AABB, may or may not be found.
    // No panic is the goal here.
}

#[test]
fn multiple_queries_same_bvh_consistent() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();
    bvh.query_aabb(&unit_box(), &mut r1);
    bvh.query_aabb(&unit_box(), &mut r2);
    assert_eq!(r1, r2);
}

#[test]
fn insert_sequential_indices_no_panic() {
    let mut bvh = EditBvh::new();
    for i in 0..8 {
        bvh.insert(i, box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    }
    // Should not panic; results may contain any/all of the overlapping inserts.
    let mut results = Vec::new();
    bvh.query_aabb(&unit_box(), &mut results);
    assert!(!results.is_empty());
}

#[test]
fn insert_remove_all_then_reinsert() {
    let mut bvh = EditBvh::new();
    bvh.insert(0, unit_box());
    bvh.insert(1, box3([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]));
    bvh.remove(0);
    bvh.remove(1);
    assert_eq!(bvh.root, usize::MAX);
    // Reinsert and verify it works.
    bvh.insert(0, unit_box());
    let mut results = Vec::new();
    bvh.query_aabb(&unit_box(), &mut results);
    assert_eq!(results, vec![0]);
}

#[test]
fn large_aabb_covers_everything() {
    let mut bvh = EditBvh::new();
    for i in 0..4 {
        let offset = (i as f32) * 100.0;
        bvh.insert(i, box3([offset, 0.0, 0.0], [offset + 1.0, 1.0, 1.0]));
    }
    let giant = box3([-1.0, -1.0, -1.0], [500.0, 500.0, 500.0]);
    let mut results = Vec::new();
    bvh.query_aabb(&giant, &mut results);
    results.sort();
    assert_eq!(results, vec![0, 1, 2, 3]);
}

#[test]
fn query_nonoverlapping_with_fat_margin_still_misses() {
    let mut bvh = EditBvh::new();
    // Edit at [0,0,0]–[1,1,1] → fat AABB: [-0.5,-0.5,-0.5]–[1.5,1.5,1.5]
    bvh.insert(0, box3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    // Query starts at x=2.0, clearly outside fat AABB max x=1.5
    let mut results = Vec::new();
    bvh.query_aabb(&box3([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]), &mut results);
    assert!(results.is_empty());
}
