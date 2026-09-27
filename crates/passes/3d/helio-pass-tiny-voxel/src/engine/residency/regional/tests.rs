use super::*;

fn split(slot: u32) -> Vec<Node> {
    let root = Node {
        low: [-32; 3],
        level: 1,
        child: 1,
    };
    let mut nodes = vec![root];
    for octant in 0..8 {
        nodes.push(Node {
            low: std::array::from_fn(|a| -32 + ((octant >> a) & 1) * 32),
            level: 0,
            child: BRICK | (slot + octant as u32),
        });
    }
    nodes
}

fn at(nodes: &[Node], cell: [i32; 3]) -> Node {
    let mut node = nodes[0];
    while node.child & BRICK == 0 {
        let half = 16 << node.level;
        let octant = (0..3).fold(0, |bits, a| {
            bits | (usize::from(cell[a] >= node.low[a] + half) << a)
        });
        node = nodes[node.child as usize + octant];
    }
    node
}

#[test]
fn ready_child_publishes_without_waiting_for_its_siblings() {
    let old = vec![Node {
        low: [-32; 3],
        level: 1,
        child: BRICK | 49,
    }];
    let target = split(80);
    let mut readiness = Readiness::new(&target, (1..9).collect());
    assert_eq!(compose(&old, &target, &readiness), old);
    for generated in 0..8 {
        readiness.generated(generated);
        let cut = compose(&old, &target, &readiness);
        assert_eq!(cut.len(), target.len());
        for octant in 0..8 {
            let cell = target[octant + 1].low.map(|v| v + 7);
            let actual = at(&cut, cell);
            if octant <= generated {
                assert_eq!(actual.child, BRICK | (80 + octant as u32));
            } else {
                assert_eq!(actual.child & SLOT_MASK, 49);
                assert_eq!(actual.level + ((actual.child >> ANCESTOR_SHIFT) & 31), 1);
            }
        }
    }
    assert_eq!(readiness.remaining[0], 0);
}

#[test]
fn unfinished_coarsening_retains_all_old_children() {
    let old = split(12);
    let target = vec![Node {
        low: [-32; 3],
        level: 1,
        child: BRICK | 90,
    }];
    let mut readiness = Readiness::new(&target, vec![0]);
    assert_eq!(compose(&old, &target, &readiness), old);
    readiness.generated(0);
    assert_eq!(compose(&old, &target, &readiness), target);
}

#[test]
fn fallback_bounds_decode_relative_to_a_non_aligned_root() {
    let root = Node {
        low: [-128; 3],
        level: 3,
        child: BRICK | 65535,
    };
    let child = Node {
        low: [32, -64, 96],
        level: 0,
        child: 0,
    };
    let link = clipped(root, child);
    assert_eq!(link.child & SLOT_MASK, 65535);
    let level = link.level + ((link.child >> ANCESTOR_SHIFT) & 31);
    let low = std::array::from_fn::<_, 3, _>(|a| {
        root.low[a] + (((child.low[a] - root.low[a]) >> (level + 5)) << (level + 5))
    });
    assert_eq!(low, root.low);
    assert_eq!(level, root.level);
}

#[test]
fn source_grid_and_root_changes_wait_for_an_atomic_complete_cut() {
    use super::super::{Pending, Plan, Residency, BRICK_CAPACITY};
    use crate::{world::Edit, World};
    use std::sync::Arc;

    for change in 0..5 {
        let world = Arc::new(World::default());
        let mut next_world = (*world).clone();
        if change == 1 {
            next_world
                .apply_edit(Edit {
                    cell: [0; 3],
                    radius: 1.0,
                    material: 0,
                })
                .unwrap();
        } else if change == 2 {
            next_world.set_voxel_size(1.0).unwrap();
        }
        // Even a semantically equal new Arc is treated conservatively.
        let next_world = if change == 0 || change == 4 {
            world.clone()
        } else {
            Arc::new(next_world)
        };
        let mut target = split(80);
        if change == 4 {
            for node in &mut target {
                node.low[0] += 64;
            }
        }
        let mut residency = Residency::new(BRICK_CAPACITY);
        residency.stats.ready = true;
        residency.active_world = Some(world.clone());
        residency.complete_nodes = vec![Node {
            low: [-32; 3],
            level: 1,
            child: BRICK | 49,
        }];
        let plan = Plan {
            nodes: target.clone(),
            leaves: Vec::new(),
            world: next_world.clone(),
            view: super::super::tests::view(glam::DVec3::ZERO),
            pixels: 1.0,
        };
        let jobs = target
            .iter()
            .skip(1)
            .map(|n| super::super::Job {
                low: n.low,
                level: n.level,
                slot: n.child & SLOT_MASK,
                pad: [0; 3],
            })
            .collect();
        let mut pending = Pending::new(
            plan,
            jobs,
            (1..9).collect(),
            Arc::new(next_world.edits.clone()),
        );
        pending.readiness.generated(0);
        pending.cursor = 1;
        residency.pending = Some(pending);
        assert_eq!(
            residency.publish().is_some(),
            change == 0,
            "change={change}"
        );
        assert!(Arc::ptr_eq(
            residency.active_world.as_ref().unwrap(),
            &world
        ));
        assert_eq!(
            residency.stats.regional_publications,
            u64::from(change == 0)
        );
        let pending = residency.pending.as_mut().unwrap();
        assert_eq!(
            residency.stats.fallback_regions,
            if change == 0 { 7 } else { 0 }
        );
        for i in 1..8 {
            pending.readiness.generated(i);
        }
        pending.cursor = 8;
        assert_eq!(residency.publish().unwrap(), target);
        assert_eq!(residency.stats.fallback_regions, 0);
        assert!(Arc::ptr_eq(
            residency.active_world.as_ref().unwrap(),
            &next_world
        ));
        assert_eq!(
            residency.active_voxel_step(),
            if change == 2 { 10 } else { 1 }
        );
    }
}

#[test]
fn mixed_refinement_and_coarsening_preserve_every_region_and_payload() {
    fn random(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *seed
    }
    fn tree(seed: &mut u32, base: u32) -> Vec<Node> {
        let mut nodes = vec![Node {
            low: [-128; 3],
            level: 3,
            child: AIR,
        }];
        let mut index = 0;
        while index < nodes.len() {
            let node = nodes[index];
            let value = random(seed);
            if node.level > 0 && value % 3 != 0 {
                nodes[index].child = nodes.len() as u32;
                let half = 16 << node.level;
                for octant in 0..8 {
                    nodes.push(Node {
                        low: std::array::from_fn(|a| node.low[a] + ((octant >> a) & 1) * half),
                        level: node.level - 1,
                        child: AIR,
                    });
                }
            } else {
                nodes[index].child = match value % 7 {
                    0 => AIR,
                    1 => SOLID,
                    _ => BRICK | (base + index as u32),
                };
            }
            index += 1;
        }
        nodes
    }
    fn payload(node: Node, root: Node) -> Node {
        if node.child == AIR || node.child == SOLID {
            return node;
        }
        let level = node.level + ((node.child >> ANCESTOR_SHIFT) & 31);
        Node {
            low: std::array::from_fn(|a| {
                root.low[a] + (((node.low[a] - root.low[a]) >> (level + 5)) << (level + 5))
            }),
            level,
            child: BRICK | (node.child & SLOT_MASK),
        }
    }
    let mut seed = 0x4ab93u32;
    for _ in 0..24 {
        let old = tree(&mut seed, 0);
        let target = tree(&mut seed, 2048);
        let jobs: Vec<_> = target
            .iter()
            .enumerate()
            .filter(|(_, n)| n.child & BRICK != 0 && n.child < SOLID)
            .map(|(i, _)| i)
            .collect();
        let mut readiness = Readiness::new(&target, jobs.clone());
        for completed in 0..=jobs.len() {
            if completed != 0 {
                readiness.generated(completed - 1);
            }
            if completed != jobs.len() && completed % 7 != 0 {
                continue;
            }
            let cut = compose(&old, &target, &readiness);
            assert!(cut.len() <= old.len() + target.len());
            for z in 0..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        let cell = [x, y, z].map(|v| -128 + v * 32 + 15);
                        let wanted = at(&target, cell);
                        let ready = wanted.child >= SOLID
                            || jobs[..completed].iter().any(|&i| target[i] == wanted);
                        let expected = if ready { wanted } else { at(&old, cell) };
                        let actual = at(&cut, cell);
                        assert_eq!(actual.child & SLOT_MASK, expected.child & SLOT_MASK);
                        if expected.child < SOLID {
                            assert_eq!(payload(actual, cut[0]), expected);
                        } else {
                            assert_eq!(actual.child, expected.child);
                        }
                    }
                }
            }
        }
    }
}
