//! Boundary-preserving group simplification using meshoptimizer.
//!
//! Collapse ordering comes from meshoptimizer; the error of a group is not
//! taken from it but measured by `deviation`.
//! Group boundaries, non-manifold edges and attribute seams stay locked;
//! material sections are simplified independently. Output references original
//! vertices, preserving their attribute bytes and exact shared positions.

use crate::deviation::point_deviations;
use crate::math::{edge_key, V3};
use crate::mesh::{Tri, VertexStore};
use std::collections::BTreeMap;

pub struct SimplifyResult {
    pub tris: Vec<([u32; 3], u32)>,
    pub new_verts: Vec<(V3, u32)>,
}

// Include non-manifold multiplicities: locking vertices alone does not stop
// a simplifier from removing a duplicate face incident to a locked edge.
fn boundary_signature(vs: &VertexStore, tris: &[[u32; 3]], strict: bool) -> BTreeMap<u64, usize> {
    let mut edges = BTreeMap::new();
    for t in tris {
        for e in 0..3 {
            *edges
                .entry(edge_key(
                    vs.pid[t[e] as usize],
                    vs.pid[t[(e + 1) % 3] as usize],
                ))
                .or_insert(0) += 1;
        }
    }
    edges.retain(|_, n| *n == 1 || (strict && *n > 2));
    edges
}

/// Simplify a group to about `target` triangles with its boundary locked.
///
/// `budget` is an object-space deviation allowance: input vertices that the
/// result leaves further than `budget` from the simplified surface (typically
/// the tips of thin spikes, whose quadric cost is deceptively low) are locked
/// and the group is re-simplified, so one hard feature cannot make the whole
/// group's error (and that of all its ancestors) explode.
///
/// `whole` marks a single group covering every remaining cluster: there is no
/// neighbour to crack against, so only open (single-use) edges are protected
/// and non-manifold edges may simplify like any other.
pub fn simplify_group(vs: &VertexStore, tris: &[Tri], target: usize, budget: f32, whole: bool) -> SimplifyResult {
    let mut edges = Vec::with_capacity(tris.len() * 3);
    for t in tris {
        for e in 0..3 {
            edges.push(edge_key(
                vs.pid[t.v[e] as usize],
                vs.pid[t.v[(e + 1) % 3] as usize],
            ));
        }
    }
    edges.sort_unstable();
    let mut boundary = Vec::new();
    let mut i = 0;
    while i < edges.len() {
        let mut end = i + 1;
        while end < edges.len() && edges[end] == edges[i] {
            end += 1;
        }
        if end - i == 1 || (!whole && end - i > 2) {
            boundary.push((edges[i] >> 32) as u32);
            boundary.push(edges[i] as u32);
        }
        i = end;
    }
    boundary.sort_unstable();
    boundary.dedup();

    let mut sections: BTreeMap<u32, Vec<[u32; 3]>> = BTreeMap::new();
    for t in tris {
        sections.entry(t.mat).or_default().push(t.v);
    }
    let mut output = Vec::new();
    for (material, section) in sections {
        let mut ids: Vec<u32> = section.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        let positions: Vec<_> = ids.iter().map(|&g| vs.pos[g as usize]).collect();
        let mut locks: Vec<_> = ids
            .iter()
            .map(|&g| vs.seam[g as usize] || boundary.binary_search(&vs.pid[g as usize]).is_ok())
            .collect();
        let indices: Vec<_> = section
            .iter()
            .flatten()
            .map(|g| ids.binary_search(g).unwrap() as u32)
            .collect();
        let vertices = meshopt::VertexDataAdapter::new(bytemuck::cast_slice(&positions), 12, 0)
            .expect("packed float3 positions");
        let mut section_error = 0.0f32;
        let expected = boundary_signature(vs, &section, !whole);
        let nm_in = if whole { nonmanifold(vs, &section) } else { 0 };
        let mut reduced = section.clone();
        let mut budget_rounds = 0;
        for _ in 0..16 {
            let result = meshopt::simplify_with_locks(
                &indices,
                &vertices,
                &locks,
                (target * section.len() / tris.len()).max(1) * 3,
                f32::MAX,
                meshopt::SimplifyOptions::LockBorder | meshopt::SimplifyOptions::ErrorAbsolute,
                Some(&mut section_error),
            );
            reduced = result
                .chunks_exact(3)
                .map(|t| [ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]])
                .collect();
            let actual = boundary_signature(vs, &reduced, !whole);
            if expected == actual && !reduced.is_empty() && (!whole || nonmanifold(vs, &reduced) <= nm_in) {
                budget_rounds += 1;
                if budget.is_infinite() || budget_rounds > 3 {
                    break;
                }
                let out: Vec<[V3; 3]> = reduced
                    .iter()
                    .map(|t| t.map(|v| vs.pos[v as usize]))
                    .collect();
                let dev = point_deviations(&positions, &out);
                let mut any = false;
                for (l, d) in locks.iter_mut().zip(dev) {
                    if d > budget && !*l {
                        *l = true;
                        any = true;
                    }
                }
                if !any {
                    break;
                }
                continue;
            }
            // Freeze the one-ring around changed topology, then retry the
            // original section. Preserve as much safe reduction as possible.
            let mut affected = std::collections::BTreeSet::new();
            for (&edge, count) in expected.iter().chain(actual.iter()) {
                if expected.get(&edge) != actual.get(&edge) {
                    let _ = count;
                    affected.insert((edge >> 32) as u32);
                    affected.insert(edge as u32);
                }
            }
            for t in &section {
                if t.iter().any(|&v| affected.contains(&vs.pid[v as usize])) {
                    for v in t {
                        locks[ids.binary_search(v).unwrap()] = true;
                    }
                }
            }
        }
        if !reduced.is_empty() && expected == boundary_signature(vs, &reduced, !whole) && (!whole || nonmanifold(vs, &reduced) <= nm_in) {
            // The simplifier's own error estimate is ignored: the DAG builder
            // measures the real deviation against the original surface
            // (`deviation.rs`).
            output.extend(reduced.into_iter().map(|t| (t, material)));
        } else {
            // Malformed/non-manifold input must retain its geometry instead
            // of opening a crack between independently selected groups.
            output.extend(section.into_iter().map(|t| (t, material)));
        }
    }
    SimplifyResult {
        tris: output,
        new_verts: Vec::new(),
    }
}

/// Number of edges used by more than two triangles.
fn nonmanifold(vs: &VertexStore, tris: &[[u32; 3]]) -> usize {
    let mut edges = BTreeMap::new();
    for t in tris {
        for e in 0..3 {
            *edges.entry(edge_key(vs.pid[t[e] as usize], vs.pid[t[(e + 1) % 3] as usize])).or_insert(0usize) += 1;
        }
    }
    edges.values().filter(|&&n| n > 2).count()
}
