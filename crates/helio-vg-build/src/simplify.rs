//! Group simplification using meshoptimizer.
//!
//! Collapse ordering comes from meshoptimizer; the error of a group is not
//! taken from it but measured by `deviation`.
//!
//! A group's triangles fall into two classes:
//! * **anchored** geometry (connected surfaces that touch other groups, or big
//!   open/non-manifold components): boundary, non-manifold edges and attribute
//!   seams stay locked, so neighbouring groups never crack.
//! * **floating** pieces (small connected components that lie entirely inside
//!   the group, e.g. needle/leaf cards, branch fragments): nothing outside the
//!   group references them, so nothing is locked. They are reduced by edge
//!   collapse with island pruning and, when that stalls (disconnected pieces
//!   cannot collapse into each other), by proximity merging (topology-free
//!   vertex clustering), so canopies coarsen into fewer/larger cards.
//!
//! Material sections are always simplified independently. Output references
//! original vertices, preserving their attribute bytes and exact positions.

use crate::deviation::point_deviations;
use crate::math::{edge_key, V3};
use crate::mesh::{Tri, VertexStore};
use std::collections::BTreeMap;

/// A connected component with at most this many triangles that touches no
/// other group is "floating".
pub const FLOATING_MAX_TRIS: usize = 1536;
/// Appearance error of a floating-dominated group (see
/// [`SimplifyResult::coverage_tris`]): `COVERAGE_FACTOR` x the spacing of its
/// output triangles, `r * cbrt(4pi/3 / n)` for `n` triangles in a group sphere
/// of radius `r`. Below that scale the extra triangles of a canopy add detail
/// the viewer cannot resolve (a node smaller than ~1/COVERAGE_FACTOR spacings
/// per pixel).
pub const COVERAGE_FACTOR: f32 = 0.05;
/// ... and never below this many linear footprints (sqrt of mean triangle area) of
/// the output: triangles smaller than a pixel add nothing.
pub const COVERAGE_FOOTPRINT: f32 = 2.0;
/// Edge collapse that ends above this ratio of the target hands over to
/// proximity merging.
const SLOPPY_TRIGGER: (usize, usize) = (115, 100);

pub struct SimplifyResult {
    pub tris: Vec<([u32; 3], u32)>,
    pub new_verts: Vec<(V3, u32)>,
    /// Diagnostics: locked / total vertices of the input group.
    pub locked: usize,
    pub verts: usize,
    /// Diagnostics: input triangles in floating pieces.
    pub floating: usize,
    /// Floating output triangles when the group is (>= 90%) floating pieces
    /// (needle/leaf cards, branch fragments); `None` for all other groups.
    /// For those the DAG builder replaces the Hausdorff distance (a vanished
    /// card leaves a hole its own size while its neighbours are further away
    /// than that, so the distance says little about a canopy) by the spacing
    /// error above.
    pub coverage_tris: Option<(usize, f32)>,
}

/// Edges of one DAG level whose global use count is not 2 (open, non-manifold),
/// used to tell a group-boundary edge (shared with another group) from a
/// genuinely open one.
pub struct GlobalEdges {
    special: Vec<(u64, u32)>,
}

impl GlobalEdges {
    pub fn build<'a>(vs: &VertexStore, tris: impl Iterator<Item = &'a [u32; 3]>) -> Self {
        use rayon::prelude::*;
        let mut keys: Vec<u64> = Vec::new();
        for t in tris {
            for e in 0..3 {
                keys.push(edge_key(vs.pid[t[e] as usize], vs.pid[t[(e + 1) % 3] as usize]));
            }
        }
        keys.par_sort_unstable();
        let mut special = Vec::new();
        let mut i = 0;
        while i < keys.len() {
            let mut j = i + 1;
            while j < keys.len() && keys[j] == keys[i] {
                j += 1;
            }
            if j - i != 2 {
                special.push((keys[i], (j - i) as u32));
            }
            i = j;
        }
        Self { special }
    }

    fn count(&self, key: u64) -> u32 {
        match self.special.binary_search_by_key(&key, |e| e.0) {
            Ok(i) => self.special[i].1,
            Err(_) => 2,
        }
    }
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

fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

/// Per triangle: does it belong to a floating piece? `edges` = sorted edge
/// keys of the group (one entry per use).
fn floating_flags(vs: &VertexStore, tris: &[Tri], edges: &[u64], global: Option<&GlobalEdges>) -> Vec<bool> {
    let cross = |key: u64| -> bool {
        let Some(global) = global else { return false };
        let lo = edges.partition_point(|&e| e < key);
        let hi = edges.partition_point(|&e| e <= key);
        (hi - lo) as u32 != global.count(key)
    };
    let mut pids: Vec<u32> = tris.iter().filter(|t| vs.floating[t.v[0] as usize]).flat_map(|t| t.v.map(|v| vs.pid[v as usize])).collect();
    pids.sort_unstable();
    pids.dedup();
    let local = |v: u32| pids.binary_search(&vs.pid[v as usize]).unwrap() as u32;
    let mut parent: Vec<u32> = (0..pids.len() as u32).collect();
    for t in tris.iter().filter(|t| vs.floating[t.v[0] as usize]) {
        for k in 1..3 {
            let a = find(&mut parent, local(t.v[0]));
            let b = find(&mut parent, local(t.v[k]));
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let mut touches = vec![false; pids.len()];
    let roots: Vec<Option<u32>> = tris
        .iter()
        .map(|t| vs.floating[t.v[0] as usize].then(|| find(&mut parent, local(t.v[0]))))
        .collect();
    for (t, r) in tris.iter().zip(&roots) {
        let Some(r) = r else { continue };
        for e in 0..3 {
            let key = edge_key(vs.pid[t.v[e] as usize], vs.pid[t.v[(e + 1) % 3] as usize]);
            if cross(key) {
                touches[*r as usize] = true;
            }
        }
    }
    roots.iter().map(|r| r.is_some_and(|r| !touches[r as usize])).collect()
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
///
/// `global` gives the level's open/non-manifold edge counts (needed to tell
/// floating pieces from group-boundary geometry); `None` when the group is
/// the whole level.
pub fn simplify_group(
    vs: &VertexStore,
    tris: &[Tri],
    target: usize,
    budget: f32,
    whole: bool,
    global: Option<&GlobalEdges>,
) -> SimplifyResult {
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

    // Without level-wide edge counts nothing can be proven isolated.
    let floating = if global.is_some() || whole {
        floating_flags(vs, tris, &edges, global)
    } else {
        vec![false; tris.len()]
    };
    let n_floating = floating.iter().filter(|&&f| f).count();


    // (material, floating) sections.
    let mut sections: BTreeMap<(u32, bool), Vec<[u32; 3]>> = BTreeMap::new();
    for (t, &f) in tris.iter().zip(&floating) {
        sections.entry((t.mat, f)).or_default().push(t.v);
    }
    let mut output = Vec::new();
    let (mut n_locked, mut n_verts) = (0usize, 0usize);
    for ((material, is_floating), section) in sections {
        let want = (target * section.len() / tris.len()).max(1);
        if is_floating {
            // Vertices on an edge shared with another material section stay put
            // (the sections are simplified independently and must still meet).
            let mut sec_edges: Vec<u64> = section
                .iter()
                .flat_map(|t| (0..3).map(move |e| edge_key(vs.pid[t[e] as usize], vs.pid[t[(e + 1) % 3] as usize])))
                .collect();
            sec_edges.sort_unstable();
            let mut pinned: Vec<u32> = Vec::new();
            let mut k = 0;
            while k < sec_edges.len() {
                let mut j = k + 1;
                while j < sec_edges.len() && sec_edges[j] == sec_edges[k] {
                    j += 1;
                }
                let all = edges.partition_point(|&e| e <= sec_edges[k]) - edges.partition_point(|&e| e < sec_edges[k]);
                if all != j - k {
                    pinned.push((sec_edges[k] >> 32) as u32);
                    pinned.push(sec_edges[k] as u32);
                }
                k = j;
            }
            pinned.sort_unstable();
            pinned.dedup();
            let (reduced, nv) = simplify_floating(vs, &section, want, &pinned);
            n_verts += nv;
            output.extend(reduced.into_iter().map(|t| (t, material)));
            continue;
        }
        let mut ids: Vec<u32> = section.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        let positions: Vec<_> = ids.iter().map(|&g| vs.pos[g as usize]).collect();
        let mut locks: Vec<_> = ids
            .iter()
            .map(|&g| vs.seam[g as usize] || boundary.binary_search(&vs.pid[g as usize]).is_ok())
            .collect();
        n_locked += locks.iter().filter(|&&l| l).count();
        n_verts += locks.len();
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
                want * 3,
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
    // Dominance is by source pieces (small components of the input mesh), also when
    // some of them currently straddle a group border and stay anchored.
    let src_floating = tris.iter().filter(|t| vs.floating[t.v[0] as usize]).count();
    let float_out: Vec<[u32; 3]> = output.iter().filter(|(t, _)| vs.floating[t[0] as usize]).map(|(t, _)| *t).collect();
    SimplifyResult {
        tris: output,
        new_verts: Vec::new(),
        locked: n_locked,
        verts: n_verts,
        floating: n_floating,
        coverage_tris: (src_floating * 10 >= tris.len() * 9 && !float_out.is_empty())
            .then(|| (float_out.len(), mean_footprint(vs, &float_out))),
    }
}

/// Reduce floating pieces of one material section to about `want` triangles:
/// unlocked edge collapse with island pruning, then proximity merging if the
/// pieces cannot collapse into each other. Returns (triangles, vertex count).
fn simplify_floating(vs: &VertexStore, section: &[[u32; 3]], want: usize, pinned: &[u32]) -> (Vec<[u32; 3]>, usize) {
    let mut ids: Vec<u32> = section.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let positions: Vec<_> = ids.iter().map(|&g| vs.pos[g as usize]).collect();
    let locks: Vec<bool> = ids.iter().map(|&g| pinned.binary_search(&vs.pid[g as usize]).is_ok()).collect();
    let indices: Vec<u32> = section
        .iter()
        .flatten()
        .map(|g| ids.binary_search(g).unwrap() as u32)
        .collect();
    let vertices = meshopt::VertexDataAdapter::new(bytemuck::cast_slice(&positions), 12, 0)
        .expect("packed float3 positions");
    let mut result = meshopt::simplify_with_locks(
        &indices,
        &vertices,
        &locks,
        want * 3,
        f32::MAX,
        meshopt::SimplifyOptions::ErrorAbsolute | meshopt::SimplifyOptions::Prune,
        None,
    );
    if result.len() / 3 * SLOPPY_TRIGGER.1 > want * SLOPPY_TRIGGER.0 {
        let merged = meshopt::simplify_sloppy_with_locks(&result, &vertices, &locks, want * 3, f32::MAX, None);
        if !merged.is_empty() {
            result = merged;
        }
    }
    let out: Vec<[u32; 3]> = result
        .chunks_exact(3)
        .map(|t| [ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]])
        .filter(|t| {
            let p = t.map(|v| vs.pid[v as usize]);
            p[0] != p[1] && p[1] != p[2] && p[0] != p[2]
        })
        .collect();
    if out.is_empty() {
        // Never drop a whole section: keep it (cannot happen with a sane target).
        return (section.to_vec(), ids.len());
    }
    (out, ids.len())
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


/// Square root of the mean triangle area: the linear footprint of one output
/// triangle (thin needles count as thin, not as long).
fn mean_footprint(vs: &VertexStore, tris: &[[u32; 3]]) -> f32 {
    let mut s = 0.0f64;
    for t in tris {
        let p = t.map(|v| vs.pos[v as usize]);
        s += 0.5 * crate::math::length(crate::math::cross(crate::math::sub(p[1], p[0]), crate::math::sub(p[2], p[0]))) as f64;
    }
    (s / tris.len().max(1) as f64).sqrt() as f32
}
