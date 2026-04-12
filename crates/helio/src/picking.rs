//! AAA-quality CPU ray picking for scene objects.
//!
//! # Architecture
//!
//! Two-phase picking:
//!
//! 1. **Broad phase** — ray vs per-object world-space AABB, sorted by entry
//!    distance and short-circuited once a closer hit is found.
//!
//! 2. **Narrow phase** — ray vs per-mesh local-space BVH of triangles using
//!    Möller-Trumbore intersection. The ray is transformed into mesh local space
//!    via the object's inverse model matrix so non-uniform scale is handled
//!    correctly.
//!
//! # Mesh BVH
//!
//! Each registered mesh gets a binary AABB-BVH built at registration time with
//! a midpoint-split strategy on the longest centroid axis (O(N log N), one-shot).
//! Leaves hold at most [`LEAF_MAX_TRIS`] triangles. Traversal uses a fixed-size
//! inline stack — no heap allocation per ray.
//!
//! # Usage
//!
//! ```ignore
//! use helio::{ScenePicker, PickHit, EditorState};
//!
//! let mut picker = ScenePicker::new();
//!
//! // Once per mesh (before creating objects):
//! let upload = cube_mesh([0.0; 3], 0.5);
//! let mesh_id = renderer.scene_mut()
//!     .insert_actor(SceneActor::mesh(upload.clone()))
//!     .as_mesh().unwrap();
//! picker.register_mesh(mesh_id, &upload);
//!
//! // After all objects are inserted (or whenever the scene changes):
//! picker.rebuild_instances(renderer.scene());
//!
//! // On left-click:
//! let (ray_o, ray_d) = EditorState::ray_from_screen(mx, my, w, h, vp_inv);
//! if let Some(hit) = picker.cast_ray(ray_o, ray_d) {
//!     editor.select(hit.object_id);
//! } else {
//!     editor.deselect();
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat3, Mat4, Vec3};

use crate::handles::{MeshId, ObjectId};
use crate::mesh::MeshUpload;
use crate::scene::Scene;

// ─────────────────────────────────────────────────────────────────────────────
// Tunables
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum triangles per BVH leaf.  Smaller → deeper tree and fewer wasted    
/// intersection tests; larger → shallower tree with cheaper traversal.
/// 4 balances stack depth and test cost well in practice.
const LEAF_MAX_TRIS: usize = 4;

/// Minimum t along the ray for a hit to be accepted.  Prevents self-intersection
/// from the ray origin sitting on or inside a triangle.
const RAY_T_MIN: f32 = 1e-4;

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

/// A BVH node: 32 bytes, two fit in one cache line.
///
/// Layout when **internal**: `left_first` = index of left child in `nodes`;
/// right child is always `left_first + 1`.
///
/// Layout when **leaf**: `left_first` = first triangle index in `tris`;
/// `count` = number of triangles in this leaf.
#[derive(Clone, Copy)]
struct BvhNode {
    min: [f32; 3],
    /// Internal: left child index.  Leaf: first triangle index into `tris`.
    left_first: u32,
    max: [f32; 3],
    /// 0 = internal node; N > 0 = leaf with N triangles.
    count: u32,
}

impl BvhNode {
    #[inline(always)]
    fn is_leaf(self) -> bool {
        self.count > 0
    }
}

/// Compact CPU triangle: 3 positions + pre-computed face normal.
#[derive(Clone, Copy)]
struct CpuTri {
    v: [[f32; 3]; 3],
    /// Pre-computed face normal (local-space, unit length).
    normal: [f32; 3],
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-mesh BVH
// ─────────────────────────────────────────────────────────────────────────────

/// A local-space AABB-BVH built from a [`MeshUpload`].
///
/// Created by [`ScenePicker::register_mesh`].  Stored inside an [`Arc`] so
/// multiple instances sharing the same mesh share one BVH without cloning.
pub(crate) struct MeshBvh {
    /// Triangles re-ordered into BVH traversal order.
    tris: Vec<CpuTri>,
    /// BVH node array; `nodes[0]` is the root.
    nodes: Vec<BvhNode>,
    /// Local-space AABB of all geometry (root node AABB, cached for convenience).
    pub(crate) local_min: Vec3,
    pub(crate) local_max: Vec3,
}

impl MeshBvh {
    /// Build a BVH from a mesh upload.  O(N log N) time, O(N) space.
    pub fn build(upload: &MeshUpload) -> Self {
        let verts = &upload.vertices;
        let inds = &upload.indices;
        let n_tris = inds.len() / 3;

        if n_tris == 0 {
            return Self {
                tris: Vec::new(),
                nodes: Vec::new(),
                local_min: Vec3::ZERO,
                local_max: Vec3::ZERO,
            };
        }

        // ── Extract triangles ─────────────────────────────────────────────────
        let mut tris: Vec<CpuTri> = Vec::with_capacity(n_tris);
        for t in 0..n_tris {
            let i0 = inds[t * 3] as usize;
            let i1 = inds[t * 3 + 1] as usize;
            let i2 = inds[t * 3 + 2] as usize;
            let p0 = Vec3::from(verts[i0].position);
            let p1 = Vec3::from(verts[i1].position);
            let p2 = Vec3::from(verts[i2].position);
            let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            tris.push(CpuTri {
                v: [p0.to_array(), p1.to_array(), p2.to_array()],
                normal: n.to_array(),
            });
        }

        // ── Per-triangle centroid and AABB ────────────────────────────────────
        let mut centroids: Vec<Vec3> = Vec::with_capacity(n_tris);
        let mut tri_mins: Vec<Vec3> = Vec::with_capacity(n_tris);
        let mut tri_maxs: Vec<Vec3> = Vec::with_capacity(n_tris);
        for tri in &tris {
            let p0 = Vec3::from(tri.v[0]);
            let p1 = Vec3::from(tri.v[1]);
            let p2 = Vec3::from(tri.v[2]);
            centroids.push((p0 + p1 + p2) / 3.0);
            tri_mins.push(p0.min(p1).min(p2));
            tri_maxs.push(p0.max(p1).max(p2));
        }

        // ── Build BVH (iterative, midpoint split on longest centroid axis) ────
        let root_min = tri_mins.iter().copied().reduce(Vec3::min).unwrap();
        let root_max = tri_maxs.iter().copied().reduce(Vec3::max).unwrap();

        // Upper bound: a full binary tree with N leaves has at most 2N - 1 nodes.
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * n_tris);
        // `tri_ids` is the permutation array that the build partitions in place.
        let mut tri_ids: Vec<u32> = (0..n_tris as u32).collect();

        // Push root as a leaf covering all triangles.
        nodes.push(BvhNode {
            min: root_min.to_array(),
            max: root_max.to_array(),
            left_first: 0,
            count: n_tris as u32,
        });

        // Iterative split using an explicit stack of node indices pending subdivision.
        let mut build_stack: Vec<usize> = vec![0];
        while let Some(node_idx) = build_stack.pop() {
            let node = nodes[node_idx];
            if node.count as usize <= LEAF_MAX_TRIS {
                continue; // Already a small-enough leaf.
            }

            let first = node.left_first as usize;
            let count = node.count as usize;
            let id_slice = &tri_ids[first..first + count];

            // Centroid AABB → longest axis to split on.
            let (c_min, c_max) = id_slice.iter().fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(mn, mx), &i| {
                    let c = centroids[i as usize];
                    (mn.min(c), mx.max(c))
                },
            );
            let extent = c_max - c_min;
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            let split = (c_min[axis] + c_max[axis]) * 0.5;

            // Dutch-flag partition: elements ≤ split go left, > split go right.
            let mut lo = first;
            let mut hi = first + count;
            while lo < hi {
                if centroids[tri_ids[lo] as usize][axis] <= split {
                    lo += 1;
                } else {
                    hi -= 1;
                    tri_ids.swap(lo, hi);
                }
            }

            // Degenerate split guard (all centroids on one side — equal axes).
            let left_count = lo - first;
            if left_count == 0 || left_count == count {
                continue; // Leave as oversized leaf rather than infinite-loop.
            }

            // Child AABBs.
            let l_first = first;
            let r_first = lo;
            let r_count = count - left_count;
            let (lmin, lmax) = slice_aabb(&tri_ids[l_first..lo], &tri_mins, &tri_maxs);
            let (rmin, rmax) = slice_aabb(&tri_ids[r_first..r_first + r_count], &tri_mins, &tri_maxs);

            let left_idx = nodes.len();
            let right_idx = left_idx + 1;
            nodes.push(BvhNode {
                min: lmin.to_array(),
                max: lmax.to_array(),
                left_first: l_first as u32,
                count: left_count as u32,
            });
            nodes.push(BvhNode {
                min: rmin.to_array(),
                max: rmax.to_array(),
                left_first: r_first as u32,
                count: r_count as u32,
            });

            // Convert the parent from leaf to internal.
            nodes[node_idx].left_first = left_idx as u32;
            nodes[node_idx].count = 0;

            build_stack.push(left_idx);
            build_stack.push(right_idx);
        }

        // Compact triangles into BVH traversal order so leaf ranges are contiguous.
        let ordered_tris: Vec<CpuTri> = tri_ids.iter().map(|&i| tris[i as usize]).collect();

        Self {
            tris: ordered_tris,
            nodes,
            local_min: root_min,
            local_max: root_max,
        }
    }

    /// Ray-BVH intersection in **local** (mesh) space.
    ///
    /// Returns `(t, face_normal_local)` for the closest forward hit, or `None`.
    ///
    /// Uses an inline fixed-size stack (no heap allocation per call).
    pub(crate) fn cast_local(&self, origin: Vec3, dir: Vec3) -> Option<(f32, Vec3)> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv_dir = safe_inv_dir(dir);
        let mut t_best = f32::MAX;
        let mut normal_best = Vec3::ZERO;

        // Fixed stack depth: BVH height ≤ 2·log₂(N/LEAF_MAX_TRIS) + 1.
        // 64 handles up to ~10 M triangles without overflow.
        let mut stack = [0u32; 64];
        let mut sp = 1usize;
        stack[0] = 0; // root

        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];

            if !ray_aabb_hit(origin, inv_dir, Vec3::from(node.min), Vec3::from(node.max), t_best) {
                continue;
            }

            if node.is_leaf() {
                let first = node.left_first as usize;
                let end = first + node.count as usize;
                for tri in &self.tris[first..end] {
                    if let Some(t) = moller_trumbore(origin, dir, tri) {
                        if t < t_best {
                            t_best = t;
                            normal_best = Vec3::from(tri.normal);
                        }
                    }
                }
            } else {
                // Push both children; the optimizer will visit the nearer one first.
                debug_assert!(sp + 2 <= stack.len(), "BVH stack overflow");
                stack[sp] = node.left_first;
                sp += 1;
                stack[sp] = node.left_first + 1;
                sp += 1;
            }
        }

        if t_best < f32::MAX {
            Some((t_best, normal_best))
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public hit result
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a successful [`ScenePicker::cast_ray`] call.
#[derive(Debug, Clone, Copy)]
pub struct PickHit {
    /// Handle of the hit scene object.
    pub object_id: ObjectId,

    /// Distance along the ray to the hit point (in world units, assuming
    /// `direction` passed to `cast_ray` was unit length).
    pub t: f32,

    /// World-space surface position of the hit.
    pub position: Vec3,

    /// World-space surface normal at the hit point (unit length, pointing
    /// toward the ray origin, i.e. front-face convention).
    pub normal: Vec3,
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-instance picking data (private)
// ─────────────────────────────────────────────────────────────────────────────

struct PickInstance {
    object_id: ObjectId,
    /// Compact key into `ScenePicker::mesh_bvhs`.
    mesh_key: u64,
    /// World transform (used to re-project local hit to world space).
    transform: Mat4,
    /// Inverse of `transform` (pre-computed once in `rebuild_instances`).
    inv_transform: Mat4,
    /// `(M^{-1})^T` upper-3×3 for transforming local normals to world space.
    normal_mat: Mat3,
    /// World-space tight AABB derived from `transform + bvh.local_{min,max}`.
    world_min: Vec3,
    world_max: Vec3,
}

// ─────────────────────────────────────────────────────────────────────────────
// ScenePicker
// ─────────────────────────────────────────────────────────────────────────────

/// CPU-side scene ray picker with per-mesh BVH acceleration.
///
/// # Lifecycle
///
/// 1. Create once: `ScenePicker::new()`
/// 2. Register each mesh **before** inserting objects: `picker.register_mesh(id, &upload)`
/// 3. Sync instances after any object add/remove: `picker.rebuild_instances(scene)`
/// 4. Per click: `picker.cast_ray(origin, dir)`
///
/// # Correctness with non-uniform scale
///
/// The ray is transformed into local mesh space, tested there, and the hit
/// point is then re-projected back to world space — so non-uniform scale
/// (e.g. `Mat4::from_scale(Vec3::new(2.0, 0.5, 1.0))`) produces exact hits.
///
/// # Notes for large scenes
///
/// The broad phase is O(N) over all registered instances, which is fast for
/// editor object counts (< ~10 000).  For world-streaming engines with 100k+
/// objects, add a scene-level BVH over the `world_min / world_max` AABBs.
pub struct ScenePicker {
    /// Per-registered-mesh BVH.  Keyed by `(slot as u64) | ((gen as u64) << 32)`.
    mesh_bvhs: HashMap<u64, Arc<MeshBvh>>,
    /// One entry per scene object that has a registered mesh.
    instances: Vec<PickInstance>,
}

impl Default for ScenePicker {
    fn default() -> Self {
        Self::new()
    }
}

impl ScenePicker {
    /// Create an empty picker.
    pub fn new() -> Self {
        Self {
            mesh_bvhs: HashMap::new(),
            instances: Vec::new(),
        }
    }

    // ── Registration ─────────────────────────────────────────────────────────

    /// Register a mesh's CPU geometry and build its local-space BVH.
    ///
    /// Call this **once per unique mesh**, with the same `MeshUpload` that was
    /// passed to `scene.insert_actor(SceneActor::mesh(...))`.  Building is
    /// O(N log N) where N is the triangle count.
    ///
    /// Meshes that are not registered here will be silently skipped during
    /// [`cast_ray`](Self::cast_ray).  This lets you exclude environmental or
    /// non-interactive geometry from the picking set.
    pub fn register_mesh(&mut self, id: MeshId, upload: &MeshUpload) {
        let key = mesh_key(id);
        self.mesh_bvhs.insert(key, Arc::new(MeshBvh::build(upload)));
    }

    // ── Instance sync ─────────────────────────────────────────────────────────

    /// Rebuild the picking instance list from the current scene.
    ///
    /// Call once after all objects are inserted, and again whenever objects are
    /// added or removed.  O(N) where N = live object count.
    ///
    /// Objects whose mesh was not registered via [`register_mesh`] are excluded.
    pub fn rebuild_instances(&mut self, scene: &Scene) {
        self.instances.clear();
        for obj in scene.iter_pickable_objects() {
            let key = mesh_key(obj.mesh_id);
            let Some(bvh) = self.mesh_bvhs.get(&key) else {
                continue; // Mesh not registered — skip (e.g. skybox, water volumes).
            };

            // Tight world AABB from local BVH root + current object transform.
            let (world_min, world_max) =
                transform_aabb(bvh.local_min, bvh.local_max, obj.transform);

            let inv = obj.transform.inverse();
            // Normal transform: transpose of inverse of the upper-left 3×3.
            let normal_mat = Mat3::from_mat4(inv).transpose();

            self.instances.push(PickInstance {
                object_id: obj.id,
                mesh_key: key,
                transform: obj.transform,
                inv_transform: inv,
                normal_mat,
                world_min,
                world_max,
            });
        }
    }

    // ── Ray cast ─────────────────────────────────────────────────────────────

    /// Cast a ray into the scene and return the **closest** hit, if any.
    ///
    /// `origin` and `dir` must be in **world space**.  `dir` should be unit
    /// length; if it is not, `t` in the returned [`PickHit`] will be in units
    /// of `dir.length()`.
    ///
    /// # Algorithm
    ///
    /// 1. Compute world-space AABB entry distances for every instance.
    /// 2. Sort candidates by entry distance (ascending).
    /// 3. For each candidate (short-circuit when entry > best hit distance):
    ///    a. Transform the normalized ray into local mesh space.
    ///    b. Traverse the per-mesh BVH with Möller-Trumbore intersection.
    ///    c. Re-project the local hit point to world space and compute world t.
    ///    d. Update the best hit if this is closer.
    ///
    /// # Performance
    ///
    /// - Broad phase: O(N) AABB tests, then sort O(N log N).
    /// - Narrow phase: O(log T) BVH traversal per AABB hit.
    /// - No heap allocation per call (BVH traversal uses a stack-allocated array).
    pub fn cast_ray(&self, origin: Vec3, dir: Vec3) -> Option<PickHit> {
        if dir.length_squared() < 1e-20 {
            return None;
        }
        let dir_n = dir.normalize();
        let inv_dir = safe_inv_dir(dir_n);

        // ── Broad phase: collect AABB-hit candidates with world t_near ────────
        let mut candidates: Vec<(f32, usize)> = self
            .instances
            .iter()
            .enumerate()
            .filter_map(|(idx, inst)| {
                ray_aabb_t(origin, inv_dir, inst.world_min, inst.world_max)
                    .map(|(t_near, _)| (t_near, idx))
            })
            .collect();

        // Sort by entry distance so we can early-exit once t_near > best hit.
        candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        // ── Narrow phase: per-mesh BVH traversal ─────────────────────────────
        let mut best_t = f32::MAX;
        let mut best_hit: Option<PickHit> = None;

        for (t_near, idx) in candidates {
            if t_near >= best_t {
                break; // All remaining candidates are further than current best.
            }

            let inst = &self.instances[idx];
            let Some(bvh) = self.mesh_bvhs.get(&inst.mesh_key) else {
                continue;
            };

            // Transform the ray into local mesh space.
            let local_origin = inst.inv_transform.transform_point3(origin);
            let local_dir = inst.inv_transform.transform_vector3(dir_n);

            if let Some((t_local, local_normal)) = bvh.cast_local(local_origin, local_dir) {
                // Re-project the local hit point to world space.  This correctly
                // handles non-uniform scale: world_hit ≠ origin + dir_n * t_local
                // when the transform includes anisotropic scale.
                let local_hit = local_origin + local_dir * t_local;
                let world_hit = inst.transform.transform_point3(local_hit);

                // World-space t (projection onto the normalized ray direction).
                let world_t = (world_hit - origin).dot(dir_n);

                if world_t > RAY_T_MIN && world_t < best_t {
                    best_t = world_t;

                    // Transform normal to world space and flip toward the ray so it
                    // always points outward (front-face convention).
                    let world_normal = (inst.normal_mat * local_normal).normalize_or_zero();
                    let world_normal = if world_normal.dot(dir_n) < 0.0 {
                        world_normal
                    } else {
                        -world_normal
                    };

                    best_hit = Some(PickHit {
                        object_id: inst.object_id,
                        t: world_t,
                        position: world_hit,
                        normal: world_normal,
                    });
                }
            }
        }

        best_hit
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compact the mesh key from a `MeshId` into a single u64.
#[inline(always)]
fn mesh_key(id: MeshId) -> u64 {
    (id.slot() as u64) | ((id.generation() as u64) << 32)
}

/// Compute per-child AABB from a slice of triangle indices.
#[inline]
fn slice_aabb(ids: &[u32], mins: &[Vec3], maxs: &[Vec3]) -> (Vec3, Vec3) {
    ids.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(mn, mx), &i| (mn.min(mins[i as usize]), mx.max(maxs[i as usize])),
    )
}

/// Compute the safe inverse of each component of a direction vector.
///
/// Uses `f32::MAX` as a sentinel for zero components to avoid NaN from
/// `0.0 * f32::INFINITY` in the AABB slab test.
#[inline(always)]
fn safe_inv_dir(dir: Vec3) -> Vec3 {
    Vec3::new(
        if dir.x.abs() > 1e-30 { 1.0 / dir.x } else { f32::MAX },
        if dir.y.abs() > 1e-30 { 1.0 / dir.y } else { f32::MAX },
        if dir.z.abs() > 1e-30 { 1.0 / dir.z } else { f32::MAX },
    )
}

/// Ray-AABB slab test.  Returns true iff the ray hits the box strictly before
/// `t_max`.  Handles rays starting inside the box (t_enter < 0).
#[inline(always)]
fn ray_aabb_hit(origin: Vec3, inv_dir: Vec3, aabb_min: Vec3, aabb_max: Vec3, t_max: f32) -> bool {
    let t_lo = (aabb_min - origin) * inv_dir;
    let t_hi = (aabb_max - origin) * inv_dir;
    let t_enter = t_lo.min(t_hi).max_element();
    let t_exit = t_lo.max(t_hi).min_element();
    t_enter <= t_exit && t_exit >= 0.0 && t_enter < t_max
}

/// Ray-AABB slab test returning the entry and exit `t` values.
///
/// Returns `None` for a miss.  Entry `t` is clamped to 0 for rays starting
/// inside the box.
#[inline(always)]
fn ray_aabb_t(
    origin: Vec3,
    inv_dir: Vec3,
    aabb_min: Vec3,
    aabb_max: Vec3,
) -> Option<(f32, f32)> {
    let t_lo = (aabb_min - origin) * inv_dir;
    let t_hi = (aabb_max - origin) * inv_dir;
    let t_enter = t_lo.min(t_hi).max_element();
    let t_exit = t_lo.max(t_hi).min_element();
    if t_enter <= t_exit && t_exit >= 0.0 {
        Some((t_enter.max(0.0), t_exit))
    } else {
        None
    }
}

/// Möller-Trumbore ray-triangle intersection.
///
/// Returns the ray parameter `t > RAY_T_MIN` if the ray hits the triangle,
/// otherwise `None`.
#[inline(always)]
fn moller_trumbore(origin: Vec3, dir: Vec3, tri: &CpuTri) -> Option<f32> {
    let v0 = Vec3::from(tri.v[0]);
    let v1 = Vec3::from(tri.v[1]);
    let v2 = Vec3::from(tri.v[2]);
    let e1 = v1 - v0;
    let e2 = v2 - v0;

    let h = dir.cross(e2);
    let a = e1.dot(h);
    // Cull both back-faces and degenerate (parallel) cases by making the test
    // two-sided.  Uncomment `a < 1e-8` instead of `a.abs() < 1e-8` for
    // single-sided (back-face culled) picking.
    if a.abs() < 1e-8 {
        return None;
    }

    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * e2.dot(q);
    if t > RAY_T_MIN {
        Some(t)
    } else {
        None
    }
}

/// Transform a local-space AABB by a 4×4 matrix, producing a tight world-space AABB.
///
/// Worst case is 8 point transforms (all 8 box corners), giving the exact OBB
/// enclosing AABB.
fn transform_aabb(local_min: Vec3, local_max: Vec3, transform: Mat4) -> (Vec3, Vec3) {
    let mut world_min = Vec3::splat(f32::MAX);
    let mut world_max = Vec3::splat(f32::MIN);
    for &x in &[local_min.x, local_max.x] {
        for &y in &[local_min.y, local_max.y] {
            for &z in &[local_min.z, local_max.z] {
                let p = transform.transform_point3(Vec3::new(x, y, z));
                world_min = world_min.min(p);
                world_max = world_max.max(p);
            }
        }
    }
    (world_min, world_max)
}
