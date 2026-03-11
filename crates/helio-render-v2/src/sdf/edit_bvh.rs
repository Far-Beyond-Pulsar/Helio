//! Dynamic AABB Bounding Volume Hierarchy for SDF edit culling.
//!
//! Provides O(log n) spatial queries for brick classification instead of
//! brute-force O(n) iteration over all edits. Supports:
//! - Region queries: which edits overlap an AABB (for brick classification)
//! - Ray queries: which edits a ray intersects (for picking)
//! - Dynamic insert/remove for edit list changes

use glam::Vec3;

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create an AABB from center and half-extents.
    pub fn from_center_radius(center: Vec3, radius: f32) -> Self {
        let r = Vec3::splat(radius);
        Self {
            min: center - r,
            max: center + r,
        }
    }

    /// Fatten the AABB by a margin for tree stability (fewer reinsertions).
    pub fn fattened(&self, margin: f32) -> Self {
        let m = Vec3::splat(margin);
        Self {
            min: self.min - m,
            max: self.max + m,
        }
    }

    /// Test if two AABBs overlap.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Merge two AABBs into the smallest enclosing AABB.
    pub fn merge(&self, other: &Aabb) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Surface area heuristic — used for tree balancing.
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Test if a ray (origin, direction) intersects this AABB.
    /// Returns (t_enter, t_exit) or None if no intersection.
    pub fn ray_intersect(&self, origin: Vec3, dir_inv: Vec3) -> Option<(f32, f32)> {
        let t0 = (self.min - origin) * dir_inv;
        let t1 = (self.max - origin) * dir_inv;
        let tmin = t0.min(t1);
        let tmax = t0.max(t1);
        let t_enter = tmin.x.max(tmin.y).max(tmin.z);
        let t_exit = tmax.x.min(tmax.y).min(tmax.z);
        if t_enter <= t_exit && t_exit >= 0.0 {
            Some((t_enter, t_exit))
        } else {
            None
        }
    }
}

const NULL_NODE: u32 = u32::MAX;

/// A node in the dynamic AABB tree.
#[derive(Clone, Debug)]
struct BvhNode {
    aabb: Aabb,
    parent: u32,
    // For internal nodes: left and right children
    // For leaf nodes: edit_index is stored in left, right = NULL_NODE
    left: u32,
    right: u32,
    /// Index into the edit list. Only valid for leaf nodes.
    edit_index: u32,
    /// Height for balancing (0 for leaves).
    height: i32,
}

impl BvhNode {
    fn is_leaf(&self) -> bool {
        self.right == NULL_NODE
    }
}

/// Dynamic AABB tree for spatial indexing of SDF edits.
///
/// Based on the incremental SAH insertion approach used in Box2D/Bullet.
pub struct EditBvh {
    nodes: Vec<BvhNode>,
    root: u32,
    free_list: Vec<u32>,
    /// Maps edit_index -> node_index for O(1) removal.
    edit_to_node: Vec<u32>,
    /// Fattening margin to reduce reinsertions on small moves.
    fat_margin: f32,
}

impl EditBvh {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: NULL_NODE,
            free_list: Vec::new(),
            edit_to_node: Vec::new(),
            fat_margin: 0.5,
        }
    }

    /// Rebuild the BVH from scratch given a set of (edit_index, bounding_sphere) pairs.
    pub fn rebuild(&mut self, bounds: &[(Vec3, f32)]) {
        self.nodes.clear();
        self.free_list.clear();
        self.root = NULL_NODE;
        self.edit_to_node.clear();
        self.edit_to_node.resize(bounds.len(), NULL_NODE);

        for (i, &(center, radius)) in bounds.iter().enumerate() {
            let aabb = Aabb::from_center_radius(center, radius);
            self.insert(i, aabb);
        }
    }

    /// Insert a leaf for an edit with the given AABB.
    pub fn insert(&mut self, edit_index: usize, aabb: Aabb) {
        let fat_aabb = aabb.fattened(self.fat_margin);
        let leaf = self.alloc_node();

        self.nodes[leaf as usize] = BvhNode {
            aabb: fat_aabb,
            parent: NULL_NODE,
            left: NULL_NODE,
            right: NULL_NODE,
            edit_index: edit_index as u32,
            height: 0,
        };

        if edit_index >= self.edit_to_node.len() {
            self.edit_to_node.resize(edit_index + 1, NULL_NODE);
        }
        self.edit_to_node[edit_index] = leaf;

        if self.root == NULL_NODE {
            self.root = leaf;
            return;
        }

        // Find best sibling using SAH
        let mut best_sibling = self.root;
        let mut best_cost = self.nodes[self.root as usize]
            .aabb
            .merge(&fat_aabb)
            .surface_area();

        // Simple walk: for small trees this is fine
        let mut stack = vec![self.root];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            let merged = node.aabb.merge(&fat_aabb);
            let direct_cost = merged.surface_area();

            // Cost of inserting at this node
            let inherited_cost = if node.parent != NULL_NODE {
                self.inherited_cost(idx, &fat_aabb)
            } else {
                0.0
            };
            let cost = direct_cost + inherited_cost;

            if cost < best_cost {
                best_cost = cost;
                best_sibling = idx;
            }

            // Lower bound: if the inherited cost alone exceeds best, skip children
            if !node.is_leaf() && inherited_cost + fat_aabb.surface_area() < best_cost {
                stack.push(node.left);
                stack.push(node.right);
            }
        }

        // Create new internal node
        let old_parent = self.nodes[best_sibling as usize].parent;
        let new_internal = self.alloc_node();
        self.nodes[new_internal as usize] = BvhNode {
            aabb: self.nodes[best_sibling as usize].aabb.merge(&fat_aabb),
            parent: old_parent,
            left: best_sibling,
            right: leaf,
            edit_index: u32::MAX,
            height: 1,
        };

        self.nodes[best_sibling as usize].parent = new_internal;
        self.nodes[leaf as usize].parent = new_internal;

        if old_parent != NULL_NODE {
            let parent = &mut self.nodes[old_parent as usize];
            if parent.left == best_sibling {
                parent.left = new_internal;
            } else {
                parent.right = new_internal;
            }
        } else {
            self.root = new_internal;
        }

        // Walk up and refit + balance
        self.refit(new_internal);
    }

    /// Remove the leaf for a given edit index.
    pub fn remove(&mut self, edit_index: usize) {
        if edit_index >= self.edit_to_node.len() {
            return;
        }
        let leaf = self.edit_to_node[edit_index];
        if leaf == NULL_NODE {
            return;
        }
        self.edit_to_node[edit_index] = NULL_NODE;

        if leaf == self.root {
            self.root = NULL_NODE;
            self.free_node(leaf);
            return;
        }

        let parent = self.nodes[leaf as usize].parent;
        let grandparent = self.nodes[parent as usize].parent;
        let sibling = if self.nodes[parent as usize].left == leaf {
            self.nodes[parent as usize].right
        } else {
            self.nodes[parent as usize].left
        };

        if grandparent != NULL_NODE {
            // Replace parent with sibling in grandparent
            let gp = &mut self.nodes[grandparent as usize];
            if gp.left == parent {
                gp.left = sibling;
            } else {
                gp.right = sibling;
            }
            self.nodes[sibling as usize].parent = grandparent;
            self.refit(grandparent);
        } else {
            self.root = sibling;
            self.nodes[sibling as usize].parent = NULL_NODE;
        }

        self.free_node(parent);
        self.free_node(leaf);
    }

    /// Query all edit indices whose (fattened) AABBs intersect the given AABB.
    pub fn query_region(&self, aabb: &Aabb, results: &mut Vec<usize>) {
        results.clear();
        if self.root == NULL_NODE {
            return;
        }

        let mut stack = vec![self.root];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            if !node.aabb.intersects(aabb) {
                continue;
            }
            if node.is_leaf() {
                results.push(node.edit_index as usize);
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
    }

    /// Query all edit indices whose AABBs a ray intersects.
    pub fn query_ray(&self, origin: Vec3, direction: Vec3, max_dist: f32, results: &mut Vec<usize>) {
        results.clear();
        if self.root == NULL_NODE {
            return;
        }

        let dir_inv = Vec3::new(
            if direction.x.abs() > 1e-10 { 1.0 / direction.x } else { 1e10_f32.copysign(direction.x) },
            if direction.y.abs() > 1e-10 { 1.0 / direction.y } else { 1e10_f32.copysign(direction.y) },
            if direction.z.abs() > 1e-10 { 1.0 / direction.z } else { 1e10_f32.copysign(direction.z) },
        );

        let mut stack = vec![self.root];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            if let Some((t_enter, _)) = node.aabb.ray_intersect(origin, dir_inv) {
                if t_enter > max_dist {
                    continue;
                }
                if node.is_leaf() {
                    results.push(node.edit_index as usize);
                } else {
                    stack.push(node.left);
                    stack.push(node.right);
                }
            }
        }
    }

    /// Returns the number of leaf nodes (edits) in the tree.
    pub fn len(&self) -> usize {
        self.edit_to_node.iter().filter(|&&n| n != NULL_NODE).count()
    }

    pub fn is_empty(&self) -> bool {
        self.root == NULL_NODE
    }

    // ── Private helpers ─────────────────────────────────────────────────

    fn alloc_node(&mut self) -> u32 {
        if let Some(idx) = self.free_list.pop() {
            idx
        } else {
            let idx = self.nodes.len() as u32;
            self.nodes.push(BvhNode {
                aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
                parent: NULL_NODE,
                left: NULL_NODE,
                right: NULL_NODE,
                edit_index: u32::MAX,
                height: 0,
            });
            idx
        }
    }

    fn free_node(&mut self, idx: u32) {
        self.free_list.push(idx);
    }

    /// Compute inherited cost (delta in ancestor surface areas from insertion).
    fn inherited_cost(&self, node: u32, leaf_aabb: &Aabb) -> f32 {
        let mut cost = 0.0;
        let mut idx = self.nodes[node as usize].parent;
        while idx != NULL_NODE {
            let n = &self.nodes[idx as usize];
            let merged = n.aabb.merge(leaf_aabb);
            cost += merged.surface_area() - n.aabb.surface_area();
            idx = n.parent;
        }
        cost
    }

    /// Walk up from a node, refitting AABBs and balancing.
    fn refit(&mut self, start: u32) {
        let mut idx = start;
        while idx != NULL_NODE {
            idx = self.balance(idx);
            let node = &self.nodes[idx as usize];
            let left = node.left;
            let right = node.right;

            if left != NULL_NODE && right != NULL_NODE {
                let l_aabb = self.nodes[left as usize].aabb;
                let r_aabb = self.nodes[right as usize].aabb;
                let l_height = self.nodes[left as usize].height;
                let r_height = self.nodes[right as usize].height;
                self.nodes[idx as usize].aabb = l_aabb.merge(&r_aabb);
                self.nodes[idx as usize].height = l_height.max(r_height) + 1;
            }

            idx = self.nodes[idx as usize].parent;
        }
    }

    /// AVL-style balancing. Returns the (possibly new) node at this position.
    fn balance(&mut self, a: u32) -> u32 {
        if self.nodes[a as usize].is_leaf() || self.nodes[a as usize].height < 2 {
            return a;
        }

        let b = self.nodes[a as usize].left;
        let c = self.nodes[a as usize].right;
        let balance = self.nodes[c as usize].height - self.nodes[b as usize].height;

        // Rotate C up
        if balance > 1 {
            let f = self.nodes[c as usize].left;
            let g = self.nodes[c as usize].right;

            // Swap A and C
            self.nodes[c as usize].left = a;
            self.nodes[c as usize].parent = self.nodes[a as usize].parent;
            self.nodes[a as usize].parent = c;

            // Update old parent of A
            let c_parent = self.nodes[c as usize].parent;
            if c_parent != NULL_NODE {
                if self.nodes[c_parent as usize].left == a {
                    self.nodes[c_parent as usize].left = c;
                } else {
                    self.nodes[c_parent as usize].right = c;
                }
            } else {
                self.root = c;
            }

            // Rotate
            let fh = if f != NULL_NODE { self.nodes[f as usize].height } else { -1 };
            let gh = if g != NULL_NODE { self.nodes[g as usize].height } else { -1 };

            if fh > gh {
                self.nodes[c as usize].right = f;
                self.nodes[a as usize].right = g;
                if g != NULL_NODE {
                    self.nodes[g as usize].parent = a;
                }
                let al = &self.nodes[b as usize].aabb;
                let ar = if g != NULL_NODE { self.nodes[g as usize].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
                self.nodes[a as usize].aabb = al.merge(&ar);
                let bl = &self.nodes[a as usize].aabb;
                let br = &self.nodes[f as usize].aabb;
                self.nodes[c as usize].aabb = bl.merge(br);

                self.nodes[a as usize].height = self.nodes[b as usize].height.max(
                    if g != NULL_NODE { self.nodes[g as usize].height } else { 0 }
                ) + 1;
                self.nodes[c as usize].height = self.nodes[a as usize].height.max(self.nodes[f as usize].height) + 1;
            } else {
                self.nodes[c as usize].right = g;
                self.nodes[a as usize].right = f;
                if f != NULL_NODE {
                    self.nodes[f as usize].parent = a;
                }
                let al = &self.nodes[b as usize].aabb;
                let ar = if f != NULL_NODE { self.nodes[f as usize].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
                self.nodes[a as usize].aabb = al.merge(&ar);
                let bl = &self.nodes[a as usize].aabb;
                let br = if g != NULL_NODE { &self.nodes[g as usize].aabb } else { &Aabb::new(Vec3::ZERO, Vec3::ZERO) };
                self.nodes[c as usize].aabb = bl.merge(br);

                self.nodes[a as usize].height = self.nodes[b as usize].height.max(
                    if f != NULL_NODE { self.nodes[f as usize].height } else { 0 }
                ) + 1;
                self.nodes[c as usize].height = self.nodes[a as usize].height.max(
                    if g != NULL_NODE { self.nodes[g as usize].height } else { 0 }
                ) + 1;
            }

            return c;
        }

        // Rotate B up
        if balance < -1 {
            let d = self.nodes[b as usize].left;
            let e = self.nodes[b as usize].right;

            self.nodes[b as usize].left = a;
            self.nodes[b as usize].parent = self.nodes[a as usize].parent;
            self.nodes[a as usize].parent = b;

            let b_parent = self.nodes[b as usize].parent;
            if b_parent != NULL_NODE {
                if self.nodes[b_parent as usize].left == a {
                    self.nodes[b_parent as usize].left = b;
                } else {
                    self.nodes[b_parent as usize].right = b;
                }
            } else {
                self.root = b;
            }

            let dh = if d != NULL_NODE { self.nodes[d as usize].height } else { -1 };
            let eh = if e != NULL_NODE { self.nodes[e as usize].height } else { -1 };

            if dh > eh {
                self.nodes[b as usize].right = d;
                self.nodes[a as usize].left = e;
                if e != NULL_NODE {
                    self.nodes[e as usize].parent = a;
                }
                let al = if e != NULL_NODE { self.nodes[e as usize].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
                let ar = &self.nodes[c as usize].aabb;
                self.nodes[a as usize].aabb = al.merge(ar);
                let bl = &self.nodes[a as usize].aabb;
                let br = &self.nodes[d as usize].aabb;
                self.nodes[b as usize].aabb = bl.merge(br);

                self.nodes[a as usize].height = (if e != NULL_NODE { self.nodes[e as usize].height } else { 0 }).max(self.nodes[c as usize].height) + 1;
                self.nodes[b as usize].height = self.nodes[a as usize].height.max(self.nodes[d as usize].height) + 1;
            } else {
                self.nodes[b as usize].right = e;
                self.nodes[a as usize].left = d;
                if d != NULL_NODE {
                    self.nodes[d as usize].parent = a;
                }
                let al = if d != NULL_NODE { self.nodes[d as usize].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
                let ar = &self.nodes[c as usize].aabb;
                self.nodes[a as usize].aabb = al.merge(ar);
                let bl = &self.nodes[a as usize].aabb;
                let br = if e != NULL_NODE { &self.nodes[e as usize].aabb } else { &Aabb::new(Vec3::ZERO, Vec3::ZERO) };
                self.nodes[b as usize].aabb = bl.merge(br);

                self.nodes[a as usize].height = (if d != NULL_NODE { self.nodes[d as usize].height } else { 0 }).max(self.nodes[c as usize].height) + 1;
                self.nodes[b as usize].height = self.nodes[a as usize].height.max(
                    if e != NULL_NODE { self.nodes[e as usize].height } else { 0 }
                ) + 1;
            }

            return b;
        }

        a
    }
}
