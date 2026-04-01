//! Dynamic AABB Bounding Volume Hierarchy for SDF edit culling.
//!
//! Provides O(log n) spatial queries for brick classification instead of
//! brute-force O(n) iteration over all edits.

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

    pub fn from_center_radius(center: Vec3, radius: f32) -> Self {
        let r = Vec3::splat(radius);
        Self {
            min: center - r,
            max: center + r,
        }
    }

    pub fn fattened(&self, margin: f32) -> Self {
        let m = Vec3::splat(margin);
        Self {
            min: self.min - m,
            max: self.max + m,
        }
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn merge(&self, other: &Aabb) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

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

#[derive(Clone, Debug)]
struct BvhNode {
    aabb: Aabb,
    parent: u32,
    left: u32,
    right: u32,
    edit_index: u32,
    height: i32,
}

impl BvhNode {
    fn is_leaf(&self) -> bool {
        self.right == NULL_NODE
    }
}

/// Dynamic AABB tree for spatial indexing of SDF edits.
pub struct EditBvh {
    nodes: Vec<BvhNode>,
    root: u32,
    free_list: Vec<u32>,
    edit_to_node: Vec<u32>,
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

        let mut stack = vec![self.root];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            let merged = node.aabb.merge(&fat_aabb);
            let direct_cost = merged.surface_area();

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

            if !node.is_leaf() && inherited_cost + fat_aabb.surface_area() < best_cost {
                stack.push(node.left);
                stack.push(node.right);
            }
        }

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

        self.refit(new_internal);
    }

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

    /// Query all edit indices whose AABBs intersect the given AABB.
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
    pub fn query_ray(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_dist: f32,
        results: &mut Vec<usize>,
    ) {
        results.clear();
        if self.root == NULL_NODE {
            return;
        }
        let dir_inv = Vec3::new(
            if direction.x.abs() < 1e-12 {
                1e12
            } else {
                1.0 / direction.x
            },
            if direction.y.abs() < 1e-12 {
                1e12
            } else {
                1.0 / direction.y
            },
            if direction.z.abs() < 1e-12 {
                1e12
            } else {
                1.0 / direction.z
            },
        );
        let mut stack = vec![self.root];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            if let Some((t_enter, _t_exit)) = node.aabb.ray_intersect(origin, dir_inv) {
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

    // ── Private helpers ─────────────────────────────────────────────────────

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

    fn inherited_cost(&self, idx: u32, leaf_aabb: &Aabb) -> f32 {
        let mut cost = 0.0;
        let mut current = self.nodes[idx as usize].parent;
        while current != NULL_NODE {
            let node = &self.nodes[current as usize];
            let merged = node.aabb.merge(leaf_aabb);
            cost += merged.surface_area() - node.aabb.surface_area();
            current = node.parent;
        }
        cost
    }

    fn refit(&mut self, start: u32) {
        let mut current = start;
        while current != NULL_NODE {
            let left = self.nodes[current as usize].left;
            let right = self.nodes[current as usize].right;
            if left == NULL_NODE || right == NULL_NODE {
                break;
            }

            let left_aabb = self.nodes[left as usize].aabb;
            let right_aabb = self.nodes[right as usize].aabb;
            let left_h = self.nodes[left as usize].height;
            let right_h = self.nodes[right as usize].height;

            self.nodes[current as usize].aabb = left_aabb.merge(&right_aabb);
            self.nodes[current as usize].height = 1 + left_h.max(right_h);

            current = self.nodes[current as usize].parent;
        }
    }
}

impl Default for EditBvh {
    fn default() -> Self {
        Self::new()
    }
}
