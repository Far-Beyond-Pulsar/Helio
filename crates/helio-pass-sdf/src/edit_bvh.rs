//! AABB BVH tree for SDF edits.
//!
//! Maintains a dynamic AVL-balanced tree so that `query_aabb` returns only the
//! edit indices whose bounding boxes overlap a given region — used by `BrickMap`
//! to build per-brick edit lists without iterating all edits.

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

    pub fn from_point_radius(center: Vec3, radius: f32) -> Self {
        let r = Vec3::splat(radius);
        Self { min: center - r, max: center + r }
    }

    /// Returns true if this AABB overlaps with `other` (inclusive).
    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    /// Expand to contain `other`.
    pub fn union(self, other: Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Surface area heuristic helper: half surface area of the AABB.
    pub fn half_area(&self) -> f32 {
        let d = self.max - self.min;
        d.x * d.y + d.y * d.z + d.z * d.x
    }

    /// Expand by `margin` on all sides.
    pub fn expand(&self, margin: f32) -> Aabb {
        let m = Vec3::splat(margin);
        Aabb { min: self.min - m, max: self.max + m }
    }
}

const NULL: usize = usize::MAX;

#[derive(Clone)]
struct Node {
    aabb: Aabb,
    parent: usize,
    left: usize,
    right: usize,
    /// Leaf if >= 0, internal if usize::MAX - 1.
    edit_idx: usize,
    height: i32,
}

impl Node {
    fn new_leaf(aabb: Aabb, edit_idx: usize) -> Self {
        Node { aabb, parent: NULL, left: NULL, right: NULL, edit_idx, height: 0 }
    }

    fn new_internal(aabb: Aabb) -> Self {
        Node { aabb, parent: NULL, left: NULL, right: NULL, edit_idx: NULL, height: 1 }
    }

    fn is_leaf(&self) -> bool {
        self.edit_idx != NULL
    }
}

/// Dynamic AABB BVH tree for SDF edits.
pub struct EditBvh {
    nodes: Vec<Node>,
    free_list: Vec<usize>,
    pub root: usize,
    /// Maps edit index → node index in `nodes`.
    edit_to_node: Vec<Option<usize>>,
    fat_margin: f32,
}

impl EditBvh {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            root: NULL,
            edit_to_node: Vec::new(),
            fat_margin: 0.5,
        }
    }

    fn alloc_node(&mut self, node: Node) -> usize {
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx] = node;
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(node);
            idx
        }
    }

    fn free_node(&mut self, idx: usize) {
        self.free_list.push(idx);
    }

    /// Register an edit with the given index and bounding box.
    pub fn insert(&mut self, edit_idx: usize, aabb: Aabb) {
        // Ensure edit_to_node map is large enough.
        if edit_idx >= self.edit_to_node.len() {
            self.edit_to_node.resize(edit_idx + 1, None);
        }
        let fat_aabb = aabb.expand(self.fat_margin);
        let leaf = self.alloc_node(Node::new_leaf(fat_aabb, edit_idx));
        self.edit_to_node[edit_idx] = Some(leaf);
        self.insert_leaf(leaf);
    }

    /// Remove a previously inserted edit.
    pub fn remove(&mut self, edit_idx: usize) {
        if let Some(Some(leaf)) = self.edit_to_node.get(edit_idx) {
            let leaf = *leaf;
            self.edit_to_node[edit_idx] = None;
            self.remove_leaf(leaf);
            self.free_node(leaf);
        }
    }

    /// Update the AABB of an existing edit.
    pub fn update(&mut self, edit_idx: usize, new_aabb: Aabb) {
        if let Some(Some(leaf)) = self.edit_to_node.get(edit_idx) {
            let leaf = *leaf;
            let fat_aabb = new_aabb.expand(self.fat_margin);
            // If new AABB still fits within the existing fat one, no reinsert needed.
            let cur = self.nodes[leaf].aabb;
            if cur.min.x <= fat_aabb.min.x && cur.min.y <= fat_aabb.min.y && cur.min.z <= fat_aabb.min.z
            && cur.max.x >= fat_aabb.max.x && cur.max.y >= fat_aabb.max.y && cur.max.z >= fat_aabb.max.z {
                return; // Fits within existing fat AABB.
            }
            self.remove_leaf(leaf);
            self.nodes[leaf].aabb = fat_aabb;
            self.insert_leaf(leaf);
        } else {
            self.insert(edit_idx, new_aabb);
        }
    }

    /// Returns all edit indices whose AABBs overlap `query`.
    pub fn query_aabb(&self, query: &Aabb, results: &mut Vec<usize>) {
        if self.root == NULL { return; }
        let mut stack = Vec::new();
        stack.push(self.root);
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            if !node.aabb.overlaps(query) { continue; }
            if node.is_leaf() {
                results.push(node.edit_idx);
            } else {
                if node.left != NULL { stack.push(node.left); }
                if node.right != NULL { stack.push(node.right); }
            }
        }
    }

    // ---- Internal tree maintenance ----

    fn insert_leaf(&mut self, leaf: usize) {
        if self.root == NULL {
            self.root = leaf;
            self.nodes[leaf].parent = NULL;
            return;
        }
        // Find best sibling (greedy SAH).
        let leaf_aabb = self.nodes[leaf].aabb;
        let best = self.find_best_sibling(leaf_aabb);

        // Create new internal node.
        let old_parent = self.nodes[best].parent;
        let new_parent_aabb = leaf_aabb.union(self.nodes[best].aabb);
        let new_parent = self.alloc_node(Node::new_internal(new_parent_aabb));
        self.nodes[new_parent].parent = old_parent;
        self.nodes[new_parent].left = best;
        self.nodes[new_parent].right = leaf;
        self.nodes[best].parent = new_parent;
        self.nodes[leaf].parent = new_parent;

        if old_parent == NULL {
            self.root = new_parent;
        } else if self.nodes[old_parent].left == best {
            self.nodes[old_parent].left = new_parent;
        } else {
            self.nodes[old_parent].right = new_parent;
        }
        // Refit and balance up.
        self.refit_and_balance(new_parent);
    }

    fn find_best_sibling(&self, aabb: Aabb) -> usize {
        // Greedy best-first search for minimum cost insertion.
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(PartialEq)]
        struct Entry(f32, usize);
        impl Eq for Entry {}
        impl PartialOrd for Entry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }
        impl Ord for Entry {
            fn cmp(&self, other: &Self) -> Ordering {
                // Min-heap by negative cost.
                other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
            }
        }

        let mut best = self.root;
        let mut best_cost = aabb.union(self.nodes[self.root].aabb).half_area();
        let mut queue = BinaryHeap::new();
        queue.push(Entry(-best_cost, self.root));
        let leaf_area = aabb.half_area();
        while let Some(Entry(_, idx)) = queue.pop() {
            let node = &self.nodes[idx];
            let direct_cost = aabb.union(node.aabb).half_area();
            let inherited_cost = self.inherited_cost(idx);
            let total = direct_cost + inherited_cost;
            if total < best_cost {
                best_cost = total;
                best = idx;
            }
            if !node.is_leaf() {
                // Lower bound on child costs.
                let lower = leaf_area + inherited_cost + node.aabb.half_area();
                if lower < best_cost {
                    if node.left != NULL { queue.push(Entry(-lower, node.left)); }
                    if node.right != NULL { queue.push(Entry(-lower, node.right)); }
                }
            }
        }
        best
    }

    fn inherited_cost(&self, mut idx: usize) -> f32 {
        let mut cost = 0.0f32;
        while self.nodes[idx].parent != NULL {
            idx = self.nodes[idx].parent;
        }
        cost += self.nodes[idx].aabb.half_area();
        cost
    }

    fn remove_leaf(&mut self, leaf: usize) {
        if self.root == leaf {
            self.root = NULL;
            return;
        }
        let parent = self.nodes[leaf].parent;
        let grandparent = self.nodes[parent].parent;
        let sibling = if self.nodes[parent].left == leaf {
            self.nodes[parent].right
        } else {
            self.nodes[parent].left
        };

        if grandparent == NULL {
            self.root = sibling;
            if sibling != NULL {
                self.nodes[sibling].parent = NULL;
            }
        } else {
            if self.nodes[grandparent].left == parent {
                self.nodes[grandparent].left = sibling;
            } else {
                self.nodes[grandparent].right = sibling;
            }
            if sibling != NULL {
                self.nodes[sibling].parent = grandparent;
            }
            self.refit_and_balance(grandparent);
        }
        self.free_node(parent);
    }

    fn refit_and_balance(&mut self, mut idx: usize) {
        while idx != NULL {
            let left = self.nodes[idx].left;
            let right = self.nodes[idx].right;
            // Refit AABB.
            if left != NULL && right != NULL {
                self.nodes[idx].aabb = self.nodes[left].aabb.union(self.nodes[right].aabb);
                let lh = self.nodes[left].height;
                let rh = self.nodes[right].height;
                self.nodes[idx].height = 1 + lh.max(rh);
            }
            // AVL balance.
            let new_idx = self.balance(idx);
            let parent = self.nodes[new_idx].parent;
            idx = parent;
        }
    }

    fn height(&self, idx: usize) -> i32 {
        if idx == NULL { -1 } else { self.nodes[idx].height }
    }

    fn balance(&mut self, a: usize) -> usize {
        if self.nodes[a].is_leaf() || self.nodes[a].height < 2 {
            return a;
        }
        let b = self.nodes[a].left;
        let c = self.nodes[a].right;
        let balance_factor = self.height(c) - self.height(b);
        if balance_factor > 1 {
            return self.rotate_left(a, b, c);
        }
        if balance_factor < -1 {
            return self.rotate_right(a, b, c);
        }
        a
    }

    fn rotate_left(&mut self, a: usize, b: usize, c: usize) -> usize {
        let f = self.nodes[c].left;
        let g = self.nodes[c].right;
        let hf = self.height(f);
        let hg = self.height(g);
        // Swap a and c.
        self.nodes[c].left = a;
        self.nodes[c].parent = self.nodes[a].parent;
        self.nodes[a].parent = c;
        // Update grandparent.
        let cp = self.nodes[c].parent;
        if cp == NULL {
            self.root = c;
        } else if self.nodes[cp].left == a {
            self.nodes[cp].left = c;
        } else {
            self.nodes[cp].right = c;
        }
        // Choose rotation direction.
        if hf > hg {
            self.nodes[c].right = f;
            self.nodes[a].right = g;
            if f != NULL { self.nodes[f].parent = c; }
            if g != NULL { self.nodes[g].parent = a; }
            let ab = if b != NULL { self.nodes[b].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            let gb = if g != NULL { self.nodes[g].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            let fb = if f != NULL { self.nodes[f].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            if b != NULL && g != NULL { self.nodes[a].aabb = ab.union(gb); }
            self.nodes[a].height = 1 + self.height(b).max(self.height(g));
            self.nodes[c].aabb = self.nodes[a].aabb.union(fb);
            self.nodes[c].height = 1 + self.nodes[a].height.max(self.height(f));
        } else {
            self.nodes[c].right = g;
            self.nodes[a].right = f;
            if g != NULL { self.nodes[g].parent = c; }
            if f != NULL { self.nodes[f].parent = a; }
            let ab = if b != NULL { self.nodes[b].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            let fb = if f != NULL { self.nodes[f].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            let gb = if g != NULL { self.nodes[g].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            if b != NULL && f != NULL { self.nodes[a].aabb = ab.union(fb); }
            self.nodes[a].height = 1 + self.height(b).max(self.height(f));
            self.nodes[c].aabb = self.nodes[a].aabb.union(gb);
            self.nodes[c].height = 1 + self.nodes[a].height.max(self.height(g));
        }
        c
    }

    fn rotate_right(&mut self, a: usize, b: usize, c: usize) -> usize {
        let d = self.nodes[b].left;
        let e = self.nodes[b].right;
        let hd = self.height(d);
        let he = self.height(e);
        self.nodes[b].left = a;
        self.nodes[b].parent = self.nodes[a].parent;
        self.nodes[a].parent = b;
        let bp = self.nodes[b].parent;
        if bp == NULL {
            self.root = b;
        } else if self.nodes[bp].left == a {
            self.nodes[bp].left = b;
        } else {
            self.nodes[bp].right = b;
        }
        if hd > he {
            self.nodes[b].right = d;
            self.nodes[a].left = e;
            if d != NULL { self.nodes[d].parent = b; }
            if e != NULL { self.nodes[e].parent = a; }
            let cb = if c != NULL { self.nodes[c].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            let eb = if e != NULL { self.nodes[e].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            let db = if d != NULL { self.nodes[d].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            if c != NULL && e != NULL { self.nodes[a].aabb = cb.union(eb); }
            self.nodes[a].height = 1 + self.height(c).max(self.height(e));
            self.nodes[b].aabb = self.nodes[a].aabb.union(db);
            self.nodes[b].height = 1 + self.nodes[a].height.max(self.height(d));
        } else {
            self.nodes[b].right = e;
            self.nodes[a].left = d;
            if e != NULL { self.nodes[e].parent = b; }
            if d != NULL { self.nodes[d].parent = a; }
            let cb = if c != NULL { self.nodes[c].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            let db = if d != NULL { self.nodes[d].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            let eb = if e != NULL { self.nodes[e].aabb } else { Aabb::new(Vec3::ZERO, Vec3::ZERO) };
            if c != NULL && d != NULL { self.nodes[a].aabb = cb.union(db); }
            self.nodes[a].height = 1 + self.height(c).max(self.height(d));
            self.nodes[b].aabb = self.nodes[a].aabb.union(eb);
            self.nodes[b].height = 1 + self.nodes[a].height.max(self.height(e));
        }
        b
    }
}

impl Default for EditBvh {
    fn default() -> Self { Self::new() }
}
