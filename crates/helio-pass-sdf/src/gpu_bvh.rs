//! Flat GPU-compatible BVH built from SDF edit bounding spheres.
//!
//! The CPU builds this tree once whenever the edit list changes and uploads
//! the flat node array to a GPU storage buffer.  The classify shader then
//! traverses the tree once per brick — no CPU brick iteration ever occurs.
//!
//! Node layout is 32 bytes, matching WGSL `struct GpuBvhNode` exactly.
//! See `sdf_classify.wgsl` for the WGSL counterpart.

/// Sentinel value stored in `right_or_leaf` of a leaf node.
pub const LEAF_SENTINEL: u32 = 0xFFFF_FFFF;

/// Maximum number of supported edits (initial capacity; buffer grows on demand).
pub const MAX_EDIT_CAPACITY: usize = 1024;

/// GPU BVH node — 32 bytes, 16-byte aligned.
///
/// ```text
/// Bytes  0-11 : aabb_min  (vec3<f32>)
/// Bytes 12-15 : left_or_edit_idx (u32)
/// Bytes 16-27 : aabb_max  (vec3<f32>)
/// Bytes 28-31 : right_or_leaf    (u32)
/// ```
///
/// | right_or_leaf      | meaning                                    |
/// |--------------------|--------------------------------------------|
/// | `LEAF_SENTINEL`    | leaf — `left_or_edit_idx` is the edit idx  |
/// | any other value    | internal — left/right are child indices    |
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    pub aabb_min: [f32; 3],
    /// Internal: left child index.  Leaf: edit index.
    pub left_or_edit_idx: u32,
    pub aabb_max: [f32; 3],
    /// Internal: right child index.  Leaf: `LEAF_SENTINEL`.
    pub right_or_leaf: u32,
}

impl GpuBvhNode {
    pub fn leaf(min: [f32; 3], max: [f32; 3], edit_idx: u32) -> Self {
        Self { aabb_min: min, left_or_edit_idx: edit_idx, aabb_max: max, right_or_leaf: LEAF_SENTINEL }
    }
    pub fn internal(min: [f32; 3], max: [f32; 3], left: u32, right: u32) -> Self {
        Self { aabb_min: min, left_or_edit_idx: left, aabb_max: max, right_or_leaf: right }
    }
    pub fn is_leaf(&self) -> bool {
        self.right_or_leaf == LEAF_SENTINEL
    }
}

/// Build a packed flat BVH from a list of `(center, radius)` bounding sphere pairs.
///
/// Returns an array of `GpuBvhNode` with the root at index 0, ready to upload
/// to the GPU.  Returns a single degenerate leaf if `bounds` is empty so the
/// shader can safely index node 0 without a bounds check.
///
/// Construction uses surface-area heuristic (SAH) median split — fast to build
/// on the CPU for typical edit counts (≤ a few thousand).
pub fn build_flat_bvh(bounds: &[(glam::Vec3, f32)]) -> Vec<GpuBvhNode> {
    if bounds.is_empty() {
        // Degenerate empty tree: one leaf whose AABB is inside-out so every
        // brick AABB test immediately fails.
        return vec![GpuBvhNode::leaf([1e10; 3], [-1e10; 3], 0)];
    }

    let n = bounds.len();
    let mut nodes: Vec<GpuBvhNode> = Vec::with_capacity(2 * n);
    let mut indices: Vec<u32> = (0..n as u32).collect();
    build_recursive(&mut nodes, bounds, &mut indices);
    nodes
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn enclosing_aabb(bounds: &[(glam::Vec3, f32)], indices: &[u32]) -> (glam::Vec3, glam::Vec3) {
    let mut lo = glam::Vec3::splat(f32::MAX);
    let mut hi = glam::Vec3::splat(f32::MIN);
    for &i in indices {
        let (c, r) = bounds[i as usize];
        lo = lo.min(c - glam::Vec3::splat(r));
        hi = hi.max(c + glam::Vec3::splat(r));
    }
    (lo, hi)
}

fn build_recursive(
    nodes: &mut Vec<GpuBvhNode>,
    bounds: &[(glam::Vec3, f32)],
    indices: &mut [u32],
) -> u32 {
    let node_idx = nodes.len() as u32;

    if indices.len() == 1 {
        let (c, r) = bounds[indices[0] as usize];
        let lo = (c - glam::Vec3::splat(r)).to_array();
        let hi = (c + glam::Vec3::splat(r)).to_array();
        nodes.push(GpuBvhNode::leaf(lo, hi, indices[0]));
        return node_idx;
    }

    // Push placeholder; we'll fill it once children are placed.
    nodes.push(GpuBvhNode::leaf([0.0; 3], [0.0; 3], 0));

    let (lo, hi) = enclosing_aabb(bounds, indices);
    let extent = hi - lo;

    // Choose longest axis for median split.
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0usize
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    indices.sort_unstable_by(|&a, &b| {
        bounds[a as usize].0[axis]
            .partial_cmp(&bounds[b as usize].0[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = indices.len() / 2;
    let (left_idx, right_idx) = indices.split_at_mut(mid);

    let left = build_recursive(nodes, bounds, left_idx);
    let right = build_recursive(nodes, bounds, right_idx);

    nodes[node_idx as usize] = GpuBvhNode::internal(lo.to_array(), hi.to_array(), left, right);
    node_idx
}
