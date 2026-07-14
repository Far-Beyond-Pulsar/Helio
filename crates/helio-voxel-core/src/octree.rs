use crate::constants::*;
use std::fmt;

/// Six complete octree subdivisions produce 8^6 = 262,144 terminal bricks,
/// exactly the existing per-volume brick limit. A deeper complete tree cannot
/// be represented by the retained voxel path's GPU budget.
pub const MAX_OCTREE_DEPTH: u32 = max_complete_depth(MAX_BRICKS_PER_VOLUME);

const fn max_complete_depth(mut leaf_budget: u32) -> u32 {
    let mut depth = 0;
    while leaf_budget >= 8 {
        leaf_budget /= 8;
        depth += 1;
    }
    depth
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OctreeDepthError {
    pub requested: u32,
    pub maximum: u32,
}

impl fmt::Display for OctreeDepthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "requested voxel octree depth {} exceeds the supported maximum {}",
            self.requested, self.maximum
        )
    }
}

impl std::error::Error for OctreeDepthError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Empty,
    Solid(u8),
    Leaf,
    Branch,
}

#[derive(Debug, Clone)]
pub struct OctreeNode {
    pub state: NodeState,
    pub gpu_slot: u32,
    pub dirty: bool,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub children: Option<Box<[Option<Box<OctreeNode>>; 8]>>,
}

impl OctreeNode {
    pub fn new_empty(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            state: NodeState::Empty,
            gpu_slot: BRICK_EMPTY,
            dirty: false,
            aabb_min,
            aabb_max,
            children: None,
        }
    }

    pub fn new_solid(material: u8, aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            state: NodeState::Solid(material),
            gpu_slot: BRICK_SOLID,
            dirty: false,
            aabb_min,
            aabb_max,
            children: None,
        }
    }

    pub fn new_leaf(gpu_slot: u32, aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            state: NodeState::Leaf,
            gpu_slot,
            dirty: true,
            aabb_min,
            aabb_max,
            children: None,
        }
    }

    pub fn subdivide(&mut self) {
        if self.children.is_some() {
            return;
        }
        let mid = [
            (self.aabb_min[0] + self.aabb_max[0]) * 0.5,
            (self.aabb_min[1] + self.aabb_max[1]) * 0.5,
            (self.aabb_min[2] + self.aabb_max[2]) * 0.5,
        ];
        let mut children: [Option<Box<OctreeNode>>; 8] = Default::default();
        for (i, child) in children.iter_mut().enumerate() {
            let (lx, ly, lz) = ((i & 1) != 0, (i & 2) != 0, (i & 4) != 0);
            let min = [
                if lx { mid[0] } else { self.aabb_min[0] },
                if ly { mid[1] } else { self.aabb_min[1] },
                if lz { mid[2] } else { self.aabb_min[2] },
            ];
            let max = [
                if lx { self.aabb_max[0] } else { mid[0] },
                if ly { self.aabb_max[1] } else { mid[1] },
                if lz { self.aabb_max[2] } else { mid[2] },
            ];
            *child = Some(Box::new(OctreeNode::new_empty(min, max)));
        }
        self.children = Some(Box::new(children));
        self.state = NodeState::Branch;
    }

    pub fn collect_leaves<'a>(&'a self, leaves: &mut Vec<&'a OctreeNode>) {
        if let Some(ref children) = self.children {
            for child in children.iter().flatten() {
                child.collect_leaves(leaves);
            }
        } else {
            // "Leaf" here is structural: constant Empty/Solid regions and
            // allocated GPU Leaf nodes are all terminal octree nodes.
            leaves.push(self);
        }
    }

    pub fn collect_dirty_leaves<'a>(&'a self, leaves: &mut Vec<&'a OctreeNode>) {
        if self.dirty && matches!(self.state, NodeState::Leaf) {
            leaves.push(self);
        }
        if let Some(ref children) = self.children {
            for child in children.iter().flatten() {
                child.collect_dirty_leaves(leaves);
            }
        }
    }

    pub fn mark_sphere_dirty(&mut self, center: [f32; 3], radius: f32) -> bool {
        let cx = (self.aabb_min[0] + self.aabb_max[0]) * 0.5;
        let cy = (self.aabb_min[1] + self.aabb_max[1]) * 0.5;
        let cz = (self.aabb_min[2] + self.aabb_max[2]) * 0.5;
        let hx = (self.aabb_max[0] - self.aabb_min[0]) * 0.5;
        let hy = (self.aabb_max[1] - self.aabb_min[1]) * 0.5;
        let hz = (self.aabb_max[2] - self.aabb_min[2]) * 0.5;
        let dx = (center[0] - cx).abs();
        let dy = (center[1] - cy).abs();
        let dz = (center[2] - cz).abs();
        if dx > hx + radius || dy > hy + radius || dz > hz + radius {
            return false;
        }
        match &mut self.state {
            NodeState::Leaf => {
                self.dirty = true;
                true
            }
            NodeState::Branch => {
                if let Some(ref mut children) = self.children {
                    let mut any = false;
                    for child in children.iter_mut().flatten() {
                        any |= child.mark_sphere_dirty(center, radius);
                    }
                    if any {
                        self.dirty = true;
                    }
                    any
                } else {
                    false
                }
            }
            NodeState::Empty | NodeState::Solid(_) => false,
        }
    }

    pub fn node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .flat_map(|children| children.iter().flatten())
            .map(|child| child.node_count())
            .sum::<usize>()
    }

    fn ensure_empty_depth(&mut self, remaining_depth: u32) {
        if remaining_depth == 0 {
            return;
        }

        match self.state {
            NodeState::Empty => self.subdivide(),
            NodeState::Branch => {}
            // Constant solid regions and materialized GPU leaves are already
            // valid sparse terminals. Expanding them as Empty would destroy
            // terrain or discard their GPU slot.
            NodeState::Solid(_) | NodeState::Leaf => return,
        }

        if let Some(children) = self.children.as_mut() {
            for child in children.iter_mut().flatten() {
                child.ensure_empty_depth(remaining_depth - 1);
            }
        }
    }
}

#[derive(Debug)]
pub struct VoxelOctree {
    pub root: OctreeNode,
    pub brick_size: u32,
    pub voxel_size: f32,
    pub level_count: u32,
}

impl VoxelOctree {
    pub fn new(voxel_size: f32, root_extent: f32) -> Self {
        let half = root_extent * 0.5;
        Self {
            root: OctreeNode::new_empty([-half, -half, -half], [half, half, half]),
            brick_size: BRICK_SIZE,
            voxel_size,
            level_count: 1,
        }
    }

    pub fn needed_depth(world_size: f32, voxel_size: f32, brick_size: u32) -> u32 {
        if !world_size.is_finite()
            || !voxel_size.is_finite()
            || world_size <= 0.0
            || voxel_size <= 0.0
            || brick_size == 0
        {
            return 0;
        }
        let brick_world = brick_size as f32 * voxel_size;
        if !brick_world.is_finite() || world_size <= brick_world {
            return 0;
        }
        let mut depth = 0;
        let mut size = world_size;
        while size > brick_world {
            size *= 0.5;
            depth += 1;
        }
        depth
    }

    /// Ensure `target_depth` subdivision levels below the root.
    ///
    /// The retained voxel path can represent at most `MAX_OCTREE_DEPTH`
    /// complete levels. Call [`Self::try_ensure_depth`] when the requested
    /// depth is derived from untrusted or runtime data.
    pub fn ensure_depth(&mut self, target_depth: u32) {
        self.try_ensure_depth(target_depth)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn try_ensure_depth(&mut self, target_depth: u32) -> Result<(), OctreeDepthError> {
        if target_depth > MAX_OCTREE_DEPTH {
            return Err(OctreeDepthError {
                requested: target_depth,
                maximum: MAX_OCTREE_DEPTH,
            });
        }
        let current_depth = self.level_count.saturating_sub(1);
        if target_depth > current_depth {
            self.root.ensure_empty_depth(target_depth);
            self.level_count = target_depth + 1;
        }
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octree_basic() {
        let octree = VoxelOctree::new(1.0, 256.0);
        assert_eq!(octree.level_count, 1);
        assert!(matches!(octree.root.state, NodeState::Empty));
        assert_eq!(octree.node_count(), 1);
    }

    #[test]
    fn needed_depth_is_exact_and_rejects_invalid_sizes_without_looping() {
        assert_eq!(VoxelOctree::needed_depth(256.0, 1.0, 8), 5);
        assert_eq!(VoxelOctree::needed_depth(8.0, 1.0, 8), 0);
        assert_eq!(VoxelOctree::needed_depth(7.0, 1.0, 8), 0);
        assert_eq!(VoxelOctree::needed_depth(f32::INFINITY, 1.0, 8), 0);
        assert_eq!(VoxelOctree::needed_depth(256.0, -1.0, 8), 0);
        assert_eq!(VoxelOctree::needed_depth(256.0, 1.0, 0), 0);
    }

    #[test]
    fn test_mark_dirty() {
        let mut octree = VoxelOctree::new(1.0, 256.0);
        octree.root.subdivide();
        if let Some(ref mut children) = octree.root.children {
            children[0] = Some(Box::new(OctreeNode::new_leaf(0, [0.0; 3], [1.0; 3])));
        }
        assert!(octree.root.mark_sphere_dirty([0.5, 0.5, 0.5], 2.0));
    }

    #[test]
    fn test_collect_leaves() {
        for depth in 0..=4 {
            let mut octree = VoxelOctree::new(1.0, 256.0);
            octree.ensure_depth(depth);
            let mut leaves = vec![];
            octree.root.collect_leaves(&mut leaves);
            assert_eq!(leaves.len(), 8_usize.pow(depth));
            assert_eq!(octree.level_count, depth + 1);
        }
    }

    #[test]
    fn ensure_depth_is_idempotent() {
        let mut octree = VoxelOctree::new(1.0, 256.0);
        octree.ensure_depth(3);
        let node_count = octree.node_count();
        octree.ensure_depth(3);
        octree.ensure_depth(2);
        assert_eq!(octree.node_count(), node_count);
        assert_eq!(octree.level_count, 4);
    }

    #[test]
    fn ensure_depth_preserves_sparse_solid_and_gpu_leaf_terminals() {
        let mut octree = VoxelOctree::new(1.0, 256.0);
        octree.root.subdivide();
        let children = octree.root.children.as_mut().unwrap();
        children[0] = Some(Box::new(OctreeNode::new_solid(7, [-128.0; 3], [0.0; 3])));
        children[1] = Some(Box::new(OctreeNode::new_leaf(
            42,
            [0.0, -128.0, -128.0],
            [128.0, 0.0, 0.0],
        )));

        octree.try_ensure_depth(3).unwrap();

        let children = octree.root.children.as_ref().unwrap();
        assert!(matches!(
            children[0].as_deref().unwrap().state,
            NodeState::Solid(7)
        ));
        assert!(children[0].as_deref().unwrap().children.is_none());
        assert!(matches!(
            children[1].as_deref().unwrap().state,
            NodeState::Leaf
        ));
        assert_eq!(children[1].as_deref().unwrap().gpu_slot, 42);
        assert!(children[1].as_deref().unwrap().children.is_none());
    }

    #[test]
    fn checked_depth_rejects_exponential_over_budget_growth() {
        assert_eq!(8_u32.pow(MAX_OCTREE_DEPTH), MAX_BRICKS_PER_VOLUME);
        let mut octree = VoxelOctree::new(1.0, 256.0);
        assert_eq!(
            octree.try_ensure_depth(MAX_OCTREE_DEPTH + 1),
            Err(OctreeDepthError {
                requested: MAX_OCTREE_DEPTH + 1,
                maximum: MAX_OCTREE_DEPTH,
            })
        );
        assert_eq!(octree.node_count(), 1);
    }

    #[test]
    #[should_panic(expected = "exceeds the supported maximum")]
    fn compatibility_depth_api_panics_before_over_budget_allocation() {
        let mut octree = VoxelOctree::new(1.0, 256.0);
        octree.ensure_depth(MAX_OCTREE_DEPTH + 1);
    }
}
