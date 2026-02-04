use glam::{Vec3, Mat4};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationStructureType {
    BLAS, // Bottom Level
    TLAS, // Top Level
}

pub struct AccelerationStructure {
    pub structure_type: AccelerationStructureType,
    pub handle: Option<u64>,
    pub build_flags: BuildFlags,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct BuildFlags: u32 {
        const PREFER_FAST_TRACE = 1 << 0;
        const PREFER_FAST_BUILD = 1 << 1;
        const ALLOW_UPDATE = 1 << 2;
        const LOW_MEMORY = 1 << 3;
    }
}

impl AccelerationStructure {
    pub fn new_blas() -> Self {
        Self {
            structure_type: AccelerationStructureType::BLAS,
            handle: None,
            build_flags: BuildFlags::PREFER_FAST_TRACE,
        }
    }
    
    pub fn new_tlas() -> Self {
        Self {
            structure_type: AccelerationStructureType::TLAS,
            handle: None,
            build_flags: BuildFlags::PREFER_FAST_TRACE | BuildFlags::ALLOW_UPDATE,
        }
    }
}

pub struct BVH {
    pub nodes: Vec<BVHNode>,
    pub primitives: Vec<u32>,
}

pub struct BVHNode {
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub left_child: u32,
    pub right_child: u32,
    pub primitive_count: u32,
    pub first_primitive: u32,
}

impl BVH {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            primitives: Vec::new(),
        }
    }
    
    pub fn build(&mut self, positions: &[Vec3], indices: &[u32]) {
        // BVH construction implementation
    }
}

impl Default for BVH {
    fn default() -> Self {
        Self::new()
    }
}
