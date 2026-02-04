use glam::Vec3;

pub struct IKChain {
    pub bones: Vec<usize>,
    pub target: Vec3,
    pub pole_target: Option<Vec3>,
    pub iterations: u32,
}

pub struct TwoBonenIK {
    pub root_bone: usize,
    pub mid_bone: usize,
    pub tip_bone: usize,
    pub target: Vec3,
    pub pole_target: Vec3,
}

impl IKChain {
    pub fn new() -> Self {
        Self {
            bones: Vec::new(),
            target: Vec3::ZERO,
            pole_target: None,
            iterations: 10,
        }
    }
    
    pub fn solve_fabrik(&mut self) {
        // FABRIK IK solver implementation
    }
}

impl Default for IKChain {
    fn default() -> Self {
        Self::new()
    }
}
