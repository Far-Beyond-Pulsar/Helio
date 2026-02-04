pub struct ComputeCulling {
    pub enabled: bool,
    pub workgroup_size: u32,
}

impl Default for ComputeCulling {
    fn default() -> Self {
        Self {
            enabled: true,
            workgroup_size: 64,
        }
    }
}
