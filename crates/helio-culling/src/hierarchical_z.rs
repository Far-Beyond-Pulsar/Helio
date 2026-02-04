pub struct HierarchicalZ {
    pub mip_levels: u32,
    pub enabled: bool,
}

impl Default for HierarchicalZ {
    fn default() -> Self {
        Self {
            mip_levels: 8,
            enabled: true,
        }
    }
}
