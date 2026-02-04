pub struct OcclusionCulling {
    pub enabled: bool,
    pub query_pool_size: u32,
}

impl Default for OcclusionCulling {
    fn default() -> Self {
        Self {
            enabled: true,
            query_pool_size: 1024,
        }
    }
}
