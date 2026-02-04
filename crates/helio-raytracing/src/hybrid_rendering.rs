pub struct HybridRendering {
    pub rasterization_primary: bool,
    pub rt_shadows: bool,
    pub rt_reflections: bool,
    pub rt_gi: bool,
    pub rt_ao: bool,
}

impl Default for HybridRendering {
    fn default() -> Self {
        Self {
            rasterization_primary: true,
            rt_shadows: false,
            rt_reflections: false,
            rt_gi: false,
            rt_ao: false,
        }
    }
}
