// raytracing system implementation
pub struct raytracingSystem {
    enabled: bool,
}

impl raytracingSystem {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

impl Default for raytracingSystem {
    fn default() -> Self {
        Self::new()
    }
}
