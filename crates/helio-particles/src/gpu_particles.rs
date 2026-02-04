pub struct GPUParticles {
    pub max_particles: u32,
    pub simulation_buffer: Option<u32>,
    pub sort_particles: bool,
    pub collision_enabled: bool,
}

impl GPUParticles {
    pub fn new(max_particles: u32) -> Self {
        Self {
            max_particles,
            simulation_buffer: None,
            sort_particles: true,
            collision_enabled: false,
        }
    }
}
