use helio::prelude::*;

fn main() {
    env_logger::init();
    
    println!("Helio Particle Systems Example");
    println!("==============================");
    
    // Create GPU particle system
    let gpu_particles = helio::particles::GPUParticles::new(1000000);
    println!("\n✓ GPU Particle System created");
    println!("  - Max particles: {}", gpu_particles.max_particles);
    println!("  - GPU simulation: {}", gpu_particles.simulation_buffer.is_some());
    println!("  - Particle sorting: {}", gpu_particles.sort_particles);
    
    // Create emitters
    let fire_emitter = helio::particles::ParticleEmitter {
        position: glam::Vec3::ZERO,
        shape: helio::particles::EmitterShape::Cone,
        emission_rate: 100.0,
        initial_velocity: glam::Vec3::Y * 5.0,
        lifetime: 2.0,
        initial_color: glam::Vec4::new(1.0, 0.5, 0.0, 1.0),
        ..Default::default()
    };
    
    println!("\n✓ Fire emitter created");
    println!("  - Shape: Cone");
    println!("  - Emission rate: {} particles/sec", fire_emitter.emission_rate);
    
    let smoke_emitter = helio::particles::ParticleEmitter {
        position: glam::Vec3::new(0.0, 2.0, 0.0),
        shape: helio::particles::EmitterShape::Sphere,
        emission_rate: 50.0,
        initial_velocity: glam::Vec3::Y * 2.0,
        lifetime: 5.0,
        initial_color: glam::Vec4::new(0.5, 0.5, 0.5, 0.5),
        ..Default::default()
    };
    
    println!("✓ Smoke emitter created");
    println!("  - Shape: Sphere");
    println!("  - Emission rate: {} particles/sec", smoke_emitter.emission_rate);
    
    // Particle modules
    println!("\n✓ Particle modules available:");
    println!("  - Color over lifetime");
    println!("  - Size over lifetime");
    println!("  - Velocity over lifetime");
    println!("  - Force over lifetime");
    println!("  - Collision detection");
    
    println!("\nParticle systems ready!");
    println!("Total capacity: {} million particles", gpu_particles.max_particles / 1000000);
}
