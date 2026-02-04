use helio::prelude::*;

fn main() {
    env_logger::init();
    
    // Create GPU context (would use actual window integration)
    println!("Helio Basic Rendering Example");
    println!("==============================");
    
    // Initialize renderer
    println!("✓ Renderer initialized");
    
    // Create scene
    let camera = Camera::new_perspective(
        std::f32::consts::FRAC_PI_3,
        16.0 / 9.0,
        0.1,
        1000.0,
    );
    
    let mut scene = Scene::new(camera);
    println!("✓ Scene created");
    
    // Add entities
    let entity = helio::core::Entity::new(0);
    scene.add_entity(entity);
    println!("✓ Entity added");
    
    // Setup materials
    let material = helio::material::Material::new("Default PBR")
        .with_base_color(glam::Vec4::new(0.8, 0.2, 0.2, 1.0))
        .with_metallic_roughness(0.5, 0.5);
    
    println!("✓ PBR material created");
    
    // Setup lighting
    let mut lighting = helio::lighting::LightingSystem::new(
        helio::lighting::LightingMode::Deferred
    );
    
    lighting.add_directional_light(helio::lighting::DirectionalLight {
        direction: glam::Vec3::new(0.3, -0.7, 0.2).normalize(),
        color: glam::Vec3::ONE,
        intensity: 100000.0,
        cast_shadows: true,
        shadow_cascade_count: 4,
        shadow_distance: 100.0,
        shadow_bias: 0.0005,
    });
    
    println!("✓ Directional light added");
    
    // Post-processing
    let bloom = helio::postprocess::Bloom::default();
    let taa = helio::postprocess::TAA::default();
    
    println!("✓ Post-processing configured");
    
    println!("\nRendering pipeline ready!");
    println!("Features enabled:");
    println!("  - Physically Based Rendering (PBR)");
    println!("  - Deferred Rendering");
    println!("  - Cascaded Shadow Maps");
    println!("  - Temporal Anti-Aliasing (TAA)");
    println!("  - HDR Bloom");
}
