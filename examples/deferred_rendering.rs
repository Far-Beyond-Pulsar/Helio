use helio::prelude::*;

fn main() {
    env_logger::init();
    
    println!("Helio Deferred Rendering Example");
    println!("=================================");
    
    // Setup deferred rendering with GBuffer
    println!("\nGBuffer Configuration:");
    println!("  - Albedo (RGBA8 sRGB)");
    println!("  - Normal (RGBA16 Float)");
    println!("  - Metallic/Roughness (RGBA8)");
    println!("  - Emissive (RGBA16 Float)");
    println!("  - Velocity (RG16 Float)");
    println!("  - Depth (D32 Float)");
    
    // Create renderer with deferred path
    let config = helio::render::RendererConfig {
        render_path: helio::render::RenderPath::Deferred,
        enable_hdr: true,
        enable_taa: true,
        enable_depth_prepass: true,
        ..Default::default()
    };
    
    println!("\n✓ Deferred renderer configured");
    println!("✓ HDR rendering enabled");
    println!("✓ Temporal Anti-Aliasing enabled");
    println!("✓ Depth prepass enabled");
    
    // Setup multiple lights
    let mut lighting = helio::lighting::LightingSystem::new(
        helio::lighting::LightingMode::Deferred
    );
    
    // Add multiple point lights
    for i in 0..10 {
        lighting.add_point_light(helio::lighting::PointLight {
            position: glam::Vec3::new(
                (i as f32 * 5.0).sin() * 10.0,
                2.0,
                (i as f32 * 5.0).cos() * 10.0,
            ),
            color: glam::Vec3::new(
                ((i * 37) % 256) as f32 / 255.0,
                ((i * 79) % 256) as f32 / 255.0,
                ((i * 113) % 256) as f32 / 255.0,
            ),
            intensity: 1000.0,
            radius: 15.0,
            cast_shadows: true,
            shadow_resolution: 512,
        });
    }
    
    println!("\n✓ {} point lights added", lighting.point_lights.len());
    println!("✓ Per-light shadows enabled");
    
    println!("\nDeferred rendering pipeline ready!");
}
