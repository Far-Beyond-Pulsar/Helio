use helio::prelude::*;

fn main() {
    env_logger::init();
    
    println!("Helio PBR Materials Example");
    println!("===========================");
    
    // Create various PBR materials
    let materials = vec![
        helio::material::Material::new("Plastic")
            .with_base_color(glam::Vec4::new(1.0, 0.0, 0.0, 1.0))
            .with_metallic_roughness(0.0, 0.5),
        
        helio::material::Material::new("Metal")
            .with_base_color(glam::Vec4::new(0.9, 0.9, 0.9, 1.0))
            .with_metallic_roughness(1.0, 0.2),
        
        helio::material::Material::new("Gold")
            .with_base_color(glam::Vec4::new(1.0, 0.766, 0.336, 1.0))
            .with_metallic_roughness(1.0, 0.1),
        
        helio::material::Material::new("Copper")
            .with_base_color(glam::Vec4::new(0.955, 0.638, 0.538, 1.0))
            .with_metallic_roughness(1.0, 0.15),
        
        helio::material::Material::new("Rubber")
            .with_base_color(glam::Vec4::new(0.1, 0.1, 0.1, 1.0))
            .with_metallic_roughness(0.0, 0.9),
        
        helio::material::Material::new("Glass")
            .with_base_color(glam::Vec4::new(1.0, 1.0, 1.0, 0.1))
            .with_metallic_roughness(0.0, 0.0)
            .with_blend_mode(helio::material::BlendMode::Translucent),
        
        helio::material::Material::new("Emissive")
            .with_base_color(glam::Vec4::new(0.0, 0.5, 1.0, 1.0))
            .with_emissive(glam::Vec3::new(0.0, 0.5, 1.0), 10.0),
    ];
    
    println!("\nCreated {} PBR materials:", materials.len());
    for mat in &materials {
        println!("  ✓ {} (Metallic: {:.2}, Roughness: {:.2})", 
            mat.name, mat.metallic, mat.roughness);
    }
    
    // Setup IBL (Image-Based Lighting)
    let ibl = helio::lighting::IBL::new();
    println!("\n✓ Image-Based Lighting configured");
    println!("  - Environment map");
    println!("  - Irradiance map");
    println!("  - Prefiltered specular map");
    println!("  - BRDF lookup table");
    
    println!("\nPBR materials showcase ready!");
}
