use helio::prelude::*;

fn main() {
    env_logger::init();
    
    println!("Helio Shadow Systems Example");
    println!("============================");
    
    // Cascaded Shadow Maps for directional light
    let mut csm = helio::lighting::CascadedShadowMaps::new(4, 2048);
    println!("\n✓ Cascaded Shadow Maps configured");
    println!("  - {} cascades", csm.cascade_count);
    println!("  - {}x{} resolution per cascade", csm.resolution, csm.resolution);
    println!("  - PCSS soft shadows");
    
    // Virtual Shadow Maps
    let vsm = helio::lighting::VirtualShadowMaps::new();
    println!("\n✓ Virtual Shadow Maps configured");
    println!("  - {}x{} virtual resolution", vsm.resolution, vsm.resolution);
    println!("  - {} page size", vsm.page_size);
    println!("  - Clipmap enabled: {}", vsm.enable_clipmap);
    
    // Shadow Atlas for point/spot lights
    let mut atlas = helio::lighting::ShadowAtlas::new(4096);
    println!("\n✓ Shadow Atlas configured");
    println!("  - {}x{} atlas size", atlas.resolution, atlas.resolution);
    
    // Ray-traced shadows
    let rt_shadows = helio::raytracing::RayTracedShadows {
        enabled: true,
        samples: 1,
        max_distance: 100.0,
    };
    println!("\n✓ Ray-traced shadows available");
    println!("  - Hardware RT acceleration");
    println!("  - Soft shadows with area lights");
    println!("  - Sample count: {}", rt_shadows.samples);
    
    println!("\nShadow techniques ready:");
    println!("  1. Cascaded Shadow Maps (CSM)");
    println!("  2. Virtual Shadow Maps (VSM)");
    println!("  3. Shadow Atlas");
    println!("  4. Ray-Traced Shadows");
}
