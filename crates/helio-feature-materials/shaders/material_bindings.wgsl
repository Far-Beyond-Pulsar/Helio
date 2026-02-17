// Material data structures and bindings

struct MaterialData {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emissive_strength: f32,
    ao: f32,
};

// Global material ID that can be set per-object
// For now, hardcoded materials by world position
fn get_material_for_fragment(world_pos: vec3<f32>) -> MaterialData {
    var mat: MaterialData;
    mat.base_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    mat.metallic = 0.0;
    mat.roughness = 0.5;
    mat.emissive_strength = 0.0;
    mat.ao = 1.0;
    
    // Detect sky sphere by distance from origin (>40 units = sky)
    let dist_from_origin = length(world_pos);
    if (dist_from_origin > 40.0) {
        // Sky sphere - bright emissive blue gradient
        mat.base_color = vec4<f32>(0.5, 0.7, 1.0, 1.0);
        mat.emissive_strength = 2.0; // Bright emissive
        mat.metallic = 0.0;
        mat.roughness = 1.0;
    }
    
    return mat;
}
