// Material processing functions

fn apply_material_color(base_color: vec3<f32>) -> vec3<f32> {
    // For this demo, materials just enhance the base color slightly
    // In a full implementation, this would use material properties from a uniform buffer
    return base_color * vec3<f32>(1.0, 0.9, 0.85); // Slightly warm tint
}
