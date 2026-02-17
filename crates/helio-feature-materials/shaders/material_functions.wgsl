// Material processing functions with procedural textures

fn checkerboard_pattern(uv: vec2<f32>, scale: f32) -> f32 {
    let scaled_uv = uv * scale;
    let checker = floor(scaled_uv.x) + floor(scaled_uv.y);
    return fract(checker * 0.5) * 2.0;
}

fn get_texture_color(uv: vec2<f32>) -> vec3<f32> {
    // Create a procedural checkerboard texture
    let checker = checkerboard_pattern(uv, 8.0);

    // Alternate between two colors
    let color1 = vec3<f32>(0.9, 0.9, 0.9); // Light gray
    let color2 = vec3<f32>(0.3, 0.5, 0.7); // Blue-gray

    return mix(color2, color1, checker);
}

fn apply_material_color(base_color: vec3<f32>, tex_coords: vec2<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    // Get material data for this fragment
    let material = get_material_for_fragment(world_pos);
    
    // If emissive, skip texture and return bright emissive color
    if (material.emissive_strength > 0.0) {
        return material.base_color.rgb * material.emissive_strength;
    }
    
    // Normal textured material
    let texture_color = get_texture_color(tex_coords);
    return base_color * texture_color * material.base_color.rgb;
}
