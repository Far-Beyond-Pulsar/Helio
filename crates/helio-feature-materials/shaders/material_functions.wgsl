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

fn apply_material_color(base_color: vec3<f32>, tex_coords: vec2<f32>) -> vec3<f32> {
    // Sample the procedural texture
    let texture_color = get_texture_color(tex_coords);

    // Blend texture with base color
    return base_color * texture_color;
}

// Get emissive color based on object position (for demo purposes)
// In a real implementation, this would come from per-object material data
fn get_emissive_color(world_pos: vec3<f32>) -> vec3<f32> {
    // Make certain objects glow based on their position
    // Objects above y=2.0 will have a warm glow
    if (world_pos.y > 2.0) {
        return vec3<f32>(1.5, 0.8, 0.3) * 0.8; // Warm orange glow
    }
    // Objects near z=-3.0 will have a cool glow
    else if (abs(world_pos.z + 3.0) < 0.5 && world_pos.y > 0.0) {
        return vec3<f32>(0.3, 0.8, 1.5) * 0.6; // Cool blue glow
    }
    return vec3<f32>(0.0, 0.0, 0.0); // No emission
}
