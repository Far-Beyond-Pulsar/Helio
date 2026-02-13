// Basic lighting functions

fn calculate_diffuse_lighting(normal: vec3<f32>, light_dir: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let ndotl = max(dot(normal, light_dir), 0.0);
    return base_color * ndotl;
}

fn calculate_ambient_lighting(base_color: vec3<f32>, ambient_strength: f32) -> vec3<f32> {
    return base_color * ambient_strength;
}

fn apply_basic_lighting(world_normal: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    // Simple directional light from top-right
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));

    let diffuse = calculate_diffuse_lighting(world_normal, light_dir, base_color);
    let ambient = calculate_ambient_lighting(base_color, 0.2);

    return diffuse + ambient;
}
