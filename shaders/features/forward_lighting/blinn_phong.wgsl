// Forward lighting shader functions

fn calculate_lighting(world_normal: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ambient = vec3<f32>(0.2);
    let diffuse = max(dot(world_normal, light_dir), 0.0);
    
    return base_color * (ambient + diffuse * vec3<f32>(0.8));
}
