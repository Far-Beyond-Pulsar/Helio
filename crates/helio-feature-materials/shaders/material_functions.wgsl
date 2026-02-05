// Material processing functions

fn apply_material_color(base_color: vec3<f32>) -> vec3<f32> {
    return base_color * material.base_color.rgb;
}
