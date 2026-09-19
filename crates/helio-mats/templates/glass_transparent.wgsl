// Glass transparent template for the transparent pass.
// Uses `radiant_eval_transparent` signature:
//   (material_id: u32, world_pos: vec3f, world_normal: vec3f, tex_coords: vec2f) -> vec4f

fn radiant_eval_transparent(material_id: u32,
                            world_pos: vec3<f32>,
                            world_normal: vec3<f32>,
                            tex_coords: vec2<f32>) -> vec4<f32> {
    // DEBUG: bright green so we can see it
    return vec4<f32>(0.0, 1.0, 0.0, 0.8);
}