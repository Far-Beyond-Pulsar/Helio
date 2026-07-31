//! Simple color blit shader - copies from source to destination texture.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dest_texture: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn blit_color(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dimensions = textureDimensions(src_texture);
    let coord = gid.xy;
    
    if coord.x >= dimensions.x || coord.y >= dimensions.y {
        return;
    }

    let color = textureLoad(src_texture, coord, 0);
    textureStore(dest_texture, coord, color);
}
