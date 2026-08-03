//! Conservative Hi-Z seed for downlevel adapters without depth texture copies.

@group(0) @binding(0) var max_mip_0: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var min_mip_0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(max_mip_0);
    if any(id.xy >= size) {
        return;
    }
    textureStore(max_mip_0, id.xy, vec4<f32>(1.0));
    textureStore(min_mip_0, id.xy, vec4<f32>(1.0));
}
