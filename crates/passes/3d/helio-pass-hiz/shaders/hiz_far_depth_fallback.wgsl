//! Conservative Hi-Z seed for downlevel adapters without depth texture copies.
//!
//! Both pyramids read as empty: the depth clear value (helio_core::depth::CLEAR,
//! infinitely far under reversed depth), so nothing is ever occluded and no
//! ray ever hits.

const FAR_DEPTH: f32 = 0.0;

@group(0) @binding(0) var farthest_mip_0: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var nearest_mip_0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(farthest_mip_0);
    if any(id.xy >= size) {
        return;
    }
    textureStore(farthest_mip_0, id.xy, vec4<f32>(FAR_DEPTH));
    textureStore(nearest_mip_0, id.xy, vec4<f32>(FAR_DEPTH));
}
