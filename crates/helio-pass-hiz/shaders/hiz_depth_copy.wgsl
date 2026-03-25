// Copies a Depth32Float texture into mip-0 of the Hi-Z R32Float texture.
//
// This is the first step of the Hi-Z pyramid build.  The depth buffer is
// written by DepthPrepassPass as `texture_depth_2d` (Depth32Float), which
// cannot be bound as a storage texture directly.  This pass reads it via
// `textureLoad` and writes each texel into the R32Float storage texture.
//
// One compute thread per output pixel.  Workgroup 8×8 = 64 threads.

@group(0) @binding(0) var depth_src : texture_depth_2d;
@group(0) @binding(1) var hiz_mip0  : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(depth_src);
    if gid.x >= size.x || gid.y >= size.y { return; }
    let depth = textureLoad(depth_src, vec2<i32>(gid.xy), 0);
    textureStore(hiz_mip0, vec2<i32>(gid.xy), vec4<f32>(depth, 0.0, 0.0, 1.0));
}
