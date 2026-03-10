// Hi-Z (Hierarchical Z-Buffer) mipmap pyramid build pass.
//
// Two entry points:
//   1. `copy_depth`  — copies the depth prepass texture into mip 0 of the
//                      Hi-Z R32Float pyramid as a plain float value.
//   2. `downsample`  — reduces mip N-1 into mip N by taking the MAX of each
//                      2×2 block (conservative: the farthest occluder in the
//                      region, so a test object must be even farther to be culled).
//
// The Hi-Z pyramid is a separate R32Float texture (not the depth texture itself)
// so it can be bound as both a sampled texture and a storage texture.
//
// Convention: depth 0 = near, 1 = far (standard wgpu / Vulkan convention).
// Storing MAX means: a subsequent sample represents the farthest visible surface
// in the region.  An object is occluded only if its MINIMUM projected depth
// exceeds the MAX depth in the region → it is entirely behind existing geometry.

// ── Pass 1: copy depth texture → Hi-Z mip 0 ─────────────────────────────────

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var hiz_mip0:  texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn copy_depth(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let size = textureDimensions(hiz_mip0);
    if gid.x >= size.x || gid.y >= size.y { return; }

    // textureLoad on a depth texture returns the depth value directly.
    let d = textureLoad(depth_tex, vec2i(gid.xy), 0);
    textureStore(hiz_mip0, vec2i(gid.xy), vec4f(d, 0.0, 0.0, 1.0));
}

// ── Pass 2: downsample mip N-1 → mip N ──────────────────────────────────────

@group(0) @binding(0) var src_mip: texture_2d<f32>;
@group(0) @binding(1) var dst_mip: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn downsample(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let dst_size = textureDimensions(dst_mip);
    if gid.x >= dst_size.x || gid.y >= dst_size.y { return; }

    // Sample the four corresponding texels from the parent mip (mip N-1).
    // Use max to be conservative: keep the farthest (largest depth) occluder.
    let src_coord = vec2i(gid.xy) * 2;
    let src_size  = vec2i(textureDimensions(src_mip));

    let clamp_coord = |c: vec2i| -> vec2i {
        vec2i(clamp(c.x, 0, src_size.x - 1), clamp(c.y, 0, src_size.y - 1))
    };

    let d00 = textureLoad(src_mip, clamp_coord(src_coord + vec2i(0, 0)), 0).r;
    let d10 = textureLoad(src_mip, clamp_coord(src_coord + vec2i(1, 0)), 0).r;
    let d01 = textureLoad(src_mip, clamp_coord(src_coord + vec2i(0, 1)), 0).r;
    let d11 = textureLoad(src_mip, clamp_coord(src_coord + vec2i(1, 1)), 0).r;

    let max_depth = max(max(d00, d10), max(d01, d11));
    textureStore(dst_mip, vec2i(gid.xy), vec4f(max_depth, 0.0, 0.0, 1.0));
}
