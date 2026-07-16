// Hi-Z pyramid builder. Downsamples depth 2x2 -> 1.
//
// Two entry points, two reductions, one file — because the *policy* (sizes, mip
// counts, dispatch) is identical and only the reduction operator differs:
//
//   main_max — farthest depth in the footprint. Conservative for occlusion
//              culling: "is this tile definitely covered?" (OcclusionCullPass,
//              VirtualGeometryPass).
//   main_min — nearest depth in the footprint. Conservative for ray marching:
//              "could a ray hit anything in this tile?" (SsrPass).
//
// The two are not interchangeable. Marching a max-reduced pyramid makes rays
// tunnel through geometry, because coarse levels claim the nearest surface is
// farther away than it is.

struct HiZUniforms {
    src_size: vec2<u32>,
    dst_size: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params:  HiZUniforms;
@group(0) @binding(1) var src_tex:          texture_2d<f32>;
@group(0) @binding(2) var dst_tex:          texture_storage_2d<r32float, write>;

/// The 2x2 source footprint for a destination texel, clamped to the source edge
/// so odd-sized mips don't sample out of bounds.
fn gather(dst_coord: vec2<u32>) -> vec4<f32> {
    let src_coord = dst_coord * 2u;
    let last = params.src_size - 1u;
    return vec4<f32>(
        textureLoad(src_tex, vec2<i32>(min(src_coord,                     last)), 0).r,
        textureLoad(src_tex, vec2<i32>(min(src_coord + vec2<u32>(1u, 0u), last)), 0).r,
        textureLoad(src_tex, vec2<i32>(min(src_coord + vec2<u32>(0u, 1u), last)), 0).r,
        textureLoad(src_tex, vec2<i32>(min(src_coord + vec2<u32>(1u, 1u), last)), 0).r,
    );
}

@compute @workgroup_size(8, 8)
fn main_max(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= params.dst_size) { return; }
    let s = gather(gid.xy);
    let m = max(max(s.x, s.y), max(s.z, s.w));
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(m, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn main_min(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= params.dst_size) { return; }
    let s = gather(gid.xy);
    let m = min(min(s.x, s.y), min(s.z, s.w));
    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(m, 0.0, 0.0, 1.0));
}
