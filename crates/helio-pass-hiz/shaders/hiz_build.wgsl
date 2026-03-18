// Hi-Z pyramid builder. Downsamples depth using max-reduction.

struct HiZUniforms {
    src_size: vec2<u32>,
    dst_size: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params:  HiZUniforms;
@group(0) @binding(1) var src_tex:          texture_2d<f32>;
@group(0) @binding(2) var dst_tex:          texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_coord = gid.xy;
    if any(dst_coord >= params.dst_size) { return; }

    let src_coord = dst_coord * 2u;
    let s00 = textureLoad(src_tex, vec2<i32>(src_coord), 0).r;
    let s10 = textureLoad(src_tex, vec2<i32>(min(src_coord + vec2<u32>(1u, 0u), params.src_size - 1u)), 0).r;
    let s01 = textureLoad(src_tex, vec2<i32>(min(src_coord + vec2<u32>(0u, 1u), params.src_size - 1u)), 0).r;
    let s11 = textureLoad(src_tex, vec2<i32>(min(src_coord + vec2<u32>(1u, 1u), params.src_size - 1u)), 0).r;
    textureStore(dst_tex, vec2<i32>(dst_coord), vec4<f32>(max(max(s00, s10), max(s01, s11)), 0.0, 0.0, 1.0));
}
