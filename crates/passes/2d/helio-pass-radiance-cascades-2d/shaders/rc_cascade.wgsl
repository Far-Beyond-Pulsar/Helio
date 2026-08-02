// The radiance cascades merge + raymarch — a fairly direct port of the
// `rcShader` fragment shader from
// https://raw.githubusercontent.com/radiance-cascades/radiance-cascades.com/refs/heads/main/public/js/rc.js
// to a WGSL compute shader (one thread per output texel instead of one
// fragment-shader invocation per pixel; `textureLod`/`texelFetch` become
// `textureSampleLevel`/`textureLoad`).
//
// Dispatched once per cascade level, from the coarsest (highest index) down
// to 0, ping-ponging between two `cascadeExtent`-sized textures: each level
// raymarches its own probes against the distance field, then *merges* in
// the next-coarser cascade's already-computed result (`last_tex`) wherever
// its own raymarch didn't hit anything (`current.a == 0`) — this is what
// makes farther cascades (fewer, sparser probes; each covering a longer,
// outward-shifted ray interval so consecutive cascades' intervals tile
// without gaps) fill in indirect/bounced light for near cascades without
// every level needing to trace all the way to the horizon itself.
//
// `cascadeExtent` is fixed at construction (`scene_size`), matching every
// level to the same texture resolution: a level's `spacing` (probe grid
// cell size in texels) grows with its index while the number of angular
// "ray buckets" tiled across that same texel budget grows to match, so the
// total resolution needed stays constant across levels — see `size`/
// `ray_pos`/`probe_relative_position` below.

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

fn fmod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}
fn fmod2(v: vec2<f32>, y: f32) -> vec2<f32> {
    return vec2<f32>(fmod(v.x, y), fmod(v.y, y));
}
fn fmod2v(v: vec2<f32>, m: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(fmod(v.x, m.x), fmod(v.y, m.y));
}

struct CascadeUniforms {
    cascade_index: f32,
    cascade_count: f32,
    base_ray_count: f32,
    base_pixels_between_probes: f32,
    cascade_interval: f32,
    ray_interval: f32,
    interval_overlap: f32,
    is_top_cascade: f32,
    scene_size: vec2<f32>,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> cu: CascadeUniforms;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var dist_tex: texture_2d<f32>;
@group(0) @binding(3) var last_tex: texture_2d<f32>;
@group(0) @binding(4) var rc_sampler: sampler;
@group(0) @binding(5) var out_tex: texture_storage_2d<rgba16float, write>;

fn raymarch(ray_start: vec2<f32>, ray_end: vec2<f32>, scale: f32, one_over_size: vec2<f32>, min_step: f32) -> vec4<f32> {
    let ray_dir = normalize(ray_end - ray_start);
    let ray_length = length(ray_end - ray_start);
    var ray_uv = ray_start * one_over_size;
    var dist = 0.0;

    for (var i = 0; i < 256; i = i + 1) {
        if (dist >= ray_length) {
            break;
        }
        if (ray_uv.x < 0.0 || ray_uv.x > 1.0 || ray_uv.y < 0.0 || ray_uv.y > 1.0) {
            break;
        }
        let df = textureSampleLevel(dist_tex, rc_sampler, ray_uv, 0.0).r;
        if (df <= min_step) {
            return textureSampleLevel(scene_tex, rc_sampler, ray_uv, 0.0);
        }
        dist += df * scale;
        ray_uv += ray_dir * (df * scale * one_over_size);
    }
    return vec4<f32>(0.0);
}

fn get_upper_cascade_uv(index: f32, offset: vec2<f32>, spacing_base: f32, cascade_index: f32, cascade_extent: vec2<f32>) -> vec2<f32> {
    let upper_spacing = pow(spacing_base, cascade_index + 1.0);
    let upper_size = floor(cascade_extent / upper_spacing);
    let upper_position = vec2<f32>(fmod(index, upper_spacing), floor(index / upper_spacing)) * upper_size;
    let clamped = clamp(offset, vec2<f32>(0.5), upper_size - vec2<f32>(0.5));
    return (upper_position + clamped) / cascade_extent;
}

fn merge(
    current: vec4<f32>,
    index: f32,
    position: vec2<f32>,
    spacing_base: f32,
    local_offset: vec2<f32>,
    cascade_index: f32,
    cascade_count: f32,
    cascade_extent: vec2<f32>,
    base_pixels_between_probes: f32,
    is_top: bool,
) -> vec4<f32> {
    if (current.a > 0.0 || cascade_index >= max(1.0, cascade_count - 1.0) || is_top) {
        return current;
    }
    let offset = (position + local_offset) / spacing_base;
    let upper_uv = get_upper_cascade_uv(index, offset, spacing_base, cascade_index, cascade_extent);
    var lod = log2(max(base_pixels_between_probes, 1.0));
    if (base_pixels_between_probes == 1.0) {
        lod = 0.0;
    }
    let upper_sample = textureSampleLevel(last_tex, rc_sampler, upper_uv, lod).rgb;
    return current + vec4<f32>(upper_sample, 1.0);
}

@compute @workgroup_size(8, 8)
fn cs_cascade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cascade_extent = cu.scene_size;
    if (f32(gid.x) >= cascade_extent.x || f32(gid.y) >= cascade_extent.y) {
        return;
    }
    let coord = vec2<f32>(f32(gid.x), f32(gid.y));

    let base = cu.base_ray_count;
    let cascade_index = cu.cascade_index;
    let ray_count = pow(base, cascade_index + 1.0);
    let spacing_base = sqrt(base);
    let spacing = pow(spacing_base, cascade_index);

    var modifier_hack = spacing_base;
    if (base < 16.0) {
        modifier_hack = pow(cu.base_pixels_between_probes, 1.0);
    }

    let size = floor(cascade_extent / spacing);
    let probe_relative_position = fmod2v(coord, size);
    let ray_pos = floor(coord / size);

    let modified_interval = modifier_hack * cu.ray_interval * cu.cascade_interval;
    var start = 0.0;
    if (cascade_index > 0.0) {
        start = pow(base, cascade_index - 1.0) * modified_interval;
    }
    let end = ((1.0 + 3.0 * cu.interval_overlap) * pow(base, cascade_index) - pow(cascade_index, 2.0)) * modified_interval;

    let probe_center = (probe_relative_position + 0.5) * cu.base_pixels_between_probes * spacing;
    let pre_avg_amt = base;
    let base_index = (ray_pos.x + spacing * ray_pos.y) * pre_avg_amt;
    let angle_step = TAU / ray_count;

    let scale = min(cascade_extent.x, cascade_extent.y);
    let one_over_size = 1.0 / cascade_extent;
    let min_step = min(one_over_size.x, one_over_size.y) * 0.5;
    let avg_recip = 1.0 / pre_avg_amt;

    var total_radiance = vec4<f32>(0.0);
    let is_top = cu.is_top_cascade > 0.5;

    let iters = i32(pre_avg_amt);
    for (var i = 0; i < iters; i = i + 1) {
        let index = base_index + f32(i);
        let angle = (index + 0.5) * angle_step;
        let ray_dir = vec2<f32>(cos(angle), -sin(angle));
        let ray_start = probe_center + ray_dir * start;
        let ray_end = ray_start + ray_dir * end;
        let raymarched = raymarch(ray_start, ray_end, scale, one_over_size, min_step);
        let merged = merge(
            raymarched, index, probe_relative_position, spacing_base, vec2<f32>(0.5),
            cascade_index, cu.cascade_count, cascade_extent, cu.base_pixels_between_probes, is_top,
        );
        total_radiance += merged * avg_recip;
    }

    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(total_radiance.rgb, 1.0));
}
