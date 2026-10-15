//!use helio_prelude

// Scene depth and G-buffer normal debug views, drawn over the final image so
// fog, anti-aliasing, the lens and tone mapping leave the values untouched.

struct View {
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var scene_depth: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(3) var<uniform> view: View;

const MODE_DEPTH: u32 = 40u;
const MODE_NORMALS: u32 = 41u;

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VSOut;
    out.clip_pos = vec4<f32>(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

/// The buffer's texel under `uv`: the buffers are at internal resolution,
/// the target at output resolution.
fn texel(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(tex));
    let p = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2(0), size - 1);
    return textureLoad(tex, p, 0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    if view.mode == MODE_NORMALS {
        let n = texel(gbuf_normal, in.uv).xyz;
        if dot(n, n) < 1.0e-6 { return vec4(0.0, 0.0, 0.0, 1.0); }
        // World space, remapped: R = +X, G = +Y, B = +Z.
        return vec4(normalize(n) * 0.5 + 0.5, 1.0);
    }
    let d = texel(scene_depth, in.uv).r;
    // Sky (nothing written) is black.
    if !(d < 1.0) { return vec4(0.0, 0.0, 0.0, 1.0); }
    let near = cameras[0].position_near.w;
    let far = cameras[0].forward_far.w;
    let distance = helio_view_depth(d, near, far);
    // Logarithmic in distance so near and far detail both read: white at the
    // near plane, black at the far plane.
    let t = clamp(log(distance / near) / log(far / near), 0.0, 1.0);
    return vec4(vec3(1.0 - t), 1.0);
}
