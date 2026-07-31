// ── Sprite Batch ─────────────────────────────────────────────────────────────
//
// One unit quad (binding 0, per-vertex), drawn instanced against a storage of
// per-sprite transforms/UVs/tints (binding 1, per-instance). Rotation is
// applied in local space before translating to world position, so sprites
// rotate about their own center regardless of size.

struct Camera {
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;

struct VertexIn {
    @location(0) quad_pos: vec2<f32>,
    @location(1) quad_uv: vec2<f32>,
}

struct InstanceIn {
    @location(2) i_position: vec2<f32>,
    @location(3) i_size: vec2<f32>,
    @location(4) i_rotation: f32,
    @location(5) i_uv_rect: vec4<f32>,
    @location(6) i_color: vec4<f32>,
}

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(v: VertexIn, i: InstanceIn) -> VOut {
    let c = cos(i.i_rotation);
    let s = sin(i.i_rotation);
    let local = v.quad_pos * i.i_size;
    let rotated = vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);
    let world = rotated + i.i_position;

    var out: VOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.uv = mix(i.i_uv_rect.xy, i.i_uv_rect.zw, v.quad_uv);
    out.color = i.i_color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let sampled = textureSample(atlas_tex, atlas_samp, in.uv);
    let c = sampled * in.color;
    if c.a < 0.001 {
        discard;
    }
    return c;
}
