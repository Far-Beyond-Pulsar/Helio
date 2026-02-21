// Depth-only shadow pass — renders scene geometry into one shadow map layer.
// Bindings use dynamic offsets so a single bind group serves every face/light.

struct ShadowVpUniforms {
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> shadow_vp: ShadowVpUniforms;

struct ShadowObjUniforms {
    model: mat4x4<f32>,
}
@group(0) @binding(1) var<uniform> shadow_obj: ShadowObjUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) _bitangent_sign: f32,
    @location(2) _tex_coords: vec2<f32>,
    @location(3) _normal: u32,
    @location(4) _tangent: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = shadow_obj.model * vec4<f32>(position, 1.0);
    out.position = shadow_vp.view_proj * world_pos;
    return out;
}
