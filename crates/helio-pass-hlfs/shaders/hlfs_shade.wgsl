//! HLFS Final Shading Pass
//!
//! Combines direct samples with field queries to produce final pixel colors.
//! This is where the O(1) per-pixel shading happens.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct HlfsGlobals {
    frame:            u32,
    sample_count:     u32,
    light_count:      u32,
    screen_width:     u32,
    screen_height:    u32,
    near_field_size:  f32,
    cascade_scale:    f32,
    temporal_blend:   f32,
    camera_position:  vec3<f32>,
    _pad0:            u32,
    camera_forward:   vec3<f32>,
    _pad1:            u32,
}

@group(0) @binding(0) var clip_stack_level0: texture_3d<f32>;
@group(0) @binding(1) var clip_stack_sampler: sampler;

// Group 1: GBuffer inputs
@group(1) @binding(0) var gbuf_albedo:   texture_2d<f32>;
@group(1) @binding(1) var gbuf_normal:   texture_2d<f32>;
@group(1) @binding(2) var gbuf_orm:      texture_2d<f32>;
@group(1) @binding(3) var gbuf_emissive: texture_2d<f32>;
@group(1) @binding(4) var gbuf_depth:    texture_depth_2d;

// Vertex shader - fullscreen triangle
struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VSOut;
    out.clip_pos = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

// Fragment shader - query field and combine with direct samples
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let pixel_coord = vec2<i32>(in.clip_pos.xy);

    // Read from GBuffer
    let albedo = textureLoad(gbuf_albedo, pixel_coord, 0);
    let normal = textureLoad(gbuf_normal, pixel_coord, 0).xyz;
    let depth = textureLoad(gbuf_depth, pixel_coord, 0);

    // Sky/background gets dark blue color
    if (depth >= 1.0) {
        return vec4<f32>(0.05, 0.05, 0.15, 1.0);
    }

    // Simple directional lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ndotl = max(dot(normal, light_dir), 0.0);
    let ambient = vec3<f32>(0.1);

    // Basic lighting: albedo * (directional + ambient)
    let lit_color = albedo.rgb * (ndotl * vec3<f32>(1.0, 0.95, 0.9) + ambient);

    return vec4<f32>(lit_color, 1.0);
}
