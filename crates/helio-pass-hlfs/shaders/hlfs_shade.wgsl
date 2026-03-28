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
@group(0) @binding(1) var clip_stack_level1: texture_3d<f32>;
@group(0) @binding(2) var clip_stack_level2: texture_3d<f32>;
@group(0) @binding(3) var clip_stack_level3: texture_3d<f32>;
@group(0) @binding(4) var clip_stack_sampler: sampler;
@group(0) @binding(5) var pre_aa_texture: texture_2d<f32>;  // Sky + debug layers
@group(0) @binding(6) var<uniform> globals: HlfsGlobals;
@group(0) @binding(7) var<uniform> camera: Camera;
@group(0) @binding(8) var<storage, read> lights: array<GpuLight>;

struct GpuLight {
    position_range: vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index: u32,
    light_type: u32,
    inner_angle: f32,
    _pad: u32,
}

fn evaluate_light(light: GpuLight, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let light_color = light.color_intensity.xyz;
    let intensity = light.color_intensity.w;
    if (light.light_type == 0u) {
        // Directional light
        let dir = normalize(light.direction_outer.xyz);
        let ndotl = max(dot(normal, dir), 0.0);
        return light_color * intensity * ndotl;
    } else {
        let delta = light.position_range.xyz - world_pos;
        let dist = max(length(delta), 0.001);
        let dir = normalize(delta);
        let ndotl = max(dot(normal, dir), 0.0);
        let range = max(light.position_range.w, 1.0);
        let attenuation = max(0.0, 1.0 - dist / range) / (dist * dist * 0.2 + 1.0);

        if (light.light_type == 2u) {
            // Spot attack
            let spot_dir = normalize(light.direction_outer.xyz);
            let cos_angle = dot(spot_dir, -dir);
            let outer = light.direction_outer.w;
            let inner = light.inner_angle;
            let spot = smoothstep(outer, inner, cos_angle);
            return light_color * intensity * ndotl * attenuation * spot;
        }

        // point/other
        return light_color * intensity * ndotl * attenuation;
    }
}

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
    let albedo = textureLoad(gbuf_albedo, pixel_coord, 0).rgb;
    let normal = normalize(textureLoad(gbuf_normal, pixel_coord, 0).xyz * 2.0 - 1.0);
    let orm = textureLoad(gbuf_orm, pixel_coord, 0).rgb;
    let emissive = textureLoad(gbuf_emissive, pixel_coord, 0).rgb;
    let depth = textureLoad(gbuf_depth, pixel_coord, 0);

    // Sky/background pixels: sample from pre_aa (sky + debug layers)
    if (depth >= 1.0) {
        return textureLoad(pre_aa_texture, pixel_coord, 0);
    }

    // Sample hierarchical radiance field
    let uv = clamp(in.uv * 0.5 + vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(1.0));
    let field_coord = vec3<f32>(uv, depth);

    let field0 = textureSampleLevel(clip_stack_level0, clip_stack_sampler, field_coord, 0).rgb;
    let field1 = textureSampleLevel(clip_stack_level1, clip_stack_sampler, field_coord, 0).rgb;
    let field2 = textureSampleLevel(clip_stack_level2, clip_stack_sampler, field_coord, 0).rgb;
    let field3 = textureSampleLevel(clip_stack_level3, clip_stack_sampler, field_coord, 0).rgb;

    let indirect = field0 * 0.6 + field1 * 0.25 + field2 * 0.1 + field3 * 0.05;

    let roughness = clamp(orm.g, 0.02, 1.0);
    let metallic = orm.b;
    let base_albedo = albedo * (1.0 - metallic);

    // Reconstruct accurate world position from depth and camera inverse view-proj.
    let screen_size = vec2<f32>(textureDimensions(gbuf_albedo));
    let uv_01 = in.clip_pos.xy / screen_size;
    let ndc_xy = vec2<f32>(uv_01.x * 2.0 - 1.0, 1.0 - uv_01.y * 2.0);
    let world_h = camera.view_proj_inv * vec4<f32>(ndc_xy, depth, 1.0);
    let world_pos = world_h.xyz / world_h.w;

    // Direct per-light accumulation (actual scene lights)
    var direct_lighting = vec3<f32>(0.0);
    let max_lights = min(globals.light_count, 64u);
    for (var i: u32 = 0u; i < max_lights; i = i + 1u) {
        direct_lighting = direct_lighting + evaluate_light(lights[i], world_pos, normal);
    }

    let ambient = vec3<f32>(0.03);
    let specular_strength = 0.04;
    var specular = vec3<f32>(0.0);
    if (length(direct_lighting) > 0.0) {
        let phong = pow(max(dot(normal, normalize(direct_lighting)), 0.0), (1.0 / max(roughness, 0.001)) * 64.0);
        specular = vec3<f32>(specular_strength) * phong;
    }

    let base = base_albedo * (direct_lighting + ambient);
    let final_color = base + specular + indirect + emissive;

    return vec4<f32>(final_color, 1.0);
}
