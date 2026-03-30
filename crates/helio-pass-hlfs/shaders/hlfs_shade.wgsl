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
    csm_splits:      vec4<f32>,
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

const ENABLE_SHADOWS: bool = true;
const MAX_SHADOW_LIGHTS: u32 = 42u;
const ATLAS_SIZE: f32 = 1024.0;
const NORMAL_OFFSET_SCALE: f32 = 0.05;

struct LightMatrix {
    mat: mat4x4<f32>,
}

struct CascadeConfig {
    split_distance: f32,
    depth_bias: f32,
    filter_radius: f32,
    pcss_light_size: f32,
}

struct ShadowConfig {
    cascades: array<CascadeConfig, 4>,
    enable_pcss: u32,
    pcss_blocker_samples: u32,
    pcss_filter_samples: u32,
    pcf_sample_count: u32,
}

@group(0) @binding(9) var<uniform> shadow_config: ShadowConfig;
@group(0) @binding(10) var shadow_atlas: texture_depth_2d_array;
@group(0) @binding(11) var shadow_sampler: sampler_comparison;
@group(0) @binding(12) var <storage, read> shadow_matrices: array<LightMatrix>;

// Vogel disk sampling - blue-noise-like spiral pattern for high-quality PCF
fn vogel_disk_sample(sample_idx: u32, sample_count: u32, theta: f32) -> vec2<f32> {
    let GOLDEN_ANGLE = 2.39996323;
    let r = sqrt(f32(sample_idx) + 0.5) / sqrt(f32(sample_count));
    let angle = f32(sample_idx) * GOLDEN_ANGLE + theta;
    return vec2<f32>(cos(angle), sin(angle)) * r;
}

// Per-pixel hash for PCF rotation (reduces banding artifacts)
fn hash22(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    let d = dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * d);
}

fn point_light_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(0u, 1u, dir.x < 0.0);
    } else if a.y >= a.x && a.y >= a.z {
        return select(2u, 3u, dir.y < 0.0);
    } else {
        return select(4u, 5u, dir.z < 0.0);
    }
}

fn pcss_blocker_search(
    layer: u32,
    shadow_uv: vec2<f32>,
    receiver_depth: f32,
    search_radius: f32,
    blocker_samples: u32,
    theta: f32
) -> vec2<f32> {
    var blocker_sum = 0.0;
    var blocker_count = 0.0;

    for (var i = 0u; i < blocker_samples; i++) {
        let offset = vogel_disk_sample(i, blocker_samples, theta) * search_radius;
        let sample_uv = shadow_uv + offset;
        let pixel_coord = vec2<i32>(sample_uv * ATLAS_SIZE);

        if any(pixel_coord < vec2<i32>(0)) || any(pixel_coord >= vec2<i32>(i32(ATLAS_SIZE))) {
            continue;
        }

        let occluder_depth = textureLoad(shadow_atlas, pixel_coord, i32(layer), 0);
        if occluder_depth < receiver_depth - 0.0001 {
            blocker_sum += occluder_depth;
            blocker_count += 1.0;
        }
    }

    if blocker_count < 0.5 {
        return vec2<f32>(0.0, 0.0);
    }

    return vec2<f32>(blocker_sum / blocker_count, blocker_count);
}

fn pcss_penumbra_size(receiver_depth: f32, avg_blocker_depth: f32, light_size: f32) -> f32 {
    return (receiver_depth - avg_blocker_depth) / max(avg_blocker_depth, 0.001) * light_size;
}

fn sample_cascade_shadow(layer: u32, cascade_scale: f32, world_pos: vec3<f32>, frag_coord: vec2<f32>, frame: u32) -> f32 {
    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }

    let ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let receiver_depth = ndc.z;
    let theta = hash22(frag_coord + vec2<f32>(f32(frame))) * 6.28318530718;
    var lit_sum = 0.0;

    for (var i = 0u; i < shadow_config.pcf_sample_count; i++) {
        let offset = vogel_disk_sample(i, shadow_config.pcf_sample_count, theta) * (cascade_scale / ATLAS_SIZE);
        lit_sum += textureSampleCompareLevel(shadow_atlas, shadow_sampler, shadow_uv + offset, i32(layer), receiver_depth);
    }

    return lit_sum / f32(shadow_config.pcf_sample_count);
}

fn sample_cascade_shadow_pcss(layer: u32, cascade_idx: u32, world_pos: vec3<f32>, frag_coord: vec2<f32>, frame: u32) -> f32 {
    let config = shadow_config.cascades[cascade_idx];
    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }

    let ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let receiver_depth = ndc.z;
    let theta = hash22(frag_coord + vec2<f32>(f32(frame))) * 6.28318530718;

    let search_radius = config.pcss_light_size / ATLAS_SIZE;
    let blocker = pcss_blocker_search(layer, shadow_uv, receiver_depth, search_radius, shadow_config.pcss_blocker_samples, theta);

    if blocker.y < 0.5 {
        return 1.0;
    }

    let penumbra = pcss_penumbra_size(receiver_depth, blocker.x, config.pcss_light_size);
    let filter_radius = clamp(penumbra / ATLAS_SIZE, config.filter_radius / ATLAS_SIZE, config.filter_radius * 3.0 / ATLAS_SIZE);
    var lit_sum = 0.0;

    for (var i = 0u; i < shadow_config.pcss_filter_samples; i++) {
        let offset = vogel_disk_sample(i, shadow_config.pcss_filter_samples, theta) * filter_radius;
        lit_sum += textureSampleCompareLevel(shadow_atlas, shadow_sampler, shadow_uv + offset, i32(layer), receiver_depth);
    }

    return lit_sum / f32(shadow_config.pcss_filter_samples);
}

fn shadow_factor(light_idx: u32, world_pos: vec3<f32>, N: vec3<f32>, frag_coord: vec2<f32>, frame: u32) -> f32 {
    if !ENABLE_SHADOWS { return 1.0; }
    if light_idx >= MAX_SHADOW_LIGHTS { return 1.0; }

    let light = lights[light_idx];
    if light.shadow_index == 4294967295u { return 1.0; }

    var light_dir: vec3<f32>;
    if light.light_type == 0u {
        light_dir = normalize(-light.direction_outer.xyz);
    } else {
        light_dir = normalize(light.position_range.xyz - world_pos);
    }
    let NdotL = max(dot(N, light_dir), 0.0);
    let normal_offset = N * NORMAL_OFFSET_SCALE * (1.0 - NdotL);
    let biased_pos = world_pos + normal_offset;

    var layer: u32;
    if light.light_type > 0u && light.light_type < 2u {
        let to_frag = biased_pos - light.position_range.xyz;
        layer = light.shadow_index + point_light_face(to_frag);
        return sample_cascade_shadow(layer, 1.0, biased_pos, frag_coord, frame);
    } else if light.light_type == 0u {
        let dist = length(world_pos - camera.position_near.xyz);
        let splits = globals.csm_splits;

        var cascade_a = 3u;
        var cascade_b = 3u;
        var blend = 0.0;
        const BLEND_ZONE = 0.1;

        if dist < splits.x * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 0u;
        } else if dist < splits.x * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 0u;
            cascade_b = 1u;
            blend = smoothstep(splits.x * (1.0 - BLEND_ZONE / 2.0), splits.x * (1.0 + BLEND_ZONE / 2.0), dist);
        } else if dist < splits.y * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 1u;
        } else if dist < splits.y * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 1u;
            cascade_b = 2u;
            blend = smoothstep(splits.y * (1.0 - BLEND_ZONE / 2.0), splits.y * (1.0 + BLEND_ZONE / 2.0), dist);
        } else if dist < splits.z * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 2u;
        } else if dist < splits.z * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 2u;
            cascade_b = 3u;
            blend = smoothstep(splits.z * (1.0 - BLEND_ZONE / 2.0), splits.z * (1.0 + BLEND_ZONE / 2.0), dist);
        } else {
            cascade_a = 3u;
        }

        let use_pcss = shadow_config.enable_pcss != 0u && shadow_config.cascades[cascade_a].pcss_light_size > 0.0;

        let layer_a = light.shadow_index + cascade_a;
        var shadow_a: f32;
        if use_pcss {
            shadow_a = sample_cascade_shadow_pcss(layer_a, cascade_a, biased_pos, frag_coord, frame);
        } else {
            let cascade_scale_a = 1.0 + f32(cascade_a) * 1.5;
            shadow_a = sample_cascade_shadow(layer_a, cascade_scale_a, biased_pos, frag_coord, frame);
        }

        if blend <= 0.001 { return shadow_a; }

        if cascade_b != cascade_a && blend > 0.001 {
            let use_pcss_b = shadow_config.enable_pcss != 0u && shadow_config.cascades[cascade_b].pcss_light_size > 0.0;
            let layer_b = light.shadow_index + cascade_b;
            var shadow_b: f32;
            if use_pcss_b {
                shadow_b = sample_cascade_shadow_pcss(layer_b, cascade_b, biased_pos, frag_coord, frame);
            } else {
                let cascade_scale_b = 1.0 + f32(cascade_b) * 1.5;
                shadow_b = sample_cascade_shadow(layer_b, cascade_scale_b, biased_pos, frag_coord, frame);
            }
            return mix(shadow_a, shadow_b, blend);
        }

        return shadow_a;
    } else {
        layer = light.shadow_index;
        return sample_cascade_shadow(layer, 1.0, biased_pos, frag_coord, frame);
    }
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
    let normal = normalize(textureLoad(gbuf_normal, pixel_coord, 0).xyz);
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
        let vis = shadow_factor(i, world_pos, normal, in.clip_pos.xy, globals.frame);
        direct_lighting = direct_lighting + evaluate_light(lights[i], world_pos, normal) * vis;
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
