enable wgpu_ray_query;

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

struct GpuLight {
    position_range: vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index: u32,
    light_type: u32,
    inner_angle: f32,
    _pad: u32,
    god_rays_enabled: u32,
    god_rays_density: f32,
    god_rays_weight: f32,
    god_rays_decay: f32,
    god_rays_exposure: f32,
    _pad2_0: u32,
    _pad2_1: u32,
    _pad2_2: u32,
}

struct ShadowConfig {
    cascades: array<CascadeConfig, 4>,
    enable_pcss: u32,
    pcss_blocker_samples: u32,
    pcss_filter_samples: u32,
    pcf_sample_count: u32,
}

struct CascadeConfig {
    split_distance: f32,
    depth_bias: f32,
    filter_radius: f32,
    pcss_light_size: f32,
}

struct LightMatrix {
    mat: mat4x4<f32>,
}

@group(0) @binding(0) var clip_stack_level0: texture_3d<f32>;
@group(0) @binding(1) var clip_stack_level1: texture_3d<f32>;
@group(0) @binding(2) var clip_stack_level2: texture_3d<f32>;
@group(0) @binding(3) var clip_stack_level3: texture_3d<f32>;
@group(0) @binding(4) var clip_stack_sampler: sampler;
@group(0) @binding(5) var pre_aa_texture: texture_2d<f32>;
@group(0) @binding(6) var<uniform> globals: HlfsGlobals;
@group(0) @binding(7) var<uniform> camera: Camera;
@group(0) @binding(8) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(9) var<uniform> shadow_config: ShadowConfig;
@group(0) @binding(10) var shadow_atlas: texture_depth_2d_array;
@group(0) @binding(11) var shadow_sampler: sampler_comparison;
@group(0) @binding(12) var <storage, read> shadow_matrices: array<LightMatrix>;
@group(0) @binding(13) var acc_struct: acceleration_structure;

const ENABLE_SHADOWS: bool = true;
const MAX_SHADOW_LIGHTS: u32 = 42u;
const NORMAL_OFFSET_SCALE: f32 = 0.01;
const PI: f32 = 3.14159265359;

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a    = roughness * roughness;
    let a2   = a * a;
    let NdH  = max(dot(N, H), 0.0);
    let denom = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdV = max(dot(N, V), 0.0);
    let NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, roughness) * geometry_schlick_ggx(NdL, roughness);
}

fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

fn fresnel_schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let one_minus_r = vec3<f32>(1.0 - roughness);
    return F0 + (max(one_minus_r, F0) - F0) * pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

fn rt_shadow(light: GpuLight, world_pos: vec3<f32>, N: vec3<f32>) -> f32 {
    if !ENABLE_SHADOWS { return 1.0; }
    if light.shadow_index == 4294967295u { return 1.0; }

    let origin = world_pos + N * NORMAL_OFFSET_SCALE * 0.4;
    var vis = 0.0;
    if light.light_type < 0.5 {
        let mut sq: ray_query;
        rayQueryInitialize(&sq, acc_struct,
            RayDesc(0x01u, 0xFFu, 0.005, 9999.0, origin, -light.direction_outer.xyz));
        rayQueryProceed(&sq);
        if rayQueryGetCommittedIntersection(&sq).kind == RAY_QUERY_INTERSECTION_NONE {
            vis = 1.0;
        }
    } else {
        let to_light_dir = normalize(light.position_range.xyz - world_pos);
        let light_radius = 0.35;
        let perp = normalize(cross(to_light_dir, select(
            vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(to_light_dir.y) < 0.9)));
        let perp2 = cross(to_light_dir, perp);
        for (var si: u32 = 0u; si < 4u; si++) {
            let off = vec2<f32>(0.707, -0.707);
            let light_point = light.position_range.xyz + perp * off.x * light_radius + perp2 * off.y * light_radius;
            let ray_dir = normalize(light_point - world_pos);
            let ray_dist = length(light_point - world_pos);
            var sq: ray_query;
            rayQueryInitialize(&sq, acc_struct,
                RayDesc(0x01u, 0xFFu, 0.005, ray_dist - 0.005, origin, ray_dir));
            rayQueryProceed(&sq);
            if rayQueryGetCommittedIntersection(&sq).kind == RAY_QUERY_INTERSECTION_NONE {
                vis += 0.25;
            }
        }
    }
    return vis;
}

fn evaluate_light(light: GpuLight, world_pos: vec3<f32>, N: vec3<f32>, V: vec3<f32>, F0: vec3<f32>, albedo: vec3<f32>, roughness: f32, metallic: f32, sf: f32) -> vec3<f32> {
    var L: vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type == 0u {
        L = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {
        let to_light = light.position_range.xyz - world_pos;
        let dist = length(to_light);
        if dist > light.position_range.w { return vec3<f32>(0.0); }
        L = to_light / dist;
        var atten = 1.0 / (dist * dist + 0.0001);
        let normalized_dist = dist / light.position_range.w;
        atten *= max(0.0, 1.0 - normalized_dist * normalized_dist * normalized_dist * normalized_dist);
        if light.light_type == 2u {
            let cos_a = dot(-L, light.direction_outer.xyz);
            atten *= smoothstep(light.direction_outer.w, light.inner_angle, cos_a);
        }
        radiance = light.color_intensity.xyz * light.color_intensity.w * atten;
    }

    let NdL = max(dot(N, L), 0.0);
    if NdL == 0.0 { return vec3<f32>(0.0); }

    if all(radiance < vec3<f32>(0.002)) { return vec3<f32>(0.0); }

    let H = normalize(V + L);
    let D = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let kD = (1.0 - F) * (1.0 - metallic);
    let specular = D * G * F / (4.0 * max(dot(N, V), 0.0) * NdL + 0.0001);

    return (kD * albedo / PI + specular) * radiance * NdL * sf;
}

// Group 1: GBuffer inputs
@group(1) @binding(0) var gbuf_albedo:      texture_2d<f32>;
@group(1) @binding(1) var gbuf_normal:      texture_2d<f32>;
@group(1) @binding(2) var gbuf_orm:         texture_2d<f32>;
@group(1) @binding(3) var gbuf_emissive:    texture_2d<f32>;
@group(1) @binding(4) var gbuf_depth:       texture_depth_2d;
@group(1) @binding(5) var gbuf_lightmap_uv: texture_2d<f32>;

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

// Fragment shader - uses ray queries for shadow evaluation
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let pixel_coord = vec2<i32>(in.clip_pos.xy);

    let albedo = textureLoad(gbuf_albedo, pixel_coord, 0).rgb;
    let normal = normalize(textureLoad(gbuf_normal, pixel_coord, 0).xyz);
    let orm = textureLoad(gbuf_orm, pixel_coord, 0).rgb;
    let emissive = textureLoad(gbuf_emissive, pixel_coord, 0).rgb;
    let depth = textureLoad(gbuf_depth, pixel_coord, 0);

    if (depth >= 1.0) {
        return textureLoad(pre_aa_texture, pixel_coord, 0);
    }

    let uv = clamp(in.uv * 0.5 + vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(1.0));
    let field_coord = vec3<f32>(uv, depth);

    let field0 = textureSampleLevel(clip_stack_level0, clip_stack_sampler, field_coord, 0).rgb;
    let field1 = textureSampleLevel(clip_stack_level1, clip_stack_sampler, field_coord, 0).rgb;
    let field2 = textureSampleLevel(clip_stack_level2, clip_stack_sampler, field_coord, 0).rgb;
    let field3 = textureSampleLevel(clip_stack_level3, clip_stack_sampler, field_coord, 0).rgb;

    let indirect = field0 * 0.6 + field1 * 0.25 + field2 * 0.1 + field3 * 0.05;

    let roughness = clamp(orm.g, 0.02, 1.0);
    let metallic = orm.b;
    let screen_size = vec2<f32>(textureDimensions(gbuf_albedo));
    let uv_01 = in.clip_pos.xy / screen_size;
    let ndc_xy = vec2<f32>(uv_01.x * 2.0 - 1.0, 1.0 - uv_01.y * 2.0);
    let world_h = camera.view_proj_inv * vec4<f32>(ndc_xy, depth, 1.0);
    let world_pos = world_h.xyz / world_h.w;
    let V = normalize(camera.position_near.xyz - world_pos);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    var direct_lighting = vec3<f32>(0.0);
    let max_lights = min(globals.light_count, 64u);
    for (var i: u32 = 0u; i < max_lights; i = i + 1u) {
        let light = lights[i];
        if light.light_type != 0u {
            let dist = length(light.position_range.xyz - world_pos);
            if dist > light.position_range.w {
                continue;
            }
        }
        let vis = rt_shadow(light, world_pos, normal);
        direct_lighting = direct_lighting + evaluate_light(light, world_pos, normal, V, F0, albedo, roughness, metallic, vis);
    }

    let NdV = max(dot(normal, V), 0.0);
    let F_ibl = fresnel_schlick_roughness(NdV, F0, roughness);
    let kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);
    let ambient = vec3<f32>(0.03);
    let indirect_lighting = kD_ibl * (indirect + ambient) * albedo;
    let final_color = direct_lighting + indirect_lighting + emissive;

    return vec4<f32>(final_color, 1.0);
}
