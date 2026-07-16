enable wgpu_binding_array;

//! Decal apply compute shader.
//!
//! Reads depth + GBuffer textures, projects active decals onto visible surfaces,
//! and blends decal contributions into the GBuffer in-place.

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

struct GpuDecal {
    transform:          mat4x4<f32>,
    color:              vec4<f32>,
    albedo_texture_index: u32,
    normal_texture_index: u32,
    roughness_texture_index: u32,
    metalness_texture_index: u32,
    blend_mode:         u32,
    decal_type:         u32,
    fade_time:          f32,
    fade_start_delay:   f32,
    age:                f32,
    _pad0:              f32,
    _pad1:              f32,
    _pad2:              f32,
}

const DECAL_BLEND_TRANSLUCENT: u32 = 0u;
const DECAL_BLEND_ALPHA:       u32 = 1u;
const DECAL_BLEND_ADDITIVE:    u32 = 2u;
const DECAL_BLEND_MULTIPLY:    u32 = 3u;

const DECAL_TYPE_ALBEDO_NORMAL: u32 = 0u;
const DECAL_TYPE_NORMAL_ONLY:   u32 = 1u;
const DECAL_TYPE_EMISSIVE:      u32 = 2u;
const DECAL_TYPE_ALL:           u32 = 3u;

const NO_TEXTURE: u32 = 0xFFFFFFFFu;

struct DecalGlobals {
    decal_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>   camera:      Camera;
@group(0) @binding(1) var<uniform>   globals:     DecalGlobals;
@group(0) @binding(2) var<storage, read> decals: array<GpuDecal>;

@group(1) @binding(0) var            gbuf_depth:  texture_depth_2d;
@group(1) @binding(1) var            gbuf_albedo:  texture_2d<f32>;
@group(1) @binding(2) var            gbuf_normal:  texture_2d<f32>;
@group(1) @binding(3) var            gbuf_orm:     texture_2d<f32>;
@group(1) @binding(4) var            gbuf_emissive: texture_2d<f32>;

@group(1) @binding(5) var            gbuf_albedo_rw:  texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(6) var            gbuf_normal_rw:  texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var            gbuf_orm_rw:     texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(8) var            gbuf_emissive_rw: texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var            decal_textures: binding_array<texture_2d<f32>, 16>;
@group(2) @binding(1) var            decal_samplers: binding_array<sampler, 16>;

fn decal_fade_opacity(age: f32, fade_start_delay: f32, fade_time: f32) -> f32 {
    if fade_time <= 0.0 {
        return 1.0;
    }
    if age < fade_start_delay {
        return 1.0;
    }
    let t = (age - fade_start_delay) / fade_time;
    return clamp(1.0 - t, 0.0, 1.0);
}

fn sample_decal_texture(tex_index: u32, uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if tex_index == NO_TEXTURE {
        return fallback;
    }
    return textureSampleLevel(decal_textures[tex_index], decal_samplers[tex_index], uv, 0.0);
}

fn blend_over(back: vec4<f32>, front: vec4<f32>) -> vec4<f32> {
    let a = clamp(front.a, 0.0, 1.0);
    return vec4<f32>(mix(back.rgb, front.rgb, a), back.a);
}

fn blend_additive(back: vec4<f32>, front: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(back.rgb + front.rgb * front.a, back.a);
}

fn blend_multiply(back: vec4<f32>, front: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(back.rgb * mix(vec3<f32>(1.0), front.rgb, front.a), back.a);
}

fn blend_normal(back_normal: vec3<f32>, front_normal: vec3<f32>, blend_alpha: f32) -> vec3<f32> {
    return normalize(mix(back_normal, front_normal, blend_alpha));
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = vec2<i32>(id.xy);
    let size = textureDimensions(gbuf_albedo);
    if id.x >= u32(size.x) || id.y >= u32(size.y) {
        return;
    }

    let depth = textureLoad(gbuf_depth, pixel, 0);
    if depth >= 1.0 {
        return;
    }

    let uv = vec2<f32>(
        (f32(pixel.x) + 0.5) / f32(size.x),
        (f32(pixel.y) + 0.5) / f32(size.y),
    );
    let ndc = vec4<f32>(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0,
        depth,
        1.0,
    );
    let world_h = camera.view_proj_inv * ndc;
    let world_pos = world_h.xyz / world_h.w;

    let decal_count = globals.decal_count;

    for (var i = 0u; i < decal_count; i = i + 1u) {
        let decal = decals[i];
        let decal_alpha = decal_fade_opacity(decal.age, decal.fade_start_delay, decal.fade_time);
        if decal_alpha <= 0.0 {
            continue;
        }

        let decal_local = decal.transform * vec4<f32>(world_pos, 1.0);
        if abs(decal_local.x) > 1.0 || abs(decal_local.y) > 1.0 || decal_local.z < -1.0 || decal_local.z > 1.0 {
            continue;
        }

        let decal_uv = vec2<f32>(decal_local.x * 0.5 + 0.5, 0.5 - decal_local.y * 0.5);
        let decal_uv_clamped = clamp(decal_uv, vec2<f32>(0.0), vec2<f32>(1.0));

        let decal_type = decal.decal_type;
        let blend = decal.blend_mode;

        let decal_albedo = sample_decal_texture(decal.albedo_texture_index, decal_uv_clamped, vec4<f32>(1.0));
        let decal_normal_raw = sample_decal_texture(decal.normal_texture_index, decal_uv_clamped, vec4<f32>(0.5, 0.5, 1.0, 1.0));
        let decal_normal = normalize(decal_normal_raw.xyz * 2.0 - 1.0);
        let decal_roughness_val = sample_decal_texture(decal.roughness_texture_index, decal_uv_clamped, vec4<f32>(1.0)).g;
        let decal_metallic_val = sample_decal_texture(decal.metalness_texture_index, decal_uv_clamped, vec4<f32>(0.0)).r;
        let decal_emissive_val = sample_decal_texture(decal.albedo_texture_index, decal_uv_clamped, vec4<f32>(0.0));

        let tint = decal.color;
        let opacity = tint.a * decal_alpha;

        let existing_albedo = textureLoad(gbuf_albedo, pixel, 0);
        let existing_normal = textureLoad(gbuf_normal, pixel, 0);
        let existing_orm = textureLoad(gbuf_orm, pixel, 0);
        let existing_emissive = textureLoad(gbuf_emissive, pixel, 0);

        var result_albedo = existing_albedo;
        var result_normal = existing_normal;
        var result_orm = existing_orm;
        var result_emissive = existing_emissive;

        if decal_type == DECAL_TYPE_ALBEDO_NORMAL || decal_type == DECAL_TYPE_ALL {
            let decal_color = vec4<f32>(decal_albedo.rgb * tint.rgb, opacity);

            if blend == DECAL_BLEND_TRANSLUCENT || blend == DECAL_BLEND_ALPHA {
                result_albedo = blend_over(existing_albedo, decal_color);
            } else if blend == DECAL_BLEND_ADDITIVE {
                result_albedo = blend_additive(existing_albedo, decal_color);
            } else if blend == DECAL_BLEND_MULTIPLY {
                result_albedo = blend_multiply(existing_albedo, decal_color);
            }

            let existing_world_normal = normalize(existing_normal.xyz);
            let blended_normal = blend_normal(existing_world_normal, decal_normal, opacity);
            result_normal = vec4<f32>(blended_normal, existing_normal.a);
        }

        if decal_type == DECAL_TYPE_NORMAL_ONLY {
            let existing_world_normal = normalize(existing_normal.xyz);
            let blended_normal = blend_normal(existing_world_normal, decal_normal, opacity);
            result_normal = vec4<f32>(blended_normal, existing_normal.a);
        }

        if decal_type == DECAL_TYPE_EMISSIVE || decal_type == DECAL_TYPE_ALL {
            let decal_emis = vec4<f32>(decal_emissive_val.rgb * tint.rgb * opacity, existing_emissive.a);
            if blend == DECAL_BLEND_ADDITIVE {
                result_emissive = blend_additive(existing_emissive, decal_emis);
            } else {
                result_emissive = blend_over(existing_emissive, decal_emis);
            }
        }

        if decal_type == DECAL_TYPE_ALL {
            if blend == DECAL_BLEND_TRANSLUCENT || blend == DECAL_BLEND_ALPHA {
                result_orm = vec4<f32>(
                    mix(existing_orm.r, 1.0, opacity),
                    mix(existing_orm.g, decal_roughness_val, opacity),
                    mix(existing_orm.b, decal_metallic_val, opacity),
                    existing_orm.a,
                );
            }
        }

        textureStore(gbuf_albedo_rw, pixel, result_albedo);
        textureStore(gbuf_normal_rw, pixel, result_normal);
        textureStore(gbuf_orm_rw, pixel, result_orm);
        textureStore(gbuf_emissive_rw, pixel, result_emissive);
    }
}
