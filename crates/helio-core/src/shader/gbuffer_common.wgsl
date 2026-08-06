// Helio G-buffer common — the structs, flags and surface evaluation shared by
// every shader that writes the G-buffer: the main gbuffer.wgsl, the secondary
// (portal/sublevel) G-buffer pass, and the tier-2 Radiant templates composed on
// top of the base.
//
// Prepended to any shader whose first lines contain `//!use helio_gbuffer_common`
// (see helio_core::shader). A consuming shader owns its own group/binding
// declarations and its own vertex/fragment entry points; this file is pure
// declarations and pure helpers so it stays composable.
//
// Unlike the prelude, this file declares the `Camera` struct and `CAMERA_SLOTS`
// itself (the G-buffer needs no prelude helper and historically stood alone).
// It must therefore not be combined with `//!use helio_prelude` in one shader —
// both declare `CAMERA_SLOTS` and the duplicate would fail at parse time.

// must equal CAMERA_SLOTS in libhelio::camera
const CAMERA_SLOTS: u32 = 7u;

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

struct Globals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: vec4<f32>,
    rc_world_min: vec4<f32>,
    rc_world_max: vec4<f32>,
    csm_splits: vec4<f32>,
    debug_mode: u32,
    screen_width: f32,
    screen_height: f32,
    _pad0: u32,
}

/// GPU material (112 bytes, matches libhelio::GpuMaterial)
struct GpuMaterial {
    base_color:         vec4<f32>,
    emissive:           vec4<f32>,
    roughness_metallic: vec4<f32>,
    tex_base_color:     u32,
    tex_normal:         u32,
    tex_roughness:      u32,
    tex_emissive:       u32,
    tex_occlusion:      u32,
    workflow:           u32,
    flags:              u32,
    material_class:     u32,
    class_params:       vec4<f32>,
}

const FLAG_HAS_NORMAL_MAP: u32 = 1u << 3u;
const FLAG_HAS_CLEAR_COAT: u32 = 1u << 4u;
const FLAG_HAS_SUBSURFACE: u32 = 1u << 5u;
const FLAG_HAS_ANISOTROPY: u32 = 1u << 6u;

const SURFACE_FLAG_SUBSURFACE: u32 = 1u << 0u;
const SURFACE_FLAG_ANISOTROPIC: u32 = 1u << 1u;
const SURFACE_FLAG_LOW_SPECULAR: u32 = 1u << 2u;

/// Per-material texture metadata (224 bytes, matches helio::GpuMaterialTextures)
struct MaterialTextureSlot {
    texture_index: u32,
    uv_channel:    u32,
    _pad0:         u32,
    _pad1:         u32,
    offset_scale:  vec4<f32>,
    rotation:      vec4<f32>,
}

struct MaterialTextureData {
    base_color:         MaterialTextureSlot,
    normal:             MaterialTextureSlot,
    roughness_metallic: MaterialTextureSlot,
    emissive:           MaterialTextureSlot,
    occlusion:          MaterialTextureSlot,
    specular_color:     MaterialTextureSlot,
    specular_weight:    MaterialTextureSlot,
    params:             vec4<f32>,  // x=normal_scale, y=occlusion_strength, z=alpha_cutoff
}

/// Per-instance data (208 bytes). Must match `GpuInstanceData` in libhelio.
struct GpuInstanceData {
    transform:      mat4x4<f32>,  // offset   0  (64 bytes)
    normal_mat_0:   vec4<f32>,    // offset  64  — row 0 of inv-transpose 3×3
    normal_mat_1:   vec4<f32>,    // offset  80
    normal_mat_2:   vec4<f32>,    // offset  96
    bounds:         vec4<f32>,    // offset 112
    prev_model:     mat4x4<f32>,  // offset 128  (64 bytes) — previous frame model matrix
    mesh_id:        u32,          // offset 192
    material_id:    u32,          // offset 196
    flags:          u32,          // offset 200
    lightmap_index: u32,          // offset 204 — index into lightmap_atlas_regions, 0xFFFFFFFF = no lightmap
}

/// Lightmap atlas region for a mesh (32 bytes).
///
/// uv_clamp_min/max are precomputed half-texel-inset bounds that prevent bilinear
/// filtering from bleeding across neighbouring atlas region boundaries at runtime.
struct LightmapAtlasRegion {
    uv_offset:    vec2<f32>,  // Top-left corner in atlas [0,1] space
    uv_scale:     vec2<f32>,  // Extent in atlas [0,1] space
    uv_clamp_min: vec2<f32>,  // uv_offset + 0.5/atlas_size  (half-texel inner inset)
    uv_clamp_max: vec2<f32>,  // uv_offset + uv_scale - 0.5/atlas_size
}

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,  // UV0 — material/albedo channel (may tile)
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
    @location(5) lightmap_uv:    vec2<f32>,  // UV1 — dedicated lightmap channel, non-overlapping [0,1]
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position:     vec3<f32>,
    @location(1) world_normal:       vec3<f32>,
    @location(2) tex_coords:         vec2<f32>,
    @location(3) world_tangent:      vec3<f32>,
    @location(4) bitangent_sign:     f32,
    @location(5) @interpolate(flat) material_id:    u32,
    @location(6) lightmap_uv:        vec2<f32>,  // Lightmap atlas UV (or (0,0) if no lightmap)
    @location(7) prev_clip_position: vec4<f32>,  // Previous frame clip-space position (velocity)
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

// ── Fragment ─────────────────────────────────────────────────────────────────

struct GBufferOutput {
    @location(0) albedo:      vec4<f32>,
    @location(1) normal:      vec4<f32>,
    @location(2) orm:         vec4<f32>,
    @location(3) emissive:    vec4<f32>,
    @location(4) lightmap_uv: vec2<f32>,
    @location(5) sss:         vec4<f32>,
    @location(6) extra:       vec4<f32>,
    @location(7) velocity:    vec2<f32>,  // screen-space motion in pixels/frame
}

// ── Surface data passed to GBuffer packing ──────────────────────────────────

struct SurfaceData {
    albedo:              vec4<f32>,
    normal:              vec3<f32>,
    ao:                  f32,
    roughness:           f32,
    metallic:            f32,
    specular_f0:         vec3<f32>,
    emissive:            vec3<f32>,
    alpha:               f32,
    flags:               u32,
    subsurface_color:    vec3<f32>,
    subsurface_radius:   f32,
    roughness_aniso_x:   f32,
    roughness_aniso_y:   f32,
    aniso_rotation:      f32,
}

const NO_TEXTURE: u32 = 0xffffffffu;
const MATERIAL_WORKFLOW_METALLIC: u32 = 0u;
const MATERIAL_WORKFLOW_SPECULAR: u32 = 1u;

/// Select UV channel and apply texture transform
fn select_uv(slot: MaterialTextureSlot, base_uv: vec2<f32>) -> vec2<f32> {
    // TODO: support uv_channel when we have tex_coords1
    let scaled = base_uv * slot.offset_scale.zw;
    let s = slot.rotation.x;
    let c = slot.rotation.y;
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    );
    return rotated + slot.offset_scale.xy;
}

/// Sample texture from bindless array, or return fallback if NO_TEXTURE
fn sample_texture(slot: MaterialTextureSlot, base_uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if slot.texture_index == NO_TEXTURE {
        return fallback;
    }
    let uv = select_uv(slot, base_uv);
    return textureSample(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], uv);
}

fn resolve_specular_f0(
    material: GpuMaterial,
    material_tex: MaterialTextureData,
    albedo: vec3<f32>,
    metallic: f32,
    uv: vec2<f32>,
) -> vec3<f32> {
    if material.workflow == MATERIAL_WORKFLOW_SPECULAR {
        let specular_color = sample_texture(material_tex.specular_color, uv, vec4<f32>(1.0)).rgb;
        let specular_weight = sample_texture(material_tex.specular_weight, uv, vec4<f32>(1.0)).a;
        let ior = max(material.roughness_metallic.z, 1.0);
        let dielectric_f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
        return material.roughness_metallic.w * specular_weight * specular_color * dielectric_f0;
    }

    // Metallic workflow: F0 = mix(0.04, albedo, metallic)
    return clamp(
        mix(vec3<f32>(0.04), albedo, metallic),
        vec3<f32>(0.0),
        vec3<f32>(0.999),
    );
}

// ── Default PBR material evaluation ──────────────────────────────────────────

fn default_pbr_surface(material: GpuMaterial, material_tex: MaterialTextureData, input: VertexOutput) -> SurfaceData {
    let uv = input.tex_coords;
    let base_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));
    let albedo = material.base_color * base_sample;
    let alpha = albedo.a;

    let N_geom = normalize(input.world_normal);

    var N: vec3<f32>;
    if (material.flags & FLAG_HAS_NORMAL_MAP) != 0u && material_tex.normal.texture_index != NO_TEXTURE {
        let T = normalize(input.world_tangent - dot(input.world_tangent, N_geom) * N_geom);
        let B = cross(N_geom, T) * input.bitangent_sign;
        var norm_ts = sample_texture(material_tex.normal, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - 1.0;
        norm_ts = vec3<f32>(norm_ts.x * material_tex.params.x, norm_ts.y * material_tex.params.x, norm_ts.z);
        N = normalize(T * norm_ts.x + B * norm_ts.y + N_geom * norm_ts.z);
    } else {
        N = N_geom;
    }

    let orm_sample = sample_texture(material_tex.roughness_metallic, uv, vec4<f32>(1.0));
    let occlusion_sample = sample_texture(material_tex.occlusion, uv, vec4<f32>(1.0));
    let emissive_sample = sample_texture(material_tex.emissive, uv, vec4<f32>(1.0));

    var ao: f32 = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;
    var roughness: f32 = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    var metallic: f32 = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    var specular_f0: vec3<f32> = resolve_specular_f0(material, material_tex, albedo.rgb, metallic, uv);
    var emissive: vec3<f32> = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    return SurfaceData(albedo, N, ao, roughness, metallic, specular_f0, emissive, alpha,
                       0u, vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0);
}

fn compute_velocity(input: VertexOutput) -> vec2<f32> {
    let prev_ndc = input.prev_clip_position.xy / input.prev_clip_position.w;
    let prev_pixel_x = (prev_ndc.x * 0.5 + 0.5) * globals.screen_width;
    let prev_pixel_y = (0.5 - prev_ndc.y * 0.5) * globals.screen_height;
    let prev_pixel = vec2<f32>(prev_pixel_x, prev_pixel_y);
    return input.clip_position.xy - prev_pixel;
}
