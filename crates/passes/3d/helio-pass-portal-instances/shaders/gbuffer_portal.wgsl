enable wgpu_binding_array;

//! Portal-duplicate G-buffer write — draws the instances `helio-pass-portal-cull`
//! selected for one active portal, mapped through that portal's coordinate
//! space, clipped to its opening.
//!
//! Fused into the same physical render pass `helio-pass-gbuffer` opened
//! (`LoadOp::Load` on all 8 attachments — see `helio-pass-foliage-gbuffer`
//! for the precedent this follows), so it shares the real depth buffer and
//! composes correctly with everything already drawn: no separate camera, no
//! compositing step. One `multi_draw_indexed_indirect` call per active
//! portal, bound with a small dynamic-offset uniform selecting which portal.
//!
//! Vertex/material logic mirrors `helio-pass-gbuffer/shaders/gbuffer.wgsl`
//! closely (own copy — see that file for the fuller commentary on each
//! piece); the two differences are: (1) the extra `portal_space *` factor
//! composed onto the instance's own coordinate-space transform, and (2) the
//! fragment-shader world-space clip test against the portal's opening.
//! Debug-visualization modes, lightmap sampling, and the Radiant material
//! graph override hook are not reachable here — a portal duplicate always
//! renders through the plain default PBR path.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct ScreenSize {
    width:  f32,
    height: f32,
    _pad0:  f32,
    _pad1:  f32,
}

/// GPU material (112 bytes, matches libhelio::GpuMaterial) — identical to gbuffer.wgsl.
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
    params:             vec4<f32>,
}

/// Must match GpuInstanceData in libhelio exactly (208 bytes).
struct GpuInstanceData {
    transform:      mat4x4<f32>,
    normal_mat_0:   vec4<f32>,
    normal_mat_1:   vec4<f32>,
    normal_mat_2:   vec4<f32>,
    bounds:         vec4<f32>,
    prev_model:     mat4x4<f32>,
    mesh_id:        u32,
    material_id:    u32,
    flags:          u32,
    lightmap_index: u32,
}

/// Must match libhelio::GpuPortalView (80 bytes).
struct GpuPortalView {
    inverse_transform: mat4x4<f32>,
    half_extent:       vec2<f32>,
    coordinate_space:  u32,
    _pad:              u32,
}

struct PortalDrawUniform {
    portal_view_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Coordinate-space slots backed by GPU storage — must match
// `libhelio::MAX_COORDINATE_SPACES` and `helio_pass_portal_cull::PORTAL_INSTANCE_CAPACITY`.
// The latter sizes the per-portal slice stride below; kept as a const (not a
// uniform) since it never changes at runtime.
const PORTAL_INSTANCE_CAPACITY: u32 = 65536u;

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform>       screen:  ScreenSize;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;
@group(0) @binding(3) var<storage, read> coordinate_spaces:      array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read> coordinate_spaces_prev: array<mat4x4<f32>>;
// Written by helio-pass-portal-cull: per-portal compacted original instance
// slots, slice `portal_view_index` at `[portal_view_index * PORTAL_INSTANCE_CAPACITY, ...)`.
@group(0) @binding(5) var<storage, read> portal_compacted_indices: array<u32>;
@group(0) @binding(6) var<storage, read> portal_views: array<GpuPortalView>;
// Dynamic-offset uniform selecting which portal this draw call belongs to —
// same "one small uniform, rebind per draw via dynamic offset" idiom
// `helio-pass-shadow`'s FaceIndex uses for its own per-face selection.
@group(0) @binding(7) var<uniform> portal_draw: PortalDrawUniform;

@group(1) @binding(0) var<storage, read>    materials:          array<GpuMaterial>;
@group(1) @binding(1) var<storage, read>    material_textures:  array<MaterialTextureData>;
@group(1) @binding(2) var                   scene_textures:     binding_array<texture_2d<f32>, 256>;
@group(1) @binding(3) var                   scene_samplers:     binding_array<sampler, 256>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
    @location(5) lightmap_uv:    vec2<f32>,
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position:     vec3<f32>,
    @location(1) world_normal:       vec3<f32>,
    @location(2) tex_coords:         vec2<f32>,
    @location(3) world_tangent:      vec3<f32>,
    @location(4) bitangent_sign:     f32,
    @location(5) @interpolate(flat) material_id: u32,
    @location(6) prev_clip_position: vec4<f32>,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(v: Vertex, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let compacted_base = portal_draw.portal_view_index * PORTAL_INSTANCE_CAPACITY;
    let inst = instance_data[portal_compacted_indices[compacted_base + instance_index]];
    let portal = portal_views[portal_draw.portal_view_index];

    // Compose the instance's own coordinate space (identity for an ordinary
    // world-space object, or its sublevel's transform) with the portal's —
    // duplicated content is placed exactly where it actually is, then mapped
    // through the portal on top, same as the main pass places sublevel
    // members and this pass additionally maps them through one more space.
    let own_space_id  = (inst.flags >> 8u) & 0xFFu;
    let own_space      = coordinate_spaces[own_space_id];
    let own_space_prev = coordinate_spaces_prev[own_space_id];
    let portal_space      = coordinate_spaces[portal.coordinate_space];
    let portal_space_prev = coordinate_spaces_prev[portal.coordinate_space];

    let own_rot    = mat3x3<f32>(own_space[0].xyz, own_space[1].xyz, own_space[2].xyz);
    let portal_rot = mat3x3<f32>(portal_space[0].xyz, portal_space[1].xyz, portal_space[2].xyz);
    let space_rot  = portal_rot * own_rot;

    let world_pos = portal_space * (own_space * (inst.transform * vec4<f32>(v.position, 1.0)));

    let normal_mat = space_rot * mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );
    let model_mat3 = space_rot * mat3x3<f32>(
        inst.transform[0].xyz,
        inst.transform[1].xyz,
        inst.transform[2].xyz,
    );

    let prev_world = portal_space_prev * (own_space_prev * (inst.prev_model * vec4<f32>(v.position, 1.0)));
    let prev_clip  = cameras[0].prev_view_proj * prev_world;

    var out: VertexOutput;
    out.clip_position      = cameras[0].view_proj * world_pos;
    out.world_position     = world_pos.xyz;
    out.world_normal       = normalize(normal_mat * decode_snorm8x4(v.normal));
    out.world_tangent      = normalize(model_mat3 * decode_snorm8x4(v.tangent));
    out.bitangent_sign     = v.bitangent_sign;
    out.tex_coords         = v.tex_coords;
    out.material_id        = inst.material_id;
    out.prev_clip_position = prev_clip;
    return out;
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
    @location(7) velocity:    vec2<f32>,
}

const NO_TEXTURE: u32 = 0xffffffffu;
const MATERIAL_WORKFLOW_SPECULAR: u32 = 1u;

fn select_uv(slot: MaterialTextureSlot, base_uv: vec2<f32>) -> vec2<f32> {
    let scaled = base_uv * slot.offset_scale.zw;
    let s = slot.rotation.x;
    let c = slot.rotation.y;
    let rotated = vec2<f32>(scaled.x * c - scaled.y * s, scaled.x * s + scaled.y * c);
    return rotated + slot.offset_scale.xy;
}

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
    return clamp(mix(vec3<f32>(0.04), albedo, metallic), vec3<f32>(0.0), vec3<f32>(0.999));
}

fn compute_velocity(input: VertexOutput) -> vec2<f32> {
    let prev_ndc = input.prev_clip_position.xy / input.prev_clip_position.w;
    let prev_pixel_x = (prev_ndc.x * 0.5 + 0.5) * screen.width;
    let prev_pixel_y = (0.5 - prev_ndc.y * 0.5) * screen.height;
    let prev_pixel = vec2<f32>(prev_pixel_x, prev_pixel_y);
    return input.clip_position.xy - prev_pixel;
}

// TEMP DIAGNOSTIC (step 2) — clip test RE-ENABLED, material evaluation still
// replaced with forced bright emissive magenta. If the clip test's own logic
// is correct, magenta should now only appear in a small window at each
// portal instead of covering the whole screen. The real body (full PBR
// material evaluation) is preserved below in `fs_main_real`; swap the
// `@fragment` attribute back once this is resolved.
@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    let portal = portal_views[portal_draw.portal_view_index];
    let local = (portal.inverse_transform * vec4<f32>(input.world_position, 1.0)).xyz;
    if abs(local.x) > portal.half_extent.x || abs(local.y) > portal.half_extent.y || local.z > 0.0 {
        discard;
    }
    var diag_out: GBufferOutput;
    diag_out.albedo = vec4<f32>(1.0, 0.0, 1.0, 1.0);
    diag_out.normal = vec4<f32>(normalize(input.world_normal), 0.0);
    diag_out.orm = vec4<f32>(1.0, 1.0, 0.0, 0.0);
    diag_out.emissive = vec4<f32>(3.0, 0.0, 3.0, 0.0);
    diag_out.lightmap_uv = vec2<f32>(-1.0, -1.0);
    diag_out.sss = vec4<f32>(0.0);
    diag_out.extra = vec4<f32>(0.0);
    diag_out.velocity = compute_velocity(input);
    return diag_out;
}

fn fs_main_real(input: VertexOutput) -> GBufferOutput {
    // World-space clip test: keep only fragments inside this portal's
    // opening and in front of its surface (see module docs for the sign
    // convention — `PortalPose::forward()` is -Z, so "visible through" is
    // local Z <= 0).
    let portal = portal_views[portal_draw.portal_view_index];
    let local = (portal.inverse_transform * vec4<f32>(input.world_position, 1.0)).xyz;
    if abs(local.x) > portal.half_extent.x || abs(local.y) > portal.half_extent.y || local.z > 0.0 {
        discard;
    }

    let material = materials[input.material_id];
    let material_tex = material_textures[input.material_id];
    let uv = input.tex_coords;

    let base_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));
    let albedo = material.base_color * base_sample;
    if albedo.a <= 0.001 || albedo.a < material_tex.params.z { discard; }

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

    let ao = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;
    let roughness = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    let metallic = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    let specular_f0 = resolve_specular_f0(material, material_tex, albedo.rgb, metallic, uv);
    let emissive = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    var out: GBufferOutput;
    out.albedo = vec4<f32>(albedo.rgb, albedo.a);
    out.normal = vec4<f32>(N, specular_f0.r);
    out.orm = vec4<f32>(ao, roughness, metallic, specular_f0.g);
    out.emissive = vec4<f32>(emissive, specular_f0.b);
    // Sentinel: portal duplicates don't carry baked lightmap data.
    out.lightmap_uv = vec2<f32>(-1.0, -1.0);
    out.sss = vec4<f32>(0.0);
    out.extra = vec4<f32>(0.0);
    out.velocity = compute_velocity(input);
    return out;
}
