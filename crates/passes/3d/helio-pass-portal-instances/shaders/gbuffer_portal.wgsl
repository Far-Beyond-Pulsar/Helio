enable wgpu_binding_array;

//! Portal-duplicate G-buffer write for one recursion *level* — draws the
//! instances `helio-pass-portal-cull` selected whose chain is exactly
//! `level` deep, mapped through that chain's *composed* transform. One
//! `PortalInstancePass` instance per level `1..=MAX_CHAIN_DEPTH`, each
//! immediately preceded by its own `PortalMaskPass` instance for the same
//! level (Mask(1),Draw(1),Mask(2),Draw(2),Mask(3),Draw(3) — see
//! `helio-default-graphs`). See `shaders/portal_mask.wgsl`'s module doc for
//! why recursion needs this interleaving rather than one flat pass.
//!
//! Fused into the same physical render pass `helio-pass-gbuffer` opened
//! (`LoadOp::Load` on all 8 attachments — see `helio-pass-foliage-gbuffer`
//! for the precedent this follows), so it shares the real depth buffer and
//! composes correctly with everything already drawn. One
//! `multi_draw_indexed_indirect` call, same shape as the ordinary
//! non-portal G-buffer pass — one draw per mesh/material draw group. All
//! chains (every depth, and every draw group) share one compacted-instance
//! buffer; each level's draw call skips instances whose chain isn't this
//! level's own depth, and gates the rest on this level's freshly-stamped
//! mask.
//!
//! # Why one mask check is enough, even for depth 3
//!
//! `helio-pass-portal-mask`'s level-*k* stamp only marks a chain's pixels
//! at all if that chain's real quad (deepest, final portal at its own true
//! position, mapped through the parent prefix's composed transform) passed
//! a depth test against whatever's *actually been drawn by now* — which
//! for level *k* includes level *k-1*'s own already-drawn duplicate
//! content. So passing level 3's mask already implies level 3's parent
//! (level 2) was itself visible there — occlusion composes through the
//! levels for free, the same way a nearer mirror blocking a farther one
//! does in a real room. That's what lets this shader drop the old
//! per-stage world-space box entirely and trust one screen-space lookup:
//! `portal_mask` at this fragment's pixel, compared against *this specific
//! chain's* own index (not just its final portal's — two different chains
//! can reach the same portal through different parents and land in
//! different screen positions, so the portal alone can't identify which
//! one a pixel belongs to).

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

/// Must match libhelio::GpuPortalView (144 bytes).
struct GpuPortalView {
    transform:         mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
    half_extent:       vec2<f32>,
    coordinate_space:  u32,
    _pad:              u32,
}

/// Must match libhelio::GpuPortalChain (16 bytes at MAX_CHAIN_DEPTH=3).
struct GpuPortalChain {
    portals: array<u32, 3>,
    depth:   u32,
}

struct LevelUniform {
    level: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform>       screen:  ScreenSize;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;
@group(0) @binding(3) var<storage, read> coordinate_spaces:      array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read> coordinate_spaces_prev: array<mat4x4<f32>>;
// Written by helio-pass-portal-cull: shared compacted original instance
// slots — group `g`'s draw call's `first_instance`/`instance_count` (from
// `portal_indirect`) already point `@builtin(instance_index)` at exactly
// this group's region, so no per-draw offset math is needed here.
@group(0) @binding(5) var<storage, read> portal_compacted_indices: array<u32>;
@group(0) @binding(6) var<storage, read> portal_views: array<GpuPortalView>;
@group(0) @binding(7) var<storage, read> portal_chains: array<GpuPortalChain>;
// Parallel to portal_compacted_indices — which chain each compacted entry
// was selected under.
@group(0) @binding(8) var<storage, read> portal_compacted_chains: array<u32>;
// Written by helio-pass-portal-mask (this level's own instance of it, run
// immediately before this pass): per-pixel `chain_index + 1` where that
// specific chain is actually visible on screen this frame, 0 elsewhere.
@group(0) @binding(9) var portal_mask: texture_2d<u32>;
// Which recursion level this pass instance draws — instances whose chain
// isn't this deep are skipped (see vs_main).
@group(0) @binding(10) var<uniform> level_uniform: LevelUniform;

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
    @location(7) @interpolate(flat) chain_idx: u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(v: Vertex, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let slot_idx = portal_compacted_indices[instance_index];
    let chain_idx = portal_compacted_chains[instance_index];
    let inst = instance_data[slot_idx];
    let chain = portal_chains[chain_idx];

    var out: VertexOutput;
    if chain.depth != level_uniform.level {
        // Not this pass's level — degenerate rather than branch the whole
        // draw call; the compacted buffer intentionally isn't pre-split by
        // level (see the module doc), so each level's draw walks every
        // chain's instances and skips the ones that aren't its own.
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.chain_idx = chain_idx;
        return out;
    }

    // Compose the instance's own coordinate space (identity for an ordinary
    // world-space object, or its sublevel's transform) through the whole
    // chain, deepest portal first.
    let own_space_id  = (inst.flags >> 8u) & 0xFFu;
    let own_space      = coordinate_spaces[own_space_id];
    let own_space_prev = coordinate_spaces_prev[own_space_id];

    var pos      = own_space * (inst.transform * vec4<f32>(v.position, 1.0));
    var pos_prev = own_space_prev * (inst.prev_model * vec4<f32>(v.position, 1.0));
    var space_rot = mat3x3<f32>(own_space[0].xyz, own_space[1].xyz, own_space[2].xyz);

    for (var i = chain.depth; i > 0u; i--) {
        let p = portal_views[chain.portals[i - 1u]];
        let p_space = coordinate_spaces[p.coordinate_space];
        let p_space_prev = coordinate_spaces_prev[p.coordinate_space];
        pos = p_space * pos;
        pos_prev = p_space_prev * pos_prev;
        space_rot = mat3x3<f32>(p_space[0].xyz, p_space[1].xyz, p_space[2].xyz) * space_rot;
    }

    let world_pos = pos;

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

    let prev_clip = cameras[0].prev_view_proj * pos_prev;

    out.clip_position      = cameras[0].view_proj * world_pos;
    out.world_position     = world_pos.xyz;
    out.world_normal       = normalize(normal_mat * decode_snorm8x4(v.normal));
    out.world_tangent      = normalize(model_mat3 * decode_snorm8x4(v.tangent));
    out.bitangent_sign     = v.bitangent_sign;
    out.tex_coords         = v.tex_coords;
    out.material_id        = inst.material_id;
    out.prev_clip_position = prev_clip;
    out.chain_idx          = chain_idx;
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

@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    // Screen-space gate: only draw where `helio-pass-portal-mask`'s
    // level-`level_uniform.level` instance determined *this specific
    // chain* is actually visible on screen — see the module doc for why
    // this one check is sufficient at every recursion depth.
    let mask_px = vec2<i32>(input.clip_position.xy);
    let mask_value = textureLoad(portal_mask, mask_px, 0).r;
    if mask_value != input.chain_idx + 1u {
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
    out.lightmap_uv = vec2<f32>(-1.0, -1.0);
    out.sss = vec4<f32>(0.0);
    out.extra = vec4<f32>(0.0);
    out.velocity = compute_velocity(input);
    return out;
}
