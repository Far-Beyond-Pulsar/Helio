enable wgpu_binding_array;

// Secondary G-buffer fill — off-screen 8-target G-buffer + depth for a single
// portal/sublevel camera view.
//
// Full default-PBR material evaluation (base color, normal, roughness/metallic,
// occlusion, emissive maps; metallic and specular workflows), bindless-texture
// sampling identical in capability to the main `GBufferPass`. Struct layouts and
// the material-evaluation functions below are duplicated verbatim from
// `gbuffer.wgsl` / `gbuffer_common.wgsl` rather than shared through a module
// system (this codebase's existing convention — every 3D pass keeps its own
// local copies; see `helio_core::shader::mod`'s doc comment on why
// `GBUFFER_COMMON` isn't wired into `resolve()`). `sample_texture`'s exact
// body text (`return textureSample(scene_textures[...], scene_samplers[...],
// uv);`) must match `libhelio::shader::apply_webgpu_material_bindings`'s
// string-replace target verbatim — do not reformat that line.
//
// Not carried over from the main pass (intentionally out of scope — see this
// crate's `lib.rs` module doc): Radiant custom material graphs (only the
// default PBR evaluation runs; `material_class`/`class_params` are read but
// unused), baked lightmaps, debug-visualization modes, alpha-blended/
// forward-shaded materials (opaque only, matching the main pass's own
// deferred/forward split — forward-shaded content never reaches `GBufferPass`
// either).

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

/// Per-instance data (208 bytes). Must match `libhelio::GpuInstanceData`.
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

/// GPU material (112 bytes). Must match `libhelio::GpuMaterial`.
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

/// Per-material texture metadata (224 bytes). Must match `helio::GpuMaterialTextures`.
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

struct ViewParams {
    camera_slot:        u32,
    membership_filter:  u32,
    draw_count:         u32,
    _pad0:              u32,
}

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
    @location(0) world_normal:       vec3<f32>,
    @location(1) tex_coords:         vec2<f32>,
    @location(2) world_tangent:      vec3<f32>,
    @location(3) bitangent_sign:     f32,
    @location(4) @interpolate(flat) material_id: u32,
    @location(5) prev_clip_position: vec4<f32>,
}

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

struct SurfaceData {
    albedo:      vec4<f32>,
    normal:      vec3<f32>,
    ao:          f32,
    roughness:   f32,
    metallic:    f32,
    specular_f0: vec3<f32>,
    emissive:    vec3<f32>,
    alpha:       f32,
}

@group(0) @binding(0) var<storage, read> cameras:           array<Camera, CAMERA_SLOTS>;
@group(0) @binding(1) var<storage, read> instance_data:     array<GpuInstanceData>;
@group(0) @binding(2) var<storage, read> compacted_indices: array<u32>;
@group(0) @binding(3) var<uniform>       view:              ViewParams;

@group(1) @binding(0) var<storage, read> materials:          array<GpuMaterial>;
@group(1) @binding(1) var<storage, read> material_textures:  array<MaterialTextureData>;
@group(1) @binding(2) var                scene_textures:     binding_array<texture_2d<f32>, 256>;
@group(1) @binding(3) var                scene_samplers:     binding_array<sampler, 256>;

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(v: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let inst = instance_data[compacted_indices[slot]];
    let cam  = cameras[view.camera_slot];

    let world      = inst.transform  * vec4<f32>(v.position, 1.0);
    let prev_world = inst.prev_model * vec4<f32>(v.position, 1.0);

    let normal_mat = mat3x3<f32>(inst.normal_mat_0.xyz, inst.normal_mat_1.xyz, inst.normal_mat_2.xyz);
    // Tangents transform by the plain upper-3x3 of the model matrix (no
    // inverse-transpose) — same distinction `gbuffer.wgsl` draws.
    let model_mat3 = mat3x3<f32>(inst.transform[0].xyz, inst.transform[1].xyz, inst.transform[2].xyz);

    var out: VertexOutput;
    out.clip_position      = cam.view_proj * world;
    out.world_normal       = normalize(normal_mat * decode_snorm8x4(v.normal));
    out.world_tangent      = normalize(model_mat3 * decode_snorm8x4(v.tangent));
    out.bitangent_sign     = v.bitangent_sign;
    out.tex_coords         = v.tex_coords;
    out.material_id        = inst.material_id;
    out.prev_clip_position = cam.prev_view_proj * prev_world;
    return out;
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

    let ao = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;
    let roughness = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    let metallic = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    let specular_f0 = resolve_specular_f0(material, material_tex, albedo.rgb, metallic, uv);
    let emissive = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    return SurfaceData(albedo, N, ao, roughness, metallic, specular_f0, emissive, alpha);
}

@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    let material = materials[input.material_id];
    let material_tex = material_textures[input.material_id];
    let surface = default_pbr_surface(material, material_tex, input);

    if surface.alpha <= 0.001 || surface.alpha < material_tex.params.z {
        discard;
    }

    let prev_ndc = input.prev_clip_position.xy / input.prev_clip_position.w;
    // This pass's own velocity output is unused — see `lib.rs`'s module doc:
    // `ProxyCompositePass` recomputes velocity against the *main* camera from
    // the composited world position. Written as zero only so the target
    // format matches the main G-buffer's byte-for-byte.
    var out: GBufferOutput;
    out.albedo      = vec4<f32>(surface.albedo.rgb, surface.alpha);
    // F0 packed into the unused alpha channels of normal/orm/emissive,
    // matching `gbuffer.wgsl`'s convention exactly, so `DeferredLight`
    // decodes composited pixels identically to native G-buffer ones.
    out.normal      = vec4<f32>(surface.normal, surface.specular_f0.r);
    out.orm         = vec4<f32>(surface.ao, surface.roughness, surface.metallic, surface.specular_f0.g);
    out.emissive    = vec4<f32>(surface.emissive, surface.specular_f0.b);
    // Sentinel "no lightmap" — see `gbuffer.wgsl`'s identical convention.
    // Baked lightmaps through portals/sublevels are out of scope (this crate's
    // module doc).
    out.lightmap_uv = vec2<f32>(-1.0, -1.0);
    out.sss         = vec4<f32>(0.0);
    out.extra       = vec4<f32>(0.0);
    out.velocity    = vec2<f32>(0.0, 0.0);
    return out;
}
