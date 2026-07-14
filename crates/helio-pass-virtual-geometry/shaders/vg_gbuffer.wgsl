// Virtual geometry G-buffer draw pass.
//
// Identical to gbuffer.wgsl but reads instance data from the VG-specific
// instance buffer.  The indirect draw's `first_instance` is the slot in the
// VG instance buffer written by the culling pass.
//
// This shader is compiled into `helio-pass-virtual-geometry` and draws only
// VG meshlets that survived the cull compute shader.

// (primitive_index extension removed: not supported by WebGPU in browsers)

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
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec4<f32>,
    rc_world_min:      vec4<f32>,
    rc_world_max:      vec4<f32>,
    csm_splits:        vec4<f32>,
    debug_mode:        u32,
    _pad0:             u32,
    _pad1:             u32,
    _pad2:             u32,
}

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
    _pad:               u32,
}

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

struct GpuInstanceData {
    transform:    mat4x4<f32>,
    normal_mat_0: vec4<f32>,
    normal_mat_1: vec4<f32>,
    normal_mat_2: vec4<f32>,
    bounds:       vec4<f32>,
    mesh_id:      u32,
    material_id:  u32,
    flags:        u32,
    _pad:         u32,
}

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var<uniform>       globals:       Globals;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;

@group(1) @binding(0) var<storage, read> materials:          array<GpuMaterial>;
@group(1) @binding(1) var<storage, read> material_textures:  array<MaterialTextureData>;
// HELIO_WEBGPU_MATERIAL_BINDINGS

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) world_tangent:  vec3<f32>,
    @location(4) bitangent_sign: f32,
    @location(5) @interpolate(flat) material_id:    u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(v: Vertex, @builtin(instance_index) packed_slot: u32) -> VertexOutput {
    // LOD level is packed in upper 8 bits by the cull shader.
    let slot = packed_slot & 0x00FFFFFFu;
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(v.position, 1.0);

    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );
    let model_mat3 = mat3x3<f32>(
        inst.transform[0].xyz,
        inst.transform[1].xyz,
        inst.transform[2].xyz,
    );

    var out: VertexOutput;
    out.clip_position  = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat  * decode_snorm8x4(v.normal));
    out.world_tangent  = normalize(model_mat3  * decode_snorm8x4(v.tangent));
    out.bitangent_sign = v.bitangent_sign;
    out.tex_coords     = v.tex_coords;
    out.material_id    = inst.material_id;
    return out;
}

struct GBufferOutput {
    @location(0) albedo:    vec4<f32>,
    @location(1) normal:    vec4<f32>,
    @location(2) orm:       vec4<f32>,
    @location(3) emissive:  vec4<f32>,
}

const NO_TEXTURE: u32 = 0xffffffffu;
const MATERIAL_WORKFLOW_METALLIC: u32 = 0u;
const MATERIAL_WORKFLOW_SPECULAR: u32 = 1u;

fn select_uv(slot: MaterialTextureSlot, base_uv: vec2<f32>) -> vec2<f32> {
    let scaled = base_uv * slot.offset_scale.zw;
    let s = slot.rotation.x;
    let c = slot.rotation.y;
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    );
    return rotated + slot.offset_scale.xy;
}

// HELIO_WEBGPU_MATERIAL_SAMPLER

fn transform_uv_gradient(slot: MaterialTextureSlot, gradient: vec2<f32>) -> vec2<f32> {
    let scaled = gradient * slot.offset_scale.zw;
    let s = slot.rotation.x;
    let c = slot.rotation.y;
    return vec2<f32>(scaled.x * c - scaled.y * s, scaled.x * s + scaled.y * c);
}

fn sample_texture(
    slot: MaterialTextureSlot,
    base_uv: vec2<f32>,
    base_uv_dx: vec2<f32>,
    base_uv_dy: vec2<f32>,
    fallback: vec4<f32>,
) -> vec4<f32> {
    if slot.texture_index == NO_TEXTURE {
        return fallback;
    }
    let uv = select_uv(slot, base_uv);
    let uv_dx = transform_uv_gradient(slot, base_uv_dx);
    let uv_dy = transform_uv_gradient(slot, base_uv_dy);
    return sample_scene_texture(slot.texture_index, uv, uv_dx, uv_dy);
}

fn resolve_specular_f0(
    material: GpuMaterial,
    material_tex: MaterialTextureData,
    albedo: vec3<f32>,
    metallic: f32,
    uv: vec2<f32>,
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
) -> vec3<f32> {
    if material.workflow == MATERIAL_WORKFLOW_SPECULAR {
        let specular_color = sample_texture(material_tex.specular_color, uv, uv_dx, uv_dy, vec4<f32>(1.0)).rgb;
        let specular_weight = sample_texture(material_tex.specular_weight, uv, uv_dx, uv_dy, vec4<f32>(1.0)).a;
        let ior = max(material.roughness_metallic.z, 1.0);
        let dielectric_f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
        return material.roughness_metallic.w * specular_weight * specular_color * dielectric_f0;
    }
    return clamp(
        mix(vec3<f32>(0.04), albedo, metallic),
        vec3<f32>(0.0),
        vec3<f32>(0.999),
    );
}

@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    // ── Normal rendering ──────────────────────────────────────────────────────
    let material     = materials[input.material_id];
    let material_tex = material_textures[input.material_id];
    let uv = input.tex_coords;
    let uv_dx = dpdx(uv);
    let uv_dy = dpdy(uv);

    let base_sample = sample_texture(material_tex.base_color, uv, uv_dx, uv_dy, vec4<f32>(1.0));
    let albedo      = material.base_color * base_sample;
    let alpha       = albedo.a;

    if alpha <= 0.001         { discard; }
    if alpha < material_tex.params.z { discard; }

    let N_geom = normalize(input.world_normal);
    var N: vec3<f32>;
    if material_tex.normal.texture_index != NO_TEXTURE {
        let T = normalize(input.world_tangent - dot(input.world_tangent, N_geom) * N_geom);
        let B = cross(N_geom, T) * input.bitangent_sign;
        var norm_ts = sample_texture(material_tex.normal, uv, uv_dx, uv_dy, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - 1.0;
        norm_ts = vec3<f32>(norm_ts.x * material_tex.params.x, norm_ts.y * material_tex.params.x, norm_ts.z);
        N = normalize(T * norm_ts.x + B * norm_ts.y + N_geom * norm_ts.z);
    } else {
        N = N_geom;
    }

    let orm_sample      = sample_texture(material_tex.roughness_metallic, uv, uv_dx, uv_dy, vec4<f32>(1.0));
    let occlusion_sample = sample_texture(material_tex.occlusion, uv, uv_dx, uv_dy, vec4<f32>(1.0));
    let emissive_sample = sample_texture(material_tex.emissive, uv, uv_dx, uv_dy, vec4<f32>(1.0));

    let ao       = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;
    let roughness = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    let metallic  = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    let specular_f0 = resolve_specular_f0(material, material_tex, albedo.rgb, metallic, uv, uv_dx, uv_dy);
    let emissive  = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    var out: GBufferOutput;
    out.albedo  = vec4<f32>(albedo.rgb, alpha);
    out.normal  = vec4<f32>(N, -(specular_f0.r + 1.0));
    out.orm     = vec4<f32>(ao, roughness, metallic, specular_f0.g);
    out.emissive = vec4<f32>(emissive, specular_f0.b);
    return out;
}

// ── Deterministic per-triangle hash for debug colouring ─────────────────────
// Uses world-space face centroid to generate a stable per-triangle seed.
fn tri_hash(world_pos: vec3<f32>) -> u32 {
    let centroid = floor(world_pos * 1000.0);
    var h = (u32(centroid.x) * 73856093u) ^
        (u32(centroid.y) * 19349663u) ^
        (u32(centroid.z) * 83492791u);
    h ^= h >> 13u;
    h *= 2654435769u;
    h ^= h >> 16u;
    return h;
}

// ── VG triangle debug (mode 20): unlit solid colour per meshlet + per-tri variation
@fragment
fn fs_debug(input: VertexOutput) -> GBufferOutput {
    // Meshlet-level base colour from material_id hash.
    var h = (input.material_id * 2747636419u);
    h ^= h >> 16u;
    h *= 2654435769u;
    h ^= h >> 16u;
    let idx = h % 12u;

    var pal: array<vec3<f32>, 12>;
    pal[0]  = vec3<f32>(1.00, 0.18, 0.18);
    pal[1]  = vec3<f32>(1.00, 0.55, 0.00);
    pal[2]  = vec3<f32>(1.00, 0.90, 0.00);
    pal[3]  = vec3<f32>(0.35, 1.00, 0.10);
    pal[4]  = vec3<f32>(0.00, 0.90, 0.40);
    pal[5]  = vec3<f32>(0.00, 0.85, 1.00);
    pal[6]  = vec3<f32>(0.10, 0.40, 1.00);
    pal[7]  = vec3<f32>(0.55, 0.10, 1.00);
    pal[8]  = vec3<f32>(0.90, 0.10, 1.00);
    pal[9]  = vec3<f32>(1.00, 0.10, 0.60);
    pal[10] = vec3<f32>(0.00, 0.65, 0.65);
    pal[11] = vec3<f32>(1.00, 0.70, 0.10);

    // Per-triangle variation: slight brightness shift so individual triangles are visible.
    let face_centroid = input.world_position;
    let t_hash = tri_hash(face_centroid);
    let brightness = 0.7 + 0.3 * f32(t_hash % 256u) / 255.0;
    let base_color = pal[idx] * brightness;

    // Unlit: write colour directly, flat face normal, no material/lighting.
    let face_n = normalize(cross(dpdx(input.world_position), dpdy(input.world_position)));
    var out: GBufferOutput;
    out.albedo   = vec4<f32>(base_color, 1.0);
    out.normal   = vec4<f32>(face_n, -1.0);
    out.orm      = vec4<f32>(1.0, 1.0, 0.0, 0.0);
    out.emissive = vec4<f32>(base_color, 0.0);
    return out;
}

// ── VG LOD heatmap debug (mode 21) ─────────────────────────────────────────
// Nanite-style: per-triangle colours tinted by LOD level.
// Green (LOD0/full detail) → Red (LOD7/coarsest), with per-tri variation
// so individual triangles remain visible within each LOD band.

struct LodVertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) lod_level:      f32,
}

@vertex
fn vs_debug_lod(v: Vertex, @builtin(instance_index) packed_slot: u32) -> LodVertexOutput {
    let slot      = packed_slot & 0x00FFFFFFu;
    let lod_level = packed_slot >> 24u;
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(v.position, 1.0);

    var out: LodVertexOutput;
    out.clip_position  = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.lod_level      = f32(lod_level);
    return out;
}

@fragment
fn fs_debug_lod(input: LodVertexOutput) -> GBufferOutput {
    // 8-stop colour ramp: LOD0=green → LOD7=red
    var pal: array<vec3<f32>, 8>;
    pal[0] = vec3<f32>(0.10, 0.95, 0.10); // LOD 0 — bright green
    pal[1] = vec3<f32>(0.50, 0.95, 0.10); // LOD 1 — yellow-green
    pal[2] = vec3<f32>(0.85, 0.90, 0.10); // LOD 2 — yellow
    pal[3] = vec3<f32>(1.00, 0.70, 0.10); // LOD 3 — amber
    pal[4] = vec3<f32>(1.00, 0.45, 0.05); // LOD 4 — orange
    pal[5] = vec3<f32>(1.00, 0.25, 0.05); // LOD 5 — red-orange
    pal[6] = vec3<f32>(0.90, 0.10, 0.05); // LOD 6 — red
    pal[7] = vec3<f32>(0.60, 0.05, 0.30); // LOD 7 — dark magenta

    let idx = min(u32(input.lod_level + 0.5), 7u);
    let lod_color = pal[idx];

    // Per-triangle brightness variation so individual tris are visible.
    let t_hash = tri_hash(input.world_position);
    let brightness = 0.7 + 0.3 * f32(t_hash % 256u) / 255.0;
    let final_color = lod_color * brightness;

    // Unlit: emit colour directly, no material/lighting.
    let face_n = normalize(cross(dpdx(input.world_position), dpdy(input.world_position)));
    var out: GBufferOutput;
    out.albedo   = vec4<f32>(final_color, 1.0);
    out.normal   = vec4<f32>(face_n, -1.0);
    out.orm      = vec4<f32>(1.0, 1.0, 0.0, 0.0);
    out.emissive = vec4<f32>(final_color, 0.0);
    return out;
}
