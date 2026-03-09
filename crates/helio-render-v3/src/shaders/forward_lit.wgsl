// forward_lit.wgsl — forward-lit transparency pass (draws on top of deferred result).
// Uses the same Cook-Torrance BRDF as deferred, but runs per-fragment.

override ENABLE_SHADOWS: bool = false;

const PI: f32 = 3.14159265358979323846;

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,
};

struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec3<f32>,
    csm_split_count:   u32,
    rc_world_min:      vec3<f32>,
    _pad0:             u32,
};

struct GpuLight {
    position:      vec3<f32>,
    kind:          u32,
    color:         vec3<f32>,
    intensity:     f32,
    direction:     vec3<f32>,
    range:         f32,
    inner_cone:    f32,
    outer_cone:    f32,
    shadow_idx:    i32,
    _pad:          f32,
};

struct MaterialUniform {
    base_color:   vec4<f32>,
    emissive:     vec3<f32>,
    metallic:     f32,
    roughness:    f32,
    occlusion:    f32,
    alpha_cutoff: f32,
    _pad:         f32,
};

@group(0) @binding(0) var<uniform>  camera:  Camera;
@group(0) @binding(1) var<uniform>  globals: Globals;

@group(1) @binding(0) var<uniform>  material:      MaterialUniform;
@group(1) @binding(1) var           albedo_tex:    texture_2d<f32>;
@group(1) @binding(2) var           normal_tex:    texture_2d<f32>;
@group(1) @binding(3) var           tex_sampler:   sampler;
@group(1) @binding(4) var           orm_tex:       texture_2d<f32>;
@group(1) @binding(5) var           emissive_tex:  texture_2d<f32>;

// Light storage passed from deferred pass context
@group(2) @binding(0) var<storage, read> lights: array<GpuLight>;

struct VertexIn {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) uv:             vec2<f32>,
    @location(3) packed_normal:  vec4<f32>,
    @location(4) packed_tangent: vec4<f32>,
    @location(5) i0: vec4<f32>,
    @location(6) i1: vec4<f32>,
    @location(7) i2: vec4<f32>,
    @location(8) i3: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       world_pos:  vec3<f32>,
    @location(1)       uv:         vec2<f32>,
    @location(2)       world_norm: vec3<f32>,
    @location(3)       world_tan:  vec3<f32>,
    @location(4)       bitan_sign: f32,
};



@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let model    = mat4x4<f32>(in.i0, in.i1, in.i2, in.i3);
    let norm_mat = mat3x3<f32>(in.i0.xyz, in.i1.xyz, in.i2.xyz);
    let wp = model * vec4<f32>(in.position, 1.0);
    var out: VertexOut;
    out.clip_pos   = camera.view_proj * wp;
    out.world_pos  = wp.xyz;
    out.uv         = in.uv;
    out.world_norm = normalize(norm_mat * in.packed_normal.xyz);
    out.world_tan  = normalize(norm_mat * in.packed_tangent.xyz);
    out.bitan_sign = in.bitangent_sign;
    return out;
}

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn geometry_ggx(ndv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return ndv / (ndv * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    return geometry_ggx(max(dot(n,v), 0.0), roughness) * geometry_ggx(max(dot(n,l), 0.0), roughness);
}

fn fresnel(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_transparent(in: VertexOut) -> @location(0) vec4<f32> {
    let albedo_smp = textureSample(albedo_tex, tex_sampler, in.uv);
    let albedo  = albedo_smp.rgb * material.base_color.rgb;
    let alpha   = albedo_smp.a  * material.base_color.a;

    let norm_smp  = textureSample(normal_tex, tex_sampler, in.uv).rgb * 2.0 - 1.0;
    let bitangent = cross(in.world_norm, in.world_tan) * in.bitan_sign;
    let tbn       = mat3x3<f32>(in.world_tan, bitangent, in.world_norm);
    let n = normalize(tbn * norm_smp);

    let orm_smp  = textureSample(orm_tex, tex_sampler, in.uv);
    let metallic  = orm_smp.b * material.metallic;
    let roughness = orm_smp.g * material.roughness;
    let ao        = orm_smp.r * material.occlusion;
    let emissive  = textureSample(emissive_tex, tex_sampler, in.uv).rgb * material.emissive;

    let v  = normalize(camera.position - in.world_pos);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < globals.light_count; i++) {
        let light = lights[i];
        var l = vec3<f32>(0.0);
        var atten = 1.0f;

        if light.kind == 0u {
            l = normalize(-light.direction);
        } else if light.kind == 1u {
            let d = light.position - in.world_pos;
            l = normalize(d);
            atten = 1.0 / (1.0 + dot(d,d) / (light.range * light.range));
        } else {
            let d = light.position - in.world_pos;
            l = normalize(d);
            let ct = dot(-l, normalize(light.direction));
            atten = smoothstep(light.outer_cone, light.inner_cone, ct)
                / (1.0 + dot(d,d) / (light.range * light.range));
        }

        let h = normalize(v + l);
        let d_term = distribution_ggx(n, h, roughness);
        let g_term = geometry_smith(n, v, l, roughness);
        let f_term = fresnel(max(dot(h, v), 0.0), f0);
        let kd     = (1.0 - f_term) * (1.0 - metallic);
        let ndl    = max(dot(n, l), 0.0);
        let ndv    = max(dot(n, v), 0.0001);
        let spec   = d_term * g_term * f_term / max(4.0 * ndl * ndv, 0.0001);
        lo += (kd * albedo / PI + spec) * ndl * light.color * light.intensity * atten;
    }

    let ambient = globals.ambient_color * globals.ambient_intensity * albedo * ao;
    let color = lo + ambient + emissive;
    return vec4<f32>(color, alpha);
}
