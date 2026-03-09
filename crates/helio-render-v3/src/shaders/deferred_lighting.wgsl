// deferred_lighting.wgsl — Cook-Torrance PBR, shadows (PCF), RC probe lookup.

override ENABLE_SHADOWS:     bool = false;
override MAX_SHADOW_LIGHTS:  u32  = 4u;
override ENABLE_RC:          bool = false;
override ENABLE_BLOOM:       bool = false;

const PI: f32 = 3.14159265358979323846;

// ── Bind groups ───────────────────────────────────────────────────────────────

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

// Layout must match Rust GpuLight exactly (4 × vec4):
//   [pos.xyz, kind_f32]  [dir.xyz, range]  [color.xyz, intensity]  [inner_cos, outer_cos, shadow01, pad]
struct GpuLight {
    position_type:     vec4<f32>,   // xyz=position, w=kind (0.0=dir, 1.0=point, 2.0=spot)
    direction_range:   vec4<f32>,   // xyz=direction, w=range
    color_intensity:   vec4<f32>,   // xyz=color, w=intensity
    cos_angles_shadow: vec4<f32>,   // x=cos_inner, y=cos_outer, z=shadow_flag(0/1), w=pad
};

struct ShadowMatrix {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    atlas_layer: u32,
    atlas_uv_offset: vec2<f32>,
    atlas_uv_scale:  f32,
    _pad: f32,
};

@group(0) @binding(0)  var<uniform>  camera:          Camera;
@group(0) @binding(1)  var<uniform>  globals:         Globals;
@group(0) @binding(2)  var<storage, read> lights:     array<GpuLight>;
@group(0) @binding(3)  var          gbuf_albedo:      texture_2d<f32>;
@group(0) @binding(4)  var          gbuf_normal:      texture_2d<f32>;
@group(0) @binding(5)  var          gbuf_orm:         texture_2d<f32>;
@group(0) @binding(6)  var          gbuf_emissive:    texture_2d<f32>;
@group(0) @binding(7)  var          depth_tex:        texture_depth_2d;
@group(0) @binding(8)  var          shadow_atlas:     texture_depth_2d_array;
@group(0) @binding(9)  var          shadow_sampler:   sampler_comparison;
@group(0) @binding(10) var<storage, read> shadow_matrices: array<ShadowMatrix>;
@group(0) @binding(11) var          rc_cascade0:      texture_2d<f32>;
@group(0) @binding(12) var          env_cube:         texture_cube<f32>;
@group(0) @binding(13) var          linear_sampler:   sampler;

// ── Fullscreen triangle ───────────────────────────────────────────────────────

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn oct_decode(enc: vec2<f32>) -> vec3<f32> {
    let p = enc * 2.0 - 1.0;
    var n = vec3<f32>(p.xy, 1.0 - abs(p.x) - abs(p.y));
    if n.z < 0.0 {
        n.x = (1.0 - abs(n.y)) * sign(n.x);
        n.y = (1.0 - abs(n.x)) * sign(n.y);
    }
    return normalize(n);
}

fn reconstruct_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world_h = camera.view_proj_inv * ndc;
    return world_h.xyz / world_h.w;
}

// ── Cook-Torrance PBR ─────────────────────────────────────────────────────────

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let denom = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(ndv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return ndv / (ndv * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let ndv = max(dot(n, v), 0.0);
    let ndl = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(ndv, roughness) * geometry_schlick_ggx(ndl, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn cook_torrance(
    l: vec3<f32>, v: vec3<f32>, n: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, roughness: f32,
    f0: vec3<f32>
) -> vec3<f32> {
    let h = normalize(v + l);
    let d = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let kd = (1.0 - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;

    let ndl = max(dot(n, l), 0.0);
    let ndv = max(dot(n, v), 0.0001);
    let spec = d * g * f / max(4.0 * ndl * ndv, 0.0001);
    return (diffuse + spec) * ndl;
}

// ── Shadow PCF 3x3 ────────────────────────────────────────────────────────────

fn sample_shadow(
    world_pos: vec3<f32>,
    sm: ShadowMatrix,
) -> f32 {
    let lsp = sm.view_proj * vec4<f32>(world_pos, 1.0);
    var shadow_uv = lsp.xy / lsp.w * vec2<f32>(0.5, -0.5) + 0.5;
    shadow_uv = shadow_uv * sm.atlas_uv_scale + sm.atlas_uv_offset;

    var shadow = 0.0;
    let dims = vec2<f32>(textureDimensions(shadow_atlas, 0));
    let texel = 1.0 / dims;
    let ref_depth = lsp.z / lsp.w;

    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
            shadow += textureSampleCompare(
                shadow_atlas, shadow_sampler,
                shadow_uv + offset,
                i32(sm.atlas_layer),
                ref_depth - 0.001,
            );
        }
    }
    return shadow / 9.0;
}

// ── Main fragment ─────────────────────────────────────────────────────────────

struct FragOut {
    @location(0) color: vec4<f32>,
};

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> FragOut {
    let dims = vec2<f32>(textureDimensions(gbuf_albedo));
    let uv   = frag_coord.xy / dims;
    let coord = vec2<i32>(i32(frag_coord.x), i32(frag_coord.y));

    let albedo_smp = textureLoad(gbuf_albedo,  coord, 0);
    let norm_mr    = textureLoad(gbuf_normal,  coord, 0);
    let orm_smp    = textureLoad(gbuf_orm,     coord, 0);
    let emissive   = textureLoad(gbuf_emissive, coord, 0).rgb;
    let depth_val  = textureLoad(depth_tex, coord, 0);

    // Skip skybox pixels.
    if depth_val >= 1.0 { return FragOut(vec4<f32>(0.0)); }

    let albedo   = albedo_smp.rgb;
    let normal   = oct_decode(norm_mr.rg);
    let metallic = norm_mr.b;
    let roughness = norm_mr.a;
    let ao       = orm_smp.r;

    let world_pos = reconstruct_world(uv, depth_val);
    let v = normalize(camera.position - world_pos);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);

    for (var i = 0u; i < globals.light_count; i++) {
        let light = lights[i];

        // Unpack vec4 fields to named locals for readability.
        let light_pos   = light.position_type.xyz;
        let light_kind  = u32(light.position_type.w + 0.5); // round 0.0/1.0/2.0 → 0/1/2
        let light_dir   = light.direction_range.xyz;
        let light_range = light.direction_range.w;
        let light_col   = light.color_intensity.xyz;
        let light_int   = light.color_intensity.w;
        let inner_cone  = light.cos_angles_shadow.x;
        let outer_cone  = light.cos_angles_shadow.y;

        var l      = vec3<f32>(0.0);
        var atten  = 1.0f;

        if light_kind == 0u {
            // Directional
            l = normalize(-light_dir);
        } else if light_kind == 1u {
            // Point
            let to_light = light_pos - world_pos;
            l = normalize(to_light);
            let dist = length(to_light);
            atten = 1.0 / (1.0 + dist * dist / (light_range * light_range));
        } else {
            // Spot
            let to_light = light_pos - world_pos;
            l = normalize(to_light);
            let dist = length(to_light);
            let cos_theta = dot(-l, normalize(light_dir));
            atten = smoothstep(outer_cone, inner_cone, cos_theta)
                / (1.0 + dist * dist / (light_range * light_range));
        }

        var shadow = 1.0f;
        if ENABLE_SHADOWS && light.cos_angles_shadow.z > 0.5 {
            let sm = shadow_matrices[0u]; // simplified; full per-light index wired up with shadow pass
            shadow = sample_shadow(world_pos, sm);
        }

        let radiance = light_col * light_int * atten * shadow;
        lo += cook_torrance(l, v, normal, albedo, metallic, roughness, f0) * radiance;
    }

    // Ambient / RC
    var ambient = globals.ambient_color * globals.ambient_intensity * albedo * ao;
    if ENABLE_RC {
        // Simple RC cascade0 lookup (screen-space probe lookup by UV).
        let rc_smp = textureSample(rc_cascade0, linear_sampler, uv);
        ambient = rc_smp.rgb * albedo * ao;
    }

    // IBL fallback using env_cube for specular.
    let r = reflect(-v, normal);
    let env_mip = roughness * 8.0;
    let env_col = textureSampleLevel(env_cube, linear_sampler, r, env_mip).rgb;
    let fre = fresnel_schlick(max(dot(normal, v), 0.0), f0);
    let spec_ambient = env_col * fre * (1.0 - roughness) * 0.5;

    var color = lo + ambient + spec_ambient + emissive;

    // ACES tonemap (applied here so sky + forward passes match by using pre-tone HDR).
    color = color / (color + vec3<f32>(0.187)) * 1.035;

    return FragOut(vec4<f32>(color, 1.0));
}
