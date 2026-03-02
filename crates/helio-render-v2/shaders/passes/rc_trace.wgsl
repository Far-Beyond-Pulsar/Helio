// Radiance Cascades - trace + merge compute shader.
//
// Y-UP octahedral encoding (Y is the pole axis, matching scene convention).
// Atlas layout (atlas_w = probe_dim * dir_dim = 32 always):
//   atlas_x = probe_x * dir_dim  +  dir_x
//   atlas_y = (probe_y * probe_dim + probe_z) * dir_dim  +  dir_y
//
// Probe stores rgba16float: rgb = radiance, w = throughput
//   throughput = 0.0 -> ray hit geometry (opaque)
//   throughput = 1.0 -> ray missed (sky/infinite)
// Merge (coarse->fine): merged_rad = local_rad + parent_rad * local_throughput
//                       merged_thr = local_throughput * parent_throughput

enable wgpu_ray_query;

// GpuLight (matches Rust GpuLight in lighting.rs, 48 bytes)
struct GpuLight {
    position:    vec3<f32>,
    light_type:  f32,   // 0=directional, 1=point, 2=spot
    direction:   vec3<f32>,
    range:       f32,
    color:       vec3<f32>,
    intensity:   f32,
    inner_angle: f32,
    outer_angle: f32,
    _pad:        vec2<f32>,
}

struct RCDynamic {
    world_min:   vec4<f32>,
    world_max:   vec4<f32>,
    frame:       u32,
    light_count: u32,
    _pad0:       u32,
    _pad1:       u32,
}

struct CascadeStatic {
    cascade_index:    u32,
    probe_dim:        u32,
    dir_dim:          u32,
    t_max_bits:       u32,
    parent_probe_dim: u32,
    parent_dir_dim:   u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var cascade_out:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var cascade_parent: texture_2d<f32>;
@group(0) @binding(2) var<uniform>  rc_dyn:  RCDynamic;
@group(0) @binding(3) var<uniform>  rc_stat: CascadeStatic;
@group(0) @binding(4) var acc_struct: acceleration_structure;
@group(0) @binding(5) var<storage, read> lights: array<GpuLight, 16>;

// Y-up octahedral decode (Y is the pole — uv center = +Y)
fn oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f  = uv * 2.0 - 1.0;
    let af = abs(f);
    let l  = af.x + af.y;
    var n: vec3<f32>;
    if l > 1.0 {
        let sx = select(-1.0, 1.0, f.x >= 0.0);
        let sz = select(-1.0, 1.0, f.y >= 0.0);
        n = vec3<f32>((1.0 - af.y) * sx, 1.0 - l, (1.0 - af.x) * sz);
    } else {
        n = vec3<f32>(f.x, 1.0 - l, f.y);
    }
    return normalize(n);
}

// Evaluate a single light at a surface point (returns lit radiance contribution).
// Casts a shadow ray to determine visibility.
fn eval_light(li: u32, hit_pos: vec3<f32>, hit_normal: vec3<f32>) -> vec3<f32> {
    let light = lights[li];
    var to_light: vec3<f32>;
    var dist:     f32;
    var atten:    f32;

    if light.light_type < 0.5 {
        // Directional
        to_light = -light.direction;
        dist     = 1000.0;
        atten    = 1.0;
    } else {
        // Point / Spot
        let diff = light.position - hit_pos;
        dist     = length(diff);
        if dist >= light.range { return vec3<f32>(0.0); }
        to_light = diff / dist;
        atten    = clamp(1.0 - (dist / light.range), 0.0, 1.0);
        atten    = atten * atten;
        if light.light_type > 1.5 {
            // Spot cone
            let cos_angle  = dot(-to_light, light.direction);
            let cos_outer  = cos(light.outer_angle);
            let cos_inner  = cos(light.inner_angle);
            let spot_atten = clamp((cos_angle - cos_outer) / (cos_inner - cos_outer + 0.001), 0.0, 1.0);
            atten *= spot_atten;
        }
    }

    let ndotl = max(0.0, dot(hit_normal, to_light));
    if ndotl < 0.001 || atten < 0.001 { return vec3<f32>(0.0); }

    // Shadow ray
    var sq: ray_query;
    rayQueryInitialize(&sq, acc_struct,
        RayDesc(0x01u, 0xFFu, 0.003, dist - 0.003,
            hit_pos + hit_normal * 0.004, to_light));
    rayQueryProceed(&sq);
    if rayQueryGetCommittedIntersection(&sq).kind != RAY_QUERY_INTERSECTION_NONE {
        return vec3<f32>(0.0); // shadowed
    }

    return light.color * light.intensity * atten * ndotl;
}

@compute @workgroup_size(8, 8)
fn cs_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_dim = rc_stat.probe_dim;
    let dir_dim   = rc_stat.dir_dim;
    let atlas_w   = probe_dim * dir_dim;
    let atlas_h   = probe_dim * probe_dim * dir_dim;

    if gid.x >= atlas_w || gid.y >= atlas_h { return; }

    let dx  = gid.x % dir_dim;
    let px  = gid.x / dir_dim;
    let dy  = gid.y % dir_dim;
    let pyz = gid.y / dir_dim;
    let pz  = pyz % probe_dim;
    let py  = pyz / probe_dim;

    let world_size = rc_dyn.world_max.xyz - rc_dyn.world_min.xyz;
    let cell_size  = world_size / f32(probe_dim);
    let probe_pos  = rc_dyn.world_min.xyz + (vec3<f32>(f32(px), f32(py), f32(pz)) + 0.5) * cell_size;

    let dir_uv = (vec2<f32>(f32(dx), f32(dy)) + 0.5) / f32(dir_dim);
    let dir    = oct_decode(dir_uv);
    let t_max  = bitcast<f32>(rc_stat.t_max_bits);

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(0x01u, 0xFFu, 0.001, t_max, probe_pos, dir));
    rayQueryProceed(&rq);
    let isect = rayQueryGetCommittedIntersection(&rq);

    var radiance:   vec3<f32>;
    var throughput: f32;

    if isect.kind != RAY_QUERY_INTERSECTION_NONE {
        let hit_pos    = probe_pos + dir * isect.t;
        let hit_normal = select(dir, -dir, isect.front_face);

        // Accumulate all scene lights at hit point
        var light_contrib = vec3<f32>(0.0);
        for (var li: u32 = 0u; li < rc_dyn.light_count; li++) {
            light_contrib += eval_light(li, hit_pos, hit_normal);
        }

        radiance   = light_contrib;
        throughput = 0.0;
    } else {
        // All cascades: no ambient fill — sky-miss gives zero radiance,
        // throughput=1 so parent radiance propagates through on merge.
        radiance   = vec3<f32>(0.0);
        throughput = 1.0;
    }

    // Merge with parent (dispatched coarse-first so parent is already written)
    // merged_rad = local_rad + parent_rad * local_throughput
    // merged_thr = local_throughput * parent_throughput
    if rc_stat.cascade_index < 3u && rc_stat.parent_dir_dim > 0u {
        let pdim  = rc_stat.parent_dir_dim;
        let ppdim = rc_stat.parent_probe_dim;
        let ppx   = px / 2u;
        let ppy   = py / 2u;
        let ppz   = pz / 2u;

        var p_rad = vec3<f32>(0.0);
        var p_thr = 0.0;
        for (var ddx: u32 = 0u; ddx < 2u; ddx++) {
            for (var ddy: u32 = 0u; ddy < 2u; ddy++) {
                let ax = i32(ppx * pdim + dx * 2u + ddx);
                let ay = i32((ppy * ppdim + ppz) * pdim + dy * 2u + ddy);
                let s = textureLoad(cascade_parent, vec2<i32>(ax, ay), 0);
                p_rad += s.rgb;
                p_thr += s.w;
            }
        }
        p_rad *= 0.25;
        p_thr *= 0.25;

        radiance   = radiance + p_rad * throughput;
        throughput = throughput * p_thr;
    }

    textureStore(cascade_out, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(radiance, throughput));
}