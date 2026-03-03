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
@group(0) @binding(6) var cascade_history: texture_2d<f32>;

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

// Read one parent probe: average its 2×2 direction sub-bins for direction (dx,dy).
// Returns vec4(radiance, throughput).
fn read_parent_probe(ppx: u32, ppy: u32, ppz: u32,
                     dx: u32, dy: u32,
                     pdim: u32, ppdim: u32) -> vec4<f32> {
    var r = vec4<f32>(0.0);
    for (var ddx: u32 = 0u; ddx < 2u; ddx++) {
        for (var ddy: u32 = 0u; ddy < 2u; ddy++) {
            let ax = i32(ppx * pdim + dx * 2u + ddx);
            let ay = i32((ppy * ppdim + ppz) * pdim + dy * 2u + ddy);
            r += textureLoad(cascade_parent, vec2<i32>(ax, ay), 0);
        }
    }
    return r * 0.25;
}

// Evaluate a single light at a surface point with soft shadow (4 samples on a light disk).
// Gradual visibility prevents the hard snap as lights move past shadow boundaries.
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
            let cos_angle  = dot(-to_light, light.direction);
            let cos_outer  = cos(light.outer_angle);
            let cos_inner  = cos(light.inner_angle);
            let spot_atten = clamp((cos_angle - cos_outer) / (cos_inner - cos_outer + 0.001), 0.0, 1.0);
            atten *= spot_atten;
        }
    }

    let ndotl = max(0.0, dot(hit_normal, to_light));
    if ndotl < 0.001 || atten < 0.001 { return vec3<f32>(0.0); }

    // Shadow visibility — behaviour differs by light type:
    //   Directional: cast rays in exactly to_light direction (no disk spread needed,
    //                just a single hard-shadow ray since the sun is infinitely far away)
    //   Point/Spot:  soft shadow disk around the light position
    let origin = hit_pos + hit_normal * 0.004;
    var vis = 0.0;

    if light.light_type < 0.5 {
        // Directional — single ray toward the sun, t_max = effectively infinite
        var sq: ray_query;
        rayQueryInitialize(&sq, acc_struct,
            RayDesc(0x01u, 0xFFu, 0.005, 9999.0, origin, to_light));
        rayQueryProceed(&sq);
        if rayQueryGetCommittedIntersection(&sq).kind == RAY_QUERY_INTERSECTION_NONE {
            vis = 1.0;
        }
    } else {
        // Point / Spot — soft shadow: 5 rays on a disk around the light position
        let light_radius = 0.35;
        let perp  = normalize(cross(to_light, select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(to_light.y) < 0.9)));
        let perp2 = cross(to_light, perp);

        var offsets: array<vec2<f32>, 5>;
        offsets[0] = vec2<f32>( 0.0,  0.0);
        offsets[1] = vec2<f32>( 1.0,  0.0);
        offsets[2] = vec2<f32>(-1.0,  0.0);
        offsets[3] = vec2<f32>( 0.0,  1.0);
        offsets[4] = vec2<f32>( 0.0, -1.0);

        for (var si: u32 = 0u; si < 5u; si++) {
            let off         = offsets[si] * light_radius;
            let light_point = light.position + perp * off.x + perp2 * off.y;
            let ray_dir     = normalize(light_point - hit_pos);
            let ray_dist    = length(light_point - hit_pos);
            var sq: ray_query;
            rayQueryInitialize(&sq, acc_struct,
                RayDesc(0x01u, 0xFFu, 0.005, ray_dist - 0.005, origin, ray_dir));
            rayQueryProceed(&sq);
            if rayQueryGetCommittedIntersection(&sq).kind == RAY_QUERY_INTERSECTION_NONE {
                vis += 0.2; // 1/5
            }
        }
    }

    return light.color * light.intensity * atten * ndotl * vis;
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

    // Trilinear parent probe interpolation (eliminates snapping at probe cell boundaries).
    // Map child probe center (px+0.5) into parent probe space: fp = (px - 0.5) * 0.5
    // Then bilinearly blend between the 8 surrounding parent probes.
    if rc_stat.cascade_index < 3u && rc_stat.parent_dir_dim > 0u {
        let pdim  = rc_stat.parent_dir_dim;
        let ppdim = rc_stat.parent_probe_dim;

        // Fractional parent probe coordinate
        let fp = (vec3<f32>(f32(px), f32(py), f32(pz)) - 0.5) * 0.5;
        let fp_c = clamp(fp, vec3<f32>(0.0), vec3<f32>(f32(ppdim) - 1.001));
        let pi0 = vec3<u32>(u32(fp_c.x), u32(fp_c.y), u32(fp_c.z));
        let pi1 = vec3<u32>(
            min(pi0.x + 1u, ppdim - 1u),
            min(pi0.y + 1u, ppdim - 1u),
            min(pi0.z + 1u, ppdim - 1u),
        );
        let w = fp_c - floor(fp_c); // trilinear weights

        let s000 = read_parent_probe(pi0.x, pi0.y, pi0.z, dx, dy, pdim, ppdim);
        let s001 = read_parent_probe(pi0.x, pi0.y, pi1.z, dx, dy, pdim, ppdim);
        let s010 = read_parent_probe(pi0.x, pi1.y, pi0.z, dx, dy, pdim, ppdim);
        let s011 = read_parent_probe(pi0.x, pi1.y, pi1.z, dx, dy, pdim, ppdim);
        let s100 = read_parent_probe(pi1.x, pi0.y, pi0.z, dx, dy, pdim, ppdim);
        let s101 = read_parent_probe(pi1.x, pi0.y, pi1.z, dx, dy, pdim, ppdim);
        let s110 = read_parent_probe(pi1.x, pi1.y, pi0.z, dx, dy, pdim, ppdim);
        let s111 = read_parent_probe(pi1.x, pi1.y, pi1.z, dx, dy, pdim, ppdim);

        let s00 = mix(s000, s001, w.z);
        let s01 = mix(s010, s011, w.z);
        let s10 = mix(s100, s101, w.z);
        let s11 = mix(s110, s111, w.z);
        let s0  = mix(s00,  s01,  w.y);
        let s1  = mix(s10,  s11,  w.y);
        let parent = mix(s0, s1, w.x);

        radiance   = radiance + parent.rgb * throughput;
        throughput = throughput * parent.w;
    }

    // ── Temporal accumulation: EMA blend with previous frame ──────────────
    // alpha=0.15 → ~6-frame convergence. First frame (history=0) blends cleanly.
    let hist = textureLoad(cascade_history, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let alpha = 0.15;
    radiance   = mix(hist.rgb, radiance,   alpha);
    throughput = mix(hist.w,   throughput, alpha);

    textureStore(cascade_out, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(radiance, throughput));
}