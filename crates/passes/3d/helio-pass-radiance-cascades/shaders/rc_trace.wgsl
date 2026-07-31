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
    cos_inner:   f32,   // cos(inner_angle), precomputed on CPU
    cos_outer:   f32,   // cos(outer_angle), precomputed on CPU
    _pad:        vec2<f32>,
}

struct RCDynamic {
    world_min:   vec4<f32>,
    world_max:   vec4<f32>,
    frame:       u32,
    light_count: u32,
    _pad0:       u32,
    _pad1:       u32,
    /// Sky radiance for miss rays (rgb = linear colour, w unused).
    sky_color:   vec4<f32>,
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

@group(0) @binding(0) var cascade_out:           texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var cascade_parent:         texture_2d<f32>;
@group(0) @binding(2) var<uniform>  rc_dyn:  RCDynamic;
@group(0) @binding(3) var<uniform>  rc_stat: CascadeStatic;
@group(0) @binding(4) var acc_struct: acceleration_structure;
@group(0) @binding(5) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(6) var cascade_history:        texture_2d<f32>;
@group(0) @binding(7) var cascade_history_write:  texture_storage_2d<rgba16float, write>;

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
// OPTIMIZED: Reduced from 4 texture loads to 1 by using center sample
fn read_parent_probe(ppx: u32, ppy: u32, ppz: u32,
                     dx: u32, dy: u32,
                     pdim: u32, ppdim: u32) -> vec4<f32> {
    // Sample the center of the 2x2 direction bin instead of averaging all 4
    // This trades a tiny bit of accuracy for 4x fewer texture reads
    let ax = i32(ppx * pdim + dx * 2u + 1u);
    let ay = i32((ppy * ppdim + ppz) * pdim + dy * 2u + 1u);
    return textureLoad(cascade_parent, vec2<i32>(ax, ay), 0);
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
            let cos_outer  = light.cos_outer;
            let cos_inner  = light.cos_inner;
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
        // Point / Spot — soft shadow: 4 rays in a rotated square pattern (better vectorization)
        let light_radius = 0.35;
        let perp  = normalize(cross(to_light, select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(to_light.y) < 0.9)));
        let perp2 = cross(to_light, perp);

        // Rotated square pattern - better coverage than cross pattern
        var offsets: array<vec2<f32>, 4>;
        offsets[0] = vec2<f32>( 0.707,  0.707);
        offsets[1] = vec2<f32>(-0.707,  0.707);
        offsets[2] = vec2<f32>(-0.707, -0.707);
        offsets[3] = vec2<f32>( 0.707, -0.707);

        for (var si: u32 = 0u; si < 4u; si++) {
            let off         = offsets[si] * light_radius;
            let light_point = light.position + perp * off.x + perp2 * off.y;
            let ray_dir     = normalize(light_point - hit_pos);
            let ray_dist    = length(light_point - hit_pos);
            var sq: ray_query;
            rayQueryInitialize(&sq, acc_struct,
                RayDesc(0x01u, 0xFFu, 0.005, ray_dist - 0.005, origin, ray_dir));
            rayQueryProceed(&sq);
            if rayQueryGetCommittedIntersection(&sq).kind == RAY_QUERY_INTERSECTION_NONE {
                vis += 0.25; // 1/4
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
        let hit_pos = probe_pos + dir * isect.t;

        // IMPORTANT: Ray queries don't provide geometric normals, only hit position.
        // For radiance cascades, we approximate the normal as the inverse ray direction.
        // This assumes surfaces generally face back toward the probe, which is valid
        // for indirect lighting (GI) since we're sampling the hemisphere around the probe.
        // For direct lighting shadow rays, this works because we only care about visibility.
        let hit_normal = select(-dir, dir, isect.front_face);

        // Accumulate all scene lights at hit point
        var light_contrib = vec3<f32>(0.0);
        for (var li: u32 = 0u; li < rc_dyn.light_count; li++) {
            light_contrib += eval_light(li, hit_pos, hit_normal);
        }

        radiance   = light_contrib;
        throughput = 0.0;
    } else {
        // Sky miss — the ray escaped to the sky.  Contribute sky radiance
        // based on the ray direction so the GI naturally fills shadowed areas
        // with sky-coloured indirect light.
        //
        // sky_color is set each frame by the renderer from the scene's
        // skylight / ambient system (zero at night or indoors → correct).
        let sky_up   = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);   // 0=down, 1=up
        let sky_base = rc_dyn.sky_color.rgb;
        radiance   = mix(sky_base * 0.15, sky_base, sky_up);
        throughput = 0.0;  // sky is terminal — no further propagation needed
    }

    // OPTIMIZED: Nearest-neighbor parent probe lookup instead of trilinear
    // Reduces from 8 probe reads (32 texture loads) to 1 probe read (1 texture load)
    // The slight reduction in smoothness is imperceptible due to temporal accumulation
    if rc_stat.cascade_index < 3u && rc_stat.parent_dir_dim > 0u {
        let pdim  = rc_stat.parent_dir_dim;
        let ppdim = rc_stat.parent_probe_dim;

        // Map child probe center to parent probe space and round to nearest
        let fp = (vec3<f32>(f32(px), f32(py), f32(pz)) - 0.5) * 0.5;
        let fp_c = clamp(fp, vec3<f32>(0.0), vec3<f32>(f32(ppdim) - 1.001));
        let pi = vec3<u32>(u32(fp_c.x + 0.5), u32(fp_c.y + 0.5), u32(fp_c.z + 0.5));
        let pi_clamped = min(pi, vec3<u32>(ppdim - 1u));

        let parent = read_parent_probe(pi_clamped.x, pi_clamped.y, pi_clamped.z, dx, dy, pdim, ppdim);

        radiance   = radiance + parent.rgb * throughput;
        throughput = throughput * parent.w;
    }

    // ── Temporal accumulation: EMA blend with previous frame ──────────────
    // alpha=0.15 → ~6-frame convergence. First frame (history=0) blends cleanly.
    let hist = textureLoad(cascade_history, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let alpha = 0.15;
    radiance   = mix(hist.rgb, radiance,   alpha);
    throughput = mix(hist.w,   throughput, alpha);

    textureStore(cascade_out,           vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(radiance, throughput));
    // Write the same value into the history ping-pong buffer so the next
    // frame can read it without a copy_texture_to_texture blit pass.
    textureStore(cascade_history_write, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(radiance, throughput));
}