// Hybrid SSR uses the same camera, depth and signed normal contract as raster SSR.
//!use helio_prelude
enable wgpu_ray_query;

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;

@group(1) @binding(0) var gbuf_normal:          texture_2d<f32>;
@group(1) @binding(1) var gbuf_orm:             texture_2d<f32>;
@group(1) @binding(2) var gbuf_depth:           texture_depth_2d;
@group(1) @binding(3) var scene_color:          texture_2d<f32>;
@group(1) @binding(4) var hiz_min:              texture_2d<f32>;
@group(1) @binding(5) var linear_sampler:       sampler;
@group(1) @binding(6) var ssr_output:           texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var acc_struct:           acceleration_structure;
@group(2) @binding(1) var rc_cascades:          texture_2d<f32>;

const MAX_ITER:      u32 = 64u;
const START_LEVEL:   i32 = 2;
const MAX_LEVEL:     i32 = 8;
const MAX_RAY_DIST:  f32 = 100.0;
const THICKNESS:     f32 = 0.02;
const NORMAL_OFFSET: f32 = 0.002;
const FADE_START:    f32 = 0.6;

fn linearize_depth(d_01: f32) -> f32 {
    return helio_view_depth(d_01, cameras[0].position_near.w, cameras[0].forward_far.w);
}

fn level_size(level: i32) -> vec2<f32> {
    return vec2<f32>(max(vec2<u32>(1u), textureDimensions(hiz_min) >> vec2<u32>(u32(level))));
}

fn cell_of(uv: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return floor(uv * size);
}

fn min_depth(cell: vec2<f32>, level: i32) -> f32 {
    return textureLoad(hiz_min, vec2<i32>(cell), level).r;
}

fn at_depth(o: vec3<f32>, d: vec3<f32>, z: f32) -> vec3<f32> {
    return o + d * ((z - o.z) / d.z);
}

fn exit_cell(
    o: vec3<f32>,
    d: vec3<f32>,
    cell: vec2<f32>,
    size: vec2<f32>,
    cross_step: vec2<f32>,
    cross_offset: vec2<f32>,
) -> vec3<f32> {
    let boundary = (cell + cross_step) / size + cross_offset;
    let delta = (boundary - o.xy) / d.xy;
    return o + d * min(delta.x, delta.y);
}

// Transmission query contract: nearest opaque endpoint, then tint only the
// segment before it. Candidate traversal order is unspecified, so accumulating
// tint while searching for the endpoint would include panes behind that hit.
struct RayTransmissionData { header: vec4<u32>, rows: array<vec4<f32>> };
@group(2) @binding(2) var<storage, read> ray_transmission: RayTransmissionData;
struct ReflectionHit { position: vec4<f32>, throughput: vec3<f32> };
fn reflection_tint(instance: u32) -> vec3<f32> {
    if instance>=min(ray_transmission.header.y,arrayLength(&ray_transmission.rows)) {
        return vec3<f32>(0.0);
    }
    return ray_transmission.rows[instance].rgb;
}
fn ray_query_hit_position(world_pos: vec3<f32>, normal: vec3<f32>, R: vec3<f32>) -> ReflectionHit {
    let origin = world_pos + normal * 0.002;
    let transmitting=ray_transmission.header.y!=0u;
    var rq: ray_query;
    // Force candidates when metadata is present, including generic TLAS inputs
    // whose BLAS opacity may not yet match their material classification.
    rayQueryInitialize(&rq, acc_struct,
        RayDesc(select(0x01u,0x02u,transmitting), 0xFFu, 0.001, MAX_RAY_DIST, origin, R));
    while rayQueryProceed(&rq) {
        let candidate=rayQueryGetCandidateIntersection(&rq);
        if all(reflection_tint(candidate.instance_index)==vec3<f32>(0.0)) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    let hit=rayQueryGetCommittedIntersection(&rq);
    if hit.kind==RAY_QUERY_INTERSECTION_NONE {
        return ReflectionHit(vec4<f32>(0.0),vec3<f32>(1.0));
    }
    var throughput=vec3<f32>(1.0);
    if transmitting {
        var tint_query: ray_query;
        rayQueryInitialize(&tint_query,acc_struct,
            RayDesc(0x02u,0xFFu,0.001,hit.t,origin,R));
        while rayQueryProceed(&tint_query) {
            let sheet=rayQueryGetCandidateIntersection(&tint_query);
            // Strict endpoint exclusion also rejects coplanar endpoint hits.
            if sheet.t<hit.t {
                throughput*=reflection_tint(sheet.instance_index);
            }
        }
    }
    return ReflectionHit(vec4<f32>(origin+R*hit.t,1.0),throughput);
}

// Screen color is valid only if it represents this RT hit, rather than an
// unrelated foreground surface at the same projected coordinate.
fn projected_hit_color(hit: vec4<f32>, direction: vec3<f32>) -> vec4<f32> {
    if hit.w==0.0 { return vec4<f32>(0.0); }
    let clip=cameras[0].view_proj*vec4<f32>(hit.xyz,1.0);
    if clip.w<=0.0 { return vec4<f32>(0.0); }
    let uv=helio_ndc_to_uv(clip.xy/clip.w);
    if any(uv<vec2<f32>(0.0)) || any(uv>=vec2<f32>(1.0)) { return vec4<f32>(0.0); }
    let px=vec2<i32>(uv*vec2<f32>(textureDimensions(gbuf_depth)));
    let depth=textureLoad(gbuf_depth,px,0);
    if depth>=1.0 { return vec4<f32>(0.0); }
    let hit_z=-(cameras[0].view*vec4<f32>(hit.xyz,1.0)).z;
    let visible_z=linearize_depth(depth);
    if abs(hit_z-visible_z)>max(0.02,0.005*hit_z) { return vec4<f32>(0.0); }
    let normal=helio_gbuffer_normal(textureLoad(gbuf_normal,px,0).xyz);
    if dot(normal,-direction)<=0.0 { return vec4<f32>(0.0); }
    return vec4<f32>(textureLoad(scene_color,px,0).rgb,1.0);
}

fn sample_rc_reflection(world_pos: vec3<f32>, R: vec3<f32>, roughness: f32) -> vec3<f32> {
    if roughness < 0.6 { return vec3<f32>(0.0); }
    let rc_dims = textureDimensions(rc_cascades);
    if rc_dims.x < 2u || rc_dims.y < 2u { return vec3<f32>(0.0); }

    let f = R / (abs(R.x) + abs(R.y) + abs(R.z));
    let oct_uv = select(
        vec2<f32>(f.z, f.x) * 0.5 + 0.5,
        vec2<f32>(1.0 - abs(f.z), 1.0 - abs(f.x)) * 0.5,
        f.y >= 0.0,
    );
    let sx = f32(rc_dims.x);
    let sy = f32(rc_dims.y);
    let px = clamp(i32(oct_uv.x * sx), 0, i32(rc_dims.x) - 3);
    let py = clamp(i32(oct_uv.y * sy), 0, i32(rc_dims.y) - 3);

    var irradiance = vec3<f32>(0.0);
    for (var dy = 0u; dy < 2u; dy++) {
        for (var dx = 0u; dx < 2u; dx++) {
            let s = textureLoad(rc_cascades, vec2<i32>(px + i32(dx), py + i32(dy)), 0);
            irradiance += s.rgb;
        }
    }
    return irradiance * 0.25;
}

@compute @workgroup_size(8, 8)
fn cs_rt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ssr_output);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let px = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let source_dims = textureDimensions(gbuf_depth);
    let source_px = clamp(
        vec2<i32>(uv * vec2<f32>(source_dims)),
        vec2<i32>(0),
        vec2<i32>(source_dims) - vec2<i32>(1),
    );
    let depth_01 = textureLoad(gbuf_depth, source_px, 0);

    if depth_01 >= 1.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    let N = helio_gbuffer_normal(textureLoad(gbuf_normal, source_px, 0).xyz);
    let roughness = textureLoad(gbuf_orm, source_px, 0).g;
    let roughness_fade = 1.0 - smoothstep(0.4, 0.7, roughness);
    if roughness_fade <= 0.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    let world_pos = helio_world_from_depth(cameras[0].view_proj_inv, uv, depth_01);
    let V = normalize(cameras[0].position_near.xyz - world_pos);
    let R2 = reflect(-V, N);
    if dot(R2, N) <= 0.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }

    // ── Build ray ───────────────────────────────────────────────────────────
    let near = cameras[0].position_near.w;
    var start_view = (cameras[0].view * vec4<f32>(world_pos, 1.0)).xyz;
    let dir_view = normalize((cameras[0].view * vec4<f32>(R2, 0.0)).xyz);
    let n_view = (cameras[0].view * vec4<f32>(N, 0.0)).xyz;
    start_view += n_view * (-start_view.z * NORMAL_OFFSET);

    var ray_len = MAX_RAY_DIST;
    if start_view.z + dir_view.z * ray_len > -near {
        ray_len = (-near - start_view.z) / dir_view.z;
    }
    if ray_len <= 0.0 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }
    let end_view = start_view + dir_view * ray_len;

    let clip0 = cameras[0].proj * vec4<f32>(start_view, 1.0);
    let clip1 = cameras[0].proj * vec4<f32>(end_view, 1.0);
    let p0 = vec3<f32>(helio_ndc_to_uv(clip0.xy / clip0.w), clip0.z / clip0.w);
    let p1 = vec3<f32>(helio_ndc_to_uv(clip1.xy / clip1.w), clip1.z / clip1.w);
    var d = p1 - p0;

    if abs(d.x) < 1e-7 && abs(d.y) < 1e-7 {
        textureStore(ssr_output, px, vec4<f32>(0.0));
        return;
    }
    d.x = select(d.x, 1e-7, abs(d.x) < 1e-7);
    d.y = select(d.y, 1e-7, abs(d.y) < 1e-7);

    let cross_step = vec2<f32>(select(0.0, 1.0, d.x >= 0.0), select(0.0, 1.0, d.y >= 0.0));
    let cross_offset = (cross_step * 2.0 - 1.0) * 1e-6;

    // ── Hi-Z traversal ──────────────────────────────────────────────────────
    var level = START_LEVEL;
    let max_level = min(MAX_LEVEL, i32(textureNumLevels(hiz_min)) - 1);
    var tr = p0;
    {
        let size = level_size(level);
        tr = exit_cell(p0, d, cell_of(p0.xy, size), size, cross_step, cross_offset);
    }

    var iter = 0u;
    var hiz_hit = false;

    while level >= 0 && iter < MAX_ITER {
        iter += 1u;
        if any(tr.xy < vec2<f32>(0.0)) || any(tr.xy > vec2<f32>(1.0)) { break; }
        if tr.z > 1.0 { break; }

        let size = level_size(level);
        let old_cell = cell_of(tr.xy, size);
        let tile_min = min_depth(old_cell, level);

        var next = tr;
        let in_front = tr.z < tile_min;
        if in_front && d.z > 0.0 {
            next = at_depth(p0, d, tile_min);
        }

        let new_cell = cell_of(next.xy, size);
        let skip_tile = in_front && d.z <= 0.0;

        if skip_tile || any(new_cell != old_cell) {
            next = exit_cell(tr, d, old_cell, size, cross_step, cross_offset);
            level = min(max_level, level + 1);
        } else {
            level -= 1;
            if level < 0 { hiz_hit = true; }
        }
        tr = next;
    }

    // Prefer reliable screen hits in opaque scenes. Hardware fallback reuses
    // screen radiance only after checking the projected hit depth.
    var final_color=vec3<f32>(0.0);
    var final_confidence=0.0;
    if hiz_hit {
        let hit_uv=tr.xy;
        // The trace target is half resolution; depth and normals are full
        // resolution. Validate the hit against the same screen coordinate.
        let hit_px=clamp(vec2<i32>(hit_uv*vec2<f32>(source_dims)),
            vec2<i32>(0),vec2<i32>(source_dims)-vec2<i32>(1));
        let ray_z=linearize_depth(tr.z);
        let scene_z=linearize_depth(textureLoad(gbuf_depth,hit_px,0));
        if abs(ray_z-scene_z)<=max(0.02,scene_z*THICKNESS) {
            let n_hit=helio_gbuffer_normal(textureLoad(gbuf_normal,hit_px,0).xyz);
            let border=min(min(hit_uv.x,1.0-hit_uv.x),min(hit_uv.y,1.0-hit_uv.y));
            let travelled=length(hit_uv-p0.xy)/max(length(d.xy),1e-6);
            final_confidence=clamp(smoothstep(-0.15,0.15,-dot(R2,n_hit))
                *smoothstep(0.0,0.1,border)*(1.0-smoothstep(0.26,0.5,dot(R2,V)))
                *(1.0-smoothstep(FADE_START,1.0,travelled))*roughness_fade,0.0,1.0);
            final_color=textureSampleLevel(scene_color,linear_sampler,hit_uv,0.0).rgb;
        }
    }
    // Screen-space traversal cannot see panes absent from the opaque G-buffer.
    // Validate all reflection segments when transmission metadata is present.
    if final_confidence<0.5 || ray_transmission.header.y!=0u {
        let hit=ray_query_hit_position(world_pos,N,R2);
        let radiance=projected_hit_color(hit.position,R2);
        if radiance.a>0.0 {
            final_color=radiance.rgb*hit.throughput;
            final_confidence=roughness_fade;
        } else if ray_transmission.header.y!=0u {
            // Do not keep an unfiltered Hi-Z hit when hardware found a different
            // endpoint. Offscreen hit shading remains a separate missing path.
            final_color=vec3<f32>(0.0);
            final_confidence=0.0;
        } else if final_confidence==0.0 && roughness>0.6 {
            final_color=sample_rc_reflection(world_pos,R2,roughness);
            final_confidence=roughness_fade*0.2;
        }
    }
    textureStore(ssr_output,px,vec4<f32>(final_color,final_confidence));
}
