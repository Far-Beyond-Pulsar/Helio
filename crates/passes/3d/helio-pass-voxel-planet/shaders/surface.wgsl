// Primary visibility, voxel shading and sunlight.

@group(0) @binding(7) var<storage, ACCESS> hits: array<Hit>;
@group(0) @binding(8) var<storage, ACCESS> surfaces: array<Surface>;

fn interleaved_gradient(p: vec2<f32>, frame_index: f32) -> f32 {
    let q = p + 5.588238 * (frame_index % 64.0);
    return fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8)
fn primary(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(frame.screen.xy)) { return; }
    let d = pixel_ray(vec2<f32>(id.xy) + 0.5);
    let r = make_ray(camera.position_near.xyz, d);
    let sky = eye_sky(d, 0.0);
    let end = min(frame.lod.w, sky_escape(sky));
    hits[pixel_index(id.xy)] = trace(r, 0.0, end, 0.0, 1.0, frame.lod.y);
}

fn srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(c / 255.0, vec3<f32>(2.2));
}

fn palette(m: u32) -> vec3<f32> {
    switch m {
        case 1u: { return srgb(vec3<f32>(104.0, 150.0, 54.0)); }
        case 2u: { return srgb(vec3<f32>(132.0, 96.0, 64.0)); }
        case 3u: { return srgb(vec3<f32>(138.0, 135.0, 128.0)); }
        case 4u: { return srgb(vec3<f32>(216.0, 198.0, 142.0)); }
        case 5u: { return srgb(vec3<f32>(236.0, 241.0, 246.0)); }
        case 6u: { return srgb(vec3<f32>(28.0, 72.0, 92.0)); }
        case 7u: { return srgb(vec3<f32>(112.0, 107.0, 101.0)); }
        case 8u: { return srgb(vec3<f32>(200.0, 152.0, 104.0)); }
        case 9u: { return srgb(vec3<f32>(90.0, 88.0, 86.0)); }
        case 10u: { return srgb(vec3<f32>(112.0, 80.0, 50.0)); }
        case 11u: { return srgb(vec3<f32>(62.0, 112.0, 40.0)); }
        case 12u: { return srgb(vec3<f32>(166.0, 118.0, 88.0)); }
        case 13u: { return srgb(vec3<f32>(152.0, 72.0, 56.0)); }
        case 14u: { return srgb(vec3<f32>(164.0, 122.0, 76.0)); }
        case 15u: { return srgb(vec3<f32>(122.0, 122.0, 120.0)); }
        default: { return srgb(vec3<f32>(200.0, 0.0, 200.0)); }
    }
}

fn plane_normal(face: u32, axis: u32, plane: i32) -> vec3<f32> {
    let f = frame.faces[face];
    var m = f.m_a.xyz;
    var q = f.q_a.xyz;
    var idx = f.index.x;
    var frac = f.q_a.w;
    if axis == 1u {
        m = f.m_b.xyz;
        q = f.q_b.xyz;
        idx = f.index.y;
        frac = f.q_b.w;
    }
    let delta = (f32(plane - idx) - frac) * frame.layer.z;
    let sc = sincos_small(delta);
    return normalize(m * sc.y - q * sc.x);
}

fn hit_up(t: f32, d: vec3<f32>) -> vec3<f32> {
    return normalize(frame.eye.xyz + (camera.position_near.xyz + t * d) / frame.eye.w);
}

fn hit_normal(h: Hit, d: vec3<f32>) -> vec3<f32> {
    let code = (h.info >> 10u) & 7u;
    let face = (h.info >> 2u) & 7u;
    let level = (h.info >> 5u) & 31u;
    if code == 0u { return plane_normal(face, 0u, (h.i + 1) << level); }
    if code == 1u { return -plane_normal(face, 0u, h.i << level); }
    if code == 2u { return plane_normal(face, 1u, (h.j + 1) << level); }
    if code == 3u { return -plane_normal(face, 1u, h.j << level); }
    if code == 4u { return hit_up(h.t, d); }
    if code == 5u { return -hit_up(h.t, d); }
    return -d;
}

// Occupancy of a level cell; `home` is the hit's own column, which serves
// most neighbour queries without a hash lookup.
fn occupied(face: u32, level: u32, i: i32, j: i32, k: i32, home: Column, hi: i32, hj: i32) -> bool {
    let n = frame.layer_i.y >> level;
    if i < 0 || j < 0 || i >= n || j >= n { return false; }
    var c = home;
    if (i >> 3u) != (hi >> 3u) || (j >> 3u) != (hj >> 3u) {
        let record = find_column(column_key0(face, level, i >> 3u), bitcast<u32>(j >> 3u));
        if record == NONE { return false; }
        c = records[record];
        if !column_valid(c) { return false; }
    }
    let b = (k >> 3u) - c.k_lo;
    if b < 0 { return true; }
    if b >= i32(band_count(c)) { return false; }
    let s = brick_state(c, u32(b));
    if s.x == 2u { return brick_bit(s.y, u32(i & 7), u32(j & 7), u32(k & 7)); }
    return s.x == 1u;
}

fn corner_ao(side1: bool, side2: bool, corner: bool) -> f32 {
    if side1 && side2 { return 0.0; }
    return 3.0 - f32(u32(side1) + u32(side2) + u32(corner));
}

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(frame.screen.xy)) { return; }
    let index = pixel_index(id.xy);
    let h = hits[index];
    var out: Surface;
    let status = h.info & 3u;
    if status != ST_HIT {
        out.t = -1.0;
        out.flags = status;
        surfaces[index] = out;
        return;
    }
    let d = pixel_ray(vec2<f32>(id.xy) + 0.5);
    let face = (h.info >> 2u) & 7u;
    let level = (h.info >> 5u) & 31u;
    let code = (h.info >> 10u) & 7u;
    let c = records[h.record];
    let x = u32(h.i & 7);
    let y = u32(h.j & 7);
    let top = column_top(c, x, y);
    var kind = terrain_kind(top, h.k);
    var material = 0u;
    if c.edits != 0u {
        let km = apply_edits(c.edits, level, vec3<i32>(center_half(h.i, level), center_half(h.j, level), center_half(h.k, level)), kind);
        kind = km.x;
        material = km.y;
    }
    if material == 0u {
        var slope = 0;
        var lowest = top;
        if x > 0u { let t = column_top(c, x - 1u, y); slope = max(slope, abs(t - top)); lowest = min(lowest, t); }
        if x < 7u { let t = column_top(c, x + 1u, y); slope = max(slope, abs(t - top)); lowest = min(lowest, t); }
        if y > 0u { let t = column_top(c, x, y - 1u); slope = max(slope, abs(t - top)); lowest = min(lowest, t); }
        if y < 7u { let t = column_top(c, x, y + 1u); slope = max(slope, abs(t - top)); lowest = min(lowest, t); }
        // Canonical materials use the column top cell, which is resident.
        // Depth counts from the lowest neighbouring top: an exposed riser
        // above it is surface, not subsoil (coarse levels step in large
        // cells where the fine terrain is a continuous slope).
        let p = domain_point(face, h.i, h.j, level);
        let depth = max(min(top, lowest) - 1 - h.k, 0) << level;
        material = ground_material(p, (top << level) * field.header.z, depth, slope, h.k << level);
    }
    let normal = hit_normal(h, d);
    // Neighbourhood occlusion around the air cell in front of the face.
    var ao = 1.0;
    // Side faces of grass voxels show soil below a ragged grass lip. The lip
    // covers more of the face with distance, where one coarse cell stands for
    // a grassy slope of many fine steps.
    var soil_side = false;
    if code < 6u {
        let axis = code >> 1u;
        let back = select(1, -1, (code & 1u) == 1u);
        var f = vec3<i32>(h.i, h.j, h.k);
        f[axis] += back;
        var u_axis = select(0u, 1u, axis == 0u);
        var v_axis = select(2u, 1u, axis == 2u);
        if axis == 2u { u_axis = 0u; v_axis = 1u; }
        var du = vec3<i32>(0);
        var dv = vec3<i32>(0);
        du[u_axis] = 1;
        dv[v_axis] = 1;
        let s0 = occupied(face, level, f.x - du.x, f.y - du.y, f.z - du.z, c, h.i, h.j);
        let s1 = occupied(face, level, f.x + du.x, f.y + du.y, f.z + du.z, c, h.i, h.j);
        let s2 = occupied(face, level, f.x - dv.x, f.y - dv.y, f.z - dv.z, c, h.i, h.j);
        let s3 = occupied(face, level, f.x + dv.x, f.y + dv.y, f.z + dv.z, c, h.i, h.j);
        let c00 = corner_ao(s0, s2, occupied(face, level, f.x - du.x - dv.x, f.y - du.y - dv.y, f.z - du.z - dv.z, c, h.i, h.j));
        let c10 = corner_ao(s1, s2, occupied(face, level, f.x + du.x - dv.x, f.y + du.y - dv.y, f.z + du.z - dv.z, c, h.i, h.j));
        let c01 = corner_ao(s0, s3, occupied(face, level, f.x - du.x + dv.x, f.y - du.y + dv.y, f.z - du.z + dv.z, c, h.i, h.j));
        let c11 = corner_ao(s1, s3, occupied(face, level, f.x + du.x + dv.x, f.y + du.y + dv.y, f.z + du.z + dv.z, c, h.i, h.j));
        // Position of the hit inside the face.
        let fr = face_ray(face, make_ray(camera.position_near.xyz, d));
        let scale = 1.0 / f32(1 << level);
        var cell = vec3<f32>(
            (f32(fr.idx.x) + face_coord(fr, 0u, h.t)) * scale,
            (f32(fr.idx.y) + face_coord(fr, 1u, h.t)) * scale,
            (f32(frame.layer_i.x) + layer_coord(make_ray(camera.position_near.xyz, d), h.t)) * scale,
        );
        let uv = clamp(vec2<f32>(cell[u_axis] - f32(select(select(h.i, h.j, u_axis == 1u), h.k, u_axis == 2u)),
                                 cell[v_axis] - f32(select(select(h.i, h.j, v_axis == 1u), h.k, v_axis == 2u))), vec2<f32>(0.0), vec2<f32>(1.0));
        if axis < 2u && material == M_GRASS {
            let tooth = f32(hash3(h.i, h.j, h.k * 4 + i32(floor(uv.x * 4.0)), 0x5bd1e995u) & 7u) / 7.0;
            // Continuous in distance (not level), so level changes show no band.
            let lip = 0.22 + 0.1 * tooth + 0.68 * (1.0 - 1.0 / max(h.t / frame.lod.x, 1.0));
            soil_side = uv.y < 1.0 - lip;
        }
        let a = mix(mix(c00, c10, uv.x), mix(c01, c11, uv.x), uv.y) / 3.0;
        ao = mix(0.42, 1.0, a);
        // Crisp voxel edges while a cell covers several pixels.
        let pixel = h.t * 2.0 / (camera.proj[1][1] * frame.screen.y);
        let size = frame.layer.y * f32(1 << level);
        let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
        let fade = clamp((size / pixel - 3.0) / 6.0, 0.0, 1.0);
        ao *= 1.0 - 0.14 * fade * (1.0 - smoothstep(0.0, 0.12, edge));
    }
    // Per-voxel pigment variation and gentle patches (Lay of the Land look).
    let hv = hash3(h.i, h.j, h.k + i32(face) * 7919 + i32(level) * 104729, 0x68bc21ebu);
    let jitter = f32(hv & 255u) / 255.0;
    let tint = f32(hash3(h.i >> 3u, h.j >> 3u, i32(face), 0x1b873593u) & 255u) / 255.0;
    var albedo = palette(select(material, M_DIRT, soil_side)) * (0.86 + 0.24 * jitter);
    if material == M_GRASS && !soil_side {
        albedo *= mix(vec3<f32>(1.08, 1.0, 0.72), vec3<f32>(0.82, 1.02, 0.95), tint);
        if code == 4u { albedo *= 1.06; } else { albedo *= 0.9; }
    }
    out.t = h.t;
    let a8 = vec4<u32>(vec4<f32>(clamp(pow(albedo, vec3<f32>(1.0 / 2.2)), vec3<f32>(0.0), vec3<f32>(1.0)), ao) * 255.0 + 0.5);
    out.albedo_ao = a8.x | (a8.y << 8u) | (a8.z << 16u) | (a8.w << 24u);
    out.normal = oct_encode(normal);
    out.flags = ST_HIT | (material << 8u) | (level << 16u);
    surfaces[index] = out;
}

@group(0) @binding(9) var sun_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var scene_depth: texture_depth_2d;

struct SunSample {
    valid: bool,
    position: vec3<f32>,
    normal: vec3<f32>,
    footprint: f32,
}

// Surface point that receives sunlight at pixel `p`: the terrain hit, or the
// mesh depth when a mesh covers the terrain.
fn sun_sample(p: vec2<u32>) -> SunSample {
    var out: SunSample;
    out.valid = false;
    let sun = normalize(frame.sun.xyz);
    let s = surfaces[pixel_index(p)];
    let d = pixel_ray(vec2<f32>(p) + 0.5);
    let depth = textureLoad(scene_depth, vec2<i32>(p), 0);
    let uv = (vec2<f32>(p) + 0.5) / frame.screen.xy;
    if (s.flags & 3u) == ST_HIT {
        out.position = camera.position_near.xyz + s.t * d;
        out.normal = oct_decode(s.normal);
        out.valid = true;
        let clip = camera.view_proj * vec4<f32>(out.position, 1.0);
        if clip.w > 0.0 && clip.z / clip.w > depth + 1e-6 && depth > 0.0 && depth < 1.0 {
            let world = camera.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
            out.position = world.xyz / world.w;
            out.normal = sun;
        }
    } else if depth > 0.0 && depth < 1.0 {
        let world = camera.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        out.position = world.xyz / world.w;
        out.normal = sun;
        out.valid = true;
    }
    out.footprint = length(out.position - camera.position_near.xyz) * 2.0 / (camera.proj[1][1] * frame.screen.y);
    return out;
}

fn sun_visibility(s: SunSample) -> f32 {
    if !s.valid || frame.sun.w < 0.5 { return 1.0; }
    let sun = normalize(frame.sun.xyz);
    if dot(s.normal, sun) <= 0.0 { return 0.0; }
    let dist = length(s.position);
    let level = level_for(dist);
    let eps = frame.layer.y * f32(1 << level) * 0.02 + dist * 2e-6;
    let blocker = trace(make_ray(s.position + s.normal * eps, sun), 0.0, frame.lod.w, dist, 1.0, 0.0);
    return select(0.0, 1.0, (blocker.info & 3u) == ST_MISS);
}

// One sunlight ray per 2x2 block at a representative pixel that rotates each
// frame (TAA resolves the pattern). Pixels on a different surface than the
// representative trace their own ray, so silhouettes stay exact.
@compute @workgroup_size(8, 8)
fn sunlight(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen = vec2<u32>(frame.screen.xy);
    let origin = id.xy * 2u;
    if any(origin >= screen) { return; }
    let f = u32(frame.screen.z);
    let rep = min(origin + vec2<u32>(f & 1u, (f >> 1u) & 1u), screen - 1u);
    let rs = sun_sample(rep);
    let rv = sun_visibility(rs);
    let sun = normalize(frame.sun.xyz);
    for (var q = 0u; q < 4u; q++) {
        let p = origin + vec2<u32>(q & 1u, q >> 1u);
        if any(p >= screen) { continue; }
        var v = rv;
        if any(p != rep) {
            let qs = sun_sample(p);
            let same = qs.valid == rs.valid && (!qs.valid
                || (dot(qs.normal, rs.normal) > 0.999 && distance(qs.position, rs.position) < 2.5 * max(qs.footprint, rs.footprint)));
            if !same { v = sun_visibility(qs); }
        }
        textureStore(sun_out, vec2<i32>(p), vec4<f32>(v, sun));
    }
}

