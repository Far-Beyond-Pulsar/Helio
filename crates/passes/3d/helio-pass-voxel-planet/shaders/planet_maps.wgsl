// Planet maps: climate and river lookups; mirror of src/planet_maps.rs.
// The maps' header is `terrain.maps` (`planet_maps::MAP_HEADER` rows), their
// words the engine's `terrain_tables`.
const MAP_FLAG_CLIMATE: i32 = 1;
const MAP_FLAG_RIVERS: i32 = 2;
const MAP_GHOST: i32 = 2;
const MAP_RECORD_WORDS: u32 = 8u;

fn map_word(i: u32) -> u32 {
    return terrain_tables[i >> 2u][i & 3u];
}

// Slot (face) and Q16 texel coordinates of domain point `p` on the river
// grid (`planet_maps::coords`).
fn map_coords(p: vec3<i32>) -> vec3<i32> {
    let half = terrain.maps[2].x;
    var slot = 0u;
    var pn = half;
    var pa = p.x;
    var pb = -p.z;
    if half <= 0 {
        let a = abs(p);
        if a.x >= a.y && a.x >= a.z {
            slot = select(0u, 1u, p.x < 0);
        } else if a.y >= a.z {
            slot = select(2u, 3u, p.y < 0);
        } else {
            slot = select(4u, 5u, p.z < 0);
        }
        pn = dot(face_axis(slot, 0u), p);
        pa = dot(face_axis(slot, 1u), p);
        pb = dot(face_axis(slot, 2u), p);
    }
    let n = u32(max(pn, 1));
    let bits = 32u - countLeadingZeros(n);
    let r = landform_recip_q30(map_norm(n, n, bits));
    let side = u32(terrain.maps[0].y) << 15u;
    return vec3<i32>(i32(slot), map_texel(pa, n, bits, r, side), map_texel(pb, n, bits, r, side));
}

fn map_norm(x: u32, n: u32, bits: u32) -> u32 {
    let c = min(x, n);
    if bits > 30u { return c >> (bits - 30u); }
    return c << (30u - bits);
}

fn map_texel(c: i32, n: u32, bits: u32, r: u32, side: u32) -> i32 {
    let m = i32(min(mul_shr(map_norm(u32(abs(c)), n, bits), r, 30u), Q30));
    let u = select(m, -m, c < 0);
    return i32(mul_shr(bitcast<u32>(u + i32(Q30)), side, 30u));
}

// Climate at domain point `p`, height `height` (mm): temperature (1/128 °C)
// and moisture (Q16) (`planet_maps::climate`).
fn terrain_climate(p: vec3<i32>, height: i32) -> vec2<i32> {
    let h = terrain.maps;
    let fall = div_floor(height, 1000) * h[2].w / 1000;
    if (h[0].x & MAP_FLAG_CLIMATE) == 0 {
        return vec2<i32>(h[2].y - fall, h[2].z);
    }
    let c = map_coords(p);
    let nc = h[0].w;
    let sc = h[1].w;
    let f = (c.yz >> vec2<u32>(1u)) - vec2<i32>(32768);
    let i = clamp(f >> vec2<u32>(16u), vec2<i32>(-1), vec2<i32>(nc - 1));
    let w = clamp(f - (i << vec2<u32>(16u)), vec2<i32>(0), vec2<i32>(65536)) >> vec2<u32>(8u);
    let at = u32(h[1].y) + u32(c.x) * u32(sc * sc) + u32((i.y + 1) * sc + i.x + 1);
    let w00 = map_word(at);
    let w10 = map_word(at + 1u);
    let w01 = map_word(at + u32(sc));
    let w11 = map_word(at + u32(sc) + 1u);
    let t = vec4<i32>(bitcast<i32>(w00 << 16u) >> 16u, bitcast<i32>(w10 << 16u) >> 16u, bitcast<i32>(w01 << 16u) >> 16u, bitcast<i32>(w11 << 16u) >> 16u);
    let m = vec4<i32>(vec4<u32>(w00, w10, w01, w11) >> vec4<u32>(16u));
    let ta = t.x + (((t.y - t.x) * w.x) >> 8u);
    let tb = t.z + (((t.w - t.z) * w.x) >> 8u);
    let ma = m.x + (((m.y - m.x) * w.x) >> 8u);
    let mb = m.z + (((m.w - m.z) * w.x) >> 8u);
    return vec2<i32>(ta + (((tb - ta) * w.y) >> 8u) - fall, ma + (((mb - ma) * w.y) >> 8u));
}

// The valley wall's rise `x` past a floodplain's edge, its foot curved
// (`planet_maps::wall_rise`).
fn map_wall_rise(x: i32, floodplain: i32, wall: i32) -> i32 {
    let w = clamp(floodplain / 2, 1, 32767);
    if x < w {
        return mul_shr_signed(mul_shr_signed(x, wall * 25, 7u), (x << 15u) / w, 16u);
    }
    return mul_shr_signed(x - w / 2, wall * 25, 7u);
}

// Meanders: the bend fields (`planet_maps::bends`).
const MEANDER_SHIFT_0: u32 = 20u;
const MEANDER_SHIFT_1: u32 = 16u;
const MEANDER_MAX_0: i32 = 67200;
const MEANDER_MAX_1: i32 = 4200;

fn map_bends(p: vec3<i32>, seed: u32) -> array<vec3<i32>, 2> {
    return array<vec3<i32>, 2>(
        vec3<i32>(noise(p, MEANDER_SHIFT_0, seed ^ 0x2f6b1c43u), noise(p, MEANDER_SHIFT_0, seed ^ 0x7a1d93e5u), noise(p, MEANDER_SHIFT_0, seed ^ 0x51c8e0a7u)),
        vec3<i32>(noise(p, MEANDER_SHIFT_1, seed ^ 0x0b4e7f29u), noise(p, MEANDER_SHIFT_1, seed ^ 0x6c932d51u), noise(p, MEANDER_SHIFT_1, seed ^ 0x3e58a6bdu)));
}

// One river segment's carving surface at `p` and whether `p` lies in its
// channel (`planet_maps::segment`), for cells `cell` domain units wide.
fn map_segment(at: u32, column: vec3<i32>, bend: array<vec3<i32>, 2>, cell: i32) -> vec2<i32> {
    let a = terrain_tables[at >> 2u];
    let b = terrain_tables[(at >> 2u) + 1u];
    let node = bitcast<vec3<i32>>(a.xyz);
    let geometry = a.w;
    let wall = i32((geometry >> 2u) & 255u);
    let floodplain = i32((geometry >> 10u) & 2047u) * 640;
    let channel = i32((geometry >> 21u) & 2047u) * 40;
    // Meanders (`planet_maps::segment`).
    let a1 = min(floodplain / 3, MEANDER_MAX_0);
    let a2 = min(channel / 3, MEANDER_MAX_1);
    let p = column + vec3<i32>(
        scale_q16(bend[0].x, a1) + scale_q16(bend[1].x, a2),
        scale_q16(bend[0].y, a1) + scale_q16(bend[1].y, a2),
        scale_q16(bend[0].z, a1) + scale_q16(bend[1].z, a2));
    let e = vec3<i32>(bitcast<i32>(b.x << 16u) >> 16u, bitcast<i32>(b.x) >> 16u, bitcast<i32>(b.y << 16u) >> 16u) << vec3<u32>(15u);
    let length = i32(b.y >> 16u) << 6u;
    let d = p - node;
    let along = mul_shr_signed(d.x, e.x, 30u) + mul_shr_signed(d.y, e.y, 30u) + mul_shr_signed(d.z, e.z, 30u);
    let s = clamp(along, 0, length);
    let w = d - vec3<i32>(mul_shr_signed(s, e.x, 30u), mul_shr_signed(s, e.y, 30u), mul_shr_signed(s, e.z, 30u));
    let n = unit_q30(w);
    let dist = max(mul_shr_signed(w.x, n.x, 30u) + mul_shr_signed(w.y, n.y, 30u) + mul_shr_signed(w.z, n.z, 30u), 0);
    var t = 0;
    if length > 0 { t = min((s << 8u) / max(length >> 8u, 1), 65536); }
    let bed = bitcast<i32>(b.z) - mul_shr_signed(i32(b.w & 0x3fffffu), t, 16u);
    var z = bed + map_wall_rise(max(dist - floodplain, 0), floodplain, wall);
    var inside = 0;
    if dist < channel {
        let q = (dist << 14u) / channel;
        let depth = i32(b.w >> 22u) * 100;
        let cut = depth - mul_shr_signed(depth, (q * q) >> 14u, 14u);
        let resolved = clamp(((channel << 7u) / cell) - 256, 0, 256);
        z -= (cut * resolved) >> 8u;
        inside = select(0, 1, resolved > 0);
    }
    return vec2<i32>(z, inside);
}

// Height `h` of the column at `p` (reference level `level`) carved by the
// rivers, and whether it lies in a channel (`planet_maps::carve`).
fn map_carve(p: vec3<i32>, level: u32, h: i32) -> vec2<i32> {
    let header = terrain.maps;
    if (header[0].x & MAP_FLAG_RIVERS) == 0 { return vec2<i32>(h, 0); }
    let c = map_coords(p);
    let n = header[0].y;
    let s = header[0].z;
    let ix = min(c.y >> 16u, n - 1);
    let iy = min(c.z >> 16u, n - 1);
    let index = u32(header[1].x) + u32(c.x) * u32(s * s);
    let cell = 8 << min(level, 24u);
    var best = 0x7fffffff;
    var inside = 0;
    var bent = false;
    var bend = array<vec3<i32>, 2>(vec3<i32>(0), vec3<i32>(0));
    for (var dy = -MAP_GHOST; dy <= MAP_GHOST; dy++) {
        for (var dx = -MAP_GHOST; dx <= MAP_GHOST; dx++) {
            let tx = ix + dx;
            let ty = iy + dy;
            let word = map_word(index + u32((ty + MAP_GHOST) * s + tx + MAP_GHOST));
            if word == 0u { continue; }
            let reach = i32(word >> 24u) << 12u;
            if abs(c.y - ((tx << 16u) + 32768)) > reach || abs(c.z - ((ty << 16u) + 32768)) > reach { continue; }
            let at = u32(header[1].z) + ((word & 0xffffffu) - 1u) * MAP_RECORD_WORDS;
            if !bent {
                bend = map_bends(p, bitcast<u32>(header[3].w));
                bent = true;
            }
            let r = map_segment(at, p, bend, cell);
            best = min(best, r.x);
            if r.y != 0 && r.x <= h { inside = 1; }
        }
    }
    if best == 0x7fffffff { return vec2<i32>(h, 0); }
    return vec2<i32>(landform_smooth_min(h, best, header[3].z), inside);
}
