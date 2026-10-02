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
    let span = sky_span(sky);
    // Fail-safe for the sky bound: a ray the bound ended that is still
    // descending there and will pass inside the terrain shell cannot be sky,
    // so it continues without the bound. In a consistent frame no ray meets
    // this; it closes rare single-frame holes at the planet's limb seen
    // during very fast altitude changes (a ring of sky around the planet
    // from orbit). One `trace` call site keeps the shader's register use.
    var t0 = span.x;
    var t1 = min(frame.lod.w, span.y);
    var hit: Hit;
    for (var attempt = 0u; attempt < 2u; attempt++) {
        hit = trace(r, t0, t1, 0.0, 1.0, frame.lod.y);
        if (hit.info & 3u) != ST_MISS || t1 >= frame.lod.w || (u32(frame.screen.w) & 4u) != 0u || !bound_cut_terrain_ray(r, t1) { break; }
        t0 = t1;
        t1 = frame.lod.w;
    }
    hits[pixel_index(id.xy)] = hit;
}

// Whether an eye ray, past ray distance `t`, still descends and will pass
// inside the terrain shell (below the world's outer radius).
fn bound_cut_terrain_ray(r: Ray, t: f32) -> bool {
    if r.ol >= 0.0 { return false; }
    if is_plane() { return true; }
    // Closest approach to the planet centre at t_c = -rho * ol, radius
    // rho * sqrt(1 - ol^2); the shell's outer radius is rho + layer.w.
    let rho = frame.eye.w;
    if -rho * r.ol <= t { return false; }
    let k = 1.0 + frame.layer.w / rho;
    return 1.0 - r.ol * r.ol < k * k;
}

fn srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(c / 255.0, vec3<f32>(2.2));
}

fn palette(m: u32) -> vec3<f32> {
    return frame.palette[min(m, 15u)].rgb;
}

// Grass colour from dry through meadow to lush green by world-space
// patches (continuous across levels). The patch octave (25.6 m wavelength)
// fades out as a pixel's footprint approaches it, so distant terrain shows
// its average. The fade follows the footprint, not the level: fading by
// level stepped the patch contrast at every level boundary, which showed as
// rings sweeping outward while ascending.
fn grass_albedo(p: vec3<i32>, pixel: f32) -> vec3<f32> {
    let broad = f32(noise(p, 13u, 0x3c6ef372u)) / f32(NOISE_ONE);
    let patches = f32(noise(p, 9u, 0xa54ff53au)) / f32(NOISE_ONE) * clamp((12.8 - pixel) / 6.4, 0.0, 1.0);
    let t = clamp(0.58 + frame.detail.x * (0.6 * broad + 0.14 * patches), 0.0, 1.0);
    let dry = frame.grass[0].rgb;
    let meadow = frame.grass[1].rgb;
    let lush = frame.grass[2].rgb;
    return select(mix(meadow, lush, t * 2.0 - 1.0), mix(dry, meadow, t * 2.0), t < 0.5);
}

fn plane_normal(face: u32, axis: u32, plane: i32) -> vec3<f32> {
    let f = frame.faces[face];
    if is_plane() { return select(f.m_b.xyz, f.m_a.xyz, axis == 0u); }
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
    if is_plane() { return frame.eye.xyz; }
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
    if (c.info & INFO_HEIGHTFIELD) != 0u {
        return k < column_top(c, u32(i & 7), u32(j & 7));
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

// Climate changes over metres to kilometres, so distant 2x2 footprints share
// a canonical height evaluation. Base-level hits use their resident top;
// distant depth discontinuities retain their own query. Occupancy is unchanged.
@group(0) @binding(20) var<storage, read_write> climate_height_cache: array<i32>;

fn climate_at(h: Hit, xy: vec2<u32>) -> i32 {
    let dir = pixel_ray(vec2<f32>(xy) + 0.5);
    let face = (h.info >> 2u) & 7u;
    let ray = make_ray(camera.position_near.xyz, dir);
    let cell = locate(ray, face_ray(face, ray), h.t, 0u);
    let level = (h.info >> 5u) & 31u;
    if frame.hints.y != 0u && (cell.i >> level) == h.i && (cell.j >> level) == h.j {
        let c = records[h.record];
        var top = column_top(c, u32(h.i & 7), u32(h.j & 7));
        // The climate bound proof uses the original floor-quantized field,
        // whereas relief occupancy publishes a conservative ceil top.
        if (c.info & INFO_RELIEF) != 0u && column_relief_fraction(c, u32(h.i & 7), u32(h.j & 7)) != 0u { top -= 1; }
        // Tall edited/steep bands can truncate their byte-packed column
        // tops; those columns retain the full canonical query.
        if column_tops_fit(c) && climate_height_reusable(top, level) {
            return i32(f32(top) * f32(world.grid.y) * f32(1u << level));
        }
    }
    return terrain_height(domain_point(face, cell.i, cell.j, 0u), u32(world.grid.w));
}

@compute @workgroup_size(8, 8)
fn climate(@builtin(global_invocation_id) id: vec3<u32>) {
    let anchor_xy = id.xy * 2u;
    if any(anchor_xy >= vec2<u32>(frame.screen.xy)) { return; }
    let anchor = hits[pixel_index(anchor_xy)];
    var height = 0;
    if (anchor.info & 3u) == ST_HIT && ((anchor.info >> 5u) & 31u) > 0u {
        height = climate_at(anchor, anchor_xy);
    }
    // Resolve discontinuities here as well. Keeping the terrain generator
    // out of shade avoids carrying its registers through material/AO work.
    for (var q = 0u; q < 4u; q++) {
        let xy = anchor_xy + vec2<u32>(q & 1u, q >> 1u);
        if any(xy >= vec2<u32>(frame.screen.xy)) { continue; }
        let h = hits[pixel_index(xy)];
        if (h.info & 3u) != ST_HIT || ((h.info >> 5u) & 31u) == 0u { continue; }
        var own_height = height;
        if ((anchor.info >> 5u) & 31u) == 0u || (anchor.info & 3u) != ST_HIT
            || ((anchor.info >> 2u) & 7u) != ((h.info >> 2u) & 7u)
            || abs(h.t - anchor.t) > max(1.0, h.t * 0.01) {
            own_height = climate_at(h, xy);
        }
        climate_height_cache[pixel_index(xy)] = own_height;
    }
}

// Climate has completed before shade. Two-pixel offsets sample distinct 2x2
// anchors, avoiding the shared-height plateaus inside each block. Reject
// incomplete, edited and discontinuous neighbourhoods instead of inventing
// relief across silhouettes, cuts, chart edges or streaming boundaries.
override FAR_RELIEF: bool = false;

// Difference of stored filtered heights in coarse-cell units. Subtract the
// integer tops before conversion to retain small slopes at large elevations.
fn column_relief_delta_q16(c: Column, a: vec2<u32>, b: vec2<u32>, top_delta: i32) -> i32 {
    let fa = column_relief_fraction(c, a.x, a.y);
    let fb = column_relief_fraction(c, b.x, b.y);
    let ca = select(0, i32(fa) - 65536, fa != 0u);
    let cb = select(0, i32(fb) - 65536, fb != 0u);
    return top_delta * 65536 + ca - cb;
}

fn relief_compatible(xy: vec2<u32>, center: Hit) -> bool {
    let h = hits[pixel_index(xy)];
    let anchor_xy = (xy >> vec2<u32>(1u)) << vec2<u32>(1u);
    let anchor = hits[pixel_index(anchor_xy)];
    let face = (center.info >> 2u) & 7u;
    for (var n = 0u; n < 2u; n++) {
        var candidate = h;
        if n == 1u { candidate = anchor; }
        if (candidate.info & 3u) != ST_HIT || ((candidate.info >> 5u) & 31u) == 0u
            || ((candidate.info >> 2u) & 7u) != face { return false; }
        if abs(candidate.t - center.t) > max(1.0, center.t * 0.02) { return false; }
        let column = records[candidate.record];
        if (column.info & INFO_TOPOLOGY) != 0u || !column_tops_fit(column) { return false; }
    }
    return true;
}

// Match climate's actual source pixel. At a depth break it computes an own
// height instead of the anchor height; derivatives must use that sample's
// position as well, or the stencil silently changes length at level edges.
fn relief_source_pixel(xy: vec2<u32>) -> vec2<u32> {
    let anchor_xy = (xy >> vec2<u32>(1u)) << vec2<u32>(1u);
    let anchor = hits[pixel_index(anchor_xy)];
    let h = hits[pixel_index(xy)];
    if ((anchor.info >> 5u) & 31u) == 0u || (anchor.info & 3u) != ST_HIT
        || ((anchor.info >> 2u) & 7u) != ((h.info >> 2u) & 7u)
        || abs(h.t - anchor.t) > max(1.0, h.t * 0.01) { return xy; }
    return anchor_xy;
}

fn cached_relief_normal(xy: vec2<u32>, center: Hit, up: vec3<f32>) -> vec4<f32> {
    if !FAR_RELIEF || frame.hints.z == 0u { return vec4<f32>(0.0); }
    let extent = vec2<u32>(frame.screen.xy);
    if any(xy < vec2<u32>(2u)) || any(xy + vec2<u32>(2u) >= extent) { return vec4<f32>(0.0); }
    let left = xy - vec2<u32>(2u, 0u);
    let right = xy + vec2<u32>(2u, 0u);
    let above = xy - vec2<u32>(0u, 2u);
    let below = xy + vec2<u32>(0u, 2u);
    if !relief_compatible(xy, center) || !relief_compatible(left, center)
        || !relief_compatible(right, center) || !relief_compatible(above, center)
        || !relief_compatible(below, center) { return vec4<f32>(0.0); }
    let source_left = relief_source_pixel(left);
    let source_right = relief_source_pixel(right);
    let source_above = relief_source_pixel(above);
    let source_below = relief_source_pixel(below);
    let pa = hits[pixel_index(source_right)].t * pixel_ray(vec2<f32>(source_right) + 0.5)
        - hits[pixel_index(source_left)].t * pixel_ray(vec2<f32>(source_left) + 0.5);
    let pb = hits[pixel_index(source_below)].t * pixel_ray(vec2<f32>(source_below) + 0.5)
        - hits[pixel_index(source_above)].t * pixel_ray(vec2<f32>(source_above) + 0.5);
    let a = pa - up * dot(up, pa);
    let b = pb - up * dot(up, pb);
    let aa = dot(a, a);
    let bb = dot(b, b);
    let ab = dot(a, b);
    let determinant = aa * bb - ab * ab;
    // Avoid unstable derivatives at grazing views and degenerate pixels.
    if min(aa, bb) < 1e-8 || determinant <= aa * bb * 0.01 { return vec4<f32>(0.0); }
    let dh_a = f32(climate_height_cache[pixel_index(right)] - climate_height_cache[pixel_index(left)]) * 0.001;
    let dh_b = f32(climate_height_cache[pixel_index(below)] - climate_height_cache[pixel_index(above)]) * 0.001;
    let gradient = (dh_a * (bb * a - ab * b) + dh_b * (aa * b - ab * a)) / determinant;
    // Fade confidence across steep gradients instead of abruptly changing
    // lighting and material support at the same slope boundary.
    let confidence = canonical_relief_confidence(dot(gradient, gradient))
        * canonical_stencil_weight(center.t, frame.lod.x, frame.lod.y);
    return vec4<f32>(normalize(up - gradient), confidence);
}

fn canonical_relief_confidence(gradient_squared: f32) -> f32 {
    return 1.0 - smoothstep(4.0, 9.0, gradient_squared);
}

// Compatible stencil hits and their anchors may be nearer by this depth
// tolerance and use the finest end of primary's dither. Keep their canonical
// derivative contribution zero through that L0 boundary, then introduce it
// gradually; the local column slope continues to shade this transition.
fn canonical_stencil_weight(distance: f32, lod0: f32, dither: f32) -> f32 {
    let tolerance = max(1.0, distance * 0.02);
    let nearest_selected = (distance - tolerance) * (1.0 - 0.5 * dither);
    return smoothstep(lod0, lod0 * 1.25, nearest_selected);
}

// Average authored detail only as it becomes unresolved by the render grid.
// The transition straddles one pixel; resolvable faces keep their contrast.
fn detail_filter_weight(projected_cell: f32) -> f32 {
    return 1.0 - smoothstep(0.75, 1.25, projected_cell);
}

// A visible angular wall is still a wall. Only unresolved natural risers
// borrow a continuous height-field normal; radial terrain tops retain it.
fn canonical_relief_face_weight(code: u32, projected_cell: f32, selected_level: bool) -> f32 {
    if code == 4u { return 1.0; }
    if code >= 4u { return 0.0; }
    if selected_level { return 1.0; }
    return detail_filter_weight(projected_cell);
}

// Convert a physical tangent gradient to the generator's slope convention:
// eighths of a radial voxel per angular index step. On the cube-sphere,
// moving one chart coordinate holds the other cell plane fixed. The chart
// tangents and angular/radial scale therefore matter away from face centres.
fn canonical_relief_slope(face: u32, up: vec3<f32>, gradient: vec3<f32>, radius: f32) -> f32 {
    if is_plane() {
        let f = frame.faces[face];
        return 8.0 * frame.layer.z / frame.layer.y * max(abs(dot(gradient, f.m_a.xyz)), abs(dot(gradient, f.m_b.xyz)));
    }
    let n = vec3<f32>(face_axis(face, 0u));
    let a = vec3<f32>(face_axis(face, 1u));
    let b = vec3<f32>(face_axis(face, 2u));
    let un = dot(up, n);
    let ma = normalize(a * un - n * dot(up, a));
    let mb = normalize(b * un - n * dot(up, b));
    let qa = normalize(n * un + a * dot(up, a));
    let qb = normalize(n * un + b * dot(up, b));
    let ti = cross(mb, up);
    let tj = cross(ma, up);
    let di = dot(gradient, ti) * dot(qa, up) / dot(ma, ti);
    let dj = dot(gradient, tj) * dot(qb, up) / dot(mb, tj);
    return 8.0 * radius * frame.layer.z / frame.layer.y * max(abs(di), abs(dj));
}

// A natural riser remains surface material even inside a topology column.
// Subsoil on a side requires a resident base-solid neighbour removed by the
// edit journal. No generator query is needed to certify this cut face.
fn removed_air_neighbour(h: Hit, c: Column, face: u32, level: u32, code: u32) -> bool {
    var ij = vec2<i32>(h.i, h.j);
    if code < 2u { ij.x += select(-1, 1, code == 0u); }
    else { ij.y += select(-1, 1, code == 2u); }
    let n = cells_at(level);
    if any(ij < vec2<i32>(0)) || any(ij >= vec2<i32>(n)) { return false; }
    var neighbour = c;
    if (ij.x >> 3) != (h.i >> 3) || (ij.y >> 3) != (h.j >> 3) {
        let record = find_column(column_key0(face, level, ij.x >> 3), bitcast<u32>(ij.y >> 3));
        if record == NONE { return false; }
        neighbour = records[record];
    }
    if !column_valid(neighbour) || !column_tops_fit(neighbour) || (neighbour.info & INFO_TOPOLOGY) == 0u { return false; }
    let base_top = column_top(neighbour, u32(ij.x & 7), u32(ij.y & 7));
    if terrain_kind(base_top, h.k) == 0u { return false; }
    let centre = vec3<i32>(center_half(ij.x, level), center_half(ij.y, level), center_half(h.k, level));
    return apply_edits(neighbour.edits, level, centre, 1u).x == 0u;
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
    let edited = material != 0u;
    var speck = false;
    var slope = 0;
    // Appearance is sampled at the ray's base-grid footprint, not the
    // centre of an increasingly large level cell. Climate uses the canonical
    // unrounded height at coarse levels: a 3 km snowfield must not become
    // sea-level grass when the radial level size exceeds its elevation.
    // Fine columns already have the canonical top at base-cell precision;
    // avoid rerunning the generator and ray-to-grid mapping for each pixel.
    var p: vec3<i32>;
    var climate_height = top * world.grid.y;
    if level == 0u {
        p = domain_point(face, h.i, h.j, 0u);
    } else {
        let ray = make_ray(camera.position_near.xyz, d);
        let appearance_cell = locate(ray, face_ray(face, ray), h.t, 0u);
        p = domain_point(face, appearance_cell.i, appearance_cell.j, 0u);
        climate_height = climate_height_cache[index];
    }
    let pixel = h.t * 2.0 / (camera.proj[1][1] * frame.screen.y);
    let size = frame.layer.y * f32(1 << level);
    var canonical_up = vec3<f32>(0.0);
    var canonical_relief = vec4<f32>(0.0);
    let base_filter_w = detail_filter_weight(frame.layer.y / pixel);
    // Primary can select either adjacent level under its bounded dither.
    // A still coarser resident column is streaming fallback, whose resolvable
    // walls must keep their actual face normal.
    let selected_level = level <= level_for(h.t * (1.0 + 0.5 * frame.lod.y));
    let authored_relief_w = base_filter_w;
    let relief_face_w = canonical_relief_face_weight(code, size / pixel, selected_level);
    if FAR_RELIEF && frame.hints.z != 0u && level > 0u && authored_relief_w > 0.0
        && relief_face_w > 0.0 && (!edited || (c.info & INFO_RELIEF) != 0u) {
        canonical_up = hit_up(h.t, d);
        canonical_relief = cached_relief_normal(id.xy, h, canonical_up);
    }
    let canonical_w = canonical_relief.w * authored_relief_w * relief_face_w;
    if !edited {
        var lowest = top;
        if x > 0u { lowest = min(lowest, column_top(c, x - 1u, y)); }
        if x < 7u { lowest = min(lowest, column_top(c, x + 1u, y)); }
        if y > 0u { lowest = min(lowest, column_top(c, x, y - 1u)); }
        if y < 7u { lowest = min(lowest, column_top(c, x, y + 1u)); }
        let tx0 = column_top(c, 0u, y);
        let tx7 = column_top(c, 7u, y);
        let ty0 = column_top(c, x, 0u);
        let ty7 = column_top(c, x, 7u);
        slope = block_slope_of(tx0, tx7, ty0, ty7);
        if level >= 1u && column_tops_fit(c) && (c.info & INFO_RELIEF) != 0u {
            let di = column_relief_delta_q16(c, vec2<u32>(7u, y), vec2<u32>(0u, y), tx7 - tx0);
            let dj = column_relief_delta_q16(c, vec2<u32>(x, 7u), vec2<u32>(x, 0u), ty7 - ty0);
            // Same truncation as block_slope_of, eighths per coarse cell.
            // Packed tops differ by at most 255, so the Q16 delta fits i32.
            slope = max(abs(di), abs(dj)) / 57344;
        }
        if canonical_w > 0.0 {
            let gradient = canonical_up - canonical_relief.xyz / dot(canonical_relief.xyz, canonical_up);
            let radius = length(frame.eye.xyz * frame.eye.w + camera.position_near.xyz + h.t * d);
            let canonical_slope = canonical_relief_slope(face, canonical_up, gradient, radius);
            // Lighting and material thresholds share the same derivative and
            // confidence; changing LOD blocks cannot silently change only rock/snow.
            slope = i32(mix(f32(slope), canonical_slope, canonical_w));
        }
        // Canonical materials use the column top cell, which is resident.
        // Depth counts from the lowest neighbouring top: an exposed riser
        // above it is surface, not subsoil (coarse levels step in large
        // cells where the fine terrain is a continuous slope). A natural side face
        // is always above the top of the air-side neighbour, which may lie
        // in the next column (not resident here): side faces are surface.
        // Measuring only in-column neighbours gave column-border risers
        // subsoil (stone at coarse levels): grey bands sweeping with the LOD
        // rings. Proven edit cuts use their own column's depth instead.
        var depth = select(max(min(top, lowest) - 1 - h.k, 0) << level, 0, code < 4u);
        if (c.info & INFO_TOPOLOGY) != 0u {
            if code >= 4u || removed_air_neighbour(h, c, face, level, code) {
                depth = max(top - 1 - h.k, 0) << level;
            }
        }
        material = ground_material(p, climate_height, depth, slope, h.k << level);
        speck = (material & M_SPECK) != 0u;
        material &= M_ID;
    }
    var normal = hit_normal(h, d);
    // Filtered appearance (after "Filtered appearance for voxels", HPG
    // 2023): a cell covering about a pixel stands for a smooth slope of many
    // finer steps, so it is lit with the macro normal of the column's height
    // field and shows surface material on its risers. Cells several pixels
    // wide keep crisp faces; the blend follows the pixel footprint, so level
    // changes show no seam.
    let coarse_w = detail_filter_weight(size / pixel);
    let authored_w = select(0.0, base_filter_w, FAR_RELIEF && frame.hints.z != 0u);
    // Visibility may temporarily use a coarser column. Its enlarged cell
    // edges are not visible authored voxels, even when the normal stencil rejects.
    let appearance_w = max(coarse_w, authored_w);
    // Generated base tops do not describe edit walls, cave ceilings or floors.
    // Paint-only and ignored tiny lists keep their existing filtering.
    let normal_filter_w = select(coarse_w, base_filter_w * relief_face_w, FAR_RELIEF && frame.hints.z != 0u);
    let smooth_w = select(normal_filter_w, 0.0, (c.info & INFO_TOPOLOGY) != 0u);
    var lift = 0u;
    if smooth_w > 0.0 {
        let x0 = select(x - 1u, 0u, x == 0u);
        let x1 = min(x + 1u, 7u);
        let y0 = select(y - 1u, 0u, y == 0u);
        let y1 = min(y + 1u, 7u);
        let di = column_top(c, x1, y) - column_top(c, x0, y);
        let dj = column_top(c, x, y1) - column_top(c, x, y0);
        var gi = f32(di) / f32(x1 - x0);
        var gj = f32(dj) / f32(y1 - y0);
        if level >= 1u && column_tops_fit(c) && (c.info & INFO_RELIEF) != 0u {
            gi = f32(column_relief_delta_q16(c, vec2<u32>(x1, y), vec2<u32>(x0, y), di)) / (65536.0 * f32(x1 - x0));
            gj = f32(column_relief_delta_q16(c, vec2<u32>(x, y1), vec2<u32>(x, y0), dj)) / (65536.0 * f32(y1 - y0));
        }
        let up = hit_up(h.t, d);
        let macro_normal = normalize(up - gi * plane_normal(face, 0u, h.i << level) - gj * plane_normal(face, 1u, h.j << level));
        normal = normalize(mix(normal, macro_normal, smooth_w));
        if code < 4u && smooth_w > 0.5 {
            if !edited {
                material = ground_material(p, climate_height, 0, slope, (top - 1) << level);
                speck = (material & M_SPECK) != 0u;
                material &= M_ID;
            }
            // Paint changes pigment, so it keeps the unpainted light origin.
            // Sunlight treats the riser as part of the slope: traced from
            // the column's top surface, not into the step above it.
            lift = u32(clamp(top - h.k, 0, 255));
        }
    }
    let raw_smooth_w = canonical_w;
    if canonical_w > 0.0 {
        normal = normalize(mix(normal, canonical_relief.xyz, canonical_w));
    }
    // Neighbourhood occlusion around the air cell in front of the face.
    var ao = 1.0;
    // Side faces of grass voxels show soil below a ragged grass lip. The lip
    // covers more of the face with distance, where one coarse cell stands for
    // a grassy slope of many fine steps.
    var soil_side = false;
    // Fully filtered cells use neither voxel AO nor face detail.
    if code < 6u && appearance_w < 1.0 {
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
        if axis < 2u && material == M_GRASS && appearance_w <= 0.5 {
            let tooth = f32(hash3(h.i, h.j, h.k * 4 + i32(floor(uv.x * 4.0)), 0x5bd1e995u) & 7u) / 7.0;
            // Continuous in distance (not level), so level changes show no band.
            let lip = 0.22 + 0.1 * tooth + 0.68 * (1.0 - 1.0 / max(h.t / frame.lod.x, 1.0));
            soil_side = uv.y < 1.0 - lip;
        }
        let a = mix(mix(c00, c10, uv.x), mix(c01, c11, uv.x), uv.y) / 3.0;
        ao = mix(mix(0.42, 1.0, a), 1.0, appearance_w);
        // Crisp voxel edges while a cell covers several pixels.
        let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
        let fade = clamp((size / pixel - 3.0) / 6.0, 0.0, 1.0);
        ao *= 1.0 - frame.detail.z * fade * (1.0 - appearance_w) * (1.0 - smoothstep(0.0, 0.12, edge));
    }
    // Per-voxel pigment variation over world-space grass patches (Lay of
    // the Land look), averaged out as *base* voxels shrink below a pixel. A
    // coarse cell stands for many base voxels, so its pigment is their mean;
    // fading by the level cell's footprint instead jumped 2x at every level
    // boundary (a sawtooth of speckle contrast: rings while ascending).
    // The same base-domain point supplies pigment on every face and level.
    // Coarse indices or level salts would choose a new colour pattern at LOD.
    let hv = hash3(p.x, p.y, p.z, 0x68bc21ebu);
    let base_w = base_filter_w;
    let jitter = mix(f32(hv & 255u) / 255.0, 0.5, base_w);
    let pigment = 1.0 + frame.detail.y * (jitter - 0.5);
    var albedo = palette(select(material, M_DIRT, soil_side)) * pigment;
    if (material == M_GRASS && !soil_side) || appearance_w > 0.0 {
        var grass = grass_albedo(p, pixel) * pigment;
        if code != 4u { grass *= mix(0.9, 1.0, max(appearance_w, raw_smooth_w)); }
        if material == M_GRASS && !soil_side {
            albedo = grass;
        } else if code == 4u && speck {
            // Single-voxel flecks (mud and sand in meadows) blend into grass.
            albedo = mix(albedo, grass, appearance_w);
        }
    }
    out.t = h.t;
    let a8 = vec4<u32>(vec4<f32>(clamp(pow(albedo, vec3<f32>(1.0 / 2.2)), vec3<f32>(0.0), vec3<f32>(1.0)), ao) * 255.0 + 0.5);
    out.albedo_ao = a8.x | (a8.y << 8u) | (a8.z << 16u) | (a8.w << 24u);
    out.normal = oct_encode(normal);
    let filtered = u32(round(smooth_w * 7.0));
    out.flags = ST_HIT | (material << 8u) | (level << 16u) | (filtered << 21u) | (lift << 24u);
    surfaces[index] = out;
}

@group(0) @binding(9) var sun_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var scene_depth: texture_depth_2d;

struct SunSample {
    valid: bool,
    position: vec3<f32>,
    normal: vec3<f32>,
    footprint: f32,
    // Level of the terrain hit (-1: a mesh surface).
    level: i32,
    // Filtered-appearance weight of the hit cell (see `shade`).
    filtered: f32,
}

// Surface point that receives sunlight at pixel `p`: the terrain hit, or the
// mesh depth when a mesh covers the terrain.
fn sun_sample(p: vec2<u32>) -> SunSample {
    var out: SunSample;
    out.valid = false;
    out.level = -1;
    out.filtered = 0.0;
    let sun = normalize(frame.sun.xyz);
    let s = surfaces[pixel_index(p)];
    let d = pixel_ray(vec2<f32>(p) + 0.5);
    let depth = textureLoad(scene_depth, vec2<i32>(p), 0);
    let uv = (vec2<f32>(p) + 0.5) / frame.screen.xy;
    if (s.flags & 3u) == ST_HIT {
        out.position = camera.position_near.xyz + s.t * d;
        out.normal = oct_decode(s.normal);
        out.valid = true;
        out.level = i32((s.flags >> 16u) & 31u);
        out.filtered = f32((s.flags >> 21u) & 7u) / 7.0;
        let clip = camera.view_proj * vec4<f32>(out.position, 1.0);
        if clip.w > 0.0 && clip.z / clip.w > depth + 1e-6 && depth > 0.0 && depth < 1.0 {
            let world = camera.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
            out.position = world.xyz / world.w;
            out.normal = sun;
            out.level = -1;
            out.filtered = 0.0;
        }
    } else if depth > 0.0 && depth < 1.0 {
        let world = camera.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        out.position = world.xyz / world.w;
        out.normal = sun;
        out.valid = true;
    }
    if out.level >= 0 {
        // Filtered risers receive the light of their column's top surface.
        let lift = f32(s.flags >> 24u) * frame.layer.y * f32(1 << u32(out.level));
        out.position += hit_up(s.t, d) * lift;
    }
    out.footprint = length(out.position - camera.position_near.xyz) * 2.0 / (camera.proj[1][1] * frame.screen.y);
    return out;
}

fn sun_visibility(s: SunSample) -> f32 {
    if !s.valid || frame.sun.w < 0.5 { return 1.0; }
    let sun = normalize(frame.sun.xyz);
    if dot(s.normal, sun) <= 0.0 { return 0.0; }
    let dist = length(s.position);
    var level = level_for(dist);
    // Sun rays choose levels by `t + offset`. Start at the level the primary
    // ray hit (its dither may differ from the distance's level): a coarser
    // start could lie inside the coarser surface and shadow itself.
    var offset = dist;
    if s.level >= 0 {
        level = u32(s.level);
        let lo = select(0.0, frame.lod.x * exp2(f32(s.level) - 1.0) * 1.001, s.level > 0);
        offset = clamp(dist, lo, frame.lod.x * exp2(f32(s.level)) * 0.999);
    }
    let cell = frame.layer.y * f32(1 << level);
    let eps = cell * 0.02 + dist * 2e-6;
    // A filtered cell stands for a smooth slope of many small steps, which
    // casts no step shadows: skip occluders up to two cells high.
    var up = frame.eye.xyz;
    if !is_plane() { up = normalize(frame.eye.xyz + s.position / frame.eye.w); }
    let skip = s.filtered * 2.0 * cell / max(dot(up, sun), 0.15);
    let blocker = trace(make_ray(s.position + s.normal * eps, sun), skip, frame.lod.w, offset, 1.0, 0.0);
    return select(0.0, 1.0, (blocker.info & 3u) == ST_MISS);
}

// Representative samples of the workgroup's 2x2 blocks: visibility,
// position with footprint (w < 0: no surface) and normal.
var<workgroup> rep_vis: array<f32, 64>;
var<workgroup> rep_pos: array<vec4<f32>, 64>;
var<workgroup> rep_nrm: array<vec3<f32>, 64>;

// Whether pixel sample `q` lies on the surface of representative `slot`, so
// it can take that ray's visibility: same validity and, for surfaces, a
// similar normal, the same plane and a nearby point. Voxel faces meet at
// right angles, so the loose normal test still separates them while the
// smooth macro normals of distant cells pass. A filtered cell stands for a
// smooth slope of small steps (see `shade`), so its plane tolerance grows to
// two cells.
fn on_rep_surface(slot: u32, q: SunSample) -> bool {
    let pos = rep_pos[slot];
    if q.valid != (pos.w >= 0.0) { return false; }
    if !q.valid { return true; }
    let fp = max(q.footprint, pos.w);
    let d = q.position - pos.xyz;
    let cell = frame.layer.y * f32(1u << u32(max(q.level, 0)));
    let plane = 0.5 * fp + 2.0 * cell * q.filtered;
    return dot(q.normal, rep_nrm[slot]) > 0.9 && abs(dot(d, q.normal)) < plane && dot(d, d) < 16.0 * fp * fp;
}

// One sunlight ray per 2x2 block at a representative pixel that rotates each
// frame (TAA resolves the pattern). Every other pixel takes the visibility
// of a representative on its own surface: its own block's or, across a face
// edge, one of the three neighbouring blocks on its side. Only a pixel with
// no matching representative traces its own ray, so silhouettes stay exact.
@compute @workgroup_size(8, 8)
fn sunlight(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let screen = vec2<u32>(frame.screen.xy);
    let origin = id.xy * 2u;
    let inside = all(origin < screen);
    let f = u32(frame.screen.z);
    let own = lid.x + lid.y * 8u;
    let rep = min(origin + vec2<u32>(f & 1u, (f >> 1u) & 1u), max(screen, vec2<u32>(1u)) - 1u);
    var rv = 1.0;
    var rs: SunSample;
    rs.valid = false;
    if inside {
        rs = sun_sample(rep);
        rv = sun_visibility(rs);
    }
    rep_vis[own] = rv;
    rep_pos[own] = vec4<f32>(rs.position, select(-1.0, rs.footprint, rs.valid));
    rep_nrm[own] = rs.normal;
    workgroupBarrier();
    if !inside { return; }
    let sun = normalize(frame.sun.xyz);
    for (var q = 0u; q < 4u; q++) {
        let p = origin + vec2<u32>(q & 1u, q >> 1u);
        if any(p >= screen) { continue; }
        var v = rv;
        if any(p != rep) {
            let qs = sun_sample(p);
            // Own block first, then the side, vertical and diagonal
            // neighbours towards this pixel's corner.
            let side = vec2<i32>(select(-1, 1, (q & 1u) != 0u), select(-1, 1, (q >> 1u) != 0u));
            var found = false;
            for (var c = 0u; c < 4u; c++) {
                let n = vec2<i32>(lid.xy) + vec2<i32>(select(0, side.x, (c & 1u) != 0u), select(0, side.y, (c & 2u) != 0u));
                if any(n < vec2<i32>(0)) || any(n > vec2<i32>(7)) { continue; }
                let slot = u32(n.x) + u32(n.y) * 8u;
                if on_rep_surface(slot, qs) {
                    v = rep_vis[slot];
                    found = true;
                    break;
                }
            }
            if !found { v = sun_visibility(qs); }
        }
        textureStore(sun_out, vec2<i32>(p), vec4<f32>(v, sun));
    }
}

