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
    return frame.materials[min(m, MATERIAL_SLOTS - 1u)].colour.rgb;
}

// Material on the sides of `m`'s surface cells below its lip (soil under
// turf); `m` itself when it has none.
fn material_lip(m: u32) -> u32 { return frame.materials[min(m, MATERIAL_SLOTS - 1u)].links.x; }
fn material_fleck(m: u32) -> u32 { return frame.materials[min(m, MATERIAL_SLOTS - 1u)].links.y; }
fn material_speck_host(m: u32) -> u32 { return frame.materials[min(m, MATERIAL_SLOTS - 1u)].links.z; }
fn material_fleck_share(m: u32) -> f32 { return f32(frame.materials[min(m, MATERIAL_SLOTS - 1u)].links.w) / 65536.0; }

// Whether hit `h` of column `c` (whose top there is `top`) lies on the
// natural ground surface: any cell of a height-field column; in a column with
// caves, overhangs or edits, its top cell and the risers of the steps down
// to its neighbours (side faces whose air side lies above the neighbour's
// top). Cave walls (their air side lies below it), ceilings and other cells
// are not; a face that a dig exposed is not either (the caller checks
// `removed_air_neighbour`). Undersides within two cells of the top are the
// lips of ledges the overhangs lean out (as thin as the level shows them);
// deeper ones (cave ceilings, thick roofs) are not. Edited columns used to
// lose all natural appearance (brown soil dashes and contour lines around
// every edit), and cave-region risers below the top cell were lit as crisp
// walls: dark, self-shadowed specks across every coarse hillside.
fn natural_surface_hit(h: Hit, c: Column, top: i32) -> bool {
    if (c.info & (INFO_TOPOLOGY | INFO_GENERATED)) == 0u || h.k >= top - 1 { return true; }
    let code = (h.info >> 10u) & 7u;
    if code == 5u { return h.k >= top - 3; }
    if code >= 4u { return false; }
    return h.k >= air_side_top(h, c, (h.info >> 2u) & 7u, (h.info >> 5u) & 31u, code, top);
}

// Natural material filtering for a hit in layer `k` of column `c` (top
// `top` there): natural columns, and the natural top surface of generated
// cave and overhang columns (`natural_surface_hit`). Those got edit-cut
// appearance: grass risers showed bare soil under a thin lip, brown stripes
// down every gentle slope of a cave region.
fn natural_material_at(edited: bool, natural_hit: bool, c: Column) -> bool {
    return natural_material_filter_allowed(edited, c)
        || (!edited && (c.info & (INFO_TOPOLOGY | INFO_GENERATED)) != 0u && natural_hit);
}

fn natural_material_filter_allowed(edited: bool, c: Column) -> bool {
    return !edited && (c.info & (INFO_TOPOLOGY | INFO_GENERATED)) == 0u;
}


// A material's filtered single-voxel flecks keep their share of its colour
// (`stone`, the rock's own colour there) instead of a fresh full-contrast
// hash choice per pixel.
fn filtered_rock_flecks(albedo: vec3<f32>, stone: vec3<f32>, pigment: f32, rock: u32, weight: f32) -> vec3<f32> {
    let share = material_fleck_share(rock);
    let mean = (1.0 - share) * stone + share * pigment * palette(material_fleck(rock));
    return mix(albedo, mean, weight);
}

// Colour of material `m`: its world-space patches from dry through middle
// to lush when it varies (continuous across levels), else its colour. The
// patch octave (25.6 m wavelength) fades out as a pixel's footprint
// approaches it, so distant terrain shows its average. The fade follows
// the footprint, not the level: fading by level stepped the patch contrast
// at every level boundary, which showed as rings sweeping outward while
// ascending.
// Each drawn cell (`cell`, its volume point) also moves along the
// dry-to-lush ramp by its own hash, weighted by `voxel` (`mosaic_weight`):
// blades of different hue, so turf reads as made of voxels at every
// distance (Lay of the Land look).
fn material_albedo(m: u32, p: vec3<i32>, cell: vec3<i32>, pixel: f32, voxel: f32) -> vec3<f32> {
    let material = frame.materials[min(m, MATERIAL_SLOTS - 1u)];
    if material.patches[0].w <= 0.0 { return material.colour.rgb; }
    let broad = f32(noise(p, 15u, 0x3c6ef372u)) / f32(NOISE_ONE);
    let patches = f32(noise(p, 11u, 0xa54ff53au)) / f32(NOISE_ONE) * clamp((12.8 - pixel) / 6.4, 0.0, 1.0);
    // Clumps of 6.4 m and 1.6 m: texture that reads from tens to hundreds
    // of metres, where single voxels have averaged out (the ground looked
    // flat and blurred there). Each fades as it shrinks below a few pixels.
    var clumps = 0.0;
    if pixel < 3.2 {
        clumps += 0.35 * f32(noise(p, 9u, 0x510e527fu)) / f32(NOISE_ONE) * clamp((3.2 - pixel) / 1.6, 0.0, 1.0);
    }
    if pixel < 0.8 {
        clumps += 0.3 * f32(noise(p, 7u, 0x9b05688cu)) / f32(NOISE_ONE) * clamp((0.8 - pixel) / 0.4, 0.0, 1.0);
    }
    let blade = f32(hash3(cell.x, cell.y, cell.z, 0x1b873593u) & 255u) / 255.0 - 0.5;
    let t = clamp(0.52 + frame.detail.x * (0.7 * broad + 0.16 * patches + clumps) + frame.detail.y * blade * voxel, 0.0, 1.0);
    let dry = material.patches[0].rgb;
    let middle = material.patches[1].rgb;
    let lush = material.patches[2].rgb;
    return select(mix(middle, lush, t * 2.0 - 1.0), mix(dry, middle, t * 2.0), t < 0.5);
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

// Keep the large planetary origin in integers until it has been removed.
// Adding a ~50M base index in f32 first loses several authored voxel fractions.
fn face_local_cell(eye_index: vec3<i32>, hit_index: vec3<i32>, relative: vec3<f32>, level: u32) -> vec3<f32> {
    let origin_delta = (hit_index << vec3<u32>(level)) - eye_index;
    return (relative - vec3<f32>(origin_delta)) / f32(1u << level);
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
    return column_cell(c, u32(i & 7), u32(j & 7), k) == 1u;
}

fn corner_ao(side1: bool, side2: bool, corner: bool) -> f32 {
    if side1 && side2 { return 0.0; }
    return 3.0 - f32(u32(side1) + u32(side2) + u32(corner));
}

// Materials classify one slope field, whatever level draws them: central
// differences of relief heights 3.2 m each way (two level-4 cells, one
// level-5 cell), interpolated bilinearly between cell centres. Levels finer
// than 4 read level-4 columns (`material_slope` in planet.rs); level 5 the
// same baseline from its own cells; coarser levels their own cells, whose
// pixels cover more than that baseline anyway. Measuring at each level's own
// scale changed rock and snow with distance (grey hillsides far away that
// became grass on approach).
const MATERIAL_SLOPE_LEVEL: u32 = 4u;
const NO_HEIGHT: i32 = -2147483647 - 1;

// The (up to) 2x2 resident columns of `level` covering cells [lo, hi].
struct SlopeColumns {
    face: u32,
    level: u32,
    origin: vec2<i32>,
    records: vec4<u32>,
}

fn slope_record(face: u32, level: u32, ci: i32, cj: i32) -> u32 {
    let record = find_column(column_key0(face, level, ci), bitcast<u32>(cj));
    if record == NONE { return NONE; }
    let m = records[record];
    if !column_valid(m) { return NONE; }
    return record;
}

fn slope_columns(face: u32, level: u32, lo: vec2<i32>, hi: vec2<i32>) -> SlopeColumns {
    var out: SlopeColumns;
    out.face = face;
    out.level = level;
    out.origin = lo >> vec2<u32>(3u);
    let far = hi >> vec2<u32>(3u);
    out.records.x = slope_record(face, level, out.origin.x, out.origin.y);
    out.records.y = select(NONE, slope_record(face, level, far.x, out.origin.y), far.x != out.origin.x);
    out.records.z = select(NONE, slope_record(face, level, out.origin.x, far.y), far.y != out.origin.y);
    out.records.w = select(NONE, slope_record(face, level, far.x, far.y), any(far != out.origin));
    return out;
}

// Relief height of cell (mi, mj) in Q16 cells of the columns' level, or
// NO_HEIGHT while its column is not resident.
fn slope_height_q16(cols: SlopeColumns, mi: i32, mj: i32) -> i32 {
    let q = vec2<i32>(mi, mj) >> vec2<u32>(3u);
    let k = u32(q.x != cols.origin.x) + 2u * u32(q.y != cols.origin.y);
    let record = cols.records[k];
    if record == NONE { return NO_HEIGHT; }
    let m = records[record];
    let x = u32(mi & 7);
    let y = u32(mj & 7);
    let f = column_relief_fraction(m, x, y);
    return column_top(m, x, y) * 65536 + select(0, i32(f) - 65536, f != 0u);
}

// Slope at cell (mi, mj) in eighths of a cell per cell from central
// differences `d` cells each way, or -1.
fn slope_at(cols: SlopeColumns, mi: i32, mj: i32, d: i32) -> i32 {
    let e = slope_height_q16(cols, mi + d, mj);
    let w = slope_height_q16(cols, mi - d, mj);
    let n = slope_height_q16(cols, mi, mj + d);
    let s = slope_height_q16(cols, mi, mj - d);
    if min(min(e, w), min(n, s)) == NO_HEIGHT { return -1; }
    return max(abs(e - w), abs(n - s)) / (16384 * d);
}

// The material slope of hit cell (i, j) of `level`, at the base cell at the
// hit cell's centre, or `own` while the columns it reads are not resident.
fn material_slope(face: u32, i: i32, j: i32, level: u32, own: i32) -> i32 {
    let s = max(level, MATERIAL_SLOPE_LEVEL);
    let d = select(1, 2, s == MATERIAL_SLOPE_LEVEL);
    // Base cell at the hit cell's centre, in half base cells relative to
    // the first cell centre of level `s`.
    let half = 1 << level;
    let ri = ((i << level) + (half >> 1)) * 2 + 1 - (1 << s);
    let rj = ((j << level) + (half >> 1)) * 2 + 1 - (1 << s);
    let ai = ri >> (s + 1u);
    let aj = rj >> (s + 1u);
    // Weights in 32nds (exact at level 4, as planet.rs truncates them).
    let wi = (ri - (ai << (s + 1u))) >> (s - 4u);
    let wj = (rj - (aj << (s + 1u))) >> (s - 4u);
    let cols = slope_columns(face, s, vec2<i32>(ai - d, aj - d), vec2<i32>(ai + 1 + d, aj + 1 + d));
    let s00 = slope_at(cols, ai, aj, d);
    let s10 = slope_at(cols, ai + 1, aj, d);
    let s01 = slope_at(cols, ai, aj + 1, d);
    let s11 = slope_at(cols, ai + 1, aj + 1, d);
    if min(min(s00, s10), min(s01, s11)) < 0 { return own; }
    return ((32 - wi) * (32 - wj) * s00 + wi * (32 - wj) * s10 + (32 - wi) * wj * s01 + wi * wj * s11) >> 10u;
}

// The natural ground under base cell `base` from the resident columns of
// level `s`: its gradient (radial cells per cell) and exact height (mm),
// continuous across cells, columns and risers. Central differences of the
// exact surface (relief heights plus surface offsets) one cell each way at
// the four level-`s` cell centres around the base cell, and their heights,
// interpolated bilinearly. A voxel staircase has flat treads metres long on
// gentle ground, so differences of its stored tops are zero there and spike
// at the risers: the exact surface is what shading reads. Columns are
// looked up and loaded once (the hit's own column without a lookup); while
// a neighbour streams, its cells are clamped into the hit column. The
// height is the material height: a 3 km snowfield must not become sea-level
// grass when a level cell exceeds its elevation.
struct GroundField {
    gradient: vec2<f32>,
    height: i32,
}

fn ground_field(face: u32, s: u32, base: vec2<i32>, home: u32, home_column: vec2<i32>) -> GroundField {
    let r = base * 2 + 1 - (1 << s);
    let a = r >> vec2<u32>(s + 1u);
    let w = vec2<f32>(r - (a << vec2<u32>(s + 1u))) / f32(2 << s);
    let origin = (a - 1) >> vec2<u32>(3u);
    let far = (a + 2) >> vec2<u32>(3u);
    let own = records[home];
    var columns: array<Column, 4>;
    var present = array<bool, 4>(false, false, false, false);
    for (var k = 0; k < 4; k++) {
        let q = origin + vec2<i32>(k & 1, k >> 1u);
        if any(q > far) { continue; }
        var record = home;
        if any(q != home_column) { record = slope_record(face, s, q.x, q.y); }
        if record != NONE {
            columns[k] = records[record];
            present[k] = true;
        }
    }
    // Exact surface heights (Q16 cells) of the 12 cells the differences
    // read: rows a.y - 1 ..= a.y + 2 of columns a.x - 1 ..= a.x + 2, but the
    // corners; and the first centre's height in millimetres.
    let mm_per_cell = f32(world.grid.y) * exp2(f32(s));
    var heights: array<i32, 16>;
    var height0 = 0;
    for (var n = 0; n < 16; n++) {
        let d = vec2<i32>(n & 3, n >> 2u) - 1;
        if (d.x == -1 || d.x == 2) && (d.y == -1 || d.y == 2) { continue; }
        var cell = a + d;
        let q = cell >> vec2<u32>(3u);
        let k = u32(q.x != origin.x) + 2u * u32(q.y != origin.y);
        var m = columns[k];
        if !present[k] {
            m = own;
            cell = clamp(cell, home_column * 8, home_column * 8 + 7);
        }
        let x = u32(cell.x & 7);
        let y = u32(cell.y & 7);
        let f = column_relief_fraction(m, x, y);
        let top = column_top(m, x, y);
        // Height above the top cell's top (cells): the relief's cut plus the
        // surface offset.
        let below = select(0.0, f32(i32(f) - 65536) / 65536.0, f != 0u)
            + column_surface_offset(m, x, y, s);
        heights[n] = top * 65536 + i32(round(below * 65536.0));
        if n == 5 { height0 = (top << s) * world.grid.y + i32(round(below * mm_per_cell)); }
    }
    // Central differences at the four cell centres (Q16 heights two cells
    // apart; wrapping differences stay exact), interpolated bilinearly.
    var g: array<vec2<f32>, 4>;
    for (var c = 0; c < 4; c++) {
        let at = (c & 1) + 1 + ((c >> 1u) + 1) * 4;
        g[c] = vec2<f32>(f32(heights[at + 1] - heights[at - 1]), f32(heights[at + 4] - heights[at - 4])) / 131072.0;
    }
    var out: GroundField;
    out.gradient = mix(mix(g[0], g[1], w.x), mix(g[2], g[3], w.x), w.y);
    let rise = vec3<f32>(f32(heights[6] - heights[5]), f32(heights[9] - heights[5]), f32(heights[10] - heights[5])) / 65536.0;
    let blend = rise.x * w.x * (1.0 - w.y) + rise.y * (1.0 - w.x) * w.y + rise.z * w.x * w.y;
    out.height = height0 + i32(round(blend * mm_per_cell));
    return out;
}

// Average authored detail only as it becomes unresolved by the render grid.
// The transition straddles one pixel; resolvable faces keep their contrast.
fn detail_filter_weight(projected_cell: f32) -> f32 {
    return 1.0 - smoothstep(0.75, 1.25, projected_cell);
}

// Visibility of the voxel mosaic (each drawn cell's own hue and
// brightness) for a cell of `projected_cell` pixels: kept down to about half
// a pixel, so every distance shows voxels (LOD cells are 1-2 pixels wide)
// and temporal reconstruction averages what is smaller.
fn mosaic_weight(projected_cell: f32) -> f32 {
    return smoothstep(0.35, 0.7, projected_cell);
}

// Lighting of base voxel steps: a step a few pixels wide is still a texture
// of thin riser lines and step shadows, densest just before the coarser,
// relief-smooth levels take over. Their macro normal fades in from four
// pixels down, so the exact voxels near the eye blend into smooth ground
// instead of a band of stripes.
fn step_filter_weight(projected_cell: f32) -> f32 {
    return 1.0 - smoothstep(1.0, 2.0, projected_cell);
}

// Independent hash detail aliases along a face's compressed projected axis.
// Use its actual projected support, including near-tangent subpixel faces.
// Area support keeps resolved long faces distinct from hash detail.
// Interior/unknown face code 6 has no geometric normal to project.
fn appearance_projection(incidence: f32, code: u32) -> vec2<f32> {
    let cosine = select(clamp(abs(incidence), 0.0, 1.0), 1.0, code >= 6u);
    return vec2<f32>(cosine, sqrt(cosine));
}

// Coplanar grazing samples are far apart along the view direction even
// within one screen pixel. Keep the perpendicular footprint unchanged and
// bound the along-view extension to four times the original world radius.
fn shadow_reuse_distance_squared(delta: vec3<f32>, eye_to_sample: vec3<f32>, normal: vec3<f32>,
    query_filtered: f32, representative_filtered: f32) -> f32 {
    if query_filtered <= 0.0 || representative_filtered <= 0.0 { return dot(delta, delta); }
    let view = normalize(eye_to_sample);
    let cosine = clamp(abs(dot(normal, view)), 0.25, 1.0);
    if cosine >= 1.0 { return dot(delta, delta); }
    let along = dot(delta, view);
    let perpendicular = delta - view * along;
    return dot(perpendicular, perpendicular) + along * along * cosine * cosine;
}

// A visible angular wall is still a wall. Only unresolved natural risers
// borrow the ground's smooth normal; radial terrain tops take it; undersides
// never do.
fn smooth_face_weight(code: u32, projected_cell: f32, selected_level: bool) -> f32 {
    if code == 4u { return 1.0; }
    if code >= 4u { return 0.0; }
    if selected_level { return 1.0; }
    return detail_filter_weight(projected_cell);
}

// The physical tangent gradient of chart derivatives (radial cells per
// chart cell). Cube-sphere chart directions are oblique away from the face
// centre; solve their two constraints instead of treating the plane normals
// as orthogonal.
fn chart_gradient(face: u32, up: vec3<f32>, derivative: vec2<f32>, radius: f32) -> vec3<f32> {
    var vi: vec3<f32>;
    var vj: vec3<f32>;
    if is_plane() {
        let f = frame.faces[face];
        vi = f.m_a.xyz * (frame.layer.z / frame.layer.y);
        vj = f.m_b.xyz * (frame.layer.z / frame.layer.y);
    } else {
        let n = vec3<f32>(face_axis(face, 0u));
        let a = vec3<f32>(face_axis(face, 1u));
        let b = vec3<f32>(face_axis(face, 2u));
        let un = dot(up, n);
        let ua = dot(up, a);
        let ub = dot(up, b);
        let scale = radius * frame.layer.z / frame.layer.y;
        vi = (a - up * ua) * (scale * (un + ua * ua / un));
        vj = (b - up * ub) * (scale * (un + ub * ub / un));
    }
    let aa = dot(vi, vi);
    let bb = dot(vj, vj);
    let ab = dot(vi, vj);
    let determinant = aa * bb - ab * ab;
    return (derivative.x * (bb * vi - ab * vj) + derivative.y * (aa * vj - ab * vi)) / determinant;
}

// Radial support of a ray/plane pixel differential. The angular half-width
// bounds the linear approximation when the pixel cone crosses tangency.
fn radial_material_span(pixel: f32, distance: f32, ray: vec3<f32>, up: vec3<f32>, normal: vec3<f32>) -> f32 {
    let numerator = cross(ray, cross(up, normal));
    let epsilon = max(0.5 * pixel / max(distance, 0.05), 0.00000095367431640625);
    return pixel * length(numerator) / max(abs(dot(normal, ray)), epsilon);
}

// A side lip varies along radial UV, not the face's arbitrary compressed
// axis. Average its threshold over the actual ray/plane radial support only
// when unresolved. The clipped interval conditions colour on this face;
// visibility and resolved voxel edges remain the primary hit's responsibility.
fn soil_lip_coverage(v: f32, lip: f32, pixel: f32, distance: f32, ray: vec3<f32>,
    up: vec3<f32>, normal: vec3<f32>, radial_size: f32, allowed: bool) -> f32 {
    let point = select(0.0, 1.0, v < 1.0 - lip);
    if !allowed { return point; }
    let span = radial_material_span(pixel, distance, ray, up, normal) / radial_size;
    let weight = detail_filter_weight((1.0 - lip) / max(span, 0.000001));
    if weight <= 0.0 { return point; }
    let lo = max(v - 0.5 * span, 0.0);
    let hi = min(v + 0.5 * span, 1.0);
    let coverage = clamp((min(hi, 1.0 - lip) - lo) / max(hi - lo, 0.000001), 0.0, 1.0);
    return mix(point, coverage, weight);
}

// Relief changes geometry inside the last coarse cell. Natural surface
// strata use that stored authored top, not the enclosing coarse voxel.
// Cuts, deep samples and resolvable walls keep their actual hit layer.
fn surface_material_layer(top: i32, fraction: u32, level: u32, hit_layer: i32,
    depth: i32, code: u32, filtered: f32, relief: bool, topology: bool) -> i32 {
    if !relief || topology || depth != 0 || (code != 4u && (code >= 4u || filtered <= 0.5)) {
        return hit_layer;
    }
    if fraction == 0u { return (top << level) - 1; }
    var remainder: u32;
    if level <= 16u { remainder = fraction >> (16u - level); }
    else { remainder = fraction << (level - 16u); }
    return ((top - 1) << level) + i32(remainder) - 1;
}

// A natural riser remains surface material even inside a topology column.
// Subsoil on a side requires a resident base-solid neighbour removed by the
// edit journal. No generator query is needed to certify this cut face: the
// neighbour is the hit's air side, so a Remove brush containing it cut it.
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
    if !column_valid(neighbour) || (neighbour.info & INFO_TOPOLOGY) == 0u { return false; }
    // Air the heightfield had is not a cut.
    if terrain_kind(column_top(neighbour, u32(ij.x & 7), u32(ij.y & 7)), h.k) == 0u { return false; }
    let centre = vec3<i32>(center_half(ij.x, level), center_half(ij.y, level), center_half(h.k, level));
    return latest_edit(neighbour.edits, level, centre, domain_point(face, ij.x, ij.y, level), OPS_REMOVE) != NONE;
}

// Generated top of the cell across side face `code` (the hit's air side),
// or `fallback` when that column is not resident. A cave wall lies far
// below it; a natural riser does not.
fn air_side_top(h: Hit, c: Column, face: u32, level: u32, code: u32, fallback: i32) -> i32 {
    var ij = vec2<i32>(h.i, h.j);
    if code < 2u { ij.x += select(-1, 1, code == 0u); }
    else { ij.y += select(-1, 1, code == 2u); }
    let n = cells_at(level);
    if any(ij < vec2<i32>(0)) || any(ij >= vec2<i32>(n)) { return fallback; }
    var neighbour = c;
    if (ij.x >> 3) != (h.i >> 3) || (ij.y >> 3) != (h.j >> 3) {
        let record = find_column(column_key0(face, level, ij.x >> 3), bitcast<u32>(ij.y >> 3));
        if record == NONE { return fallback; }
        neighbour = records[record];
        if !column_valid(neighbour) { return fallback; }
    }
    return column_top(neighbour, u32(ij.x & 7), u32(ij.y & 7));
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
    var material = 0u;
    // A cell an Add brush built is authored geometry: its faces keep their
    // actual normals at any distance (painted ground stays natural).
    var built = false;
    if (c.info & INFO_EDIT_MATERIALS) != 0u {
        let flags = latest_edit(c.edits, level, vec3<i32>(center_half(h.i, level), center_half(h.j, level), center_half(h.k, level)),
            domain_point(face, h.i, h.j, level), OPS_MATERIAL);
        if flags != NONE {
            material = (flags >> 8u) & 255u;
            built = ((flags >> 4u) & 3u) == 1u;
        }
    }
    let edited = material != 0u;
    // A face a dig exposed (its air-side cell was removed) shows the cut.
    let exposed = (c.info & INFO_TOPOLOGY) != 0u && c.edits != 0u && code < 4u
        && removed_air_neighbour(h, c, face, level, code);
    let natural_hit = natural_surface_hit(h, c, top);
    let natural_material = natural_material_at(edited, natural_hit, c) && !exposed;
    var speck = false;
    var slope = 0;
    var debug_depth = 0;
    // Appearance is sampled at the ray's base-grid footprint, not the
    // centre of an increasingly large level cell: the base cell's 3D volume
    // point, height included (sampling the column's 2D point gave every
    // voxel of a cliff the same colour and material choice: vertical streaks
    // down every steep face). It is also where the ground field is read.
    var p: vec3<i32>;
    var base_ij = vec2<i32>(h.i, h.j);
    if level == 0u {
        p = volume_point(face, h.i, h.j, h.k, 0u);
    } else {
        let ray = make_ray(camera.position_near.xyz, d);
        let appearance_cell = locate(ray, face_ray(face, ray), h.t, 0u);
        // A quarter layer into the solid: a top face's point lies on the
        // boundary between two layers.
        let code_now = (h.info >> 10u) & 7u;
        let nudge = select(0.0, select(-0.25, 0.25, code_now == 5u), code_now >= 4u && code_now < 6u);
        let k = frame.layer_i.x + i32(floor(layer_coord(ray, h.t) + nudge));
        p = volume_point(face, appearance_cell.i, appearance_cell.j, k, 0u);
        base_ij = vec2<i32>(appearance_cell.i, appearance_cell.j);
    }
    let ground = ground_field(face, level, base_ij, h.record, vec2<i32>(h.i, h.j) >> vec2<u32>(3u));
    let actual_normal = hit_normal(h, d);
    let pixel = h.t * 2.0 / (camera.proj[1][1] * frame.screen.y);
    // Material filtering is appearance only: explicit brush materials and
    // topology cuts retain the canonical procedural classification.
    let size = frame.layer.y * f32(1 << level);
    var projection = vec2<f32>(1.0);
    if !edited && (c.info & INFO_TOPOLOGY) == 0u {
        projection = appearance_projection(dot(actual_normal, d), code);
    }
    let hash_filter_w = detail_filter_weight(frame.layer.y / pixel * projection.x);
    // Base steps filter by the size their faces project to: a riser seen
    // from above is a fraction of a voxel tall on screen. Filtering by the
    // voxel's size kept risers 1-2 voxels wide lit as walls, one-pixel dark
    // contour lines across every gentle slope.
    let base_filter_w = step_filter_weight(frame.layer.y / pixel * projection.x);
    // Primary can select either adjacent level under its bounded dither.
    // A still coarser resident column is streaming fallback, whose resolvable
    // walls must keep their actual face normal.
    let selected_level = level <= level_for(h.t * (1.0 + 0.5 * frame.lod.y));
    let material_relief = (c.info & INFO_RELIEF) != 0u;
    var material_fraction = 0u;
    if material_relief { material_fraction = column_relief_fraction(c, x, y); }
    let coarse_w = detail_filter_weight(size / pixel);
    // Visibility may temporarily use a coarser column. Its enlarged cell
    // edges are not visible authored voxels.
    let appearance_w = max(coarse_w, base_filter_w);
    // Generated base tops do not describe edit walls, cave ceilings or floors.
    // A natural underside (a ledge's lip) faces down: filtered, it stands for
    // the ground's slope like a riser (unfiltered lips one or two pixels wide
    // were black specks under every leaning ledge of a hillside seen from
    // below); resolved, it keeps its own shadowed face.
    let ledge_lip = code == 5u && natural_hit;
    let face_w = select(smooth_face_weight(code, size / pixel, selected_level), 1.0, ledge_lip);
    let smooth_w = select(0.0, base_filter_w * face_w, natural_hit && !exposed && !built);
    // A grazing face can have subpixel area while its long edge is resolved.
    // Keep the resolved face normal. Pigment and corner occlusion can alias
    // along the compressed axis even while that face's long edge is resolved.
    // Natural ground is a staircase of base voxels standing for a smooth
    // slope: its steps are lit partly with the slope's normal (`step
    // softness`, appearance detail.w), cast no step shadows and keep turf on
    // their risers, so they read as voxel texture instead of black contour
    // lines. Edits and cave walls keep crisp faces.
    let natural_step = !edited && !exposed && natural_hit && !ledge_lip;
    let soft_w = select(0.0, frame.detail.w, natural_step);
    let shade_smooth_w = max(smooth_w, soft_w);
    let ao_appearance_w = max(max(appearance_w, soft_w), detail_filter_weight(size / pixel * projection.x));
    // The ground's smooth normal (`ground_field`).
    let up = hit_up(h.t, d);
    let radius = frame.eye.w + height_rel(make_ray(camera.position_near.xyz, d), h.t);
    let ground_normal = normalize(up - chart_gradient(face, up, ground.gradient, radius));
    if natural_material {
        material_weathered_skin = true;
        let material_up = up;
        let material_normal = select(material_up, ground_normal, smooth_w > 0.0);
        // Bound the grazing expansion; voxel detail and normal gates keep
        // their own footprint. This scalar approximation is material-only.
        material_footprint = pixel / max(abs(dot(material_normal, d)), 0.25);
        material_radial_span = radial_material_span(pixel, h.t, d, material_up, material_normal);
    }
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
        if level == 0u && (shade_smooth_w > 0.0 || hash_filter_w > 0.0)
            && natural_material {
            // An unresolved terrace ensemble's radial support follows the
            // ground's smooth normal, not a resolved voxel face's.
            material_radial_span = max(material_radial_span,
                radial_material_span(pixel, h.t, d, up, ground_normal));
        }
        slope = material_slope(face, h.i, h.j, level, block_slope_of(tx0, tx7, ty0, ty7));
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
        if column_generated(c) {
            // Generated caves and overhangs: tops are the generated tops, so
            // depth counts from the air side's top. Cave walls, floors and
            // ceilings are buried; natural risers and lips are surface.
            var air_top = min(top, lowest);
            if code < 4u { air_top = air_side_top(h, c, face, level, code, air_top); }
            depth = max(air_top - 1 - h.k, 0) << level;
        } else if (c.info & INFO_TOPOLOGY) != 0u {
            if code >= 4u || exposed {
                depth = max(top - 1 - h.k, 0) << level;
            }
        }
        // Filtered natural sides use the top material. Select those final
        // inputs once instead of evaluating and discarding a lower sample.
        let top_material = code < 4u && smooth_w > 0.5;
        let material_depth = select(depth, 0, top_material);
        debug_depth = material_depth;
        let sample_layer = select(h.k << level, (top - 1) << level, top_material);
        let material_layer = surface_material_layer(top, material_fraction, level,
            sample_layer, material_depth, code, smooth_w, material_relief,
            (c.info & INFO_TOPOLOGY) != 0u);
        material = ground_material(p, column_surface(c, x, y), ground.height, material_depth, slope, material_layer);
        speck = (material & M_SPECK) != 0u;
        material &= M_ID;
    }
    var normal = actual_normal;
    // Filtered appearance (after "Filtered appearance for voxels", HPG
    // 2023): a cell covering about a pixel stands for a smooth slope of many
    // finer steps, so it is lit with the macro normal of the column's height
    // field and shows surface material on its risers. Cells several pixels
    // wide keep crisp faces; the blend follows the pixel footprint, so level
    // changes show no seam.
    var lift = 0u;
    if shade_smooth_w > 0.0 {
        normal = normalize(mix(normal, ground_normal, shade_smooth_w));
    }
    if code < 4u && shade_smooth_w > 0.5 {
        // Paint keeps the unpainted light origin. Filtered natural risers
        // receive light at the top surface; topology and oversized fallback
        // wall guards already suppress shade_smooth_w above.
        lift = u32(clamp(top - h.k, 0, 255));
    }
    // Neighbourhood occlusion around the air cell in front of the face.
    var ao = 1.0;
    // Side faces of grass voxels show soil below a ragged grass lip. The lip
    // covers more of the face with distance, where one coarse cell stands for
    // a grassy slope of many fine steps.
    var soil_side = false;
    var soil_coverage = 0.0;
    // Grazing faces can have fully filtered AO while their grass lip remains
    // resolved. Only those lips still need face coordinates in that case.
    if code < 6u && appearance_w < 1.0 &&
        (ao_appearance_w < 1.0 || ((code >> 1u) < 2u && material_lip(material) != material)) {
        let axis = code >> 1u;
        // The face's basis from scalar selects: FXC rejects local vector
        // l-values indexed by a runtime value (`v[axis] = ..`).
        let axis_dir = vec3<i32>(select(0, 1, axis == 0u), select(0, 1, axis == 1u), select(0, 1, axis == 2u));
        let du = vec3<i32>(select(1, 0, axis == 0u), select(0, 1, axis == 0u), 0);
        let dv = vec3<i32>(0, select(0, 1, axis == 2u), select(1, 0, axis == 2u));
        // Position of the hit inside the face.
        let fr = face_ray(face, make_ray(camera.position_near.xyz, d));
        let cell = face_local_cell(vec3<i32>(fr.idx, frame.layer_i.x), vec3<i32>(h.i, h.j, h.k),
            vec3<f32>(face_coord(fr, 0u, h.t), face_coord(fr, 1u, h.t),
                layer_coord(make_ray(camera.position_near.xyz, d), h.t)), level);
        let uv = clamp(vec2<f32>(dot(cell, vec3<f32>(du)), dot(cell, vec3<f32>(dv))), vec2<f32>(0.0), vec2<f32>(1.0));
        if axis < 2u && material_lip(material) != material {
            let tooth = f32(hash3(h.i, h.j, h.k * 4 + i32(floor(uv.x * 4.0)), 0x5bd1e995u) & 7u) / 7.0;
            // Continuous in distance (not level), so level changes show no band.
            let distance_fade = 1.0 - 1.0 / max(h.t / frame.lod.x, 1.0);
            let cut_lip = 0.22 + 0.1 * tooth + 0.68 * distance_fade;
            // Natural turf wraps the riser (a soil edge drew a brown contour
            // stripe on every step of a hill); brush cuts show their soil.
            let lip = select(cut_lip, 1.0, natural_material);
            soil_side = uv.y < 1.0 - lip;
            soil_coverage = soil_lip_coverage(uv.y, lip, pixel, h.t, d,
                hit_up(h.t, d), actual_normal, size, natural_material);
        }
        if ao_appearance_w < 1.0 {
            let back = select(1, -1, (code & 1u) == 1u);
            let f = vec3<i32>(h.i, h.j, h.k) + axis_dir * back;
            let s0 = occupied(face, level, f.x - du.x, f.y - du.y, f.z - du.z, c, h.i, h.j);
            let s1 = occupied(face, level, f.x + du.x, f.y + du.y, f.z + du.z, c, h.i, h.j);
            let s2 = occupied(face, level, f.x - dv.x, f.y - dv.y, f.z - dv.z, c, h.i, h.j);
            let s3 = occupied(face, level, f.x + dv.x, f.y + dv.y, f.z + dv.z, c, h.i, h.j);
            let c00 = corner_ao(s0, s2, occupied(face, level, f.x - du.x - dv.x, f.y - du.y - dv.y, f.z - du.z - dv.z, c, h.i, h.j));
            let c10 = corner_ao(s1, s2, occupied(face, level, f.x + du.x - dv.x, f.y + du.y - dv.y, f.z + du.z - dv.z, c, h.i, h.j));
            let c01 = corner_ao(s0, s3, occupied(face, level, f.x - du.x + dv.x, f.y - du.y + dv.y, f.z - du.z + dv.z, c, h.i, h.j));
            let c11 = corner_ao(s1, s3, occupied(face, level, f.x + du.x + dv.x, f.y + du.y + dv.y, f.z + du.z + dv.z, c, h.i, h.j));
            let a = mix(mix(c00, c10, uv.x), mix(c01, c11, uv.x), uv.y) / 3.0;
            ao = mix(mix(0.42, 1.0, a), 1.0, ao_appearance_w);
            // Crisp voxel edges while a cell covers several pixels.
            let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
            let fade = clamp((size / pixel - 3.0) / 6.0, 0.0, 1.0);
            ao *= 1.0 - frame.detail.z * fade * (1.0 - ao_appearance_w) * (1.0 - smoothstep(0.0, 0.12, edge));
        }
    }
    // Per-voxel pigment variation over world-space grass patches (Lay of
    // the Land look), averaged out as *base* voxels shrink below a pixel. A
    // coarse cell stands for many base voxels, so its pigment is their mean;
    // fading by the level cell's footprint instead jumped 2x at every level
    // boundary (a sawtooth of speckle contrast: rings while ascending).
    // The same base-domain point supplies pigment on every face and level.
    // Coarse indices or level salts would choose a new colour pattern at LOD.
    let cell_p = volume_point(face, h.i, h.j, h.k, level);
    let mosaic_w = mosaic_weight(size / pixel * projection.y);
    let hv = hash3(cell_p.x, cell_p.y, cell_p.z, 0x68bc21ebu);
    let jitter = mix(0.5, f32(hv & 255u) / 255.0, mosaic_w);
    let pigment = 1.0 + frame.detail.y * (jitter - 0.5);
    let lip = material_lip(material);
    // Every material's colour varies by its patches and voxels (turf hues,
    // weathered stone), not only turf: rock used to be one flat grey.
    let surface = material_albedo(material, p, cell_p, pixel, mosaic_w) * pigment;
    var albedo = select(surface, palette(lip) * pigment, soil_side);
    if (lip != material && soil_coverage < 1.0) || appearance_w > 0.0 {
        let host = material_speck_host(material);
        if lip != material {
            // Keep the resolved soil lip, then average its coverage only as
            // authored voxels become sub-pixel. A boolean cutoff at half the
            // filter weight made the lip switch to turf along a distance ring.
            albedo = mix(surface, palette(lip) * pigment, soil_coverage * (1.0 - appearance_w));
        } else if code == 4u && speck && host != material {
            // Filtered single-voxel specks (mud and sand in meadows) blend
            // into their host material.
            albedo = mix(albedo, material_albedo(host, p, cell_p, pixel, mosaic_w) * pigment, appearance_w);
        }
    }
    if material_mix.x >= 0.0 {
        albedo = pigment * (material_mix.x * palette(material_mix_ids.x)
            + material_mix.y * palette(material_mix_ids.y)
            + material_mix.z * palette(material_mix_ids.z)
            + material_mix.w * palette(material_mix_ids.w));
    } else if material_coverage >= 0.0 && material_fleck_base == M_AIR
        && natural_material {
        albedo = pigment * mix(palette(material_coverage_ids.x), palette(material_coverage_ids.y), material_coverage);
    } else if material_fleck_base != M_AIR && natural_material {
        // Band support is independent of single-voxel fleck support. Keep a
        // resolved fleck while filtering unresolved coverage around it.
        if material_coverage >= 0.0 && material != material_fleck(material_fleck_base) {
            albedo = pigment * mix(palette(material_coverage_ids.x), palette(material_coverage_ids.y), material_coverage);
        }
        if hash_filter_w > 0.0 {
            let stone = select(material_albedo(material_fleck_base, p, cell_p, pixel, mosaic_w) * pigment, albedo,
                material_coverage >= 0.0 && material != material_fleck(material_fleck_base));
            albedo = filtered_rock_flecks(albedo, stone, pigment, material_fleck_base, hash_filter_w);
        }
    }
    let debug_view = frame.hints.w >> 8u;
    if debug_view != 0u {
        // Diagnostics (`Settings::debug_view`): level colours.
        let colours = array<vec3<f32>, 8>(vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.9, 0.6, 0.1), vec3<f32>(0.8, 0.9, 0.1), vec3<f32>(0.2, 0.8, 0.2),
            vec3<f32>(0.1, 0.8, 0.8), vec3<f32>(0.2, 0.3, 0.9), vec3<f32>(0.6, 0.2, 0.9), vec3<f32>(0.9, 0.2, 0.7));
        let shown = select(level, level_for(h.t), debug_view == 3u);
        albedo = colours[min(shown, 7u)] * (0.45 + 0.55 * shade_smooth_w);
        if debug_view == 4u {
            // Column kinds: generated volume red, edit topology orange,
            // relief green, plain blue.
            var kind_colour = vec3<f32>(0.2, 0.3, 0.9);
            if column_generated(c) { kind_colour = vec3<f32>(0.9, 0.2, 0.2); }
            else if (c.info & INFO_TOPOLOGY) != 0u { kind_colour = vec3<f32>(0.9, 0.6, 0.1); }
            else if (c.info & INFO_RELIEF) != 0u { kind_colour = vec3<f32>(0.2, 0.8, 0.2); }
            albedo = kind_colour * (0.45 + 0.55 * shade_smooth_w);
        }
        if debug_view == 5u {
            // Faces and burial: red sides, blue undersides, green buried
            // (material depth > 0).
            albedo = vec3<f32>(select(0.15, 0.9, code < 4u), select(0.15, 0.9, debug_depth > 0), select(0.15, 0.9, code == 5u));
        }
        ao = 1.0;
        if debug_view == 2u { normal = hit_up(h.t, d); }
    }
    out.t = h.t;
    let a8 = vec4<u32>(vec4<f32>(clamp(pow(albedo, vec3<f32>(1.0 / 2.2)), vec3<f32>(0.0), vec3<f32>(1.0)), ao) * 255.0 + 0.5);
    out.albedo_ao = a8.x | (a8.y << 8u) | (a8.z << 16u) | (a8.w << 24u);
    out.normal = oct_encode(normal);
    let filtered = select(u32(round(shade_smooth_w * 7.0)), 7u, natural_step);
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

// A riser hit already contains its within-cell radial fraction. Lift only
// to the resident surface, including its authored relief remainder, rather
// than adding whole cells and overshooting the surface by that fraction.
fn filtered_shadow_lift(c: Column, h: Hit, r: Ray) -> f32 {
    if !natural_surface_hit(h, c, column_top(c, u32(h.i & 7), u32(h.j & 7))) { return 0.0; }
    let level = (h.info >> 5u) & 31u;
    var fraction = 0u;
    if (c.info & INFO_RELIEF) != 0u {
        fraction = column_relief_fraction(c, u32(h.i & 7), u32(h.j & 7));
    }
    let receiver_top = relief_height(c, h.i, h.j, level, fraction);
    return max(receiver_top - height_rel(r, h.t), 0.0);
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
    if out.level >= 0 && (s.flags >> 24u) != 0u {
        // Filtered risers receive the light of their column's top surface.
        let h = hits[pixel_index(p)];
        let lift = filtered_shadow_lift(records[h.record], h, make_ray(camera.position_near.xyz, d));
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
    // Terrain no level holds yet (streaming) is unknown, not an occluder:
    // counting it drew the loading columns' outlines as shadows.
    let status = blocker.info & 3u;
    return select(0.0, 1.0, status == ST_MISS || status == ST_LOADING);
}

// Representative samples of the workgroup's 2x2 blocks: visibility,
// position with footprint (w < 0: no surface) and normal.
var<workgroup> rep_vis: array<f32, 64>;
var<workgroup> rep_pos: array<vec4<f32>, 64>;
var<workgroup> rep_nrm: array<vec3<f32>, 64>;
var<workgroup> rep_filtered: array<f32, 64>;

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
    // Both samples must represent filtered terrain. Topology surfaces keep
    // zero support and the original world-distance cap, including edit walls.
    let separation = shadow_reuse_distance_squared(d, q.position - camera.position_near.xyz,
        q.normal, q.filtered, rep_filtered[slot]);
    return dot(q.normal, rep_nrm[slot]) > 0.9 && abs(dot(d, q.normal)) < plane && separation < 16.0 * fp * fp;
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
    rep_filtered[own] = rs.filtered;
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


// Sky visibility: the share of the sky dome a terrain point sees past the
// terrain around it, as ambient occlusion at terrain scale (valleys, cliff
// feet, overhangs, caves). The corner term in `shade` covers single cells;
// screen-space AO skips terrain. Rays are fixed and cosine-weighted around
// the local vertical, counting only those that leave the face (the sky's
// irradiance already gives each face its share of the dome by its normal:
// a wall in the open is not occluded by the rock behind it). So it varies
// smoothly over the terrain, and a block's estimate serves every pixel near
// it. A face turned down (ceilings, overhang undersides) sees no dome, but
// the ground under it, which that irradiance holds: what decides its light
// is whether the space under it is enclosed, so its rays run all around,
// just below level (cave walls stop them, an overhang lets them out). Rays reach
// SKY_RANGE_CELLS cells of the hit's level: occlusion keeps its size in
// cells, a coarse level standing for a wider neighbourhood.
const SKY_RAYS: u32 = 6u;
const SKY_RANGE_CELLS: f32 = 48.0;

fn sky_visibility(s: SunSample) -> f32 {
    if !s.valid || s.level < 0 { return 1.0; }
    let dist = length(s.position);
    let level = u32(s.level);
    let lo = select(0.0, frame.lod.x * exp2(f32(s.level) - 1.0) * 1.001, s.level > 0);
    let offset = clamp(dist, lo, frame.lod.x * exp2(f32(s.level)) * 0.999);
    let cell = frame.layer.y * f32(1 << level);
    let eps = cell * 0.02 + dist * 2e-6;
    let range = SKY_RANGE_CELLS * cell;
    var up = frame.eye.xyz;
    if !is_plane() { up = normalize(frame.eye.xyz + s.position / frame.eye.w); }
    let ceiling = dot(s.normal, up) < -0.25;
    let t1 = normalize(select(cross(up, vec3<f32>(0.0, 0.0, 1.0)), cross(up, vec3<f32>(1.0, 0.0, 0.0)), abs(up.z) > 0.9));
    let t2 = cross(up, t1);
    // A filtered cell stands for a slope of small steps: its own steps do
    // not occlude it (as for sunlight).
    let skip = s.filtered * 2.0 * cell;
    let origin = s.position + s.normal * eps;
    var open = 0.0;
    var total = 0.0;
    for (var i = 0u; i < SKY_RAYS; i++) {
        let u = (f32(i) + 0.5) / f32(SKY_RAYS);
        let phi = f32(i) * 2.39996323;
        let sin_t = sqrt(u);
        let around = t1 * cos(phi) + t2 * sin(phi);
        var dir = normalize(around * sin_t + up * sqrt(1.0 - u));
        if ceiling { dir = normalize(around - up * 0.02); }
        else if dot(dir, s.normal) <= 0.0 { continue; }
        let hit = trace(make_ray(origin, dir), skip, range, offset, 1.0, 0.0);
        // A ray into terrain no level holds yet (streaming) tells nothing.
        if (hit.info & 3u) == ST_LOADING { continue; }
        total += 1.0;
        if (hit.info & 3u) == ST_MISS {
            open += 1.0;
        } else {
            // Distant occluders shade less: no hard edge at the range.
            open += smoothstep(0.5, 1.0, hit.t / range);
        }
    }
    return select(1.0, open / total, total > 0.0);
}

// Squared distance from terrain sample `q` to representative `slot`'s
// point, or -1 when either is not terrain.
fn rep_distance_squared(slot: u32, q: SunSample) -> f32 {
    let pos = rep_pos[slot];
    if !q.valid || q.level < 0 || pos.w < 0.0 { return -1.0; }
    let d = q.position - pos.xyz;
    return dot(d, d);
}

// One sky-visibility estimate per 4x4 block at a representative pixel that
// rotates each frame (TAA resolves the pattern). Every terrain pixel takes
// the nearest representative on terrain among its own block and the three
// blocks towards its corner: sky visibility varies slowly, and the rotation
// averages the choice at silhouettes. Tracing there instead (a few pixels
// per warp, each with the longest rays of the frame) cost more than all the
// representatives together. Only a pixel with no terrain representative
// around it traces its own rays. The result is multiplied into the
// terrain's ambient occlusion before the GBuffer publishes it.
@compute @workgroup_size(8, 8)
fn skylight(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let screen = vec2<u32>(frame.screen.xy);
    let origin = id.xy * 4u;
    let inside = all(origin < screen);
    let f = u32(frame.screen.z);
    let own = lid.x + lid.y * 8u;
    let rep = min(origin + vec2<u32>(f & 3u, (f >> 2u) & 3u), max(screen, vec2<u32>(1u)) - 1u);
    var rv = 1.0;
    var rs: SunSample;
    rs.valid = false;
    if inside {
        rs = sun_sample(rep);
        rv = sky_visibility(rs);
    }
    rep_vis[own] = rv;
    rep_pos[own] = vec4<f32>(rs.position, select(-1.0, rs.footprint, rs.valid && rs.level >= 0));
    workgroupBarrier();
    if !inside || (frame.hints.w >> 8u) != 0u { return; }
    for (var q = 0u; q < 16u; q++) {
        let local = vec2<u32>(q & 3u, q >> 2u);
        let p = origin + local;
        if any(p >= screen) { continue; }
        let index = pixel_index(p);
        if (surfaces[index].flags & 3u) != ST_HIT { continue; }
        var v = rv;
        if any(p != rep) {
            let qs = sun_sample(p);
            let side = vec2<i32>(select(-1, 1, local.x >= 2u), select(-1, 1, local.y >= 2u));
            var nearest = -1.0;
            for (var c = 0u; c < 4u; c++) {
                let n = vec2<i32>(lid.xy) + vec2<i32>(select(0, side.x, (c & 1u) != 0u), select(0, side.y, (c & 2u) != 0u));
                if any(n < vec2<i32>(0)) || any(n > vec2<i32>(7)) { continue; }
                let slot = u32(n.x) + u32(n.y) * 8u;
                let d2 = rep_distance_squared(slot, qs);
                if d2 >= 0.0 && (nearest < 0.0 || d2 < nearest) {
                    nearest = d2;
                    v = rep_vis[slot];
                }
            }
            if nearest < 0.0 { v = sky_visibility(qs); }
        }
        let packed = surfaces[index].albedo_ao;
        let ao = f32(packed >> 24u) / 255.0 * v;
        surfaces[index].albedo_ao = (packed & 0x00ffffffu) | (u32(ao * 255.0 + 0.5) << 24u);
    }
}
