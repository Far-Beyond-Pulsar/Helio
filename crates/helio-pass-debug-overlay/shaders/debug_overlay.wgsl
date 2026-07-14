struct Params {
    screen_w: u32,
    screen_h: u32,
    big_cols: u32,
    big_rows: u32,
    small_cols: u32,
    small_rows: u32,
    num_bars: u32,
    num_pies: u32,
    num_lines: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage> big_grid: array<u32>;
@group(0) @binding(2) var font_tex: texture_2d<f32>;
@group(0) @binding(3) var font_sampler: sampler;
@group(0) @binding(4) var<storage> bar_data: array<vec4f>;
@group(0) @binding(5) var<storage> pie_data: array<vec4f>;
@group(0) @binding(6) var<storage> line_data: array<vec4f>;
@group(0) @binding(7) var<storage> small_grid: array<u32>;

const BIG_CHAR_W: u32 = 14u;
const BIG_CHAR_H: u32 = 14u;
const BIG_ROW_H: u32 = 24u;
const SM_CHAR_W: u32 = 8u;
const SM_CHAR_H: u32 = 8u;
const SM_ROW_H: u32 = 12u;
const FONT_CHAR_W: u32 = 8u;
const FONT_CHAR_H: u32 = 8u;
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

fn is_glyph_set(char_code: u32, local_x: u32, local_y: u32, out_w: u32, out_h: u32) -> bool {
    if char_code < 32u || char_code > 126u { return false; }
    let idx = char_code - 32u;
    let atlas_col = idx % 16u;
    let atlas_row = idx / 16u;
    let src_x = local_x * FONT_CHAR_W / out_w;
    let src_y = local_y * FONT_CHAR_H / out_h;
    let tx = f32(atlas_col * FONT_CHAR_W + src_x) + 0.5;
    let ty = f32(atlas_row * FONT_CHAR_H + src_y) + 0.5;
    let tex_w = f32(textureDimensions(font_tex).x);
    let tex_h = f32(textureDimensions(font_tex).y);
    let sample = textureSampleLevel(font_tex, font_sampler, vec2(tx / tex_w, ty / tex_h), 0.0);
    return sample.r > 0.5;
}

fn render_big(px: u32, py: u32) -> vec4f {
    let col = px / BIG_CHAR_W;
    let row = py / BIG_ROW_H;
    if col >= params.big_cols || row >= params.big_rows { return vec4f(0.0); }
    let char_code = big_grid[row * params.big_cols + col];
    if char_code == 0u { return vec4f(0.0); }
    let local_x = px % BIG_CHAR_W;
    let local_y = py % BIG_ROW_H;
    if local_y >= BIG_CHAR_H { return vec4f(0.0); }
    if !is_glyph_set(char_code, local_x, local_y, BIG_CHAR_W, BIG_CHAR_H) { return vec4f(0.0); }
    return vec4f(0.85, 0.85, 0.85, 1.0);
}

fn render_small(px: u32, py: u32) -> vec4f {
    let col = px / SM_CHAR_W;
    let row = py / SM_ROW_H;
    if col >= params.small_cols || row >= params.small_rows { return vec4f(0.0); }
    let char_code = small_grid[row * params.small_cols + col];
    if char_code == 0u { return vec4f(0.0); }
    let local_x = px % SM_CHAR_W;
    let local_y = py % SM_ROW_H;
    if local_y >= SM_CHAR_H { return vec4f(0.0); }
    if !is_glyph_set(char_code, local_x, local_y, SM_CHAR_W, SM_CHAR_H) { return vec4f(0.0); }
    return vec4f(0.85, 0.85, 0.85, 1.0);
}

fn dist_to_seg(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let lsq = dx * dx + dy * dy;
    if lsq < 0.0001 { return sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1)); }
    let t = clamp(((px - x1) * dx + (py - y1) * dy) / lsq, 0.0, 1.0);
    let cx = x1 + t * dx;
    let cy = y1 + t * dy;
    return sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
}

fn render_bar(px: f32, py: f32) -> vec4f {
    for (var i = 0u; i < params.num_bars; i = i + 1u) {
        let bar = bar_data[i];
        if px >= bar.x && px < bar.x + bar.z && py >= bar.y && py < bar.y + bar.w {
            return bar_data[i + params.num_bars];
        }
    }
    return vec4f(0.0);
}

fn render_line(px: f32, py: f32) -> vec4f {
    for (var i = 0u; i < params.num_lines; i = i + 1u) {
        let seg = line_data[i];
        if dist_to_seg(px, py, seg.x, seg.y, seg.z, seg.w) < 1.5 {
            return line_data[i + params.num_lines];
        }
    }
    return vec4f(0.0);
}

fn render_pie(px: f32, py: f32) -> vec4f {
    let count = params.num_pies;
    if count == 0u { return vec4f(0.0); }
    let cx = pie_data[0u].x;
    let cy = pie_data[0u].y;
    let radius = pie_data[0u].z;
    let dx = px - cx;
    let dy = py - cy;
    let dist = sqrt(dx * dx + dy * dy);
    if dist > radius { return vec4f(0.0); }
    var angle = atan2(dy, dx);
    if angle < 0.0 { angle = angle + TAU; }
    var slice_idx = 0xFFFFFFFFu;
    var prev_end = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let end_angle = pie_data[i].w;
        if angle <= end_angle && angle > prev_end { slice_idx = i; }
        prev_end = end_angle;
    }
    if slice_idx == 0xFFFFFFFFu { return vec4f(0.0); }
    return pie_data[slice_idx + count];
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let px = pos.x;
    let py = pos.y;
    let ux = u32(px);
    let uy = u32(py);
    if ux >= params.screen_w || uy >= params.screen_h { discard; }

    let line_color = render_line(px, py);
    if line_color.a > 0.0 { return line_color; }

    let pie_color = render_pie(px, py);
    if pie_color.a > 0.0 { return pie_color; }

    let bar_color = render_bar(px, py);
    if bar_color.a > 0.0 { return bar_color; }

    // Big text (main body)
    let big = render_big(ux, uy);
    if big.a > 0.0 { return big; }

    // Small text (labels, axis annotations)
    let sm = render_small(ux, uy);
    if sm.a > 0.0 { return sm; }

    discard;
    return vec4f(0.0);
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var pos = array(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    return VertexOutput(vec4f(pos[vi], 0.0, 1.0));
}
