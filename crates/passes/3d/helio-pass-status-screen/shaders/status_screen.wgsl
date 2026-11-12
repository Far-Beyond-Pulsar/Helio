// The status screen: one fullscreen pass that draws a near-black field and one
// red dialog from a rounded-rectangle signed distance field, in physical
// pixels, so it is the same at any size and DPI. No images, no glow.
//
// Layers, back to front: the void with a soft vignette, the contact shadow,
// the panel (fill, 1px edge, top rim light), chips and rules, text from the
// engine's 8x8 glyph atlas, the icon, and grain last to stop banding.
//
// The structs below are the layout of `src/abi.rs`; `tests/abi.rs` compares
// them. Colours are sRGB (the design's), converted for an sRGB target at the end.

struct Params {
    screen: vec2<f32>,
    unit: f32,
    flags: u32,
    dialog: vec4<f32>,
    shape: vec4<f32>,
    glyphs: vec4<u32>,
    counts: vec4<u32>,
    text: vec4<f32>,
    icon: vec4<f32>,
    void_color: vec4<f32>,
    shadow: vec4<f32>,
    panel_a: vec4<f32>,
    panel_b: vec4<f32>,
    edge: vec4<f32>,
    rim: vec4<f32>,
    chip_border: vec4<f32>,
    icon_ring: vec4<f32>,
    icon_ink: vec4<f32>,
}

struct Line {
    place: vec4<f32>,
    color: vec4<f32>,
    text: vec4<u32>,
}

struct Rect {
    bounds: vec4<f32>,
    color: vec4<f32>,
    shape: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lines: array<Line>;
@group(0) @binding(2) var<storage, read> rects: array<Rect>;
@group(0) @binding(3) var<storage, read> chars: array<u32>;
@group(0) @binding(4) var font_tex: texture_2d<f32>;
@group(0) @binding(5) var font_sampler: sampler;

const FLAG_SRGB_TARGET: u32 = 1u;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return VertexOutput(vec4<f32>(corners[vertex], 0.0, 1.0));
}

// Signed distance to a rounded box centred on the origin, in pixels (< 0 inside).
fn sd_round_box(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let r = min(radius, min(half_size.x, half_size.y));
    let q = abs(p) - half_size + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Pixel coverage of a shape from its signed distance.
fn coverage(d: f32) -> f32 {
    return clamp(0.5 - d, 0.0, 1.0);
}

fn over(base: vec3<f32>, color: vec4<f32>, amount: f32) -> vec3<f32> {
    return mix(base, color.rgb, color.a * amount);
}

// Distance from `p` to the segment `a`-`b`.
fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// Coverage of the glyph in `code` at `uv` (0..1 over its cell). Sampled
// bilinearly inside the cell only, so neighbours in the atlas never bleed.
fn glyph_coverage(code: u32, uv: vec2<f32>) -> f32 {
    if code < params.glyphs.x {
        return 0.0;
    }
    let index = code - params.glyphs.x;
    let texels = f32(params.glyphs.z);
    let cell = vec2<f32>(f32(index % params.glyphs.y), f32(index / params.glyphs.y)) * texels;
    let inside = clamp(uv * texels, vec2<f32>(0.5), vec2<f32>(texels - 0.5));
    let size = vec2<f32>(textureDimensions(font_tex));
    return textureSampleLevel(font_tex, font_sampler, (cell + inside) / size, 0.0).r;
}

fn text_alpha(p: vec2<f32>, line: Line) -> f32 {
    let advance = line.place.z;
    let height = line.place.w;
    let rel = p - line.place.xy;
    if rel.y < 0.0 || rel.y >= height || rel.x < 0.0 {
        return 0.0;
    }
    let index = u32(floor(rel.x / advance));
    if index >= line.text.y {
        return 0.0;
    }
    let width = height * params.text.x;
    let gx = rel.x - f32(index) * advance - (advance - width) * 0.5;
    if gx < 0.0 || gx >= width {
        return 0.0;
    }
    let code = chars[line.text.x + index] & 0xffu;
    let c = glyph_coverage(code, vec2<f32>(gx / width, rel.y / height));
    // Bilinear coverage rises by one per texel: the edge is as wide as a
    // pixel is in texels.
    let texel_px = f32(params.glyphs.z) / min(width, height);
    let threshold = select(params.text.y, params.text.z, (line.text.z & 1u) != 0u);
    let w = max(0.5 * texel_px, 0.05);
    return smoothstep(threshold - w, threshold + w, c);
}

// The icon, drawn in a 48 unit square like the design's SVG.
fn icon_color(base: vec3<f32>, p: vec2<f32>) -> vec3<f32> {
    let size = params.icon.z;
    let k = size / 48.0;
    let q = (p - params.icon.xy) / k;
    if abs(q.x) > 26.0 || abs(q.y) > 26.0 {
        return base;
    }
    var color = base;
    let ring = abs(length(q) - 21.0);
    if params.icon.w < 0.5 {
        // Ring, an exclamation mark.
        color = over(color, params.icon_ring, coverage((ring - 1.1) * k));
        let bar = sd_segment(q, vec2<f32>(0.0, -10.5), vec2<f32>(0.0, 3.0)) - 1.6;
        let dot_ = length(q - vec2<f32>(0.0, 9.5)) - 2.2;
        color = over(color, params.icon_ink, coverage(min(bar, dot_) * k));
    } else {
        // Dashed ring (3 on, 5 off along the circle, from 3 o'clock, clockwise), a dash.
        let angle = atan2(q.y, q.x);
        let arc = (angle + select(0.0, 6.2831853, angle < 0.0)) * 21.0;
        let m = arc - floor(arc / 8.0) * 8.0;
        let dash = clamp(min(m, 3.0 - m) * k + 0.5, 0.0, 1.0);
        color = over(color, params.icon_ring, coverage((ring - 1.0) * k) * dash);
        let bar = sd_segment(q, vec2<f32>(-9.0, 0.0), vec2<f32>(9.0, 0.0)) - 1.5;
        color = over(color, params.icon_ink, coverage(bar * k));
    }
    return color;
}

fn hash(p: vec2<u32>) -> f32 {
    var h = p.x * 374761393u + p.y * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xffffu) / 65535.0;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy;

    // The void, with a soft vignette (0 at the centre, 1 at the corners).
    let corner = length((p / params.screen - vec2<f32>(0.5)) * 2.0) * 0.70710678;
    var color = params.void_color.rgb * (1.0 - params.void_color.a * smoothstep(0.6, 1.0, corner));

    let half_size = params.dialog.zw * 0.5;
    let center = params.dialog.xy + half_size;
    let radius = params.shape.x;
    let border = params.shape.y;
    let local = p - center;
    let d = sd_round_box(local, half_size, radius);

    // The contact shadow: a Gaussian falloff of the distance, offset down.
    let sigma = params.shape.z;
    let ds = sd_round_box(local - vec2<f32>(0.0, params.shape.w), half_size, radius);
    if ds < 3.0 * sigma {
        color = over(color, params.shadow, 0.5 * (1.0 - tanh(0.85 * ds / sigma)));
    }

    if d < 1.0 {
        // The panel: a 168 degree gradient.
        let dir = vec2<f32>(sin(2.9321531), -cos(2.9321531));
        let span = abs(params.dialog.z * dir.x) + abs(params.dialog.w * dir.y);
        let t = clamp(dot(local, dir) / span + 0.5, 0.0, 1.0);
        let panel = mix(params.panel_a.rgb, params.panel_b.rgb, t);
        let fill = coverage(d);
        var surface = panel;

        // The 1px edge, inside the shape.
        let inner = coverage(d + border);
        surface = over(surface, params.edge, 1.0 - inner);

        // The rim light: inside the edge, and not inside it moved down a pixel.
        let padding = coverage(d + border);
        let lowered = coverage(sd_round_box(local - vec2<f32>(0.0, 1.0), half_size, radius) + border);
        surface = over(surface, params.rim, padding * (1.0 - lowered));

        // Rectangles: chips, rules, dots.
        for (var i = 0u; i < params.counts.y; i = i + 1u) {
            let r = rects[i];
            if p.x < r.bounds.x - 2.0 || p.y < r.bounds.y - 2.0
                || p.x > r.bounds.x + r.bounds.z + 2.0 || p.y > r.bounds.y + r.bounds.w + 2.0 {
                continue;
            }
            let rc = r.bounds.xy + r.bounds.zw * 0.5;
            let rd = sd_round_box(p - rc, r.bounds.zw * 0.5, r.shape.x);
            var alpha = coverage(rd);
            if r.shape.y > 0.5 && r.shape.y < 1.5 {
                alpha = alpha * (1.0 - clamp((p.x - r.bounds.x) / r.bounds.z, 0.0, 1.0));
            }
            surface = over(surface, r.color, alpha);
            if r.shape.y > 1.5 {
                surface = over(surface, params.chip_border, alpha * (1.0 - coverage(rd + 1.0)));
            }
        }

        // Text.
        for (var i = 0u; i < params.counts.x; i = i + 1u) {
            let line = lines[i];
            let alpha = text_alpha(p, line);
            if alpha > 0.0 {
                surface = over(surface, line.color, alpha);
            }
        }

        surface = icon_color(surface, p);
        color = mix(color, surface, fill);
    }

    // Grain, last: about one 8-bit step, so the dark gradients do not band.
    color = color + vec3<f32>((hash(vec2<u32>(frag.xy)) - 0.5) * (1.5 / 255.0));
    color = max(color, vec3<f32>(0.0));
    if (params.flags & FLAG_SRGB_TARGET) != 0u {
        color = srgb_to_linear(color);
    }
    return vec4<f32>(color, 1.0);
}
