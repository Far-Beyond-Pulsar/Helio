// The editor's selection outline, over the final image.
//
// The mask holds the selected geometry's depth (reversed Z, 0 where nothing
// selected was drawn), unjittered, at the internal resolution; the scene
// depth is at the internal resolution too. Both are read by the output
// pixel's UV, and the outline's width is in output pixels, so it holds its
// width whatever the internal resolution.
//
// A pixel outside the selection is on the outline when a selected texel lies
// within `WIDTH` output pixels of it (the line sits outside the silhouette,
// as Unreal's does). The nearest such selected depth is compared with the
// scene's at the pixel: in front of it, the line is drawn at full strength;
// behind it (something in front hides the selected object there), dimmed.

@group(0) @binding(0) var selection_mask: texture_2d<f32>;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;

// The line: yellow, two output pixels wide, a third as strong where hidden.
const COLOR: vec3<f32> = vec3<f32>(1.0, 0.78, 0.06);
const WIDTH: f32 = 2.0;
const HIDDEN_ALPHA: f32 = 0.35;
// Reversed Z: a selected surface within this fraction of the scene's depth
// counts as the visible surface (the mask is unjittered, the scene is not).
const DEPTH_TOLERANCE: f32 = 0.002;

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VSOut;
    out.clip_pos = vec4<f32>(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

/// The depth texel under `uv` (0 outside the texture).
fn depth_at(tex: texture_2d<f32>, uv: vec2<f32>) -> f32 {
    if any(uv < vec2(0.0)) || any(uv >= vec2(1.0)) {
        return 0.0;
    }
    let size = vec2<i32>(textureDimensions(tex));
    let p = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2(0), size - 1);
    return textureLoad(tex, p, 0).r;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Inside the selection: the object itself shows, untouched.
    if depth_at(selection_mask, in.uv) > 0.0 {
        discard;
    }
    // One output pixel in UV (the full-screen triangle's UV is linear).
    let pixel = vec2<f32>(abs(dpdx(in.uv.x)), abs(dpdy(in.uv.y)));
    // The nearest selected depth within `WIDTH` pixels: two rings of eight.
    var nearest = 0.0;
    for (var ring = 1; ring <= 2; ring = ring + 1) {
        let r = WIDTH * f32(ring) / 2.0;
        for (var i = 0; i < 8; i = i + 1) {
            let angle = f32(i) * 0.7853982;
            let offset = vec2<f32>(cos(angle), sin(angle)) * r * pixel;
            nearest = max(nearest, depth_at(selection_mask, in.uv + offset));
        }
    }
    if nearest <= 0.0 {
        discard;
    }
    let scene = depth_at(scene_depth, in.uv);
    let visible = nearest >= scene * (1.0 - DEPTH_TOLERANCE);
    return vec4<f32>(COLOR, select(HIDDEN_ALPHA, 1.0, visible));
}
