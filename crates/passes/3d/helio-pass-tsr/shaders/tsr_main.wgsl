// Temporal Super-Resolution (TSR) — compute-free fullscreen resolve
//
// Algorithm:
//   1. Depth-based reprojection → history UV
//   2. Pixel classification (LARGE_MOTION, DISOCCLUSION, SPECULAR_SHIMMER, EDGE)
//   3. Neighbourhood sampling in YCoCg space (5×5 tap for Quality/Native, 3×3 otherwise)
//   4. AABB clamping of the history sample
//   5. Adaptive temporal blend driven by classification
//   6. Display sharpening in a separate blit after storing this resolve
//
// References:
//   UE5 TSR — https://docs.unrealengine.com/en-US/temporal-super-resolution/
//   FSR 2.x — https://gpuopen.com/fidelityfx-super-resolution-2
//   Playdead temporal — https://github.com/playdeadgames/temporal (MIT)

// ── Classification bit flags ──────────────────────────────────────────────────
const CLASS_NONE:             u32 = 0u;
const CLASS_LARGE_MOTION:     u32 = 1u;   // |velocity| > threshold → shorter accumulation
const CLASS_DISOCCLUSION:     u32 = 2u;   // depth mismatch → discard history
const CLASS_SPECULAR_SHIMMER: u32 = 4u;   // luminance variance > threshold → smooth
const CLASS_EDGE:             u32 = 8u;   // depth discontinuity → aggressive clamping

// ── Constants ─────────────────────────────────────────────────────────────────
const C_POS_INFTY:              f32 = 1.0e32;
const C_NEG_INFTY:              f32 = -1.0e32;
const MIN_HISTORY_BLEND_RATE:   f32 = 0.04;
const MAX_HISTORY_BLEND_RATE:   f32 = 1.0;
const LARGE_MOTION_THRESHOLD:   f32 = 0.01;  // UV-space velocity magnitude
const DISOCCLUSION_THRESHOLD:   f32 = 0.01;  // relative perspective depth difference
const SHIMMER_VAR_THRESHOLD:    f32 = 0.03;
const EDGE_DEPTH_THRESHOLD:     f32 = 0.02;

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var current_frame:  texture_2d<f32>;  // pre-AA at internal res
@group(0) @binding(1) var history_frame:  texture_2d<f32>;  // previous TSR output at output res
@group(0) @binding(2) var depth_tex:      texture_depth_2d; // depth at internal res
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var point_sampler:  sampler;

struct CameraUniforms {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}
@group(0) @binding(5) var<storage, read> cameras: array<CameraUniforms, 2>;

struct TsrUniform {
    jitter_offset:  vec2<f32>, // sub-pixel jitter in [-0.5, 0.5)
    reactivity:     f32,       // 0 = full history, 1 = no history
    reset:          u32,       // 1 on first frame / after reset_history()
    time_delta:     f32,       // seconds since last frame
    tap_radius:     u32,       // 1 = 3×3, 2 = 5×5
    previous_jitter: vec2<f32>, // previous projection translation in render pixels
    clip_to_previous: mat4x4<f32>, // composed in f64 on the CPU
}
@group(0) @binding(6) var<uniform> tsr: TsrUniform;
@group(0) @binding(7) var history_depth: texture_2d<f32>; // centered RG32Float depth interval, never color alpha
@group(0) @binding(8) var history_mean: texture_2d<f32>; // raw sample YCoCg mean, A=count
@group(0) @binding(9) var history_variance: texture_2d<f32>;

// The default graph uses forward device depth. Compare both surfaces in the
// PREVIOUS projection, including camera translation. The 1-depth scale avoids
// accepting nearly every occlusion once device depth approaches one. The small
// absolute floor covers f32 projection rounding, not a world-space LOD change.
fn history_depth_matches(expected: f32, stored: f32) -> bool {
    if (expected >= 1.0) != (stored >= 1.0) { return false; }
    return abs(expected - stored) <= max(0.00000012, (1.0 - min(expected, stored)) * DISOCCLUSION_THRESHOLD);
}

// ── Vertex passthrough ────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// ── Colour space helpers ──────────────────────────────────────────────────────

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(rgb, vec3<f32>( 0.25,  0.5,  0.25)),
        dot(rgb, vec3<f32>( 0.5,   0.0, -0.5 )),
        dot(rgb, vec3<f32>(-0.25,  0.5, -0.25)),
    );
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ycocg.x + ycocg.y - ycocg.z,
        ycocg.x            + ycocg.z,
        ycocg.x - ycocg.y - ycocg.z,
    );
}

// ── Reversible Reinhard tonemapper ────────────────────────────────────────────

fn max3(v: vec3<f32>) -> f32 { return max(v.r, max(v.g, v.b)); }
fn tonemap(c: vec3<f32>)         -> vec3<f32> { return c / (max3(c) + 1.0); }
fn reverse_tonemap(c: vec3<f32>) -> vec3<f32> { return c / (1.0 - max3(c) + 1.0e-8); }

// Reconstruct color and validity from the same Catmull-Rom footprint. A lone
// nearest-depth test rejects ordinary edge coverage; color filtering without
// geometric validity leaks unrelated surfaces. Signed cubic weights preserve
// resolved detail without applying a second bilinear blur to every frame.
fn cubic_weights(f:f32)->vec4<f32> {
    return vec4<f32>(f*(-0.5+f*(1.0-0.5*f)),1.0+f*f*(-2.5+1.5*f),
        f*(0.5+f*(2.0-1.5*f)),f*f*(-0.5+0.5*f));
}
struct HistorySample {
    color:vec3<f32>, weight:f32,
    mean:vec3<f32>, count:f32,
    variance:vec3<f32>,
}
fn sample_geometric_history(uv: vec2<f32>, expected_depth: f32) -> HistorySample {
    let dims = vec2<i32>(textureDimensions(history_frame));
    let point = uv * vec2<f32>(dims) - 0.5;
    let base = vec2<i32>(floor(point));
    let fraction = fract(point);
    let wx=cubic_weights(fraction.x);let wy=cubic_weights(fraction.y);
    var color = vec3<f32>(0.0);
    var weight = 0.0;
    var mean_sum=vec4<f32>(0.0);var second_sum=vec3<f32>(0.0);var moment_weight=0.0;
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let pixel = clamp(base + vec2<i32>(x-1,y-1), vec2<i32>(0), dims-1);
            let depth_range = textureLoad(history_depth,pixel,0).rg;
            let stored_depth = clamp(expected_depth,depth_range.x,depth_range.y);
            let w = wx[x]*wy[y];
            if history_depth_matches(expected_depth,stored_depth) {
                color += textureLoad(history_frame,pixel,0).rgb * w;
                weight += w;
                // Pool moments with positive weights. Signed cubic colour
                // reconstruction must not create a negative sample variance.
                let mw=max(w,0.0);let mean=textureLoad(history_mean,pixel,0);
                let variance=textureLoad(history_variance,pixel,0).rgb;
                mean_sum+=mean*mw;
                second_sum+=(variance+mean.rgb*mean.rgb)*mw;
                moment_weight+=mw;
            }
        }
    }
    let mean=mean_sum/max(moment_weight,0.00001);
    let variance=max(second_sum/max(moment_weight,0.00001)-mean.rgb*mean.rgb,vec3<f32>(0.0));
    return HistorySample(max(color/max(weight,0.00001),vec3<f32>(0.0)),weight,mean.rgb,mean.a,variance);
}

// ── Neighbourhood statistics ──────────────────────────────────────────────────

struct Neighbourhood {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    avg:      vec3<f32>,
    variance: f32,      // luminance variance
    depth_range: vec2<f32>, // min, max depth in neighbourhood
}

// Gather neighbourhood statistics in tonemapped YCoCg space.
// tap_radius: 1 = 3×3 (9 samples), 2 = 5×5 (25 samples).
fn gather_neighbourhood(
    tex: texture_2d<f32>,
    depth: texture_depth_2d,
    uv: vec2<f32>,
    texel: vec2<f32>,
    tap_radius: i32,
) -> Neighbourhood {
    var aabb_min = vec3<f32>(C_POS_INFTY);
    var aabb_max = vec3<f32>(C_NEG_INFTY);
    var w_sum   = 0.0;
    var l1      = vec3<f32>(0.0);
    var l2      = vec3<f32>(0.0);
    var d_min   = C_POS_INFTY;
    var d_max   = C_NEG_INFTY;

    for (var y = -tap_radius; y <= tap_radius; y++) {
        for (var x = -tap_radius; x <= tap_radius; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let s  = textureSampleLevel(tex, point_sampler, uv + offset, 0.0).rgb;
            let q  = rgb_to_ycocg(tonemap(s));
            // Distance-based weight (centre-heavy)
            let dist = abs(f32(x)) + abs(f32(y));
            let w  = exp(-0.5 * dist);

            aabb_min = min(aabb_min, q);
            aabb_max = max(aabb_max, q);
            w_sum   += w;
            l1      += w * q;
            l2      += w * q * q;

            let d = textureSample(depth, point_sampler, uv + offset);
            d_min = min(d_min, d);
            d_max = max(d_max, d);
        }
    }

    l1 /= w_sum;
    l2 /= w_sum;

    let variance_vec = max(l2 - l1 * l1, vec3<f32>(0.0));
    let luma_variance = variance_vec.x; // Y channel variance = luminance variance

    var n: Neighbourhood;
    n.aabb_min   = aabb_min;
    n.aabb_max   = aabb_max;
    n.avg        = l1;
    n.variance   = luma_variance;
    n.depth_range = vec2<f32>(d_min, d_max);
    return n;
}

// ── Pixel classification ──────────────────────────────────────────────────────

// Classify this pixel and return a bitmask of CLASS_* flags.
fn classify_pixel(
    velocity:      vec2<f32>, // screen-space velocity (UV per frame)
    valid_depth:   bool,
    n:             Neighbourhood,
) -> u32 {
    var flags = CLASS_NONE;

    // Large motion: sub-pixel accumulation breaks down
    if length(velocity) > LARGE_MOTION_THRESHOLD {
        flags |= CLASS_LARGE_MOTION;
    }

    // Disocclusion: reprojected history depth mismatches current depth
    if !valid_depth {
        flags |= CLASS_DISOCCLUSION;
    }

    // Specular shimmer: high luminance variance in neighbourhood
    if n.variance > SHIMMER_VAR_THRESHOLD {
        flags |= CLASS_SPECULAR_SHIMMER;
    }

    // Edge: large depth range in neighbourhood
    if (n.depth_range.y - n.depth_range.x) > EDGE_DEPTH_THRESHOLD {
        flags |= CLASS_EDGE;
    }

    return flags;
}

// ── Adaptive blend factor ─────────────────────────────────────────────────────

// Maps classification flags to a [MIN, MAX] blend factor.
// blend_factor = how much weight the CURRENT frame gets (1 = no history).
fn compute_blend_factor(
    flags:      u32,
    reactivity: f32,
    time_delta: f32,
    history_ycocg: vec3<f32>,
    n: Neighbourhood,
) -> f32 {
    var base = MIN_HISTORY_BLEND_RATE;

    // Forced fast-blend cases
    if (flags & CLASS_DISOCCLUSION) != 0u {
        return 1.0;
    } else if (flags & CLASS_LARGE_MOTION) != 0u {
        base = 0.25;
    }

    // Clamp history luminance to AABB; if already inside, keep history
    let clamped = clamp(history_ycocg, n.aabb_min, n.aabb_max);
    let dist    = length(history_ycocg - clamped);
    // Extra push toward current when history is far outside the AABB
    let aabb_push = clamp(dist * 8.0, 0.0, 0.5);
    base += aabb_push;

    // Shimmer: temporal smoothing (reduce blend so history accumulates)
    if (flags & CLASS_SPECULAR_SHIMMER) != 0u {
        base = max(base - 0.05, MIN_HISTORY_BLEND_RATE);
    }

    // Reactivity override (camera cut / scene change via set_reactivity)
    base = mix(base, MAX_HISTORY_BLEND_RATE, reactivity);

    // Frame-rate independent adaptation: more blend if long time has passed
    let time_factor = 1.0 - exp(-time_delta * 4.0);
    base = mix(base, 0.5, time_factor * 0.5);

    return clamp(base, MIN_HISTORY_BLEND_RATE, MAX_HISTORY_BLEND_RATE);
}

// Display sharpening runs after the unsharpened resolve is stored.

// ── Main fragment shader ──────────────────────────────────────────────────────

// Current color and stored history are centered images, while both camera
// matrices include projection jitter. Deproject the current sample in its
// jittered image, then remove previous projection jitter from the history UV.
fn reproject_history_point(center_uv:vec2<f32>, depth:f32, in_dims:vec2<f32>)->vec3<f32> {
    let current_uv=center_uv+tsr.jitter_offset*vec2<f32>(1.0,-1.0)/in_dims;
    let ndc=vec2<f32>(current_uv.x*2.0-1.0,1.0-current_uv.y*2.0);
    let previous=tsr.clip_to_previous*vec4<f32>(ndc,depth,1.0);
    if previous.w<=0.0 {return vec3<f32>(-1.0);}
    let previous_ndc=previous.xy/previous.w;
    let previous_uv=vec2<f32>(previous_ndc.x*0.5+0.5,0.5-previous_ndc.y*0.5);
    return vec3<f32>(previous_uv-tsr.previous_jitter*vec2<f32>(1.0,-1.0)/in_dims, previous.z/previous.w);
}
fn reproject_history(center_uv:vec2<f32>, depth:f32, in_dims:vec2<f32>)->vec2<f32> {
    return reproject_history_point(center_uv,depth,in_dims).xy;
}
// Depth is a point sample from the internal raster, whereas the resolved
// color pixel is centered and may lie between raster samples. Obtain motion
// from the actual depth texel, then carry that displacement to the color pixel.
// Deprojecting point-sampled depth at a different UV invents a different surface.
fn reproject_sample_motion(center_uv:vec2<f32>, depth:f32, in_dims:vec2<f32>)->vec3<f32> {
    let jitter_uv=tsr.jitter_offset*vec2<f32>(1.0,-1.0)/in_dims;
    let pixel=clamp(floor((center_uv+jitter_uv)*in_dims),vec2<f32>(0.0),in_dims-1.0);
    let sample_center=(pixel+0.5)/in_dims-jitter_uv;
    let previous=reproject_history_point(sample_center,depth,in_dims);
    return vec3<f32>(previous.xy+center_uv-sample_center,previous.z);
}
struct ResolveOutput {
    @location(0) color: vec4<f32>,
    @location(1) depth: vec2<f32>,
    @location(2) mean: vec4<f32>,
    @location(3) variance: vec4<f32>,
}
fn fresh_resolve(color:vec3<f32>,depth:vec2<f32>)->ResolveOutput {
    let sample=rgb_to_ycocg(tonemap(color));
    return ResolveOutput(vec4<f32>(color,1.0),depth,vec4<f32>(sample,1.0),vec4<f32>(0.0));
}

// The current color reconstruction is bilinear. Retain the depth range of
// exactly its contributing texels, including partial voxel/sky coverage. A
// single nearest depth cannot describe that mixed image sample. This range
// changes history validity only; it does not change the current geometry.
fn current_depth_range(uv:vec2<f32>)->vec2<f32> {
    let dims=vec2<i32>(textureDimensions(depth_tex));
    let point=uv*vec2<f32>(dims)-0.5;
    let base=vec2<i32>(floor(point));let fraction=fract(point);
    var lo=1.0;var hi=0.0;
    for(var y=0;y<2;y++) {for(var x=0;x<2;x++) {
        let weight=select(1.0-fraction.x,fraction.x,x==1)*select(1.0-fraction.y,fraction.y,y==1);
        if weight>0.00001 {
            let pixel=clamp(base+vec2<i32>(x,y),vec2<i32>(0),dims-1);
            let depth=textureLoad(depth_tex,pixel,0);
            lo=min(lo,depth);hi=max(hi,depth);
        }
    }}
    return vec2<f32>(lo,hi);
}

@fragment
fn fs_main(in: VertexOutput) -> ResolveOutput {
    let in_dims  = vec2<f32>(textureDimensions(current_frame));
    let out_dims = vec2<f32>(textureDimensions(history_frame));
    let in_texel = 1.0 / in_dims;

    // ── Jitter correction ─────────────────────────────────────────────────────
    let jitter_uv = tsr.jitter_offset * vec2<f32>(1.0, -1.0) / in_dims;
    let cur_uv    = in.uv + jitter_uv;

    // ── Current frame sample (jitter-corrected) ───────────────────────────────
    let current_rgb = textureSampleLevel(current_frame, linear_sampler, cur_uv, 0.0).rgb;

    let depth_val = textureSample(depth_tex, point_sampler, cur_uv);
    let depth_range = current_depth_range(cur_uv);

    // ── RESET path ────────────────────────────────────────────────────────────
    if tsr.reset != 0u {
        return fresh_resolve(current_rgb, depth_range);
    }

    // ── Depth-based reprojection → history UV ─────────────────────────────────
    let previous = reproject_sample_motion(in.uv, depth_val, in_dims);
    let history_uv = previous.xy;

    // If reprojected UV is out of screen, use current frame only
    if any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0)) {
        return fresh_resolve(current_rgb, depth_range);
    }

    let history = sample_geometric_history(history_uv,previous.z);
    let history_rgb = history.color;

    // ── Neighbourhood statistics ───────────────────────────────────────────────
    let tap_radius = i32(tsr.tap_radius);
    var n = gather_neighbourhood(current_frame, depth_tex, cur_uv, in_texel, tap_radius);
    if history.weight>0.01 && history.count>=2.0 {
        // A single spatial neighbourhood can omit valid subpixel face colours.
        // Retain measured temporal coverage noise in both clipping and blend
        // validation, while geometrically invalid history remains rejected.
        let allowance=3.0*sqrt(history.variance);
        n.aabb_min-=allowance;n.aabb_max+=allowance;
    }

    // ── Screen-space velocity (UV-space) ──────────────────────────────────────
    let velocity = history_uv - in.uv;

    // ── Pixel classification ──────────────────────────────────────────────────
    let flags = classify_pixel(velocity, history.weight > 0.01, n);

    // ── Tonemap for stable accumulation ───────────────────────────────────────
    let current_tm  = rgb_to_ycocg(tonemap(current_rgb));
    let history_tm  = rgb_to_ycocg(tonemap(history_rgb));

    // ── AABB clamp (YCoCg) → prevents ghosting ─────────────────────────────────
    var aabb_min = n.aabb_min;
    var aabb_max = n.aabb_max;

    // Widen AABB slightly for specular shimmer (avoid over-clamping moving highlights)
    if (flags & CLASS_SPECULAR_SHIMMER) != 0u {
        let widen = vec3<f32>(0.03);
        aabb_min -= widen;
        aabb_max += widen;
    }

    // Narrow AABB on edges to reduce bleeding across depth discontinuities
    if (flags & CLASS_EDGE) != 0u {
        let avg = (aabb_min + aabb_max) * 0.5;
        let half_extent = (aabb_max - aabb_min) * 0.35;
        aabb_min = avg - half_extent;
        aabb_max = avg + half_extent;
    }

    let clamped_history = clamp(history_tm, aabb_min, aabb_max);

    // ── Adaptive blend ─────────────────────────────────────────────────────────
    let blend = compute_blend_factor(flags, tsr.reactivity, tsr.time_delta, history_tm, n);

    // ── Blend ─────────────────────────────────────────────────────────────────
    let blended_ycocg = mix(clamped_history, current_tm, blend);
    let blended_rgb   = ycocg_to_rgb(blended_ycocg);
    let result_linear = reverse_tonemap(clamp(blended_rgb, vec3<f32>(0.0), vec3<f32>(1.0)));

    // The display blit sharpens this image without modifying history.
    let count=select(1.0,min(history.count+1.0,32.0),history.weight>0.01 && tsr.reactivity<1.0);
    let moment_blend=max(1.0/count,tsr.reactivity);
    let delta=current_tm-history.mean;
    let mean=mix(history.mean,current_tm,moment_blend);
    let variance=(1.0-moment_blend)*(history.variance+moment_blend*delta*delta);
    return ResolveOutput(vec4<f32>(result_linear, 1.0), depth_range,
        vec4<f32>(mean,count),vec4<f32>(variance,0.0));
}
