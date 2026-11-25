// How a surface reflects light (`helio_frame_api::gbuffer::BSDF_SNIPPET`,
// `//!use frame_bsdf`, with `//!use frame_surface`): the BSDF of a
// `SurfaceData`, the one every lighting model, every hit's consumer and the
// reference path tracer evaluate, so a pixel and a hit are lit alike.
//
//   bsdf_eval(s, wo, wi)      f(wo, wi) |cos(n, wi)|: the radiance leaving
//                             toward `wo` per unit radiance arriving from `wi`
//                             over a unit solid angle (what a light's
//                             radiance is multiplied by)
//   bsdf_pdf(s, wo, wi)       the solid-angle density `bsdf_sample` draws
//                             `wi` with (for multiple importance sampling)
//   bsdf_sample(s, wo, u)     one direction from three uniform numbers, its
//                             weight `bsdf_eval / bsdf_pdf` and its density
//   bsdf_diffuse_reflectance(s, cos_o)
//                             the diffuse lobe's reflectance of light from
//                             every direction above (an irradiance's factor)
//   bsdf_prepare(s), bsdf_eval_prepared(p, wo, wi)
//                             `bsdf_eval` with the surface's frame made once,
//                             for a model that lights a point by many lights
//                             (the deferred lighting: building the frame a
//                             light cost it 0.1 ms in indoor_cathedral_gi)
//
// `wo` and `wi` are unit vectors leaving the surface (toward the eye, toward
// the light), about the surface's normal as it is: a writer that knows the
// side a surface is seen from turns a two-sided surface's normal toward it
// (the G-buffer's drawers by `front_facing`, a ray's consumer by the hit's
// face: `bsdf_facing`).
//
// The model is the engine's lighting model as it was before this contract
// (the deleted `pbr_eval.wgsl` and the deferred lighting's
// `pbr_direct_light`), term for term, so moving it here changed no image:
//
// - **Specular**: GGX with alpha = roughness^2 and `+1e-4` in the
//   distribution's denominator, Smith-Schlick masking with Karis's analytic
//   remapping `k = (roughness + 1)^2 / 8`, `D G F / (4 n.wo n.wi + 1e-4)`;
//   `SURFACE_FLAG_ANISOTROPIC`: the anisotropic GGX of
//   `roughness_aniso_x/_y` along a tangent turned by `aniso_rotation`, with
//   its own masking. Schlick's Fresnel from `specular_f0` at `wo.h`.
// - **Diffuse**: Lambert, `albedo (1 - metallic) (1 - F(wo.h)) / pi`;
//   `SURFACE_FLAG_SUBSURFACE` wraps it (`w = 0.2 + 0.4 (1 - roughness)`)
//   and mixes in `subsurface_color / pi`.
// - **Through a thin two-sided surface** (`SURFACE_FLAG_TWO_SIDED`): light
//   from behind passes as `subsurface_color / pi` (Lambert on the far side).
//
// Changed from that model only where it returned more light than it
// received (the furnace test, `tests/bsdf_gpu.rs`):
//
// 1. **Grazing reflectance**: Schlick's grazing reflectance is 1 for any
//    real material and none for a reflectance under 2% (`bsdf_fresnel`;
//    Unreal's `F_Schlick`). The old model gave an F0-0 surface a grazing
//    specular lobe on top of its full diffuse one: 4.8% gained.
// 2. **The diffuse lobe's share**: `1 - F(wo.h)` leaves the diffuse lobe
//    more than the specular one takes at grazing views of mid-rough
//    dielectrics (1.8% gained at F0 0.04, 3.4% at 0.02, roughness 0.3,
//    n.wo 0.2). The diffuse lobe is scaled by `1 - bsdf_diffuse_loss`, a
//    bound on that gain measured offline over F0, roughness and n.wo; it is
//    0 at normal views, rough and smooth surfaces and F0 past 7%.
// 3. **Wrapped diffuse** (`SURFACE_FLAG_SUBSURFACE`): the wrapped lobe is
//    normalised (`(1 + w) / (1 + 2w)`: it returned `(1 + 2w) / (1 + w)`,
//    up to 37% more), the specular lobe keeps its own cosine, and the
//    scattered light is what the specular lobe leaves (`1 - F`).
//
// Sampling: the diffuse and through lobes by the cosine, the specular lobe
// by GGX's visible normals (Heitz 2018; spherical caps, Dupuy and Benyoub
// 2023). Importance sampling needs no particular BSDF: every density is the
// mixture's and a sample's weight is `bsdf_eval / bsdf_pdf`, so the estimate
// is unbiased for the function evaluated whichever lobe drew it. No
// multiple scattering between microfacets: rough metals reflect less than
// they receive.

const BSDF_PI: f32 = 3.14159265358979;
// The smoothest surface sampled: a delta lobe would make the visible normals
// a single direction (the evaluation takes roughness as it is).
const BSDF_MIN_ROUGHNESS: f32 = 0.045;

// `bsdf_sample`'s lobes.
const BSDF_LOBE_NONE: u32 = 0u;
const BSDF_LOBE_DIFFUSE: u32 = 1u;
const BSDF_LOBE_SPECULAR: u32 = 2u;
const BSDF_LOBE_THROUGH: u32 = 3u;

struct BsdfSample {
    // Leaving the surface (unit), toward where the light comes from.
    wi: vec3<f32>,
    // `bsdf_eval(s, wo, wi) / bsdf_pdf(s, wo, wi)`; zero: no sample.
    weight: vec3<f32>,
    // `bsdf_pdf(s, wo, wi)`.
    pdf: f32,
    // `BSDF_LOBE_*` that drew it.
    lobe: u32,
}

// The surface's frame: tangent, bitangent, normal (unit, frame space). The
// tangent is the anisotropic one (`aniso_rotation` about the normal from the
// projection of +y, or +x where the normal is near +y) for an anisotropic
// surface, any for the others.
struct BsdfFrame {
    t: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
}

fn bsdf_frame(s: SurfaceData) -> BsdfFrame {
    let n = s.normal;
    if (s.flags & SURFACE_FLAG_ANISOTROPIC) != 0u {
        var t = cross(cross(n, vec3<f32>(0.0, 1.0, 0.0)), n);
        if dot(t, t) < 1e-6 {
            t = cross(cross(n, vec3<f32>(1.0, 0.0, 0.0)), n);
        }
        t = normalize(t);
        let b = cross(n, t);
        let turned = t * cos(s.aniso_rotation) + b * sin(s.aniso_rotation);
        return BsdfFrame(turned, cross(n, turned), n);
    }
    // Duff et al. 2017, "Building an Orthonormal Basis, Revisited".
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let c = n.x * n.y * a;
    return BsdfFrame(vec3<f32>(1.0 + sign * n.x * n.x * a, sign * c, -sign * n.x), vec3<f32>(c, sign + n.y * n.y * a, -n.y), n);
}

fn bsdf_to_local(f: BsdfFrame, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(v, f.t), dot(v, f.b), dot(v, f.n));
}

fn bsdf_to_world(f: BsdfFrame, v: vec3<f32>) -> vec3<f32> {
    return f.t * v.x + f.b * v.y + f.n * v.z;
}

// `s` as seen from `wo` on a hit whose geometric normal is `face`: a
// two-sided surface met from behind turned toward `wo`, as a drawer turns
// it by `front_facing`.
fn bsdf_facing(s: SurfaceData, face: vec3<f32>, wo: vec3<f32>) -> SurfaceData {
    var out = s;
    if (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u && dot(face, wo) < 0.0 {
        out.normal = -s.normal;
    }
    return out;
}

// The alphas `bsdf_sample` draws visible normals with (along the tangent,
// along the bitangent); also a ray cone's spread for a specular bounce.
fn bsdf_alphas(s: SurfaceData) -> vec2<f32> {
    let least = BSDF_MIN_ROUGHNESS * BSDF_MIN_ROUGHNESS;
    if (s.flags & SURFACE_FLAG_ANISOTROPIC) != 0u {
        return max(vec2<f32>(s.roughness_aniso_x, s.roughness_aniso_y), vec2<f32>(least));
    }
    let r = max(s.roughness, BSDF_MIN_ROUGHNESS);
    return vec2<f32>(r * r);
}

fn bsdf_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn bsdf_pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

// Schlick's Fresnel at `cos_theta` from reflectance `f0` at normal
// incidence, toward a grazing reflectance of 1 for any real material and of
// nothing for a reflectance under 2% (fix 1).
fn bsdf_fresnel(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let f90 = clamp(50.0 * max(f0.r, max(f0.g, f0.b)), 0.0, 1.0);
    return f0 + (f90 - f0) * bsdf_pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

// The share of the diffuse lobe taken off so that it and the specular lobe
// together return no more than they receive (fix 2): a bound, over
// roughness, `n.wo` and the reflectance, on what `1 - F(wo.h)` over-counts,
// measured by quadrature of this model. Zero almost everywhere; at most
// 3.9% (F0 2%, roughness 0.3, n.wo 0.05 to 0.25).
fn bsdf_diffuse_loss(s: SurfaceData, cos_o: f32) -> f32 {
    let f0 = max(s.specular_f0.r, max(s.specular_f0.g, s.specular_f0.b));
    let reflectance = min(f0 / 0.02, max((0.068 - f0) / 0.048, 0.0));
    let r = s.roughness;
    let by_roughness = clamp((r - 0.195) / 0.075, 0.0, 1.0) * clamp((0.66 - r) / 0.34, 0.0, 1.0);
    let by_view = min(0.58 + 12.0 * cos_o, 1.0) * clamp((0.58 - cos_o) / 0.32, 0.0, 1.0);
    return 0.0385 * reflectance * by_roughness * by_view;
}

// The diffuse lobe's reflectance under light from every direction above
// the surface, seen from `cos_o` (n.wo): what an irradiance is multiplied by
// (a lighting model's indirect diffuse and ambient light, which come from
// all directions), as `bsdf_eval`'s diffuse lobe integrates it. Its
// `1 - F(wo.h)` averaged over cosine-distributed light is
// `1 - f0 - (f90 - f0) P(cos_o)`, P the mean of `(1 - wo.h)^5`:
// `0.0283 (1 - cos_o)^2.5` within 0.001 of quadrature (0.028 at grazing
// views, below 0.0001 at normal ones). A wrapped lobe (`SURFACE_FLAG_SUBSURFACE`)
// takes light by `(cos + w) / (1 + 2w)`: a cosine share `1 / (1 + 2w)` and
// a uniform share `2w / (1 + 2w)`, whose mean of `(1 - wo.h)^5` is
// `0.048 (1 - cos_o)^2` within 0.001. Metals reflect none of it.
fn bsdf_diffuse_reflectance(s: SurfaceData, cos_o: f32) -> vec3<f32> {
    let f90 = clamp(50.0 * max(s.specular_f0.r, max(s.specular_f0.g, s.specular_f0.b)), 0.0, 1.0);
    let o = clamp(1.0 - cos_o, 0.0, 1.0);
    var p = 0.0283 * pow(o, 2.5);
    var albedo = s.albedo.rgb;
    if (s.flags & SURFACE_FLAG_SUBSURFACE) != 0u {
        albedo = mix(albedo, s.subsurface_color, 0.5);
        let w = bsdf_wrap(s);
        p = (p + 2.0 * w * 0.048 * o * o) / (1.0 + 2.0 * w);
    }
    let f = s.specular_f0 + (f90 - s.specular_f0) * p;
    return albedo * (1.0 - f) * (1.0 - s.metallic) * (1.0 - bsdf_diffuse_loss(s, cos_o));
}

// A surface with its frame, made once where many lights light it (a
// lighting model's pixel): `bsdf_eval_prepared` then builds no frame a
// light. `bsdf_eval(s, ...)` is `bsdf_eval_prepared(bsdf_prepare(s), ...)`.
struct BsdfPrepared {
    s: SurfaceData,
    frame: BsdfFrame,
}

fn bsdf_prepare(s: SurfaceData) -> BsdfPrepared {
    return BsdfPrepared(s, bsdf_frame(s));
}

// GGX's specular lobe times the cosine, `F` given, the half vector `h`.
fn bsdf_specular(p: BsdfPrepared, cos_o: f32, cos_i: f32, h: vec3<f32>, f: vec3<f32>) -> vec3<f32> {
    let s = p.s;
    let n = s.normal;
    let cos_h = max(dot(n, h), 0.0);
    var d: f32;
    var g: f32;
    if (s.flags & SURFACE_FLAG_ANISOTROPIC) != 0u {
        // `c^2 + (1 - c^2)(cos^2 phi / ax^2 + sin^2 phi / ay^2)` with phi the
        // half vector's angle about the normal: its tangent and bitangent
        // components carry `sin theta cos phi` and `sin theta sin phi`.
        let ht = dot(h, p.frame.t);
        let hb = dot(h, p.frame.b);
        let ax = s.roughness_aniso_x;
        let ay = s.roughness_aniso_y;
        let e = cos_h * cos_h + ht * ht / (ax * ax) + hb * hb / (ay * ay);
        d = 1.0 / (BSDF_PI * ax * ay * e * e + 0.0001);
        g = cos_o / (cos_o + sqrt(ax * ax + (1.0 - ax * ax) * cos_o * cos_o) + 0.0001)
            * (cos_i / (cos_i + sqrt(ay * ay + (1.0 - ay * ay) * cos_i * cos_i) + 0.0001));
    } else {
        let a = s.roughness * s.roughness;
        let a2 = a * a;
        let e = cos_h * cos_h * (a2 - 1.0) + 1.0;
        d = a2 / (BSDF_PI * e * e + 0.0001);
        let r = s.roughness + 1.0;
        let k = r * r / 8.0;
        g = cos_o / (cos_o * (1.0 - k) + k + 0.0001) * (cos_i / (cos_i * (1.0 - k) + k + 0.0001));
    }
    return f * (d * g / (4.0 * cos_o * cos_i + 0.0001) * cos_i);
}

// The wrap of a subsurface surface's diffuse lobe.
fn bsdf_wrap(s: SurfaceData) -> f32 {
    return 0.2 + 0.4 * (1.0 - s.roughness);
}

fn bsdf_eval(s: SurfaceData, wo: vec3<f32>, wi: vec3<f32>) -> vec3<f32> {
    return bsdf_eval_prepared(bsdf_prepare(s), wo, wi);
}

fn bsdf_eval_prepared(p: BsdfPrepared, wo: vec3<f32>, wi: vec3<f32>) -> vec3<f32> {
    let s = p.s;
    let n = s.normal;
    let cos_i = dot(n, wi);
    if cos_i <= 0.0 {
        // From behind: only what a thin two-sided surface lets through.
        if (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u && cos_i < -1e-4 {
            return s.subsurface_color / BSDF_PI * -cos_i;
        }
        return vec3<f32>(0.0);
    }
    let cos_o = max(dot(n, wo), 0.0);
    let h = normalize(wo + wi);
    let f = bsdf_fresnel(max(dot(h, wo), 0.0), s.specular_f0);
    let specular = bsdf_specular(p, cos_o, cos_i, h, f);
    let left = (1.0 - f) * (1.0 - s.metallic) * (1.0 - bsdf_diffuse_loss(s, cos_o));
    if (s.flags & SURFACE_FLAG_SUBSURFACE) != 0u {
        // Fix 3: the wrapped lobe normalised over the hemisphere it lights,
        // the scattered light out of what the specular lobe leaves.
        let w = bsdf_wrap(s);
        let wrapped = (cos_i + w) / (1.0 + 2.0 * w);
        return left * mix(s.albedo.rgb, s.subsurface_color, 0.5) / BSDF_PI * wrapped + specular;
    }
    return left * s.albedo.rgb / BSDF_PI * cos_i + specular;
}

// A cosine-distributed local direction about +z.
fn bsdf_cosine_direction(u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = 2.0 * BSDF_PI * u.y;
    return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
}

// A visible normal of GGX seen from local `wo` (above the surface), from two
// uniform numbers: Dupuy and Benyoub 2023, "Sampling Visible GGX Normals
// with Spherical Caps".
fn bsdf_sample_visible_normal(wo: vec3<f32>, alpha: vec2<f32>, u: vec2<f32>) -> vec3<f32> {
    let v = normalize(vec3<f32>(wo.x * alpha.x, wo.y * alpha.y, wo.z));
    let phi = 2.0 * BSDF_PI * u.x;
    let z = (1.0 - u.y) * (1.0 + v.z) - v.z;
    let sin_theta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    let h = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), z) + v;
    return normalize(vec3<f32>(h.x * alpha.x, h.y * alpha.y, max(h.z, 0.0)));
}

// GGX's distribution of normals at local half vector `h` (unit), for the
// visible normals' density.
fn bsdf_ggx_d(h: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let e = vec3<f32>(h.x / alpha.x, h.y / alpha.y, h.z);
    let d = dot(e, e);
    return 1.0 / (BSDF_PI * alpha.x * alpha.y * d * d);
}

// Smith's Lambda for GGX along local direction `w`.
fn bsdf_smith_lambda(w: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let z2 = max(w.z * w.z, 1e-12);
    let a2 = (alpha.x * alpha.x * w.x * w.x + alpha.y * alpha.y * w.y * w.y) / z2;
    return 0.5 * (sqrt(1.0 + a2) - 1.0);
}

// The share of `bsdf_sample`'s draws each lobe takes (diffuse, specular,
// through): in proportion to an estimate of what it reflects toward `wo`.
fn bsdf_lobe_weights(s: SurfaceData, cos_o: f32) -> vec3<f32> {
    let f = bsdf_fresnel(cos_o, s.specular_f0);
    var albedo = s.albedo.rgb;
    if (s.flags & SURFACE_FLAG_SUBSURFACE) != 0u {
        albedo = mix(albedo, s.subsurface_color, 0.5);
    }
    let diffuse = bsdf_luminance(albedo * (1.0 - s.metallic) * (1.0 - f));
    let through = select(0.0, bsdf_luminance(s.subsurface_color), (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u);
    let w = vec3<f32>(diffuse, bsdf_luminance(f), through);
    let sum = w.x + w.y + w.z;
    if !(sum > 0.0) { return vec3<f32>(0.0); }
    return w / sum;
}

// `wo` in the surface's frame, held just above its horizon (a shading
// normal can turn from the eye; the visible normals need a view above it).
fn bsdf_local_view(frame: BsdfFrame, wo: vec3<f32>) -> vec3<f32> {
    let l = bsdf_to_local(frame, wo);
    return normalize(vec3<f32>(l.xy, max(l.z, 1e-4)));
}

fn bsdf_pdf(s: SurfaceData, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    let frame = bsdf_frame(s);
    let lo = bsdf_local_view(frame, wo);
    let lobes = bsdf_lobe_weights(s, lo.z);
    let cos_i = dot(s.normal, wi);
    // The specular lobe's visible normals reflect `wo` below the horizon
    // too (those samples carry no light): its density is counted wherever
    // the half vector is a microfacet's, so a direction the light through
    // a thin surface shares is weighed by both lobes' densities.
    let h = normalize(lo + bsdf_to_local(frame, wi));
    var specular = 0.0;
    if h.z > 0.0 {
        let alpha = bsdf_alphas(s);
        let g1 = 1.0 / (1.0 + bsdf_smith_lambda(lo, alpha));
        specular = g1 * bsdf_ggx_d(h, alpha) / (4.0 * lo.z);
    }
    let diffuse = select(lobes.z * -cos_i, lobes.x * cos_i, cos_i > 0.0) / BSDF_PI;
    return diffuse + lobes.y * specular;
}

fn bsdf_sample(s: SurfaceData, wo: vec3<f32>, u: vec3<f32>) -> BsdfSample {
    let none = BsdfSample(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, BSDF_LOBE_NONE);
    let frame = bsdf_frame(s);
    let lo = bsdf_local_view(frame, wo);
    let lobes = bsdf_lobe_weights(s, lo.z);
    if lobes.x + lobes.y + lobes.z <= 0.0 { return none; }
    var wi: vec3<f32>;
    var lobe: u32;
    if u.x < lobes.x {
        wi = bsdf_to_world(frame, bsdf_cosine_direction(u.yz));
        lobe = BSDF_LOBE_DIFFUSE;
    } else if u.x < lobes.x + lobes.y {
        let h = bsdf_sample_visible_normal(lo, bsdf_alphas(s), u.yz);
        wi = bsdf_to_world(frame, reflect(-lo, h));
        lobe = BSDF_LOBE_SPECULAR;
    } else {
        wi = -bsdf_to_world(frame, bsdf_cosine_direction(u.yz));
        lobe = BSDF_LOBE_THROUGH;
    }
    let pdf = bsdf_pdf(s, wo, wi);
    if !(pdf > 0.0) { return none; }
    return BsdfSample(wi, bsdf_eval(s, wo, wi) / pdf, pdf, lobe);
}

// The specular lobe's directional albedo split into the scale and bias of
// `f0` (Karis 2013's split sum, Lazarov's analytic fit): what a
// prefiltered environment's radiance is multiplied by.
fn bsdf_environment_split_sum(cos_o: f32, roughness: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * cos_o)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// Schlick's Fresnel averaged over a rough lobe (for an environment's
// layers over the split sum).
fn bsdf_fresnel_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * bsdf_pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}
