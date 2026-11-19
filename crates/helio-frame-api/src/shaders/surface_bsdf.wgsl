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
//
// `wo` and `wi` are unit vectors leaving the surface (toward the eye, toward
// the light). The model:
//
// - **Specular**: GGX (Trowbridge-Reitz) with alpha = roughness^2 (the
//   `SURFACE_FLAG_ANISOTROPIC` alphas `roughness_aniso_x/_y` along a tangent
//   turned by `aniso_rotation`), Smith's height-correlated masking-shadowing
//   G2 = 1 / (1 + L(wo) + L(wi)) (Heitz 2014), Schlick's Fresnel from
//   `specular_f0`. Sampled from the distribution of visible normals (Heitz
//   2018; spherical caps, Dupuy and Benyoub 2023), density
//   G1(wo) D(h) / (4 n.wo).
// - **Diffuse**: Lambert, `albedo (1 - metallic) (1 - F(wo.h)) / pi`;
//   `SURFACE_FLAG_SUBSURFACE` wraps it (`(cos + w) / (1 + w)`,
//   `w = 0.2 + 0.4 (1 - roughness)`) and mixes in `subsurface_color / pi`.
//   Cosine sampled.
// - **Through a thin two-sided surface** (`SURFACE_FLAG_TWO_SIDED`): light
//   from behind passes as `subsurface_color / pi` (Lambert on the far side).
//   Cosine sampled about the back.
//
// A surface met from behind (`wo` below the normal) is lit as its front by a
// two-sided surface (it has no back) and with `n.wo` held just above zero by
// the others, as a shading normal that turns from the eye is. The lobes are
// chosen in proportion to an estimate of what each reflects, and every
// density is the mixture's, so a sample's weight is unbiased whichever lobe
// drew it. No multiple scattering between microfacets: at high roughness the
// specular lobe loses energy (a white surface reflects less than it receives,
// never more).

const BSDF_PI: f32 = 3.14159265358979;
// The smoothest surface lit (Filament's least perceptual roughness): a
// delta lobe would make a point light's highlight a single pixel.
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

fn bsdf_frame(s: SurfaceData, n: vec3<f32>) -> BsdfFrame {
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

// The specular lobe's alphas (along the tangent, along the bitangent).
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
// incidence.
fn bsdf_fresnel(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * bsdf_pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

// GGX's distribution of normals at local half vector `h`.
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

// A cosine-distributed local direction about +z.
fn bsdf_cosine_direction(u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = 2.0 * BSDF_PI * u.y;
    return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
}

// The wrap of a subsurface surface's diffuse lobe.
fn bsdf_wrap(s: SurfaceData) -> f32 {
    return 0.2 + 0.4 * (1.0 - s.roughness);
}

// The share of `bsdf_sample`'s draws each lobe takes (diffuse, specular,
// through): in proportion to an estimate of what it reflects toward `wo`.
fn bsdf_lobe_weights(s: SurfaceData, cos_o: f32) -> vec3<f32> {
    let f = bsdf_fresnel(cos_o, s.specular_f0);
    var diffuse = bsdf_luminance(s.albedo.rgb * (1.0 - s.metallic) * (1.0 - f));
    if (s.flags & SURFACE_FLAG_SUBSURFACE) != 0u {
        diffuse = 0.5 * diffuse + 0.5 * bsdf_luminance(s.subsurface_color);
    }
    let through = select(0.0, bsdf_luminance(s.subsurface_color), (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u);
    let w = vec3<f32>(diffuse, bsdf_luminance(f), through);
    let sum = w.x + w.y + w.z;
    if !(sum > 0.0) { return vec3<f32>(0.0); }
    return w / sum;
}

// The normal the surface is lit about from `wo`: a two-sided surface met
// from behind is its front seen from the other side.
fn bsdf_normal(s: SurfaceData, wo: vec3<f32>) -> vec3<f32> {
    let flip = (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u && dot(s.normal, wo) < 0.0;
    return select(s.normal, -s.normal, flip);
}

fn bsdf_eval(s: SurfaceData, wo: vec3<f32>, wi: vec3<f32>) -> vec3<f32> {
    let n = bsdf_normal(s, wo);
    let cos_o = max(dot(n, wo), 1e-4);
    let cos_i = dot(n, wi);
    if cos_i <= 0.0 {
        // From behind: only what a thin two-sided surface lets through.
        if (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u {
            return s.subsurface_color / BSDF_PI * -cos_i;
        }
        return vec3<f32>(0.0);
    }
    let frame = bsdf_frame(s, n);
    let lo = normalize(vec3<f32>(bsdf_to_local(frame, wo).xy, cos_o));
    let li = bsdf_to_local(frame, wi);
    let h = normalize(lo + li);
    let alpha = bsdf_alphas(s);
    let f = bsdf_fresnel(max(dot(lo, h), 0.0), s.specular_f0);
    let g2 = 1.0 / (1.0 + bsdf_smith_lambda(lo, alpha) + bsdf_smith_lambda(li, alpha));
    // f_spec cos_i = F D G2 / (4 cos_o cos_i) cos_i.
    let specular = f * (bsdf_ggx_d(h, alpha) * g2 / (4.0 * cos_o));
    var diffuse = s.albedo.rgb * (1.0 - s.metallic) * (1.0 - f) / BSDF_PI;
    var cosine = cos_i;
    if (s.flags & SURFACE_FLAG_SUBSURFACE) != 0u {
        let w = bsdf_wrap(s);
        cosine = (cos_i + w) / (1.0 + w);
        diffuse = 0.5 * diffuse + 0.5 * s.subsurface_color / BSDF_PI;
        return diffuse * cosine + specular * (cosine / cos_i);
    }
    return diffuse * cosine + specular;
}

fn bsdf_pdf(s: SurfaceData, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    let n = bsdf_normal(s, wo);
    let cos_o = max(dot(n, wo), 1e-4);
    let cos_i = dot(n, wi);
    let lobes = bsdf_lobe_weights(s, cos_o);
    if cos_i <= 0.0 {
        return lobes.z * -cos_i / BSDF_PI;
    }
    let frame = bsdf_frame(s, n);
    let lo = normalize(vec3<f32>(bsdf_to_local(frame, wo).xy, cos_o));
    let h = normalize(lo + bsdf_to_local(frame, wi));
    let alpha = bsdf_alphas(s);
    let g1 = 1.0 / (1.0 + bsdf_smith_lambda(lo, alpha));
    let specular = g1 * bsdf_ggx_d(h, alpha) / (4.0 * cos_o);
    return lobes.x * cos_i / BSDF_PI + lobes.y * specular;
}

fn bsdf_sample(s: SurfaceData, wo: vec3<f32>, u: vec3<f32>) -> BsdfSample {
    let none = BsdfSample(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, BSDF_LOBE_NONE);
    let n = bsdf_normal(s, wo);
    let cos_o = max(dot(n, wo), 1e-4);
    let lobes = bsdf_lobe_weights(s, cos_o);
    if lobes.x + lobes.y + lobes.z <= 0.0 { return none; }
    let frame = bsdf_frame(s, n);
    var wi: vec3<f32>;
    var lobe: u32;
    if u.x < lobes.x {
        wi = bsdf_to_world(frame, bsdf_cosine_direction(u.yz));
        lobe = BSDF_LOBE_DIFFUSE;
    } else if u.x < lobes.x + lobes.y {
        let lo = normalize(vec3<f32>(bsdf_to_local(frame, wo).xy, cos_o));
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
