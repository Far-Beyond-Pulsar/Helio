// The surface at a lit point (`helio_frame_api::gbuffer::SURFACE_SNIPPET`,
// `//!use frame_surface`): what a material makes of a point, what a G-buffer
// texel holds, what a ray stream's hit holds, and what a lighting model
// lights. One struct with two encodings, each with one encode and one decode:
//
// - the G-buffer's targets (`helio_frame_api::gbuffer::TARGETS`):
//   `gbuffer_surface_texels` (a drawer, through `gbuffer_output` /
//   `gbuffer_depth_output`) and `gbuffer_surface_at` (a lighting model);
// - a ray stream's `SceneSurface` row (`//!use frame_ray_surface`):
//   `scene_surface_pack` (a tracer) and `scene_surface_unpack` (a consumer).
//
// So a pixel and a hit carry the same fields and are lit by the same code.
// A material class's template (the materials plugin) returns it; a kind
// with no material table fills it with `surface_data`.

// `SurfaceData::flags`.
// Subsurface scattering: wrap diffuse and a rim, tinted by
// `subsurface_color`.
const SURFACE_FLAG_SUBSURFACE: u32 = 1u;
// Anisotropic GGX by `roughness_aniso_x`/`_y`, the tangent turned by
// `aniso_rotation` (radians) about the normal.
const SURFACE_FLAG_ANISOTROPIC: u32 = 2u;
const SURFACE_FLAG_LOW_SPECULAR: u32 = 4u;
// A thin two-sided surface (leaves): light from behind passes through it,
// `subsurface_color` the share that does, by colour (what lies between the
// light and the surface is the light's visibility).
const SURFACE_FLAG_TWO_SIDED: u32 = 8u;

struct SurfaceData {
    // Base colour (linear RGB) and coverage (1 where a surface is: a hit,
    // an opaque pixel).
    albedo:              vec4<f32>,
    // The shading normal (unit, frame space).
    normal:              vec3<f32>,
    // The material's own occlusion of indirect light.
    ao:                  f32,
    roughness:           f32,
    metallic:            f32,
    // Reflectance at normal incidence.
    specular_f0:         vec3<f32>,
    // Radiance it gives off.
    emissive:            vec3<f32>,
    alpha:               f32,
    // `SURFACE_FLAG_*`.
    flags:               u32,
    // Subsurface tint, or (`SURFACE_FLAG_TWO_SIDED`) the share of the light
    // behind it that passes through, by colour.
    subsurface_color:    vec3<f32>,
    subsurface_radius:   f32,
    roughness_aniso_x:   f32,
    roughness_aniso_y:   f32,
    aniso_rotation:      f32,
}

// A plain surface: opaque, no material occlusion, the metallic workflow's
// reflectance (`mix(0.04, albedo, metallic)`), no class extensions.
fn surface_data(albedo: vec3<f32>, normal: vec3<f32>, roughness: f32, metallic: f32, emissive: vec3<f32>) -> SurfaceData {
    let f0 = clamp(mix(vec3<f32>(0.04), albedo, metallic), vec3<f32>(0.0), vec3<f32>(0.999));
    return SurfaceData(vec4<f32>(albedo, 1.0), normal, 1.0, roughness, metallic, f0, emissive, 1.0,
                       0u, vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0);
}

// `s` as a thin two-sided surface that lets `through` of the light behind
// it pass (by colour; none: `s` as it is).
fn surface_with_transmission(s: SurfaceData, through: vec3<f32>) -> SurfaceData {
    var out = s;
    if any(through > vec3<f32>(0.0)) {
        out.flags |= SURFACE_FLAG_TWO_SIDED;
        out.subsurface_color = through;
    }
    return out;
}

// What a surface lets through from behind, by colour (none: zero).
fn surface_transmission(s: SurfaceData) -> vec3<f32> {
    return select(vec3<f32>(0.0), s.subsurface_color, (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u);
}

// ── The G-buffer's encoding ─────────────────────────────────────────────────
// Targets 0-3, 5 and 6 of `helio_frame_api::gbuffer::TARGETS` (the surface
// tag, 4, and the motion, 7, are the drawer's own):
//   albedo   (Rgba8Unorm)  rgb albedo, a alpha
//   normal   (Rgba16Float) xyz shading normal, w F0.r
//   orm      (Rgba8Unorm)  ao, roughness, metallic, F0.g
//   emissive (Rgba16Float) rgb emissive, a F0.b
//   sss      (Rgba16Float) rgb subsurface colour, a subsurface radius
//   extra    (Rgba16Float) aniso x, aniso y, aniso rotation, flags
// (flags as a float: Rgba16Float holds small integers exactly).
struct GBufferSurfaceTexels {
    albedo:   vec4<f32>,
    normal:   vec4<f32>,
    orm:      vec4<f32>,
    emissive: vec4<f32>,
    sss:      vec4<f32>,
    extra:    vec4<f32>,
}

fn gbuffer_surface_texels(s: SurfaceData) -> GBufferSurfaceTexels {
    return GBufferSurfaceTexels(
        vec4<f32>(s.albedo.rgb, s.alpha),
        vec4<f32>(s.normal, s.specular_f0.r),
        vec4<f32>(s.ao, s.roughness, s.metallic, s.specular_f0.g),
        vec4<f32>(s.emissive, s.specular_f0.b),
        vec4<f32>(s.subsurface_color, s.subsurface_radius),
        vec4<f32>(s.roughness_aniso_x, s.roughness_aniso_y, s.aniso_rotation, f32(s.flags)),
    );
}

// The surface a G-buffer texel holds (its six surface targets, loaded).
fn gbuffer_surface_at(t: GBufferSurfaceTexels) -> SurfaceData {
    return SurfaceData(
        t.albedo,
        normalize(t.normal.xyz),
        t.orm.r,
        t.orm.g,
        t.orm.b,
        vec3<f32>(t.normal.w, t.orm.a, t.emissive.a),
        t.emissive.rgb,
        t.albedo.a,
        u32(round(max(t.extra.a, 0.0))),
        t.sss.rgb,
        t.sss.a,
        t.extra.r,
        t.extra.g,
        t.extra.b,
    );
}
