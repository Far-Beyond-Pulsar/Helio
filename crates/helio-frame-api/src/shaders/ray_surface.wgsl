// A ray stream's surface rows (`helio_frame_api::rays::stream::SURFACE_SNIPPET`,
// `//!use frame_ray_surface`, with `//!use frame_ray_stream` for the
// packings and `//!use frame_surface` for `SurfaceData`): what a nearest-hit
// stream's tracer writes at each hit and its consumer reads.

// The surface at a nearest hit, written by the tracer that found it: a
// `SurfaceData` (`//!use frame_surface`, the surface a G-buffer texel holds)
// packed into 32 bytes, plus the face's normal (a hit needs it to leave the
// surface; a G-buffer has none). `scene_surface_pack` and
// `scene_surface_unpack` are the one encode and the one decode:
//   normal      the shading normal (`scene_pack_unit`, on the side the ray met)
//   face        the face's normal (the same)
//   base_color  RGBA8: albedo rgb, metallic
//   params      RGBA8: roughness, ao, subsurface radius as r / (1 + r), flags
//   emissive    RGB9E5 (`scene_pack_radiance`)
//   specular    F0, 10 bits a channel (unorm), 2 bits zero
//   subsurface  RGB9E5: subsurface colour, or the share let through from
//               behind (`SURFACE_FLAG_TWO_SIDED`)
//   anisotropy  aniso x (8 bits unorm), aniso y (8), rotation (16: its
//               fraction of a turn)
// The alpha a row unpacks to is 1 (a hit is where a surface is). A tracer
// that has nothing for a field writes `surface_data`'s neutral value.
struct SceneSurface {
    normal: u32,
    face: u32,
    base_color: u32,
    params: u32,
    emissive: u32,
    specular: u32,
    subsurface: u32,
    anisotropy: u32,
}

const SCENE_TURN: f32 = 6.28318530718;

fn scene_surface_pack(s: SurfaceData, face: vec3<f32>) -> SceneSurface {
    let radius = max(s.subsurface_radius, 0.0);
    let f0 = vec3<u32>(clamp(s.specular_f0, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0 + 0.5);
    let turn = fract(s.aniso_rotation / SCENE_TURN);
    return SceneSurface(
        scene_pack_unit(s.normal),
        scene_pack_unit(face),
        pack4x8unorm(vec4<f32>(s.albedo.rgb, s.metallic)),
        (pack4x8unorm(vec4<f32>(s.roughness, s.ao, radius / (1.0 + radius), 0.0)) & 0x00ffffffu) | ((s.flags & 0xffu) << 24u),
        scene_pack_radiance(s.emissive),
        f0.x | (f0.y << 10u) | (f0.z << 20u),
        scene_pack_radiance(s.subsurface_color),
        (pack4x8unorm(vec4<f32>(s.roughness_aniso_x, s.roughness_aniso_y, 0.0, 0.0)) & 0xffffu) | (u32(clamp(turn, 0.0, 1.0) * 65535.0 + 0.5) << 16u),
    );
}

fn scene_surface_unpack(row: SceneSurface) -> SurfaceData {
    let base = unpack4x8unorm(row.base_color);
    let params = unpack4x8unorm(row.params);
    let radius = params.z / max(1.0 - params.z, 1.0 / 255.0);
    let f0 = vec3<f32>(f32(row.specular & 1023u), f32((row.specular >> 10u) & 1023u), f32((row.specular >> 20u) & 1023u)) / 1023.0;
    let aniso = unpack4x8unorm(row.anisotropy & 0xffffu);
    return SurfaceData(
        vec4<f32>(base.rgb, 1.0),
        scene_unpack_unit(row.normal),
        params.y,
        params.x,
        base.a,
        f0,
        scene_unpack_radiance(row.emissive),
        1.0,
        row.params >> 24u,
        scene_unpack_radiance(row.subsurface),
        radius,
        aniso.x,
        aniso.y,
        f32(row.anisotropy >> 16u) / 65535.0 * SCENE_TURN,
    );
}

// The face's normal of a hit's surface (on the side the ray met).
fn scene_surface_face(row: SceneSurface) -> vec3<f32> {
    return scene_unpack_unit(row.face);
}
