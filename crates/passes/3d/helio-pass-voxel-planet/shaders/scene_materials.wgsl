// Voxel materials drawn with the host's scene materials: Helio's material
// rows (`GpuMaterial`) and its descriptor-indexed texture table, bound as
// the publish pipeline's group 2 (`chunks::scene`). Spliced over the
// `SCENE_MATERIALS` line of chunk_draw.wgsl when the device has binding
// arrays.

struct SceneGpuMaterial {
    base_color:         vec4<f32>,
    emissive:           vec4<f32>,
    roughness_metallic: vec4<f32>,
    tex_base_color:     u32,
    tex_normal:         u32,
    tex_roughness:      u32,
    tex_emissive:       u32,
    tex_occlusion:      u32,
    workflow:           u32,
    flags:              u32,
    material_class:     u32,
    class_params:       vec4<f32>,
}

const SCENE_NO_TEXTURE: u32 = 0xffffffffu;
// `GpuMaterial::flags`: the normal texture is a normal map
// (`helio_mats::FLAG_HAS_NORMAL_MAP`).
const SCENE_HAS_NORMAL_MAP: u32 = 1u << 3u;

// Per voxel material: the scene material rows of its top, side and bottom
// faces (x, y, z; SCENE_NO_TEXTURE: none).
struct SceneMap {
    faces: array<vec4<u32>, 32>,
}

@group(2) @binding(0) var<storage, read> scene_map: SceneMap;
@group(2) @binding(1) var<storage, read> scene_materials: array<SceneGpuMaterial>;
@group(2) @binding(2) var scene_textures: binding_array<texture_2d<f32>, MATERIAL_TEXTURES>;
@group(2) @binding(3) var scene_samplers: binding_array<sampler, MATERIAL_TEXTURES>;

// A material texture's coordinate along a face axis: base cells from `b`
// (a cell's first base cell) plus its span `span` of base cells at `t`
// across it, over `tile` base cells a texture. Exact at any planet size
// (the cell index is taken modulo the tile first); a texture wholly inside
// the span is not wrapped (no seam row at the cell's far edge).
fn tile_coord(b: i32, t: f32, span: f32, tile: i32) -> f32 {
    let start = f32(((b % tile) + tile) % tile);
    let x = (start + t * span) / f32(tile);
    return select(fract(x), min(x, 0.99999), x <= 1.0);
}

// Material `m`'s surface on a face of cell `cell` (its index at `level`) of
// `level` (`axis` 2 is the radial one: a top face is its sign 1, a bottom
// its sign 0). `in_cell` is the hit's place in the cell. A texture spans
// the material's tile (`SceneMap`, whole base cells; one by default) and
// runs on across cells and levels, so surfaces keep their size at any voxel
// size and distance; the mip level follows its texels' size in pixels
// (filtered, never derivative based, which would break across cells). The
// normal map's normal is in the face's texture frame (x along u, y against
// v, z out: OpenGL's), straight out without one.
fn scene_surface(m: u32, axis: u32, sign: u32, level: u32, cell: vec3<i32>, in_cell: vec3<f32>, pixel_m: f32) -> SceneSurface {
    let none = SceneSurface(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 1.0, false);
    let faces = scene_map.faces[min(m, 31u)];
    var row = faces.y;
    if axis == 2u {
        row = select(faces.z, faces.x, sign == 1u);
    }
    if row == SCENE_NO_TEXTURE || row >= arrayLength(&scene_materials) {
        return none;
    }
    let material = scene_materials[row];
    let tile = max(i32(round(bitcast<f32>(faces.w) / params.planet.y)), 1);
    let span = f32(1u << min(level, 15u));
    let base = cell << vec3<u32>(min(level, 15u));
    // u along the face's first index, v down its second (down a side).
    var uv = vec2<f32>(tile_coord(base.x, in_cell.x, span, tile), tile_coord(base.y, in_cell.y, span, tile));
    if axis == 0u {
        uv = vec2<f32>(tile_coord(base.y, in_cell.y, span, tile), 1.0 - tile_coord(base.z, in_cell.z, span, tile));
    } else if axis == 1u {
        uv = vec2<f32>(tile_coord(base.x, in_cell.x, span, tile), 1.0 - tile_coord(base.z, in_cell.z, span, tile));
    }
    uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(0.99999));
    // A texture's pixels on screen; a texel a half pixel wide (not a whole
    // one): TSR averages the jittered samples, so the detail the mip would
    // have averaged away is rebuilt.
    let tile_px = f32(tile) * params.planet.y / max(pixel_m, 1e-6);
    var colour = material.base_color.rgb;
    var texel = vec3<f32>(1.0);
    if material.tex_base_color != SCENE_NO_TEXTURE && material.tex_base_color < MATERIAL_TEXTURES {
        texel = scene_sample(material.tex_base_color, uv, tile_px).rgb;
        colour *= texel;
    }
    var roughness = material.roughness_metallic.x;
    var occlusion = 1.0;
    if material.tex_roughness != SCENE_NO_TEXTURE && material.tex_roughness < MATERIAL_TEXTURES {
        // Occlusion-roughness-metallic packing: roughness is green.
        let orm = scene_sample(material.tex_roughness, uv, tile_px);
        roughness *= orm.g;
        if material.tex_occlusion == material.tex_roughness { occlusion = orm.r; }
    }
    if material.tex_occlusion != SCENE_NO_TEXTURE && material.tex_occlusion < MATERIAL_TEXTURES && material.tex_occlusion != material.tex_roughness {
        occlusion = scene_sample(material.tex_occlusion, uv, tile_px).r;
    }
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    if (material.flags & SCENE_HAS_NORMAL_MAP) != 0u && material.tex_normal != SCENE_NO_TEXTURE && material.tex_normal < MATERIAL_TEXTURES {
        let n = scene_sample(material.tex_normal, uv, tile_px).rgb * 2.0 - 1.0;
        normal = n / max(length(n), 1e-4);
    }
    return SceneSurface(colour, roughness, material.emissive.rgb * material.emissive.w * texel, normal, occlusion, true);
}

// Texture `slot` at `uv`, its mip by the texture's width on screen
// (`tile_px` pixels).
fn scene_sample(slot: u32, uv: vec2<f32>, tile_px: f32) -> vec4<f32> {
    let size = f32(textureDimensions(scene_textures[slot]).x);
    let lod = max(log2(size / max(tile_px, 1e-6)) - 1.0, 0.0);
    return textureSampleLevel(scene_textures[slot], scene_samplers[slot], uv, lod);
}
