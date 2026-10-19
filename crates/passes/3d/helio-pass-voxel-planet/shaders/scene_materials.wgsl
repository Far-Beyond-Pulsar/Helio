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

// Per voxel material: the scene material rows of its top, side and bottom
// faces (x, y, z; SCENE_NO_TEXTURE: none).
struct SceneMap {
    faces: array<vec4<u32>, 32>,
}

@group(2) @binding(0) var<uniform> scene_map: SceneMap;
@group(2) @binding(1) var<storage, read> scene_materials: array<SceneGpuMaterial>;
@group(2) @binding(2) var scene_textures: binding_array<texture_2d<f32>, MATERIAL_TEXTURES>;
@group(2) @binding(3) var scene_samplers: binding_array<sampler, MATERIAL_TEXTURES>;

// Material `m`'s surface on a face of a cell of `level` (`axis` 2 is the
// radial one: a top face is its sign 1, a bottom its sign 0). `in_cell` is
// the hit's place in the cell. A coarse level's cell repeats the texture once
// per base cell it spans, so blocks keep their size at any distance; the
// mip level follows the voxel's size in pixels (filtered, never derivative
// based, which would break across cells).
fn scene_surface(m: u32, axis: u32, sign: u32, level: u32, in_cell: vec3<f32>, pixel_m: f32) -> SceneSurface {
    let none = SceneSurface(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), false);
    let faces = scene_map.faces[min(m, 31u)];
    var row = faces.y;
    if axis == 2u {
        row = select(faces.z, faces.x, sign == 1u);
    }
    if row == SCENE_NO_TEXTURE || row >= arrayLength(&scene_materials) {
        return none;
    }
    let material = scene_materials[row];
    var uv = in_cell.xy;
    if axis == 0u {
        uv = vec2<f32>(in_cell.y, 1.0 - in_cell.z);
    } else if axis == 1u {
        uv = vec2<f32>(in_cell.x, 1.0 - in_cell.z);
    }
    let repeats = f32(1u << min(level, 15u));
    let tiled = uv * repeats;
    var wrapped = min(tiled, vec2<f32>(0.99999));
    if repeats > 1.0 {
        wrapped = fract(tiled);
    }
    var colour = material.base_color.rgb;
    var texel = vec3<f32>(1.0);
    if material.tex_base_color != SCENE_NO_TEXTURE && material.tex_base_color < MATERIAL_TEXTURES {
        let size = f32(textureDimensions(scene_textures[material.tex_base_color]).x);
        let voxel_px = params.planet.y / max(pixel_m, 1e-6);
        let lod = max(log2(size / max(voxel_px, 1e-6)), 0.0);
        texel = textureSampleLevel(scene_textures[material.tex_base_color], scene_samplers[material.tex_base_color], wrapped, lod).rgb;
        colour *= texel;
    }
    var roughness = material.roughness_metallic.x;
    if material.tex_roughness != SCENE_NO_TEXTURE && material.tex_roughness < MATERIAL_TEXTURES {
        // Occlusion-roughness-metallic packing: roughness is green.
        roughness *= textureSampleLevel(scene_textures[material.tex_roughness], scene_samplers[material.tex_roughness], wrapped, 0.0).g;
    }
    return SceneSurface(colour, roughness, material.emissive.rgb * material.emissive.w * texel, true);
}
