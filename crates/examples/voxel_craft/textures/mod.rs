//! Block textures and the scene materials that carry them.
//!
//! Every block face is an ordinary Helio scene material (`MaterialComponent`
//! with textures in the SceneDB texture store), the same rows meshes use.
//! The voxel planet pass is told which row each of its material ids shows
//! on its top, sides and bottom ([`VoxelMaterialMap`]) and reads them
//! through the renderer's material table, so a game can swap any block for
//! a graph-compiled material without touching the terrain.
//!
//! Most blocks are physically based materials baked from ambientCG's (CC0,
//! `assets/textures/pbr/LICENSE`, `tools/bake_textures.py`): colour, a normal
//! map and occlusion-roughness-metalness. Leaves, cactus and glowstone are
//! tiles of the Modernized 128x pack (MIT, `assets/textures/
//! LICENSE-Modernized-128x`); [`compose`] builds the leaves from its sprigs.

mod compose;

use crate::blocks::{Block, Surface};
use helio_pass_gbuffer::MaterialComponent;
use helio_pass_voxel_planet::chunks::scene::{VoxelMaterialMap, NO_MATERIAL};
use helio_pass_voxel_planet::engine::TerrainAppearance;
use image::RgbaImage;
use pulsar_scenedb::{gpu::TextureStore, Entity, World};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

macro_rules! tiles {
    ($($name:literal),* $(,)?) => {
        /// The embedded PNG of a pack tile.
        fn tile_bytes(name: &str) -> Option<&'static [u8]> {
            match name {
                $($name => Some(include_bytes!(concat!("../assets/textures/", $name, ".png"))),)*
                _ => None,
            }
        }
    };
}

tiles!("glowstone", "cactus_side", "cactus_top", "cactus_bottom", "oak_leaves", "birch_leaves", "spruce_leaves", "jungle_leaves");

macro_rules! baked {
    ($($name:literal),* $(,)?) => {
        /// The embedded maps of a baked material: colour, normal, ORM.
        fn baked_bytes(name: &str) -> Option<[&'static [u8]; 3]> {
            match name {
                $($name => Some([
                    include_bytes!(concat!("../assets/textures/pbr/", $name, "_color.jpg")),
                    include_bytes!(concat!("../assets/textures/pbr/", $name, "_normal.jpg")),
                    include_bytes!(concat!("../assets/textures/pbr/", $name, "_orm.jpg")),
                ]),)*
                _ => None,
            }
        }
    };
}

baked!(
    "grass_top", "grass_side", "dirt", "stone", "andesite", "sand", "snow", "gravel", "sandstone", "sandstone_top", "clay", "bricks",
    "cobblestone", "stone_bricks", "oak_planks", "dark_oak_planks", "oak_log", "birch_log", "spruce_log", "jungle_log", "oak_log_top",
    "birch_log_top", "spruce_log_top", "jungle_log_top",
);

fn decode(bytes: &[u8]) -> RgbaImage {
    image::load_from_memory(bytes).expect("embedded texture decodes").to_rgba8()
}

fn tile(name: &str) -> RgbaImage {
    decode(tile_bytes(name).unwrap_or_else(|| panic!("tile {name} is not embedded")))
}

/// The maps a surface shows: its colour (sRGB), and a baked material's
/// normal map and occlusion-roughness-metalness (linear).
struct SurfaceMaps {
    colour: RgbaImage,
    normal_orm: Option<(RgbaImage, RgbaImage)>,
}

fn surface_maps(surface: Surface) -> SurfaceMaps {
    match surface {
        Surface::Pbr(name) => {
            let [colour, normal, orm] = baked_bytes(name).unwrap_or_else(|| panic!("material {name} is not embedded"));
            SurfaceMaps { colour: decode(colour), normal_orm: Some((decode(normal), decode(orm))) }
        }
        Surface::File(name) => SurfaceMaps { colour: tile(name), normal_orm: None },
        Surface::Leaves(name, green) => SurfaceMaps { colour: compose::leaves(&tile(name), green, name.len() as u32 * 7919), normal_orm: None },
    }
}

/// What loading the blocks made.
pub struct BlockTextures {
    /// The SceneDB texture store to hand to the renderer's scene handle.
    pub store: Arc<RwLock<TextureStore>>,
    /// Which scene material rows each voxel material shows.
    pub map: VoxelMaterialMap,
    /// Flat colours (texture averages) for ground that has no scene
    /// material, and for devices that cannot index textures.
    pub appearance: TerrainAppearance,
    /// Each block's side face material, for things that look like blocks (the
    /// block in hand).
    sides: HashMap<Block, Entity>,
}

impl BlockTextures {
    /// The scene material showing `block`'s side.
    pub fn side_material(&self, block: Block) -> Option<Entity> {
        self.sides.get(&block).copied()
    }
}

/// Registers every block's textures and materials in `world`.
pub fn load(device: &wgpu::Device, queue: &wgpu::Queue, world: &mut World) -> BlockTextures {
    let mut store = TextureStore::new(256);
    let mut appearance = TerrainAppearance::default();
    let mut map = VoxelMaterialMap::default();
    // A material row for each distinct (surface, block look).
    let mut rows: HashMap<String, Entity> = HashMap::new();
    let mut sides = HashMap::new();
    for block in Block::ALL {
        let mut faces = [NO_MATERIAL; 3];
        let mut average = [0.0f32; 3];
        for (face, surface) in block.surfaces().into_iter().enumerate() {
            let key = format!("{surface:?}/{:?}", block.emission().is_some());
            let maps = surface_maps(surface);
            let row = *rows.entry(key).or_insert_with(|| {
                let base = upload(device, queue, &mut store, &maps.colour, Texels::Colour);
                let (emissive, strength) = block.emission().unwrap_or(([0.0; 3], 0.0));
                // A baked material's roughness is its map's.
                let roughness = if maps.normal_orm.is_some() { 1.0 } else { block.roughness() };
                let mut material = MaterialComponent::new([1.0; 4], roughness, 0.0, emissive, strength);
                material.tex_base_color = base;
                if let Some((normal, orm)) = &maps.normal_orm {
                    material.tex_normal = upload(device, queue, &mut store, normal, Texels::Normal);
                    let orm = upload(device, queue, &mut store, orm, Texels::Linear);
                    material.tex_roughness = orm;
                    material.tex_occlusion = orm;
                    material.flags |= helio_mats::FLAG_HAS_NORMAL_MAP;
                }
                let entity = world.spawn();
                world.insert(entity, material);
                entity
            });
            faces[face] = row.index();
            if face == 1 {
                sides.insert(block, row);
            }
            let a = compose::average(&maps.colour);
            // Weights: the sides are most of what is seen of a block.
            let w = if face == 1 { 0.6 } else { 0.2 };
            for c in 0..3 {
                average[c] += a[c] * w;
            }
        }
        // A texture a block face.
        map.set(block.material(), faces[0], faces[1], faces[2], 0.0);
        let slot = block.material() as usize;
        appearance.materials[slot].colour = [average[0], average[1], average[2], block.roughness()];
        appearance.materials[slot].patches = None;
        appearance.materials[slot].lip = None;
        appearance.materials[slot].fleck = None;
        appearance.materials[slot].speck_host = None;
        // Leaves let a third of the light through (global illumination).
        appearance.materials[slot].transmission = if block.surfaces().iter().any(|s| matches!(s, Surface::Leaves(..))) { 0.3 } else { 0.0 };
        // Glowing blocks light their surroundings (global illumination).
        appearance.materials[slot].emission = block.emission().map_or(0.0, |(_, strength)| strength);
    }
    // The planet's generated fleck and speck blends reference these slots.
    for slot in [helio_pass_voxel_planet::terrain::material::STONE, helio_pass_voxel_planet::terrain::material::DARK_STONE, helio_pass_voxel_planet::terrain::material::SANDSTONE] {
        appearance.materials[slot as usize].fleck = None;
    }
    // Blocks look like their textures, not like the generator's patches.
    appearance.detail = [0.0, 0.12, 0.08, 0.0];
    // Shade keeps its colour: a cool fill like the sky's bounce, and sky
    // occlusion never closing past 0.38 (canopies stay readable).
    appearance.shade = [0.20, 0.25, 0.34, 0.38];
    BlockTextures { store: Arc::new(RwLock::new(store)), map, appearance, sides }
}

/// A global material sampler for blocks: smooth up close (or crisp pixels
/// with `VOXEL_CRAFT_PIXELATED=1`), trilinear far away.
pub fn configure_sampler(renderer: &mut helio::Renderer) {
    let magnify = if std::env::var_os("VOXEL_CRAFT_PIXELATED").is_some() { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
    renderer.set_material_sampler(&wgpu::SamplerDescriptor {
        label: Some("Voxel Craft block sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: magnify,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
}

fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// What a texture's texels are, for its format and mips.
#[derive(Clone, Copy, PartialEq)]
enum Texels {
    /// sRGB colour (mips averaged in linear light).
    Colour,
    /// Linear data (occlusion, roughness, metalness).
    Linear,
    /// A tangent-space normal map (mips averaged, then made unit again).
    Normal,
}

/// Uploads `image` with a full mip chain (2x2 boxes averaged as `texels`
/// says) and returns its texture slot.
fn upload(device: &wgpu::Device, queue: &wgpu::Queue, store: &mut TextureStore, image: &RgbaImage, texels: Texels) -> u32 {
    let srgb = texels == Texels::Colour;
    let decode = |c: usize, v: u8| {
        let x = f32::from(v) / 255.0;
        match texels {
            Texels::Colour if c < 3 => srgb_to_linear(x),
            Texels::Normal if c < 3 => x * 2.0 - 1.0,
            _ => x,
        }
    };
    let encode = |c: usize, v: f32| match texels {
        Texels::Colour if c < 3 => linear_to_srgb(v),
        Texels::Normal if c < 3 => v * 0.5 + 0.5,
        _ => v,
    };
    let (mut width, mut height) = image.dimensions();
    assert!(width.is_power_of_two() && width == height, "block tiles are square powers of two");
    let levels = width.ilog2() + 1;
    let slot = store
        .register(
            device,
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Voxel Craft block"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            image.as_raw(),
        )
        .expect("register block texture");
    let texture = store.texture(slot).expect("registered texture");
    let mut pixels: Vec<[f32; 4]> = image.pixels().map(|p| std::array::from_fn(|c| decode(c, p[c]))).collect();
    for level in 1..levels {
        let (w, h) = (width / 2, height / 2);
        let mut next = vec![[0.0f32; 4]; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let texel = &mut next[(y * w + x) as usize];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let s = pixels[((y * 2 + dy) * width + x * 2 + dx) as usize];
                    for c in 0..4 {
                        texel[c] += s[c] * 0.25;
                    }
                }
                if texels == Texels::Normal {
                    let n = (texel[0] * texel[0] + texel[1] * texel[1] + texel[2] * texel[2]).sqrt().max(1e-6);
                    for c in 0..3 {
                        texel[c] /= n;
                    }
                }
            }
        }
        let bytes: Vec<u8> = next.iter().flat_map(|p| (0..4).map(move |c| (encode(c, p[c]).clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        pixels = next;
        width = w;
        height = h;
    }
    slot
}
