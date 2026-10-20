//! The block palette: what each block is made of (a voxel planet material
//! id) and which surface it shows on its top, sides and bottom.

use helio_pass_voxel_planet::terrain::material;

/// A look drawn on a block face. Most are baked physically based materials
/// (colour, normal and occlusion-roughness-metalness maps); a few are
/// tiles of the pixel pack, and leaves are composed from its sprigs
/// (`textures::compose`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surface {
    /// A baked material (`assets/textures/pbr`, `tools/bake_textures.py`),
    /// by name.
    Pbr(&'static str),
    /// A tile of the pack, by file stem.
    File(&'static str),
    /// A dense leaf tile built from the pack's leaf sprig, tinted.
    Leaves(&'static str, [f32; 3]),
}

/// Every placeable block. The voxel planet stores a block as its material id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Block {
    Grass,
    Dirt,
    Stone,
    Sand,
    Snow,
    Gravel,
    Sandstone,
    Andesite,
    OakLog,
    OakLeaves,
    Clay,
    Bricks,
    OakPlanks,
    Cobblestone,
    BirchLog,
    BirchLeaves,
    SpruceLog,
    SpruceLeaves,
    JungleLog,
    JungleLeaves,
    Cactus,
    Glowstone,
    StoneBricks,
    DarkOakPlanks,
}

impl Block {
    pub const ALL: [Block; 24] = [
        Block::Grass,
        Block::Dirt,
        Block::Stone,
        Block::Sand,
        Block::Snow,
        Block::Gravel,
        Block::Sandstone,
        Block::Andesite,
        Block::OakLog,
        Block::OakLeaves,
        Block::Clay,
        Block::Bricks,
        Block::OakPlanks,
        Block::Cobblestone,
        Block::BirchLog,
        Block::BirchLeaves,
        Block::SpruceLog,
        Block::SpruceLeaves,
        Block::JungleLog,
        Block::JungleLeaves,
        Block::Cactus,
        Block::Glowstone,
        Block::StoneBricks,
        Block::DarkOakPlanks,
    ];

    /// The blocks on the hotbar, in order.
    pub const HOTBAR: [Block; 9] = [
        Block::Cobblestone,
        Block::OakPlanks,
        Block::StoneBricks,
        Block::Bricks,
        Block::OakLog,
        Block::Glowstone,
        Block::Sand,
        Block::Dirt,
        Block::Snow,
    ];

    /// The voxel planet material id this block is stored as.
    pub fn material(self) -> u32 {
        match self {
            Block::Grass => material::GRASS,
            Block::Dirt => material::DIRT,
            Block::Stone => material::STONE,
            Block::Sand => material::SAND,
            Block::Snow => material::SNOW,
            Block::Gravel => material::GRAVEL,
            Block::Sandstone => material::SANDSTONE,
            Block::Andesite => material::DARK_STONE,
            Block::OakLog => material::WOOD,
            Block::OakLeaves => material::LEAVES,
            Block::Clay => material::CLAY,
            Block::Bricks => material::BRICK,
            Block::OakPlanks => material::PLANKS,
            Block::Cobblestone => material::COBBLE,
            // Ids past the planet's built-in materials are this game's own.
            Block::BirchLog => 22,
            Block::BirchLeaves => 23,
            Block::SpruceLog => 24,
            Block::SpruceLeaves => 25,
            Block::JungleLog => 26,
            Block::JungleLeaves => 27,
            Block::Cactus => 28,
            Block::Glowstone => 29,
            Block::StoneBricks => 30,
            Block::DarkOakPlanks => 31,
        }
    }

    /// The block a voxel material id is, if it is one.
    pub fn from_material(id: u32) -> Option<Block> {
        Block::ALL.into_iter().find(|b| b.material() == id)
    }

    pub fn name(self) -> &'static str {
        match self {
            Block::Grass => "Grass",
            Block::Dirt => "Dirt",
            Block::Stone => "Stone",
            Block::Sand => "Sand",
            Block::Snow => "Snow",
            Block::Gravel => "Gravel",
            Block::Sandstone => "Sandstone",
            Block::Andesite => "Andesite",
            Block::OakLog => "Oak Log",
            Block::OakLeaves => "Oak Leaves",
            Block::Clay => "Clay",
            Block::Bricks => "Bricks",
            Block::OakPlanks => "Oak Planks",
            Block::Cobblestone => "Cobblestone",
            Block::BirchLog => "Birch Log",
            Block::BirchLeaves => "Birch Leaves",
            Block::SpruceLog => "Spruce Log",
            Block::SpruceLeaves => "Spruce Leaves",
            Block::JungleLog => "Jungle Log",
            Block::JungleLeaves => "Jungle Leaves",
            Block::Cactus => "Cactus",
            Block::Glowstone => "Glowstone",
            Block::StoneBricks => "Stone Bricks",
            Block::DarkOakPlanks => "Dark Oak Planks",
        }
    }

    /// The surfaces of the top, side and bottom faces.
    pub fn surfaces(self) -> [Surface; 3] {
        use Surface::{File, Pbr};
        let all = |s| [s, s, s];
        let log = |side, end| [Pbr(end), Pbr(side), Pbr(end)];
        match self {
            Block::Grass => [Pbr("grass_top"), Pbr("grass_side"), Pbr("dirt")],
            Block::Dirt => all(Pbr("dirt")),
            Block::Stone => all(Pbr("stone")),
            Block::Sand => all(Pbr("sand")),
            Block::Snow => all(Pbr("snow")),
            Block::Gravel => all(Pbr("gravel")),
            Block::Sandstone => [Pbr("sandstone_top"), Pbr("sandstone"), Pbr("sandstone_top")],
            Block::Andesite => all(Pbr("andesite")),
            Block::OakLog => log("oak_log", "oak_log_top"),
            Block::OakLeaves => all(Surface::Leaves("oak_leaves", [0.62, 0.95, 0.42])),
            Block::Clay => all(Pbr("clay")),
            Block::Bricks => all(Pbr("bricks")),
            Block::OakPlanks => all(Pbr("oak_planks")),
            Block::Cobblestone => all(Pbr("cobblestone")),
            Block::BirchLog => log("birch_log", "birch_log_top"),
            Block::BirchLeaves => all(Surface::Leaves("birch_leaves", [0.78, 1.0, 0.45])),
            Block::SpruceLog => log("spruce_log", "spruce_log_top"),
            Block::SpruceLeaves => all(Surface::Leaves("spruce_leaves", [0.38, 0.72, 0.46])),
            Block::JungleLog => log("jungle_log", "jungle_log_top"),
            Block::JungleLeaves => all(Surface::Leaves("jungle_leaves", [0.45, 1.0, 0.36])),
            Block::Cactus => [File("cactus_top"), File("cactus_side"), File("cactus_bottom")],
            Block::Glowstone => all(File("glowstone")),
            Block::StoneBricks => all(Pbr("stone_bricks")),
            Block::DarkOakPlanks => all(Pbr("dark_oak_planks")),
        }
    }

    /// Light the block gives off, as an emissive colour and strength.
    pub fn emission(self) -> Option<([f32; 3], f32)> {
        match self {
            Block::Glowstone => Some(([1.0, 0.78, 0.45], 3.5)),
            _ => None,
        }
    }

    /// Roughness of the block's faces.
    pub fn roughness(self) -> f32 {
        match self {
            Block::Snow => 0.7,
            Block::Glowstone => 0.35,
            Block::Sand | Block::Dirt | Block::Gravel | Block::Clay => 0.95,
            Block::Stone | Block::Andesite | Block::Cobblestone | Block::StoneBricks | Block::Bricks => 0.88,
            _ => 0.82,
        }
    }
}
