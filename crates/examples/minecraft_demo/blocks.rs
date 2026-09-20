//! Minecraft-style block registry used by the dedicated demo.

pub type BlockId = u8;

pub const AIR: BlockId = 0;
pub const GRASS: BlockId = 1;
pub const DIRT: BlockId = 2;
pub const STONE: BlockId = 3;
pub const SAND: BlockId = 5;
pub const WATER: BlockId = 6;
pub const BEDROCK: BlockId = 7;
pub const COAL_ORE: BlockId = 8;
pub const IRON_ORE: BlockId = 9;
pub const GOLD_ORE: BlockId = 10;
pub const LOG: BlockId = 11;
pub const LEAVES: BlockId = 12;

/// Demo-owned material data. The renderer consumes palettes but does not
/// prescribe block colors or roughness.
pub fn default_palette() -> Vec<[f32; 4]> {
    vec![
        [0.0, 0.0, 0.0, 1.0], [0.24, 0.62, 0.18, 0.92],
        [0.48, 0.28, 0.12, 1.0], [0.34, 0.36, 0.38, 0.88],
        [0.65, 0.22, 0.08, 0.7], [0.78, 0.68, 0.36, 0.96],
        [0.12, 0.36, 0.78, 0.18], [0.12, 0.12, 0.14, 0.9],
        [0.08, 0.08, 0.08, 0.86], [0.54, 0.56, 0.6, 0.72],
        [0.88, 0.62, 0.12, 0.5], [0.42, 0.22, 0.08, 1.0],
        [0.16, 0.52, 0.12, 0.86],
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Air,
    Solid,
    Fluid,
    Foliage,
}

pub fn kind(id: BlockId) -> BlockKind {
    match id {
        AIR => BlockKind::Air,
        WATER => BlockKind::Fluid,
        LEAVES => BlockKind::Foliage,
        _ => BlockKind::Solid,
    }
}

pub fn is_opaque(id: BlockId) -> bool {
    matches!(kind(id), BlockKind::Solid)
}
