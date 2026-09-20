//! Host-configurable Minecraft terrain generation settings.

use super::blocks::*;
use helio_pass_voxel_mesh::VOXEL_MODE_CUBES;

#[derive(Clone, Debug)]
pub struct TerrainSettings {
    pub dimensions: [u32; 3],
    pub seed: u32,
    pub sea_level: i32,
    pub base_height: i32,
    pub height_amplitude: i32,
    pub terrain_frequency: f32,
    pub cave_frequency: f32,
    pub cave_threshold: f32,
    pub ore_frequency: f32,
    pub tree_frequency: f32,
    pub enable_caves: bool,
    pub enable_ores: bool,
    pub enable_trees: bool,
    pub enable_water: bool,
    pub enable_bedrock: bool,
    pub dirt_depth: u32,
    pub beach_depth: u32,
}

impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            // Deliberately not the renderer's legacy 64³ size. This stays
            // within the demo pass's resident-brick budget while exercising
            // per-volume dimensions.
            dimensions: [80, 64, 80],
            seed: 1337,
            sea_level: 25,
            base_height: 29,
            height_amplitude: 13,
            terrain_frequency: 0.055,
            cave_frequency: 0.11,
            cave_threshold: 0.72,
            ore_frequency: 0.19,
            tree_frequency: 0.08,
            enable_caves: true,
            enable_ores: true,
            enable_trees: true,
            enable_water: true,
            enable_bedrock: true,
            dirt_depth: 4,
            beach_depth: 2,
        }
    }
}

/// Converts host-facing settings into the SceneDB component's fixed GPU row.
pub fn encode_component_settings(settings: &TerrainSettings, component: &mut helio_pass_voxel_mesh::VoxelComponent) {
    component.set_seed(settings.seed);
    component.generation = [
        settings.base_height as f32,
        settings.height_amplitude as f32,
        settings.sea_level as f32,
        settings.terrain_frequency,
        settings.cave_frequency,
        settings.cave_threshold,
        settings.ore_frequency,
        settings.tree_frequency,
        settings.dirt_depth as f32,
        settings.beach_depth as f32,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    component.generation_flags = [
        settings.enable_caves as u32
            | ((settings.enable_ores as u32) << 1)
            | ((settings.enable_trees as u32) << 2)
            | ((settings.enable_water as u32) << 3)
            | ((settings.enable_bedrock as u32) << 4),
        0,
        0,
    ];
    component.palette = default_palette();
    component.set_render_mode(VOXEL_MODE_CUBES);
}
