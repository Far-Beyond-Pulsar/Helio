//! Grass and flowers on the world's grass, from the engine's foliage: blocky
//! blades (thin boxes, `FoliageKind::Block`) keyed to the grass material, so a
//! meadow at 10 cm voxels reads as voxel grass. Placed on the terrain the
//! voxel planet draws (its foliage ground), gone where it is dug.

use helio_pass_foliage_place::components::{FoliageTypeComponent, FoliageWindComponent};
use helio_pass_foliage_place::{pack_kind_and_flags, FoliageKind, FoliageMaterial, GpuFoliageType, FOLIAGE_FLAG_RECEIVES_INTERACTION};
use helio_pass_voxel_planet::terrain::material;
use pulsar_scenedb::World;

/// Blades a square metre the foliage arena is sized for (every type's together).
pub const DENSITY: f32 = 150.0;
/// The resident ring's radius: the blades end (`lod_distances`) well inside it.
pub const RING_RADIUS_M: f32 = 48.0;

/// A blocky blade type of `colour` (sRGB) on grass.
fn blades(density: f32, height_m: [f32; 2], width_m: [f32; 2], colour: [u8; 3], lod: [f32; 4]) -> FoliageTypeComponent {
    let linear = colour.map(|v| (f32::from(v) / 255.0).powf(2.2));
    let mut t = GpuFoliageType {
        density,
        height_range: height_m,
        width_range: width_m,
        // Any slope the grass voxels lie on (their tops are level steps).
        slope_range: [-1.0, 1.0],
        altitude_range: [f32::MIN, f32::MAX],
        lod_distances: lod,
        wind_response: [0.0, 0.25, 0.6],
        interaction_stiffness: 6.0,
        kind_and_flags: pack_kind_and_flags(FoliageKind::Block, FOLIAGE_FLAG_RECEIVES_INTERACTION),
        ground_mask: 1 << material::GRASS,
        ..GpuFoliageType::default()
    };
    t.set_material(Some(FoliageMaterial { base_color: linear, roughness: 0.9, metallic: 0.0 }));
    t.into()
}

/// The meadow's types and its wind, as SceneDB rows.
pub fn plant(world: &mut World) {
    let grass_lod = [6.0, 14.0, 26.0, 40.0];
    let flower_lod = [6.0, 12.0, 20.0, 28.0];
    for row in [
        // Tall and short grass, a shade lighter than the turf they stand on.
        blades(55.0, [0.14, 0.38], [0.02, 0.03], [86, 120, 56], grass_lod),
        blades(85.0, [0.05, 0.14], [0.02, 0.03], [78, 112, 52], grass_lod),
        // Flowers: yellow, white and violet.
        blades(1.6, [0.16, 0.28], [0.03, 0.04], [232, 206, 64], flower_lod),
        blades(1.2, [0.12, 0.22], [0.03, 0.04], [236, 236, 228], flower_lod),
        blades(0.8, [0.18, 0.30], [0.03, 0.04], [150, 96, 196], flower_lod),
    ] {
        let entity = world.spawn();
        world.insert(entity, row);
    }
    let wind = world.spawn();
    world.insert(wind, FoliageWindComponent { direction_speed: [1.0, 0.0, 0.3, 1.5], gust: [0.5, 0.2, 0.0, 0.05], time_prev_time: [0.0, 0.0], _pad: [0.0; 2] });
}
