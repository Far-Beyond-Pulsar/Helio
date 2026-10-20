//! A meadow for voxel worlds from the engine's foliage: grass and flowers as blocky
//! blades (`FoliageKind::Block`: leaning stacks of 5 cm cubes on one grid, gathered in
//! tufts) keyed to the grass material, grown on the terrain the chunk pass draws (its
//! foliage ground) and gone where it is dug. Hosts spawn [`Meadow::types`] and
//! [`Meadow::wind`] as SceneDB rows and size the renderer's foliage with
//! [`Meadow::blades_per_m2`] and [`Meadow::ring_radius_m`]. Tiles are 8 m in the frame
//! of the cube face under the eye, so the cost is the same on any planet.

use helio_pass_foliage_place::components::{FoliageTypeComponent, FoliageWindComponent};
use helio_pass_foliage_place::{pack_kind_and_flags, FoliageKind, FoliageMaterial, GpuFoliageType, FOLIAGE_FLAG_RECEIVES_INTERACTION};

use crate::terrain::material;

/// The grass's cube edge, metres.
pub const CUBE_M: f32 = 0.05;

/// A meadow's foliage rows and the renderer settings it is drawn with.
pub struct Meadow {
    /// The blade types, one SceneDB row each.
    pub types: Vec<FoliageTypeComponent>,
    /// The wind, one SceneDB row.
    pub wind: FoliageWindComponent,
    /// Blades a square metre the foliage arena is sized for (every type's together):
    /// a candidate every 8 cm, each owning one or two cubes. Tufts thin their rims,
    /// so about two thirds of them grow. `RendererConfig::foliage_blades_per_m2`.
    pub blades_per_m2: f32,
    /// The resident ring's radius: the blades end (`lod_distances`) well inside it.
    /// `RendererConfig::foliage_ring_radius`.
    pub ring_radius_m: f32,
}

/// A blocky blade type of `colour` (sRGB) on `ground`, stacks `height_m` tall.
fn blades(ground: u32, density: f32, height_m: [f32; 2], colour: [u8; 3], lod: [f32; 4]) -> FoliageTypeComponent {
    let linear = colour.map(|v| (f32::from(v) / 255.0).powf(2.2));
    let mut t = GpuFoliageType {
        density,
        height_range: height_m,
        width_range: [CUBE_M; 2],
        // Any slope the grass voxels lie on (their tops are level steps).
        slope_range: [-1.0, 1.0],
        altitude_range: [f32::MIN, f32::MAX],
        lod_distances: lod,
        wind_response: [0.0, 0.25, 0.6],
        interaction_stiffness: 6.0,
        kind_and_flags: pack_kind_and_flags(FoliageKind::Block, FOLIAGE_FLAG_RECEIVES_INTERACTION),
        ground_mask: 1 << ground,
        ..GpuFoliageType::default()
    };
    t.set_material(Some(FoliageMaterial { base_color: linear, roughness: 0.9, metallic: 0.0 }));
    t.into()
}

/// Tall and short grass with yellow, white and violet flowers on [`material::GRASS`].
pub fn meadow() -> Meadow {
    let grass = material::GRASS;
    let grass_lod = [5.0, 12.0, 24.0, 36.0];
    let flower_lod = [5.0, 10.0, 18.0, 26.0];
    Meadow {
        types: vec![
            // Tall and short grass, a shade lighter than the turf they stand on.
            blades(grass, 80.0, [0.15, 0.30], [86, 120, 56], grass_lod),
            blades(grass, 60.0, [0.05, 0.15], [78, 112, 52], grass_lod),
            blades(grass, 1.6, [0.10, 0.20], [232, 206, 64], flower_lod),
            blades(grass, 1.2, [0.10, 0.15], [236, 236, 228], flower_lod),
            blades(grass, 0.8, [0.10, 0.20], [150, 96, 196], flower_lod),
        ],
        wind: FoliageWindComponent { direction_speed: [1.0, 0.0, 0.3, 1.5], gust: [0.5, 0.2, 0.0, 0.05], time_prev_time: [0.0, 0.0], _pad: [0.0; 2] },
        blades_per_m2: 150.0,
        ring_radius_m: 48.0,
    }
}
