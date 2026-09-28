//! The CPU voxel world of a terrain entity, and its scripting surface.
//!
//! A terrain component names a registered generator, its seed and settings,
//! and an ordered edit journal. [`world_recipe`] turns that into the world
//! both the renderer and gameplay code use, so a block script, the sculpt
//! tool and the renderer always agree on every cell. Positions are world
//! metres; the terrain entity sits at the origin.
//!
//! Scripts (blueprints) call the world methods on
//! [`VoxelTerrainComponent`]: `get_block`, `set_block`, `fill_sphere`,
//! `fill_cube`, `raycast_distance` and `voxel_size`. Material ids are the
//! engine terrain materials (`helio_pass_voxel_planet::terrain::material`);
//! 0 is air.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use glam::DVec3;
use helio_pass_voxel_planet::{
    grid::Shape,
    terrain,
    Brush, BrushOp, BrushShape, Planet, PlanetRecipe, TerrainSource,
};
/// Terrain material ids (`material::GRASS`, ...) and their names.
pub use helio_pass_voxel_planet::terrain::material;
use helio_voxel_data::{VoxelBrushEdit, VoxelBrushOp, VoxelBrushShape};
use pulsar_scene_model::components::Transform;
use pulsar_scenedb::{Entity, World};

use super::{VoxelTerrainComponent, VoxelWorldShape};

/// The world of a terrain form and generator.
pub fn world_recipe(shape: VoxelWorldShape, planet_radius: f64, plane_size: f64, voxel_size: f64, source: TerrainSource) -> PlanetRecipe {
    PlanetRecipe {
        shape: match shape {
            VoxelWorldShape::Sphere => Shape::Sphere,
            VoxelWorldShape::Plane => Shape::Plane,
            VoxelWorldShape::InfinitePlane => Shape::InfinitePlane,
        },
        radius_m: planet_radius,
        plane_size_m: plane_size,
        voxel_size_m: voxel_size,
        terrain: source,
        ..PlanetRecipe::default()
    }
}

/// A journal edit as a world brush.
pub fn planet_brush(edit: &VoxelBrushEdit) -> Brush {
    Brush {
        center: edit.center,
        radius: edit.radius,
        shape: match edit.shape {
            VoxelBrushShape::Sphere => BrushShape::Sphere,
            VoxelBrushShape::Cube => BrushShape::Cube,
        },
        op: match edit.op {
            VoxelBrushOp::Remove => BrushOp::Remove,
            VoxelBrushOp::Add => BrushOp::Add,
            VoxelBrushOp::Paint => BrushOp::Paint,
        },
        material: edit.material,
    }
}

/// Engine class of the component holding a generator's settings, as the
/// generator declares it.
pub fn generator_settings_component(id: &str, version: u32) -> Option<String> {
    terrain::find(id, version).and_then(|generator| generator.info().settings_component)
}

/// The generator settings (JSON) of a terrain entity: its settings
/// component serialized when present, else `generator_parameters`.
pub fn generator_settings(world: &World, entity: Entity, component: &VoxelTerrainComponent) -> String {
    generator_settings_component(&component.generator.id, component.generator.version)
        .and_then(|class| pulsar_world_registry::get_world_component_as_engine_class(&class, world, entity))
        .and_then(|settings| settings.to_json().ok())
        .map(|json| json.to_string())
        .unwrap_or_else(|| component.generator_parameters.clone())
}

/// Whether the terrain is streamed from a registered terrain generator (as
/// opposed to externally supplied sample data).
pub fn is_generated(component: &VoxelTerrainComponent) -> bool {
    terrain::find(&component.generator.id, component.generator.version).is_some()
}

/// The recipe of a terrain entity, with its transform's uniform scale.
fn entity_recipe(world: &World, entity: Entity, component: &VoxelTerrainComponent) -> Result<PlanetRecipe, String> {
    let transform = world.get::<Transform>(entity).copied().unwrap_or_default();
    if transform.position.iter().any(|v| v.abs() > 1.0e-6) || transform.rotation.iter().any(|v| v.abs() > 1.0e-5) {
        return Err("a voxel world is centred on the world origin; move the entity to (0, 0, 0) without rotation".into());
    }
    let [sx, sy, sz] = transform.scale.map(f64::from);
    if !sx.is_finite() || sx <= 0.0 || (sx - sy).abs() > 1.0e-5 || (sx - sz).abs() > 1.0e-5 {
        return Err("voxel worlds require a positive uniform scale".into());
    }
    let source = TerrainSource {
        generator: component.generator.id.clone(),
        version: component.generator.version,
        seed: component.seed,
        settings: generator_settings(world, entity, component),
    };
    Ok(world_recipe(component.shape, component.planet_radius * sx, component.plane_size * sx, component.voxel_size * sx, source))
}

struct CachedWorld {
    recipe: PlanetRecipe,
    edits: Vec<VoxelBrushEdit>,
    planet: Arc<Planet>,
}

/// Worlds by terrain entity. A newer journal that only appends edits
/// extends the cached world.
static WORLDS: Mutex<Option<HashMap<u64, CachedWorld>>> = Mutex::new(None);

/// The CPU world of a terrain entity: its generated terrain with every
/// journal edit applied.
pub fn terrain_world(world: &World, entity: Entity) -> Result<Arc<Planet>, String> {
    let component = world.get::<VoxelTerrainComponent>(entity).ok_or("the entity has no voxel terrain")?;
    if !component.enabled {
        return Err("the voxel terrain is disabled".into());
    }
    if !is_generated(component) {
        return Err(format!("unknown terrain generator {} v{}", component.generator.id, component.generator.version));
    }
    let recipe = entity_recipe(world, entity, component)?;
    let mut worlds = WORLDS.lock().unwrap_or_else(|e| e.into_inner());
    let worlds = worlds.get_or_insert_with(HashMap::new);
    let key = entity.bits();
    if let Some(cached) = worlds.get_mut(&key) {
        if cached.recipe == recipe && component.edits.starts_with(&cached.edits) {
            let new = &component.edits[cached.edits.len()..];
            if !new.is_empty() {
                // Scripts append edit after edit: extend the world in place
                // while no caller still holds it, else a copy.
                if Arc::get_mut(&mut cached.planet).is_none() {
                    cached.planet = Arc::new((*cached.planet).clone());
                }
                let planet = Arc::get_mut(&mut cached.planet).expect("uniquely owned");
                for edit in new {
                    planet.apply(planet_brush(edit))?;
                }
                cached.edits.extend_from_slice(new);
            }
            return Ok(Arc::clone(&cached.planet));
        }
    }
    let mut planet = Planet::new(recipe.clone())?;
    for edit in &component.edits {
        planet.apply(planet_brush(edit))?;
    }
    let planet = Arc::new(planet);
    if worlds.len() >= 16 && !worlds.contains_key(&key) {
        worlds.clear();
    }
    worlds.insert(key, CachedWorld { recipe, edits: component.edits.clone(), planet: Arc::clone(&planet) });
    Ok(planet)
}

/// Append edits to a terrain's journal after checking that each applies.
pub fn append_edits(world: &mut World, entity: Entity, edits: Vec<VoxelBrushEdit>) -> Result<(), String> {
    let grid = *terrain_world(world, entity)?.grid();
    let component = world.get::<VoxelTerrainComponent>(entity).ok_or("the entity has no voxel terrain")?;
    if !component.editable {
        return Err("the voxel terrain is not editable".into());
    }
    for edit in &edits {
        if edit.op != VoxelBrushOp::Remove && (edit.material == material::AIR || edit.material >= material::COUNT) {
            return Err(format!("material {} is not a solid terrain material (1 to {})", edit.material, material::COUNT - 1));
        }
        planet_brush(edit).resolve(&grid)?;
    }
    let mut component = world.get_mut::<VoxelTerrainComponent>(entity).ok_or("the entity has no voxel terrain")?;
    component.edits.extend(edits);
    component.source_revision = component.source_revision.wrapping_add(1);
    Ok(())
}

/// The edit that makes the cell containing `p` exactly `material` (0 air).
pub fn block_edit(planet: &Planet, p: DVec3, material: u32) -> VoxelBrushEdit {
    let grid = planet.grid();
    let (cell, _) = grid.locate(p);
    VoxelBrushEdit {
        center: grid.cell_center(cell).to_array(),
        // Contains only its own cell centre.
        radius: grid.voxel_size() * 0.5,
        shape: VoxelBrushShape::Cube,
        op: if material == material::AIR { VoxelBrushOp::Remove } else { VoxelBrushOp::Add },
        material,
    }
}

/// The edit that fills (or, with material 0, clears) a sphere or cube.
pub fn shape_edit(center: DVec3, radius: f64, shape: VoxelBrushShape, material: u32) -> Result<VoxelBrushEdit, String> {
    if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return Err("a fill needs a finite centre and a positive radius".into());
    }
    Ok(VoxelBrushEdit {
        center: center.to_array(),
        radius,
        shape,
        op: if material == material::AIR { VoxelBrushOp::Remove } else { VoxelBrushOp::Add },
        material,
    })
}

/// A camera pose framing a voxel world from `eye`: `height` metres above
/// the ground under the eye (on a finite plane, the nearest point well
/// inside it), looking along the local horizon in the direction of
/// `forward`, tilted slightly down. Returns the position and view
/// direction.
pub fn frame_view(planet: &Planet, eye: DVec3, forward: DVec3, height: f64) -> (DVec3, DVec3) {
    let grid = planet.grid();
    let column = match grid.shape() {
        Shape::Sphere => eye.try_normalize().unwrap_or(DVec3::Y) * grid.radius(),
        Shape::Plane => {
            let half = f64::from(grid.cells()) * grid.voxel_size() * 0.4;
            DVec3::new(eye.x.clamp(-half, half), 0.0, eye.z.clamp(-half, half))
        }
        Shape::InfinitePlane => DVec3::new(eye.x, 0.0, eye.z),
    };
    let position = planet.surface_point(column, height);
    let up = grid.up(position);
    let along = forward - up * forward.dot(up);
    let along = along.try_normalize().unwrap_or_else(|| up.any_orthonormal_vector());
    let tilt = 0.2f64;
    (position, (along * tilt.cos() - up * tilt.sin()).normalize())
}

fn position(x: f64, y: f64, z: f64) -> Result<DVec3, String> {
    let p = DVec3::new(x, y, z);
    if p.is_finite() { Ok(p) } else { Err("positions must be finite".into()) }
}

// Scripting surface. Blocks are the world's exact base cells.
#[pulsar_scenedb::component_methods]
impl VoxelTerrainComponent {
    /// Material of the block containing the point (0 when it is air).
    #[world_method(pure, category = "Voxel")]
    fn get_block(world: &World, entity: Entity, x: f64, y: f64, z: f64) -> Result<u32, String> {
        let planet = terrain_world(world, entity)?;
        let (cell, _) = planet.grid().locate(position(x, y, z)?);
        Ok(planet.material(cell))
    }

    /// Make the block containing the point `material` (0 removes it).
    #[world_method(category = "Voxel")]
    fn set_block(world: &mut World, entity: Entity, x: f64, y: f64, z: f64, material: u32) -> Result<(), String> {
        let planet = terrain_world(world, entity)?;
        let edit = block_edit(&planet, position(x, y, z)?, material);
        append_edits(world, entity, vec![edit])
    }

    /// Fill every block whose centre lies within `radius` of the point with
    /// `material` (0 digs a hole).
    #[world_method(category = "Voxel")]
    fn fill_sphere(world: &mut World, entity: Entity, x: f64, y: f64, z: f64, radius: f64, material: u32) -> Result<(), String> {
        let edit = shape_edit(position(x, y, z)?, radius, VoxelBrushShape::Sphere, material)?;
        append_edits(world, entity, vec![edit])
    }

    /// Fill every block whose centre lies in the cube of half size
    /// `half_size` around the point, aligned with the ground.
    #[world_method(category = "Voxel")]
    fn fill_cube(world: &mut World, entity: Entity, x: f64, y: f64, z: f64, half_size: f64, material: u32) -> Result<(), String> {
        let edit = shape_edit(position(x, y, z)?, half_size, VoxelBrushShape::Cube, material)?;
        append_edits(world, entity, vec![edit])
    }

    /// Distance along the ray to the first solid block, or -1 when none is
    /// hit within `max_distance`.
    #[world_method(pure, category = "Voxel")]
    #[allow(clippy::too_many_arguments)]
    fn raycast_distance(
        world: &World,
        entity: Entity,
        x: f64,
        y: f64,
        z: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        max_distance: f64,
    ) -> Result<f64, String> {
        let planet = terrain_world(world, entity)?;
        let direction = position(dx, dy, dz)?;
        if direction.length_squared() == 0.0 {
            return Err("the ray direction is zero".into());
        }
        Ok(planet.raycast(position(x, y, z)?, direction, max_distance).map_or(-1.0, |hit| hit.distance))
    }

    /// Edge length of one block in metres.
    #[world_method(pure, category = "Voxel")]
    fn voxel_size(world: &World, entity: Entity) -> Result<f64, String> {
        Ok(terrain_world(world, entity)?.grid().voxel_size())
    }
}
