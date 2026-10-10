//! Pluggable terrain generation.
//!
//! A [`TerrainGenerator`] turns a seed and its settings (JSON) into a
//! [`TerrainField`] for one world grid: the surface height and ground
//! material of every column. The field is evaluated twice, identically: on
//! the CPU for collision, ray casts, edits and gameplay queries, and by its
//! WGSL [`TerrainProgram`] on the GPU for streaming and rendering. Worlds
//! name their generator in a [`TerrainSource`]; generators register once
//! per process with [`register`], and [`generators`] lists them for editor
//! pickers.
//!
//! # Writing a generator
//!
//! The WGSL program defines
//!
//! ```wgsl
//! struct TerrainConstants { /* the bytes of TerrainProgram::constants */ }
//! fn terrain_height(p: vec3<i32>, level: u32) -> i32
//! fn ground_material(p: vec3<i32>, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32
//! ```
//!
//! and optionally a surface word per column cell, computed once when the
//! column is generated and stored with it (8 bits): whatever the materials
//! need besides height (biome, sediment, crater age), so shading never
//! runs the generator per pixel.
//!
//! ```wgsl
//! fn terrain_surface(p: vec3<i32>, level: u32, height: i32) -> u32
//! ```
//!
//! A program whose height and surface word share their work (or that is
//! large) may also define both at once for generation, which then calls it
//! at a single site (shader compilers inline every call, so each call site
//! of a large program costs compile time; `display` asks for the program's
//! coarse display variant, if it has one, and may be ignored):
//!
//! ```wgsl
//! fn terrain_column(p: vec3<i32>, level: u32, display: bool) -> vec2<i32> // (height, surface word)
//! ```
//!
//! and optionally, for volumetric terrain (caves, overhangs, arches),
//!
//! ```wgsl
//! fn terrain_extent(p: vec3<i32>, level: u32) -> vec2<i32>
//! fn terrain_density(p: vec3<i32>, q: vec3<i32>, level: u32, top: i32, height: i32, lean_height: i32, k: i32) -> i32
//! fn terrain_clearance(q: vec3<i32>, level: u32, step: f32) -> i32 // optional
//! ```
//!
//! `terrain_clearance` is a conservative count of level cells along a lane,
//! from the cell at volume point `q` (the point moving `step` units per
//! cell), in which the volume cannot change a cell below the heightfield's
//! lean band (caves cannot carve there); generation steps over them. Without
//! it every cell of the extent is evaluated.
//!
//! and, to lean its surface (overhangs that are a continuous deformation of
//! the heightfield: nothing floats),
//!
//! ```wgsl
//! fn terrain_lean(level: u32) -> vec2<i32> // (reach, spacing) in level cells
//! fn terrain_lean_offset(p: vec3<i32>, i: i32, j: i32, k: i32, level: u32) -> vec2<i32> // Q8 level cells
//! ```
//!
//! The engine evaluates the field's height on a lattice of nodes every
//! `spacing` cells, at the offset point under each cell, and passes it to
//! `terrain_density` as `lean_height` ([`lean_height`]).
//!
//! `terrain_extent` bounds, in level cells, how far below the heightfield
//! top (first air layer) and how far above it the column's cells may
//! differ from the heightfield; `(0, 0)` means none (the default).
//! `terrain_density` is the signed distance from the centre of layer `k`
//! to the surface, in [`DENSITY_ONE`] units per level cell, positive inside
//! solid, in a column whose heightfield top is `top` (its field height
//! `height`, mm: the exact surface the volume folds, so that coarse levels
//! show it, not a floor of their cell); `q` is the cell centre's seamless
//! 3D domain point (`volume_point`) and `lean_height` the leaning
//! heightfield's height under it (`height` without lean). The cell is solid
//! where it is positive. Outside the extent its sign must be the
//! heightfield's (`k < top`). Without them the engine uses the heightfield
//! ([`heightfield_density`]). Distances, not just solid or air, are what a
//! smooth surface (interpolated between cell centres) needs.
//!
//! The program reads its constants from the `terrain` uniform. It may call the
//! integer noise library (`shaders/noise.wgsl`, mirrored in
//! [`crate::noise`]) and the material ids (`M_*`, see [`material`]).
//! Heights are integer [`HEIGHT_ONE`] units (millimetres) above the datum;
//! `p` is the column centre in domain units ([`crate::grid::DOMAIN_UNIT`],
//! 1.25 cm) on the sphere of the planet's radius (a plane: its horizontal
//! position), independent of the world's voxel size, and `level` the
//! column's footprint, `2^level`
//! reference cells wide, so a field can omit detail finer than it. The CPU
//! and GPU functions must return equal values for every input:
//! [`crate::engine::verify_field`] compares them, and [`check_field`] checks
//! the declared bounds against sampled columns.
use crate::grid::{Grid, PLANE_FACE};
use crate::planet::Planet;
use glam::IVec3;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

/// Heights are integer millimetres above the datum (sea level).
pub const HEIGHT_ONE: i32 = 1000;
/// Largest `TerrainConstants` uniform a program may declare.
pub const MAX_CONSTANT_BYTES: usize = 4096;

/// Engine material ids (`M_*` in WGSL). A ground material may carry
/// [`material::SPECK`].
pub mod material {
    pub const AIR: u32 = 0;
    pub const GRASS: u32 = 1;
    pub const DIRT: u32 = 2;
    pub const STONE: u32 = 3;
    pub const SAND: u32 = 4;
    pub const SNOW: u32 = 5;
    pub const WATER: u32 = 6;
    pub const GRAVEL: u32 = 7;
    pub const SANDSTONE: u32 = 8;
    pub const DARK_STONE: u32 = 9;
    pub const WOOD: u32 = 10;
    pub const LEAVES: u32 = 11;
    pub const CLAY: u32 = 12;
    pub const BRICK: u32 = 13;
    pub const PLANKS: u32 = 14;
    pub const COBBLE: u32 = 15;
    /// Lunar ground: fine regolith, dark mare plains, bright fresh ejecta,
    /// boulders, and the basalt and anorthosite bedrock under them.
    pub const REGOLITH: u32 = 16;
    pub const MARE: u32 = 17;
    pub const EJECTA: u32 = 18;
    pub const BOULDER: u32 = 19;
    pub const BASALT: u32 = 20;
    pub const ANORTHOSITE: u32 = 21;
    /// Ids in use, air included (every terrain program shares them, so a
    /// painted material means the same on any world).
    pub const COUNT: u32 = 22;
    /// Flag on a ground material: a single-voxel fleck of a surface (mud
    /// in a meadow) that blends into grass once its cell is about a pixel
    /// wide instead of tinting the distant ground.
    pub const SPECK: u32 = 0x100;
    /// Bits of the material id.
    pub const ID: u32 = 0xff;

    /// Names of the solid materials, in id order after air.
    pub const NAMES: [&str; COUNT as usize - 1] = [
        "Grass", "Dirt", "Stone", "Sand", "Snow", "Water", "Gravel", "Sandstone", "DarkStone", "Wood", "Leaves", "Clay",
        "Brick", "Planks", "Cobble", "Regolith", "Mare", "Ejecta", "Boulder", "Basalt", "Anorthosite",
    ];

    /// The material id of a name in [`NAMES`] (ignoring case and `_`).
    pub fn from_name(name: &str) -> Option<u32> {
        let key: String = name.chars().filter(|c| *c != '_').collect::<String>().to_ascii_lowercase();
        NAMES.iter().position(|n| n.to_ascii_lowercase() == key).map(|i| i as u32 + 1)
    }
}

/// The GPU half of a [`TerrainField`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainProgram {
    /// Identity of `wgsl`: equal keys must mean equal source. Pipelines are
    /// compiled once per key, so a settings change that only changes
    /// `constants` never recompiles shaders.
    pub key: Cow<'static, str>,
    /// Defines `TerrainConstants`, `terrain_height` and `ground_material`.
    pub wgsl: Cow<'static, str>,
    /// Bytes of the `TerrainConstants` uniform (WGSL uniform layout, at
    /// most [`MAX_CONSTANT_BYTES`]).
    pub constants: Vec<u8>,
}

/// Surface height and ground material of every column of one world grid.
pub trait TerrainField: Send + Sync + 'static {
    /// Surface height ([`HEIGHT_ONE`] units above the datum) of the column
    /// centred at domain point `p` whose footprint is `2^level` reference
    /// cells.
    fn height(&self, p: IVec3, level: u32) -> i32;
    /// Material of a solid ground cell: `surface` is its column cell's
    /// surface word ([`Self::surface`]), `top_height` its column's top
    /// ([`HEIGHT_ONE`] units, a whole number of voxel layers), `depth` the
    /// cells below the column top (0 is the exposed top cell), `slope` the
    /// ground slope across the cell's 8x8 column block in eighths of a cell
    /// per cell, and `layer` the cell's layer index (0 is the layer just
    /// above the datum).
    fn ground_material(&self, p: IVec3, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32;
    /// Surface word (8 bits) of the column at `p` whose height is `height`
    /// (`terrain_surface` in WGSL, stored per column cell at generation):
    /// inputs of the materials besides height. 0 when the program has none.
    fn surface(&self, _p: IVec3, _level: u32, _height: i32) -> u32 {
        0
    }
    /// Lowest and highest height any column can have at any level.
    fn height_range(&self) -> (i32, i32);
    /// Per grid level `L >= 1`: how many level-`L` cells the top of any
    /// finer column inside a level-`L` column may rise above that column's
    /// own top, plus two. Coarse levels bound the terrain with this, so it
    /// must be conservative; [`check_field`] samples it.
    fn bound_margins(&self) -> [i32; 24];
    /// Level cells below and above the heightfield top in which the cells of
    /// the column at domain point `p` may differ from the heightfield
    /// (`terrain_extent` in WGSL). `(0, 0)`: a pure heightfield column.
    fn extent(&self, _p: IVec3, _level: u32) -> (i32, i32) {
        (0, 0)
    }
    /// Lean lattice of `level` (`terrain_lean` in WGSL): `(reach,
    /// spacing)` in level cells, `(0, 0)` without lean. See
    /// [`Self::lean_offset`].
    fn lean(&self, _level: u32) -> (i32, i32) {
        (0, 0)
    }
    /// Horizontal displacement (level cells, Q8, `|offset| < reach` cells)
    /// of the heightfield under cell `(i, j, k)` of the column at `p`
    /// (`terrain_lean_offset` in WGSL). The engine evaluates the field's
    /// height there ([`lean_height`]) and passes it to [`Self::density`]:
    /// a field leaning its surface this way is a continuous deformation of
    /// its heightfield (overhangs on steep ground, no floating pieces), as
    /// long as the offset varies slowly across the ground.
    fn lean_offset(&self, _p: IVec3, _i: i32, _j: i32, _k: i32, _level: u32) -> (i32, i32) {
        (0, 0)
    }
    /// Signed distance ([`DENSITY_ONE`] per level cell, positive inside
    /// solid) from the centre of layer `k` of the column at `p`, whose
    /// heightfield top is `top` (field height `height`, mm), to the surface;
    /// `q` is the cell's 3D domain point ([`Grid::volume_point`]) and
    /// `lean_height` the height of the heightfield displaced by
    /// [`Self::lean_offset`] under the cell (`height` without lean). The
    /// cell is solid where it is positive; outside [`Self::extent`] its sign
    /// must be [`terrain_kind`]'s (`terrain_density` in WGSL).
    fn density(&self, _p: IVec3, _q: IVec3, _level: u32, top: i32, _height: i32, _lean_height: i32, k: i32) -> i32 {
        heightfield_density(top, k)
    }
    /// Largest depth (mm) below the surface and height above it at which
    /// any level's cells may differ from the heightfield: world bounds
    /// (outer and inner radius, camera clearance) include them.
    fn volume_bounds(&self) -> (i32, i32) {
        (0, 0)
    }
    /// The generator's own material table and detail (the renderer's
    /// appearance unless the world overrides it).
    fn appearance(&self) -> TerrainAppearance {
        TerrainAppearance::default()
    }
    fn program(&self) -> TerrainProgram;
}

/// How one terrain material looks. Shading knows materials only through
/// these properties (no material is special), so any generator can define
/// its own: regolith and basalt on a moon, coloured sands elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialAppearance {
    /// sRGB colour in [0, 1] and perceptual roughness.
    pub colour: [f32; 4],
    /// Dry, middle and lush world-space patch colours (sRGB), when the
    /// material varies by patches (turf).
    pub patches: Option<[[f32; 3]; 3]>,
    /// Material showing on the sides of this material's surface cells below
    /// a lip (soil under turf).
    pub lip: Option<u8>,
    /// Material of this one's single-voxel flecks and their share, averaged
    /// into its colour once flecks are below a pixel.
    pub fleck: Option<(u8, f32)>,
    /// Material a filtered single-voxel speck of this one blends into.
    pub speck_host: Option<u8>,
}

impl Default for MaterialAppearance {
    fn default() -> Self {
        Self { colour: [1.0, 0.0, 1.0, 0.9], patches: None, lip: None, fleck: None, speck_host: None }
    }
}

/// Material slots of a world's appearance table (ids 0..MATERIALS; the
/// built-in ones are [`material::NAMES`]).
pub const MATERIALS: usize = 32;

/// Art controls, independent of occupancy, terrain recipes and edit journals.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TerrainAppearance {
    /// Per material id (see [`MaterialAppearance`]).
    pub materials: [MaterialAppearance; MATERIALS],
    /// Patch contrast, voxel pigment contrast, edge darkening, step
    /// softness (share of the slope's normal in natural voxel steps' light).
    pub detail: [f32; 4],
}

impl Default for TerrainAppearance {
    /// The built-in generators' materials (`terrain::material`).
    fn default() -> Self {
        use crate::terrain::material::*;
        // Air and unused slots are magenta (a missing material shows).
        let colours: [[u8; 3]; COUNT as usize] = [
            [200, 0, 200], [106, 126, 68], [128, 96, 66], [133, 139, 142],
            [203, 188, 151], [217, 228, 236], [28, 72, 92], [116, 111, 102],
            [185, 142, 104], [82, 88, 95], [101, 75, 53], [59, 102, 52],
            [155, 113, 89], [148, 77, 63], [158, 119, 79], [121, 126, 130],
            [142, 140, 135], [84, 84, 86], [196, 194, 188], [112, 110, 106],
            [58, 59, 63], [168, 166, 158],
        ];
        let roughness = [
            0.9, 0.94, 0.96, 0.84, 0.93, 0.78, 0.35, 0.9, 0.88, 0.82, 0.97, 0.94, 0.92, 0.86, 0.86, 0.85,
            0.95, 0.95, 0.93, 0.85, 0.8, 0.85,
        ];
        let unit = |c: [u8; 3]| c.map(|v| f32::from(v) / 255.0);
        let mut materials: [MaterialAppearance; MATERIALS] = std::array::from_fn(|i| match colours.get(i) {
            Some(&c) => {
                let [r, g, b] = unit(c);
                MaterialAppearance { colour: [r, g, b, roughness[i]], ..Default::default() }
            }
            None => MaterialAppearance::default(),
        });
        let turf = &mut materials[GRASS as usize];
        turf.patches = Some([unit([160, 150, 96]), unit([106, 126, 68]), unit([62, 98, 56])]);
        turf.lip = Some(DIRT as u8);
        for speck in [DIRT, SAND] {
            materials[speck as usize].speck_host = Some(GRASS as u8);
        }
        for rock in [STONE, DARK_STONE, SANDSTONE] {
            materials[rock as usize].fleck = Some((DIRT as u8, 0.125));
        }
        // Weathered and fresh stone: lichen-warm, the rock's grey, darker.
        materials[STONE as usize].patches = Some([unit([160, 157, 149]), unit([133, 139, 142]), unit([98, 104, 110])]);
        // Dark stone alternates with stone in 4.5 m strata on cliffs: a shade
        // darker, not a band of another colour (zebra stripes on outcrops).
        materials[DARK_STONE as usize].patches = Some([unit([140, 138, 131]), unit([110, 115, 119]), unit([82, 87, 94])]);
        // Lunar ground varies in maturity (fresh is brighter); fresh ejecta
        // and boulders thin out into the regolith around them.
        materials[REGOLITH as usize].patches = Some([unit([158, 156, 150]), unit([142, 140, 135]), unit([118, 117, 113])]);
        materials[MARE as usize].patches = Some([unit([98, 98, 99]), unit([84, 84, 86]), unit([68, 68, 71])]);
        materials[BOULDER as usize].patches = Some([unit([130, 128, 123]), unit([112, 110, 106]), unit([90, 89, 86])]);
        materials[EJECTA as usize].speck_host = Some(REGOLITH as u8);
        // Steps of natural ground are lit almost wholly with the slope: voxels
        // read through their own colours (pigment, hue), not as dark contour
        // lines along every riser.
        Self { materials, detail: [0.3, 0.4, 0.08, 0.9] }
    }
}

/// What a generator is, for registration and editor pickers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratorInfo {
    /// Stable id stored in worlds, e.g. `helio.terrain`.
    pub id: String,
    /// Output version: a change that alters any generated column is a new
    /// version, so saved worlds keep the terrain they were edited against.
    pub version: u32,
    pub name: String,
    pub description: String,
    /// Engine class name of the component that holds this generator's
    /// settings, if it has one; its serialized form is the settings JSON.
    pub settings_component: Option<String>,
}

pub trait TerrainGenerator: Send + Sync + 'static {
    fn info(&self) -> GeneratorInfo;
    /// The field of `grid` for `seed` and `settings` (JSON; empty means
    /// the generator's defaults).
    fn build(&self, grid: &Grid, seed: u64, settings: &str) -> Result<Arc<dyn TerrainField>, String>;
}

/// The generator a world uses: its id, version, seed and settings.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct TerrainSource {
    pub generator: String,
    /// Generator output version; 0 accepts the registered version, which
    /// `Planet::recipe` then names.
    pub version: u32,
    pub seed: u64,
    /// Generator settings (JSON); empty means its defaults.
    pub settings: String,
}

impl Default for TerrainSource {
    fn default() -> Self {
        Self {
            generator: crate::layers::ID.into(),
            version: 0,
            seed: 7,
            settings: String::new(),
        }
    }
}

type Registry = RwLock<BTreeMap<String, Arc<dyn TerrainGenerator>>>;

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut map: BTreeMap<String, Arc<dyn TerrainGenerator>> = BTreeMap::new();
        for generator in [Arc::new(crate::layers::LayersGenerator) as Arc<dyn TerrainGenerator>] {
            map.insert(generator.info().id, generator);
        }
        RwLock::new(map)
    })
}

/// Make a generator available to every world in this process.
pub fn register(generator: Arc<dyn TerrainGenerator>) -> Result<(), String> {
    let info = generator.info();
    if info.id.is_empty() || info.version == 0 {
        return Err("a terrain generator needs an id and a nonzero version".into());
    }
    let mut map = registry().write().map_err(|_| "terrain registry poisoned")?;
    if map.contains_key(&info.id) {
        return Err(format!("terrain generator {} is already registered", info.id));
    }
    map.insert(info.id, generator);
    Ok(())
}

/// The generator registered as `id`, if it has `version` (0 accepts the
/// registered version). One version of each generator is registered.
pub fn find(id: &str, version: u32) -> Option<Arc<dyn TerrainGenerator>> {
    let generator = registry().read().ok()?.get(id).cloned()?;
    (version == 0 || generator.info().version == version).then_some(generator)
}

/// Every registered generator, by name.
pub fn generators() -> Vec<GeneratorInfo> {
    let mut list: Vec<_> = registry().read().map(|map| map.values().map(|g| g.info()).collect()).unwrap_or_default();
    list.sort_by(|a, b| a.name.cmp(&b.name));
    list
}

/// Build the field of `source` for `grid`.
pub fn build(source: &TerrainSource, grid: &Grid) -> Result<Arc<dyn TerrainField>, String> {
    let generator = find(&source.generator, source.version)
        .ok_or_else(|| format!("unknown terrain generator {} v{}", source.generator, source.version))?;
    let field = generator.build(grid, source.seed, &source.settings)?;
    let (lo, hi) = field.height_range();
    let program = field.program();
    if lo > hi || program.constants.len() > MAX_CONSTANT_BYTES || program.key.is_empty() {
        return Err(format!("terrain generator {} built an invalid field", source.generator));
    }
    Ok(field)
}

/// First air layer above a column of `height` at `level` (level cells).
#[inline]
pub fn top_cells(grid: &Grid, height: i32, level: u32) -> i32 {
    height.div_euclid(grid.layer_mm() as i32) >> level
}

/// Heightfield terrain kind at a level cell before edits: 0 air, 1 solid.
#[inline]
pub fn terrain_kind(top: i32, k: i32) -> u32 {
    u32::from(k < top)
}

/// Density units per level cell ([`TerrainField::density`]).
pub const DENSITY_ONE: i32 = 256;
/// Largest density magnitude, in level cells.
pub const DENSITY_CELLS: i32 = 1 << 16;

/// Density of a heightfield cell: the distance from the centre of layer `k`
/// to the column top `top` (`heightfield_density` in WGSL).
#[inline]
pub fn heightfield_density(top: i32, k: i32) -> i32 {
    top.saturating_sub(k).clamp(-DENSITY_CELLS, DENSITY_CELLS) * DENSITY_ONE - DENSITY_ONE / 2
}

/// Generated kind of a level cell before edits, volumetric terms included
/// (the GPU's per-cell generation rule), in the column of field height
/// `height` (mm).
pub fn generated_kind(grid: &Grid, field: &dyn TerrainField, face: u8, i: i32, j: i32, k: i32, level: u32, height: i32) -> u32 {
    let top = top_cells(grid, height, level);
    let p = grid.domain_point(face, i, j, level);
    let (below, above) = field.extent(p, level);
    if (below == 0 && above == 0) || k < top - below || k >= top + above {
        return terrain_kind(top, k);
    }
    let lean = lean_height(grid, field, face, i, j, k, level, height);
    u32::from(field.density(p, grid.volume_point(face, i, j, k, level), level, top, height, lean, k) > 0)
}

/// Height (mm) of the heightfield displaced by the field's lean under cell
/// `(i, j, k)` (`generate.wgsl` mirrors it): the field's height on the
/// level's lean lattice, interpolated bilinearly at the displaced point,
/// plus the column's own detail off that lattice (`height` minus the
/// lattice at the cell). Without displacement it is exactly `height`.
///
/// The lattice is global (nodes every `spacing` cells of the face, height
/// at the node's cell), so neighbouring columns agree; a column needs the
/// 8x8 nodes from `div_floor(8 ci - reach, spacing)`, which the GPU's 64
/// lanes evaluate once per column.
pub fn lean_height(grid: &Grid, field: &dyn TerrainField, face: u8, i: i32, j: i32, k: i32, level: u32, height: i32) -> i32 {
    let (reach, spacing) = field.lean(level);
    if reach == 0 {
        return height;
    }
    let p = grid.domain_point(face, i, j, level);
    let (ox, oy) = field.lean_offset(p, i, j, k, level);
    if ox == 0 && oy == 0 {
        return height;
    }
    let last = (grid.cells() >> level) - 1;
    let (node_i, node_j) = lattice_origin(i, j, reach, spacing);
    let node = |n: i32, m: i32| {
        let (ni, nj) = (((node_i + n) * spacing).clamp(0, last), ((node_j + m) * spacing).clamp(0, last));
        lattice_height(grid, field, face, ni, nj, level)
    };
    // Q8 cell positions overflow i32 at planet-scale indices. The GPU mirror
    // computes them in wrapping i32 and only the small difference from the
    // lattice origin matters, so wrap here too: bit-identical, never a panic.
    let origin = |n: i32| n.wrapping_mul(spacing).wrapping_mul(256);
    let sample = |x: i32, y: i32| lattice_bilinear(x.wrapping_sub(origin(node_i)), y.wrapping_sub(origin(node_j)), spacing, &node);
    let (x, y) = (i.wrapping_mul(256).wrapping_add(128), j.wrapping_mul(256).wrapping_add(128));
    sample(x.wrapping_add(ox), y.wrapping_add(oy)).wrapping_add(height).wrapping_sub(sample(x, y))
}

/// First node of the lean lattice a column of 8x8 cells reads: the nodes
/// from `div_floor(8 ci - reach, spacing)` cover the column and its reach.
fn lattice_origin(i: i32, j: i32, reach: i32, spacing: i32) -> (i32, i32) {
    (((i >> 3) * 8 - reach).div_euclid(spacing), ((j >> 3) * 8 - reach).div_euclid(spacing))
}

/// Bilinear height of the lean lattice at `(x, y)` (Q8 level cells from
/// node (0, 0)), nodes `0..8` per axis.
pub(crate) fn lattice_bilinear(x: i32, y: i32, spacing: i32, node: &dyn Fn(i32, i32) -> i32) -> i32 {
    let step = spacing * 256;
    let ni = x.div_euclid(step).clamp(0, 6);
    let nj = y.div_euclid(step).clamp(0, 6);
    let fu = ((x - ni * step) / spacing).clamp(0, 256);
    let fv = ((y - nj * step) / spacing).clamp(0, 256);
    let h00 = node(ni, nj);
    let h10 = node(ni + 1, nj);
    let h01 = node(ni, nj + 1);
    let h11 = node(ni + 1, nj + 1);
    let a = h00.wrapping_add(h10.wrapping_sub(h00).wrapping_mul(fu) >> 8);
    let b = h01.wrapping_add(h11.wrapping_sub(h01).wrapping_mul(fu) >> 8);
    a.wrapping_add(b.wrapping_sub(a).wrapping_mul(fv) >> 8)
}

/// The field's height at lattice node cell `(i, j)` of `level`, cached per
/// thread (ray casts through a leaning region query the same nodes).
fn lattice_height(grid: &Grid, field: &dyn TerrainField, face: u8, i: i32, j: i32, level: u32) -> i32 {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static CACHE: RefCell<HashMap<(usize, u8, i32, i32, u32), i32>> = RefCell::new(HashMap::new());
    }
    let key = (field as *const dyn TerrainField as *const () as usize, face, i, j, level);
    if let Some(h) = CACHE.with(|c| c.borrow().get(&key).copied()) {
        return h;
    }
    let h = field.height(grid.domain_point(face, i, j, level), level + grid.level_offset());
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 1 << 16 {
            c.clear();
        }
        c.insert(key, h);
    });
    h
}

/// Generated top of a column (level cells): the first air above its highest
/// generated solid cell, edits excluded. Equals `top` outside the field's
/// volumetric extent; material depth counts from it (`generate.wgsl`).
/// `height` is the column's field height (mm).
pub fn generated_top(grid: &Grid, field: &dyn TerrainField, face: u8, i: i32, j: i32, level: u32, height: i32) -> i32 {
    let top = top_cells(grid, height, level);
    let p = grid.domain_point(face, i, j, level);
    let (below, above) = field.extent(p, level);
    if below == 0 && above == 0 {
        return top;
    }
    (top - below..top + above)
        .rev()
        .find(|&k| {
            let lean = lean_height(grid, field, face, i, j, k, level, height);
            field.density(p, grid.volume_point(face, i, j, k, level), level, top, height, lean, k) > 0
        })
        .map_or(top - below, |k| k + 1)
}

/// Ground slope of a cell in its 8x8 column block, in eighths of a cell per
/// cell: the larger top difference across the block along either axis.
/// Smooth and level-invariant, unlike neighbour steps of stepped terrain.
pub fn block_slope(top: impl Fn(i32, i32) -> i32, x: i32, y: i32) -> i32 {
    let si = (top(7, y) - top(0, y)).abs();
    let sj = (top(x, 7) - top(x, 0)).abs();
    si.max(sj) * 8 / 7
}

/// Check a world's field against its declared bounds at `samples` random
/// columns: every height inside [`TerrainField::height_range`], and every
/// finer column top within [`TerrainField::bound_margins`] of the coarser
/// column containing it. Returns the worst margin use per level.
pub fn check_field(planet: &Planet, samples: u32) -> Result<Vec<i32>, String> {
    let grid = *planet.grid();
    let field = planet.field();
    let margins = field.bound_margins();
    let (lo, hi) = field.height_range();
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut worst = vec![i32::MIN; grid.levels() as usize];
    for _ in 0..samples {
        let level = 1 + (next() % u64::from(grid.levels() - 1)) as u32;
        let cells = (grid.cells() >> level).max(1) as u64;
        let face = if grid.is_plane() { PLANE_FACE } else { (next() % 6) as u8 };
        let (i, j) = ((next() % cells) as i32, (next() % cells) as i32);
        let height = planet.column_height(face, i, j, level);
        if height < lo || height > hi {
            return Err(format!("level {level} column ({face}, {i}, {j}): height {height} outside {lo}..={hi}"));
        }
        let top = top_cells(&grid, height, level);
        for _ in 0..6 {
            let finer = (next() % u64::from(level)) as u32;
            let shift = level - finer;
            let fi = (i << shift) + (next() % (1u64 << shift)) as i32;
            let fj = (j << shift) + (next() % (1u64 << shift)) as i32;
            let fine = planet.column_top(face, fi, fj, finer);
            let excess = (((fine - 1) >> shift) + 1) - top;
            worst[level as usize] = worst[level as usize].max(excess);
            if excess > margins[level as usize] {
                return Err(format!(
                    "level {level} column ({face}, {i}, {j}): a level {finer} top rises {excess} cells, margin {}",
                    margins[level as usize]
                ));
            }
        }
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_generators_are_listed_and_duplicates_rejected() {
        let ids: Vec<_> = generators().into_iter().map(|g| g.id).collect();
        assert!(ids.contains(&crate::layers::ID.to_string()));
        assert!(register(Arc::new(crate::layers::LayersGenerator)).is_err());
        let grid = Grid::plane(crate::grid::Shape::Plane, 1024.0, 0.1).unwrap();
        let unknown = TerrainSource { generator: "example.none".into(), ..TerrainSource::default() };
        assert!(build(&unknown, &grid).is_err());
    }
}
