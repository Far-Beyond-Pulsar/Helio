//! Chunk trees against the canonical CPU planet: the WGSL terrain programs
//! bit for bit, and drawn hits near the eye (level 0) against exact CPU ray
//! casts, over natural terrain, caves, overhangs, planes and edits.
mod common;
use common::{gpu, Gpu};
use glam::{DVec3, Mat4, Vec3};
use helio_pass_voxel_planet::chunks::{residency, Capacity, ChunkWorld, ResidencyConfig};
use helio_pass_voxel_planet::engine::{PickHit, PickRequest, SharedPicks};
use helio_pass_voxel_planet::grid::Shape;
use helio_pass_voxel_planet::layers::{Layer, LayerKind, TerrainLayers};
use helio_pass_voxel_planet::terrain::material;
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Cell, Planet, PlanetRecipe, TerrainSource};

const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;

/// A camera at the eye (camera-relative) looking along `forward`.
struct View {
    eye: DVec3,
    forward: Vec3,
    size: [u32; 2],
    /// The near plane (m).
    near: f32,
}

impl View {
    fn new(eye: DVec3, forward: Vec3, size: [u32; 2]) -> Self {
        Self { eye, forward: forward.normalize(), size, near: 0.05 }
    }
    fn view_proj(&self, planet: &Planet) -> Mat4 {
        let up = planet.grid().up(self.eye).as_vec3();
        let up = if self.forward.dot(up).abs() > 0.99 { up.any_orthonormal_vector() } else { up };
        let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, self.forward, up);
        let proj = helio_core::depth::perspective(FOV_Y, self.size[0] as f32 / self.size[1] as f32, self.near);
        proj * view
    }
    /// The ray through pixel (x, y)'s centre.
    fn pixel_dir(&self, view_proj: Mat4, x: u32, y: u32) -> DVec3 {
        let ndc = glam::Vec4::new((x as f32 + 0.5) / self.size[0] as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / self.size[1] as f32 * 2.0, 0.5, 1.0);
        let w = view_proj.inverse() * ndc;
        (w.truncate() / w.w).as_dvec3().normalize()
    }
    /// A pixel's angle.
    fn pixel_angle(&self) -> f64 {
        2.0 * f64::from((FOV_Y * 0.5).tan()) / f64::from(self.size[1])
    }
    fn config(&self) -> ResidencyConfig {
        let mut config = ResidencyConfig { footprints_per_frame: 512, generate_ms: None, ..Default::default() };
        config.plan.pixel_angle = self.pixel_angle();
        config
    }
}

/// Chunks of one planet and their frame counter.
/// A test world's capacity: past the chunks the largest view settles with
/// (12901, the hollowed core; the others 8683 at most), so tests running in
/// parallel fit in a 12 GB GPU (1.1 GB a test with its device; a game's
/// default world, 2.9 GB, paged eight parallel tests out of device memory
/// and stalled them).
const TEST_CAPACITY: Capacity = Capacity { chunks: 16384, detail_pages: 2048, footprints: 1024 };

struct World {
    chunks: ChunkWorld,
    frame: u64,
}

impl World {
    fn new(gpu: &Gpu, planet: &Planet) -> Self {
        Self { chunks: ChunkWorld::with_capacity(&gpu.device, &gpu.queue, planet, TEST_CAPACITY), frame: 0 }
    }

    /// The planet's edits now (a new world for a history it cannot follow).
    fn sync(&mut self, gpu: &Gpu, planet: &Planet) {
        if !self.chunks.sync_edits(planet) {
            *self = Self::new(gpu, planet);
        }
    }

    /// One frame: residency, the draw, `picks` copied; its residency.
    fn frame(&mut self, gpu: &Gpu, planet: &Planet, view: &View, picks: Option<&SharedPicks>) -> helio_pass_voxel_planet::chunks::UpdateStats {
        let view_proj = view.view_proj(planet);
        let tan_y = f64::from((FOV_Y * 0.5).tan());
        let tan_x = tan_y * f64::from(view.size[0]) / f64::from(view.size[1]);
        let cone = residency::View { forward: view.forward.as_dvec3(), cos_half: (tan_x * tan_x + tan_y * tan_y).sqrt().atan().cos() };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut recorder = helio_core::CommandRecorder::from_encoder(&mut encoder);
        let stats = self.chunks.update(&mut recorder, view.eye, DVec3::ZERO, Some(cone), self.frame, &view.config());
        self.chunks.draw(&mut recorder, self.frame, view.eye, view_proj, view_proj, view.near, Vec3::new(0.3, 0.8, 0.4).normalize(), view.size);
        if let Some(picks) = picks {
            self.chunks.copy_picks(&mut recorder, picks);
        }
        drop(recorder);
        gpu.queue.submit([encoder.finish()]);
        self.chunks.submitted();
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        self.frame += 1;
        stats
    }

    /// Frames until residency settles; every pixel's hit then.
    fn settle(&mut self, gpu: &Gpu, planet: &Planet, view: &View) -> Vec<Option<PickHit>> {
        let started = self.frame;
        let stats = loop {
            let stats = self.frame(gpu, planet, view, None);
            if self.chunks.settled() {
                break stats;
            }
            assert!(self.frame - started < 4000, "residency did not settle: {stats:?}");
        };
        eprintln!("settled in {} frames: {} chunks drawn; {stats:?}", self.frame - started, self.chunks.chunk_count());
        self.chunks.read_hits()
    }
}

/// Drawn hits against CPU ray casts: every second pixel whose CPU hit lies
/// within the level-0 range (cells of a pixel or more) must be drawn at
/// level 0 at the same cell (or its neighbour across an edge the ray runs
/// along) and distance. A chunk is drawn in an affine frame, off the curved
/// grid by up to 512 voxel^2 / radius (a millionth of a millimetre on Earth,
/// 2 mm on a 3 km world): within that a ray grazing a step's corner may meet
/// its neighbour. (compared, mismatched).
fn compare(gpu: &Gpu, world: &mut World, planet: &Planet, view: &View) -> (usize, usize) {
    let hits = world.settle(gpu, planet, view);
    let view_proj = view.view_proj(planet);
    let near = 0.7 * planet.grid().voxel_size() / view.pixel_angle();
    let voxel = planet.grid().voxel_size();
    let frame = if planet.grid().is_plane() { 0.0 } else { 4.0 * 512.0 * voxel * voxel / planet.grid().radius() };
    let (mut compared, mut mismatched) = (0, 0);
    for y in (0..view.size[1]).step_by(2) {
        for x in (0..view.size[0]).step_by(2) {
            let Some(cpu) = planet.raycast(view.eye, view.pixel_dir(view_proj, x, y), near).filter(|c| c.distance < near) else { continue };
            compared += 1;
            let drawn = hits[(y * view.size[0] + x) as usize];
            // A ray along the edge between two solid cells meets both at
            // once: either is exact.
            let tie = |c: Cell| c.face == cpu.cell.face && (c.i - cpu.cell.i).abs() + (c.j - cpu.cell.j).abs() + (c.k - cpu.cell.k).abs() == 1 && planet.solid(c);
            let exact = drawn.is_some_and(|h| h.level == 0 && (h.cell == cpu.cell || tie(h.cell)) && (h.distance - cpu.distance).abs() <= 2e-3 + cpu.distance * 1e-5 + frame);
            if !exact {
                mismatched += 1;
                if mismatched < 8 {
                    eprintln!("pixel {x},{y}: drawn {drawn:?}, cpu {:?} at {}", cpu.cell, cpu.distance);
                }
            }
        }
    }
    eprintln!("compared {compared}, mismatched {mismatched}");
    (compared, mismatched)
}

/// [`compare`] with a new world: at most one pixel in a thousand differs.
fn matches(gpu: &Gpu, planet: &Planet, view: View) {
    let (compared, mismatched) = compare(gpu, &mut World::new(gpu, planet), planet, &view);
    assert!(compared > 1000, "{compared} pixels compared");
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// [`compare`] with a new world: every pixel agrees.
fn matches_exactly(gpu: &Gpu, planet: &Planet, view: View, at_least: usize) {
    let (compared, mismatched) = compare(gpu, &mut World::new(gpu, planet), planet, &view);
    assert!(compared > at_least, "{compared} pixels compared");
    assert_eq!(mismatched, 0);
}

/// A land direction (surface above sea level) near `face` coordinates.
fn land(planet: &Planet, face: u8, fi: f64, fj: f64) -> DVec3 {
    let grid = planet.grid();
    let n = f64::from(grid.cells());
    for step in 0..400 {
        let a = fi + 0.013 * f64::from(step % 20);
        let b = fj + 0.017 * f64::from(step / 20);
        let (i, j) = ((a.fract() * n) as i32, (b.fract() * n) as i32);
        if planet.column_top(face, i, j, 0) > 40 {
            return grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        }
    }
    panic!("no land found");
}

/// Standing on the ground at `dir`, looking out and down.
fn ground_view(planet: &Planet, dir: DVec3, size: [u32; 2]) -> View {
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    View::new(eye, (up.any_orthonormal_vector() - up * 0.35).normalize().as_vec3(), size)
}

fn xorshift(mut state: u64) -> impl FnMut() -> u64 {
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

#[test]
fn terrain_programs_are_bit_identical_to_cpu() {
    let Some(gpu) = gpu() else { return };
    let moon = TerrainLayers::moon().source(7);
    let flat = TerrainLayers { soil_depth_m: 2.0, ..TerrainLayers::flat_at(-1.25) }.source(7);
    let desert = TerrainLayers::desert().source(7);
    let mut deep = TerrainLayers::earth();
    deep.caves.depth_m = 900.0;
    let deep = deep.source(7);
    let mut cliffs = TerrainLayers::earth();
    cliffs.layers.push(Layer { coverage: 0.6, ..Layer::new(LayerKind::Cliffs) });
    let cliffs = cliffs.source(7);
    for (shape, size, terrain) in [
        (Shape::Sphere, 0.1, TerrainSource::default()),
        (Shape::Sphere, 0.3, TerrainSource::default()),
        (Shape::Sphere, 1.0, TerrainSource::default()),
        (Shape::Plane, 0.1, TerrainSource::default()),
        (Shape::InfinitePlane, 0.1, TerrainSource::default()),
        (Shape::InfinitePlane, 1.0, TerrainSource::default()),
        (Shape::Plane, 0.1, flat.clone()),
        (Shape::Sphere, 0.1, flat),
        (Shape::Sphere, 0.1, moon.clone()),
        (Shape::Sphere, 1.0, moon.clone()),
        (Shape::Plane, 0.1, moon.clone()),
        (Shape::InfinitePlane, 0.1, moon),
        (Shape::Sphere, 0.1, desert.clone()),
        (Shape::Plane, 0.3, desert),
        (Shape::Sphere, 0.1, deep),
        (Shape::Sphere, 0.1, cliffs.clone()),
        (Shape::Plane, 0.3, cliffs.clone()),
        (Shape::InfinitePlane, 1.0, cliffs),
    ] {
        let planet = Planet::new(PlanetRecipe { shape, voxel_size_m: size, plane_size_m: 5_000.0, terrain: terrain.clone(), ..Default::default() }).unwrap();
        helio_pass_voxel_planet::engine::verify_field(&gpu.device, &gpu.queue, &planet, 20_000).unwrap_or_else(|e| panic!("{} on {shape:?} at {size} m: {e}", terrain.generator));
    }
}

#[test]
fn ground_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    matches(&gpu, &planet, view);
}

/// Level-0 columns near a face's far corner (indices past 2^26), looking
/// across the face's edge 3 m away.
#[test]
fn ground_view_across_a_face_edge_matches_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *planet.grid();
    let n = grid.cells();
    let i = n - 30;
    let j = (n - n / 10..n).step_by(997).find(|&j| planet.column_top(4, i, j, 0) > 40).expect("land near the edge");
    assert!(i >= 1 << 26 && j >= 1 << 26);
    let dir = grid.direction(4, f64::from(i) + 0.5, f64::from(j) + 0.5);
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    // Toward the edge (+i): across it, the neighbouring face.
    let toward = grid.direction(4, f64::from(n) + 30.0, f64::from(j) + 0.5) - dir;
    let forward = ((toward - up * toward.dot(up)).normalize() - up * 0.35).normalize();
    matches(&gpu, &planet, View::new(eye, forward.as_vec3(), [320, 180]));
}

/// An eye inside a generated cave: an air cell with air around it, at least
/// four cells below its column's heightfield top, walls, floor and ceiling
/// within 8 m.
fn find_cave(planet: &Planet) -> Option<(DVec3, Vec3)> {
    let grid = *planet.grid();
    let field = planet.field();
    let mut next = xorshift(0x2545_F491_4F6C_DD1D);
    let sites = (0..6u8).flat_map(|face| [(0.37, 0.61), (0.2, 0.3), (0.7, 0.45)].map(|(a, b)| (face, a, b)));
    for (face, a, b) in sites {
        let (base, _) = grid.locate(planet.surface_point(land(planet, face, a, b), 0.0));
        for _ in 0..4_000 {
            let i = base.i + (next() % 8000) as i32 - 4000;
            let j = base.j + (next() % 8000) as i32 - 4000;
            let (below, _) = field.extent(grid.domain_point(base.face, i, j, 0), 0);
            if below == 0 {
                continue;
            }
            let top = planet.column_top(base.face, i, j, 0);
            let k = top - 4 - (next() % below.max(1) as u64) as i32;
            let air = |di: i32, dj: i32, dk: i32| !planet.solid(Cell::new(base.face, i + di, j + dj, k + dk));
            if (-1..=1).all(|a| (-1..=1).all(|b| (-1..=1).all(|c| air(a, b, c)))) {
                let eye = grid.cell_center(Cell::new(base.face, i, j, k));
                let up = eye.normalize();
                let forward = (up.any_orthonormal_vector() - up * 0.15).normalize();
                let side = up.cross(forward);
                if [forward, -forward, side, -side, up, -up].iter().all(|d| planet.raycast(eye, *d, 8.0).is_some()) {
                    return Some((eye, forward.as_vec3()));
                }
            }
        }
    }
    None
}

/// Generated caves render exactly from inside: walls, floor and ceiling.
#[test]
fn cave_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let (eye, forward) = find_cave(&planet).expect("a cave near the test sites");
    matches(&gpu, &planet, View::new(eye, forward, [320, 180]));
}

/// Caves 800 m deep: from inside one at a random depth.
#[test]
fn deep_cave_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let mut stack = TerrainLayers::earth();
    stack.caves.depth_m = 800.0;
    let planet = Planet::new(PlanetRecipe { terrain: stack.source(7), ..Default::default() }).unwrap();
    let (eye, forward) = find_cave(&planet).expect("a cave near the test sites");
    matches(&gpu, &planet, View::new(eye, forward, [320, 180]));
}

/// An eye over leaning ground, looking up the slope: a column whose
/// generated cells the lean moved off the heightfield by two cells or more.
fn find_overhang(planet: &Planet) -> Option<(DVec3, Vec3)> {
    let grid = *planet.grid();
    let field = planet.field();
    let mut next = xorshift(0x1D8E_4E27_C47D_124F);
    let sites = (0..6u8).flat_map(|face| [(0.37, 0.61), (0.2, 0.3), (0.7, 0.45)].map(|(a, b)| (face, a, b)));
    for (face, a, b) in sites {
        let (base, _) = grid.locate(planet.surface_point(land(planet, face, a, b), 0.0));
        for _ in 0..20_000 {
            let i = base.i + (next() % 40_000) as i32 - 20_000;
            let j = base.j + (next() % 40_000) as i32 - 20_000;
            let (_, above) = field.extent(grid.domain_point(base.face, i, j, 0), 0);
            if above < 20 {
                continue;
            }
            let top = |di: i32, dj: i32| planet.column_top(base.face, i + di, j + dj, 0);
            let (si, sj) = (top(20, 0) - top(-20, 0), top(0, 20) - top(0, -20));
            if si.abs().max(sj.abs()) < 8 {
                continue;
            }
            let t = top(0, 0);
            if (t - above..t + above).filter(|&k| planet.solid(Cell::new(base.face, i, j, k)) != (k < t)).count() < 2 {
                continue;
            }
            let eye = planet.surface_point(grid.cell_center(Cell::new(base.face, i, j, t)), 2.0);
            let up = eye.normalize();
            let (di, dj) = if si.abs() >= sj.abs() { (si.signum(), 0) } else { (0, sj.signum()) };
            let along = grid.cell_center(Cell::new(base.face, i + di, j + dj, t)) - grid.cell_center(Cell::new(base.face, i, j, t));
            let forward = ((along - up * along.dot(up)).normalize() - up * 0.3).normalize();
            return Some((eye, forward.as_vec3()));
        }
    }
    None
}

/// Generated overhangs render exactly: the generation's lean lattice is
/// the field's.
#[test]
fn overhang_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let mut stack = TerrainLayers::earth();
    stack.caves.enabled = false;
    let planet = Planet::new(PlanetRecipe { terrain: stack.source(7), ..Default::default() }).unwrap();
    let (eye, forward) = find_overhang(&planet).expect("an overhang near the test sites");
    matches(&gpu, &planet, View::new(eye, forward, [320, 180]));
}

/// Finite and infinite planes, from the ground and from above.
#[test]
fn plane_views_match_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    for shape in [Shape::Plane, Shape::InfinitePlane] {
        let planet = Planet::new(PlanetRecipe { shape, plane_size_m: 3_000.0, ..Default::default() }).unwrap();
        for (x, z, height, pitch) in [(12.3, -45.6, 1.7, -0.35), (-300.0, 250.0, 8.0, -1.2)] {
            let eye = planet.surface_point(DVec3::new(x, 0.0, z), height);
            matches(&gpu, &planet, View::new(eye, Vec3::new(0.8, pitch, 0.6), [320, 180]));
        }
    }
}

/// A dug sphere and a raised cube.
#[test]
fn brushes_render_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let eye = planet.surface_point(dir, 3.0);
    let up = eye.normalize();
    let side = up.any_orthonormal_vector();
    let ground = planet.surface_point(dir, 0.0) + side * 4.0;
    planet.apply(Brush { center: ground.to_array(), radius: 2.5, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
    planet.apply(Brush { center: (ground + side * 3.0 + up * 1.5).to_array(), radius: 0.8, shape: BrushShape::Cube, op: BrushOp::Add, material: material::BRICK, height: 0.0 }).unwrap();
    matches_exactly(&gpu, &planet, View::new(eye, (side - up * 0.6).normalize().as_vec3(), [320, 180]), 1000);
}

/// The floor of a ball 10 km across, dug from the surface: from a few metres
/// over it, looking across it.
#[test]
fn a_planet_scale_dig_renders_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 2, 0.47, 0.52);
    let ground = planet.surface_point(dir, 0.0);
    let up = ground.normalize();
    planet.apply(Brush { center: ground.to_array(), radius: 10_000.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
    let side = up.any_orthonormal_vector();
    for (off, height, pitch) in [(0.0, 3.0, -0.5), (2_500.0, 4.0, -0.4), (6_000.0, 3.0, -0.3)] {
        let floor = ground - up * 10_000.0 + side * off;
        let floor = planet.surface_point(floor + up * 9_000.0, 0.0);
        let eye = floor + up * height;
        let view = View::new(eye, (up.cross(side) + up * pitch).normalize().as_vec3(), [320, 180]);
        let (compared, mismatched) = compare(&gpu, &mut World::new(&gpu, &planet), &planet, &view);
        eprintln!("off {off}: {mismatched}/{compared}");
        assert!(compared > 1000, "{compared} pixels compared");
        assert_eq!(mismatched, 0, "off {off}");
    }
}

/// The floor of a ball dug from the surface, seen from its centre across
/// it: every pixel's ray enters its nearest box. Far boxes lie metres apart
/// in depth, where a standard (not reversed) depth, precise to
/// distance^2 / near 2^-24, cannot tell them apart: a ray seeded in the
/// farther box entered it by its side, under the nearer one's cells. A 2 mm
/// near plane: 1.5 km away, ties 70 m deep.
#[test]
fn a_dug_floor_seen_from_afar_shows_its_nearest_cells() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 2, 0.47, 0.52);
    let ground = planet.surface_point(dir, 0.0);
    let up = ground.normalize();
    let radius = 1_500.0;
    planet.apply(Brush { center: ground.to_array(), radius, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
    let side = up.any_orthonormal_vector();
    let view = View { near: 0.002, ..View::new(ground + up * 6.0, (side - up * 0.5).normalize().as_vec3(), [480, 270]) };
    let hits = World::new(&gpu, &planet).settle(&gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let (mut compared, mut beyond) = (0, 0);
    for y in (0..view.size[1]).step_by(12) {
        for x in (0..view.size[0]).step_by(12) {
            let dir = view.pixel_dir(view_proj, x, y);
            let Some(cpu) = planet.raycast(view.eye, dir, 4.0 * radius) else { continue };
            // The ball's wall and floor, well under the natural ground: at
            // the rim the ground left over the ball thins to slivers a
            // coarse cell does not hold.
            let at = view.eye + dir * cpu.distance;
            let under = planet.column_top(cpu.cell.face, cpu.cell.i, cpu.cell.j, 0) - cpu.cell.k;
            if ((at - ground).length() - radius).abs() > 1.0 || f64::from(under) * planet.grid().voxel_size() < 30.0 {
                continue;
            }
            let Some(drawn) = hits[(y * view.size[0] + x) as usize] else { continue };
            compared += 1;
            // A level's cells sit within a cell or so of the exact ground.
            let cell = planet.grid().voxel_size() * f64::from(1u32 << drawn.level);
            if (drawn.distance - cpu.distance).abs() > 3.0 * cell + 0.5 {
                beyond += 1;
                if beyond < 8 {
                    eprintln!("pixel {x},{y}: drawn {:.2} m at level {} entered {:?}, cpu {:.2} m", drawn.distance, drawn.level, drawn.entered, cpu.distance);
                }
            }
        }
    }
    eprintln!("compared {compared}, {beyond} beyond three cells");
    assert!(compared > 500, "{compared} compared");
    assert_eq!(beyond, 0);
}

/// Flat-topped boxes (what Flatten stamps): a slab removed above a level
/// and one filled below it, across a hillside.
#[test]
fn box_brushes_render_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let ground = planet.surface_point(dir, 0.0);
    for n in 0..6 {
        let at = ground + side * (f64::from(n) * 2.5);
        planet.apply(Brush { center: (at + up * 1.5).to_array(), radius: 1.3, shape: BrushShape::Cube, op: BrushOp::Remove, material: 0, height: 1.5 }).unwrap();
        planet.apply(Brush { center: (at - up * 0.75).to_array(), radius: 1.3, shape: BrushShape::Cube, op: BrushOp::Add, material: material::BRICK, height: 0.75 }).unwrap();
    }
    let eye = ground + up * 4.0 - side * 5.0 + up.cross(side) * 3.0;
    let forward = ((ground + side * 6.0) - eye).normalize().as_vec3();
    matches_exactly(&gpu, &planet, View::new(eye, forward, [320, 180]), 1000);
}

/// Destruction to any depth: at the bottom of a 600 m shaft, and 300 m up
/// it with the same world.
#[test]
fn a_deep_shaft_renders_exactly_at_any_depth() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 3, 0.41, 0.57);
    let ground = planet.surface_point(dir, 0.0);
    let up = planet.grid().up(ground);
    let mut depth = -10.0;
    while depth < 600.0 {
        planet.apply(Brush { center: (ground - up * depth).to_array(), radius: 2.0, shape: BrushShape::Cube, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
        depth += 3.0;
    }
    let mut world = World::new(&gpu, &planet);
    let side = up.any_orthonormal_vector();
    for (depth, pitch) in [(590.0, -0.6), (290.0, 0.4)] {
        let view = View::new(ground - up * depth + side * 0.7, (side + up * pitch).normalize().as_vec3(), [256, 144]);
        let (compared, mismatched) = compare(&gpu, &mut world, &planet, &view);
        assert!(compared > 1000, "{depth} m: {compared} compared");
        assert_eq!(mismatched, 0, "{depth} m");
    }
}

/// A crater 3 km wide: on its floor.
#[test]
fn a_planet_scale_crater_renders_exactly_on_its_floor() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let ground = planet.surface_point(land(&planet, 5, 0.33, 0.52), 0.0);
    let up = planet.grid().up(ground);
    planet.apply(Brush { center: ground.to_array(), radius: 3_000.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
    let eye = ground - up * (3_000.0 - 1.7);
    assert!(!planet.solid(planet.grid().locate(eye).0), "the eye is in the crater");
    let forward = (up.any_orthonormal_vector() - up * 0.4).normalize().as_vec3();
    matches_exactly(&gpu, &planet, View::new(eye, forward, [256, 144]), 1000);
}

/// The whole planet is destructible: a 3 km world with its core hollowed
/// out, seen from inside the hollow. The rock between the hollow's wall and
/// the surface takes no chunks (each tile's solid gap), the hollow is air
/// under each tile's chunks. On a world this small the chunks' affine
/// frames bend from the grid enough that a few grazing rays meet a corner's
/// neighbour.
#[test]
fn a_hollowed_core_renders_exactly_from_inside() {
    let Some(gpu) = gpu() else { return };
    let recipe = PlanetRecipe { radius_m: 3_000.0, terrain: TerrainLayers::earth().heightfield().source(7), ..Default::default() };
    let mut planet = Planet::new(recipe).unwrap();
    planet.apply(Brush { center: [0.0; 3], radius: 1_500.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
    let up = DVec3::new(0.3, 0.9, 0.2).normalize();
    let eye = up * (1_500.0 - 1.7);
    assert!(!planet.solid(planet.grid().locate(eye).0), "the eye is in the hollow");
    assert!(planet.solid(planet.grid().locate(up * 1_500.5).0), "the hollow has a wall");
    let forward = (up.any_orthonormal_vector() + up * 0.5).normalize().as_vec3();
    matches(&gpu, &planet, View::new(eye, forward, [256, 144]));
}

/// A hall 40 m across dug 300 m under the ground: from its floor, its walls
/// and ceiling render exactly, and the rock between it and the surface takes
/// no chunks (a solid gap in each tile over it).
#[test]
fn an_underground_hall_renders_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let ground = planet.surface_point(land(&planet, 2, 0.43, 0.57), 0.0);
    let up = planet.grid().up(ground);
    let centre = ground - up * 300.0;
    planet.apply(Brush { center: centre.to_array(), radius: 20.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
    let eye = centre - up * 18.0;
    assert!(!planet.solid(planet.grid().locate(eye).0), "the eye is in the hall");
    let forward = (up.any_orthonormal_vector() + up * 0.3).normalize().as_vec3();
    matches_exactly(&gpu, &planet, View::new(eye, forward, [256, 144]), 1000);
}

/// A building of single 0.1 m blocks and a field of single-block holes
/// (about 18 000 block edits).
#[test]
fn thousands_of_block_edits_render_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let grid = *planet.grid();
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let ground = planet.surface_point(dir, 0.0);
    let block = |p: DVec3, op: BrushOp| {
        let (cell, _) = grid.locate(p);
        Brush { center: grid.cell_center(cell).to_array(), radius: grid.voxel_size() * 0.5, shape: BrushShape::Cube, op, material: if op == BrushOp::Add { material::BRICK } else { 0 }, height: 0.0 }
    };
    let at = |x: i32, y: i32, z: i32| ground + side * (f64::from(x) * 0.1) + fwd * (f64::from(z) * 0.1 + 6.0) + up * (f64::from(y) * 0.1 + 0.55);
    let mut count = 0;
    for y in 0..60 {
        for x in -20..20 {
            for z in -20..20 {
                let wall = x == -20 || x == 19 || z == -20 || z == 19;
                if y % 10 == 0 || (wall && (x + z + y) % 7 != 0) {
                    planet.apply(block(at(x, y, z), BrushOp::Add)).unwrap();
                    count += 1;
                }
            }
        }
    }
    for x in -40..40 {
        for z in -30..-10 {
            if (x * 3 + z) % 2 == 0 {
                planet.apply(block(ground + side * (f64::from(x) * 0.1) + fwd * (f64::from(z) * 0.1) - up * 0.05, BrushOp::Remove)).unwrap();
                count += 1;
            }
        }
    }
    assert!(count > 12_000, "{count}");
    let eye = ground - fwd * 6.0 + up * 14.0 - side * 3.0;
    let forward = ((ground + fwd * 6.0) - eye).normalize().as_vec3();
    matches_exactly(&gpu, &planet, View::new(eye, forward, [320, 180]), 400);
}

/// Blocks of many materials over ground with single cells painted: every
/// pixel whose CPU hit is an edited cell draws that cell's own material, not
/// its leaf's.
#[test]
fn edited_cells_draw_their_own_materials() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let grid = *planet.grid();
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let ground = planet.surface_point(dir, 0.0);
    let kinds = [material::BRICK, material::PLANKS, material::COBBLE, material::WOOD, material::CLAY, material::SNOW, material::BASALT, material::SAND];
    let mut edited = std::collections::HashMap::new();
    let mut cell_brush = |planet: &mut Planet, cell: Cell, op: BrushOp, material: u32| {
        planet.apply(Brush { center: grid.cell_center(cell).to_array(), radius: grid.voxel_size() * 0.5, shape: BrushShape::Cube, op, material, height: 0.0 }).unwrap();
        edited.insert(cell, material);
    };
    // A wall of blocks, each cell its own material.
    for y in 0..24i32 {
        for x in -16..16 {
            let (cell, _) = grid.locate(ground + side * (f64::from(x) * 0.1) + fwd * 4.0 + up * (f64::from(y) * 0.1 + 0.55));
            cell_brush(&mut planet, cell, BrushOp::Add, kinds[((x * 7 + y * 3).rem_euclid(8)) as usize]);
        }
    }
    // Ground cells painted in a checkerboard in front of it.
    for x in -16..16 {
        for z in 0..30i32 {
            if (x + z) % 2 != 0 {
                continue;
            }
            let (c, _) = grid.locate(ground + side * (f64::from(x) * 0.1) + fwd * (f64::from(z) * 0.1 + 0.5));
            let cell = Cell::new(c.face, c.i, c.j, planet.column_top(c.face, c.i, c.j, 0) - 1);
            assert!(planet.solid(cell));
            cell_brush(&mut planet, cell, BrushOp::Paint, kinds[((x + z * 5).rem_euclid(8)) as usize]);
        }
    }
    let eye = ground - fwd * 2.0 + up * 3.0;
    let view = View::new(eye, ((ground + fwd * 3.0 + up * 0.8) - eye).normalize().as_vec3(), [320, 180]);
    let hits = World::new(&gpu, &planet).settle(&gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let (mut compared, mut wrong) = (0, 0);
    for y in 0..view.size[1] {
        for x in 0..view.size[0] {
            let Some(cpu) = planet.raycast(view.eye, view.pixel_dir(view_proj, x, y), 50.0) else { continue };
            let Some(&want) = edited.get(&cpu.cell) else { continue };
            assert_eq!(planet.material(cpu.cell), want);
            let Some(drawn) = hits[(y * view.size[0] + x) as usize].filter(|h| h.level == 0 && h.cell == cpu.cell) else { continue };
            compared += 1;
            if drawn.material != want {
                wrong += 1;
                if wrong < 8 {
                    eprintln!("pixel {x},{y}: {:?} drawn {}, edited {want}", cpu.cell, drawn.material);
                }
            }
        }
    }
    eprintln!("compared {compared}, wrong {wrong}");
    assert!(compared > 3000, "{compared} pixels compared");
    assert_eq!(wrong, 0);
}

/// Land columns picked for relief and mixed top materials (flat grassland
/// shows one material everywhere): the `steep` with the greatest top
/// difference over 16 cells, then the `mixed` with the most distinct top
/// materials around them, of `candidates` random ones.
fn varied_spots(planet: &Planet, candidates: usize, steep: usize, mixed: usize) -> Vec<(String, u8, i32, i32)> {
    let n = planet.grid().cells() as u64;
    let mut next = xorshift(0x9e37_79b9_7f4a_7c15);
    let (mut by_relief, mut by_kinds) = (Vec::new(), Vec::new());
    for _ in 0..candidates {
        let face = (next() % 6) as u8;
        let (i, j) = ((next() % (n - 64)) as i32 + 32, (next() % (n - 64)) as i32 + 32);
        if planet.column_top(face, i, j, 0) <= 40 {
            continue;
        }
        let (mut lo, mut hi) = (i32::MAX, i32::MIN);
        let mut kinds = std::collections::HashSet::new();
        for a in (-8..8).step_by(4) {
            for b in (-8..8).step_by(4) {
                let top = planet.column_top(face, i + a, j + b, 0);
                lo = lo.min(top);
                hi = hi.max(top);
                kinds.insert(planet.material(Cell::new(face, i + a, j + b, top - 1)));
            }
        }
        by_relief.push((hi - lo, face, i, j));
        by_kinds.push((kinds.len() as i32 * 1000 + hi - lo, face, i, j));
    }
    by_relief.sort_by(|a, b| b.0.cmp(&a.0));
    by_kinds.sort_by(|a, b| b.0.cmp(&a.0));
    let mut spots: Vec<(String, u8, i32, i32)> = by_relief.iter().take(steep).map(|s| (format!("steep {} cells", s.0), s.1, s.2, s.3)).collect();
    for s in &by_kinds {
        if spots.len() < steep + mixed && !spots.iter().any(|t| (t.1, t.2, t.3) == (s.1, s.2, s.3)) {
            spots.push((format!("{} materials", s.0 / 1000), s.1, s.2, s.3));
        }
    }
    spots
}

/// The surface material of the leaf (4x4x4 cells) holding natural cell `c`
/// when every one of its 16 columns tops in it with the same canonical
/// material: what the leaf's vote must draw on its top faces.
fn uniform_leaf_top(planet: &Planet, c: Cell) -> Option<u32> {
    let grid = planet.grid();
    let (li, lj, lk) = (c.i.div_euclid(4) * 4, c.j.div_euclid(4) * 4, c.k.div_euclid(4) * 4);
    let mut material = None;
    for a in 0..4 {
        for b in 0..4 {
            let (i, j) = (li + a, lj + b);
            let height = planet.column_height(c.face, i, j, 0);
            let top = helio_pass_voxel_planet::terrain::generated_top(grid, planet.field(), c.face, i, j, 0, height);
            if !(lk..lk + 4).contains(&(top - 1)) {
                return None;
            }
            let m = planet.material(Cell::new(c.face, i, j, top - 1));
            if material.is_some_and(|x| x != m) {
                return None;
            }
            material = Some(m);
        }
    }
    material
}

/// The drawn material against `Planet::material` over a view's level-0
/// pixels, as (pixels, wrong): top faces of uniform leaves, other cells
/// under air, cells under solid ground. Each wrong (canonical, drawn) pair
/// is counted into `confusions`.
fn material_agreement(gpu: &Gpu, planet: &Planet, view: &View, confusions: &mut std::collections::HashMap<(u32, u32), usize>) -> [(usize, usize); 3] {
    let hits = World::new(gpu, planet).settle(gpu, planet, view);
    let view_proj = view.view_proj(planet);
    let near = 0.7 * planet.grid().voxel_size() / view.pixel_angle();
    let mut counts = [(0, 0); 3];
    // Uniform leaves by leaf (16 columns' tops each).
    let mut leaves = std::collections::HashMap::new();
    for y in 0..view.size[1] {
        for x in 0..view.size[0] {
            let Some(cpu) = planet.raycast(view.eye, view.pixel_dir(view_proj, x, y), near).filter(|c| c.distance < near) else { continue };
            let Some(drawn) = hits[(y * view.size[0] + x) as usize].filter(|h| h.level == 0 && h.cell == cpu.cell) else { continue };
            let c = cpu.cell;
            let own = planet.material(c);
            let covered = planet.solid(Cell::new(c.face, c.i, c.j, c.k + 1));
            // A top face: the ray stepped down k into the cell.
            let kind = if covered {
                2
            } else if drawn.entered == Some(4) && leaves.entry((c.face, c.i.div_euclid(4), c.j.div_euclid(4), c.k.div_euclid(4))).or_insert_with(|| uniform_leaf_top(planet, c)).is_some() {
                0
            } else {
                1
            };
            counts[kind].0 += 1;
            if drawn.material != own {
                counts[kind].1 += 1;
                *confusions.entry((own, drawn.material)).or_default() += 1;
            }
        }
    }
    counts
}

/// Natural ground draws `Planet::material` exactly where the leaf's
/// materials leave no choice: the top faces of leaves whose 16 columns all
/// top in them with one material (the generation kernel evaluates
/// `ground_material` at each cell's volume point, with
/// `Planet::material_slope`).
#[test]
fn uniform_leaves_draw_canonical_materials() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *planet.grid();
    let mut confusions = std::collections::HashMap::new();
    let (mut compared, mut wrong) = (0, 0);
    for (label, face, i, j) in varied_spots(&planet, 600, 2, 1) {
        let dir = grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        let eye = planet.surface_point(dir, 1.7);
        let up = eye.normalize();
        let view = View::new(eye, (up.any_orthonormal_vector() - up * 0.35).normalize().as_vec3(), [320, 180]);
        let [uniform, ..] = material_agreement(&gpu, &planet, &view, &mut confusions);
        eprintln!("{label}: {} of {} uniform-leaf top pixels wrong", uniform.1, uniform.0);
        compared += uniform.0;
        wrong += uniform.1;
    }
    assert!(compared > 2000, "{compared} pixels compared");
    assert_eq!(wrong, 0, "{confusions:?}");
}

/// A pit's walls and floor draw `Planet::material` exactly: chunks under
/// edits give exposed cells their own natural materials where their leaf's
/// pair would draw them wrong (soil a cell deep drawn as rock).
#[test]
fn dug_walls_draw_canonical_materials() {
    let Some(gpu) = gpu() else { return };
    let natural = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *natural.grid();
    let radius = 2.5;
    let (mut compared, mut wrong, mut pages) = (0, 0, 0);
    let mut confusions: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
    for (label, face, i, j) in varied_spots(&natural, 600, 2, 1) {
        let dir = grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        let eye = natural.surface_point(dir, 1.7);
        let up = eye.normalize();
        let side = up.any_orthonormal_vector();
        let centre = natural.surface_point(dir, 0.0) + side * 4.0;
        let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
        planet.apply(Brush { center: centre.to_array(), radius, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
        let view = View::new(eye, (side - up * 0.35).normalize().as_vec3(), [320, 180]);
        let mut world = World::new(&gpu, &planet);
        let hits = world.settle(&gpu, &planet, &view);
        pages += world.frame(&gpu, &planet, &view, None).detail_pages.0;
        let view_proj = view.view_proj(&planet);
        let near = 0.7 * grid.voxel_size() / view.pixel_angle();
        let (mut n, mut w) = (0, 0);
        for y in 0..view.size[1] {
            for x in 0..view.size[0] {
                let Some(cpu) = planet.raycast(view.eye, view.pixel_dir(view_proj, x, y), near).filter(|c| c.distance < near) else { continue };
                // The pit's cells: within a cell of its sphere.
                if (grid.cell_center(cpu.cell) - centre).length() > radius + 2.0 * grid.voxel_size() {
                    continue;
                }
                let Some(drawn) = hits[(y * view.size[0] + x) as usize].filter(|h| h.level == 0 && h.cell == cpu.cell) else { continue };
                let own = planet.material(cpu.cell);
                n += 1;
                if drawn.material != own {
                    w += 1;
                    *confusions.entry((own, drawn.material)).or_default() += 1;
                }
            }
        }
        eprintln!("{label}: {w} of {n} pit pixels wrong");
        compared += n;
        wrong += w;
    }
    eprintln!("detail pages in use, summed over views: {pages}");
    assert!(compared > 5000, "{compared} pixels compared");
    assert_eq!(wrong, 0, "{confusions:?}");
}

/// How often unedited voxels draw another material than their own
/// (`Planet::material`): over level-0 pixels of views at spots picked for
/// relief and mixed materials, from 1.7 m and 6 m, then with a pit dug in
/// front (a removal sets no materials: natural cells in its walls). Split
/// by what decides the drawn material: top faces of uniform leaves (exact),
/// other cells under air (the leaf's surface vote), cells under ground
/// (the leaf's subsurface byte). A report, not a gate:
/// `cargo test --release --test chunks natural_cells -- --ignored --nocapture`.
#[test]
#[ignore]
fn natural_cells_against_the_leaf_rule() {
    let Some(gpu) = gpu() else { return };
    let natural = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *natural.grid();
    let name = |m: u32| if m == 0 || m as usize > material::NAMES.len() { format!("#{m}") } else { material::NAMES[m as usize - 1].to_string() };
    let pct = |(n, w): (usize, usize)| 100.0 * w as f64 / n.max(1) as f64;
    let sum = |c: &[(usize, usize); 3]| c.iter().fold((0, 0), |a, c| (a.0 + c.0, a.1 + c.1));
    let mut confusions = std::collections::HashMap::new();
    let mut all = [(0, 0); 3];
    for (label, face, i, j) in varied_spots(&natural, 1500, 4, 4) {
        let dir = grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        for (lift, pit) in [(1.7, false), (6.0, false), (1.7, true)] {
            let eye = natural.surface_point(dir, lift);
            let up = eye.normalize();
            let side = up.any_orthonormal_vector();
            let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
            if pit {
                let ground = natural.surface_point(dir, 0.0) + side * 4.0;
                planet.apply(Brush { center: ground.to_array(), radius: 2.5, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
            }
            let view = View::new(eye, (side - up * if lift > 2.0 { 0.6 } else { 0.35 }).normalize().as_vec3(), [320, 180]);
            let counts = material_agreement(&gpu, &planet, &view, &mut confusions);
            let total = sum(&counts);
            eprintln!(
                "{label} at {lift} m{}: {}/{} pixels wrong ({:.1}%); uniform tops {:.1}% of {}, other surface {:.1}% of {}, under ground {:.1}% of {}",
                if pit { ", pit" } else { "" },
                total.1,
                total.0,
                pct(total),
                pct(counts[0]),
                counts[0].0,
                pct(counts[1]),
                counts[1].0,
                pct(counts[2]),
                counts[2].0
            );
            for (a, c) in all.iter_mut().zip(counts) {
                *a = (a.0 + c.0, a.1 + c.1);
            }
        }
    }
    let total = sum(&all);
    eprintln!(
        "ALL: {}/{} pixels wrong ({:.2}%); uniform tops {:.2}% of {}, other surface {:.2}% of {}, under ground {:.2}% of {}",
        total.1,
        total.0,
        pct(total),
        pct(all[0]),
        all[0].0,
        pct(all[1]),
        all[1].0,
        pct(all[2]),
        all[2].0
    );
    let mut top: Vec<_> = confusions.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    for ((own, drawn), count) in top.iter().take(12) {
        eprintln!("  own {} drawn {}: {count} pixels ({:.2}%)", name(*own), name(*drawn), 100.0 * *count as f64 / total.0.max(1) as f64);
    }
}

/// Edits sealed under a running world: a crater and a mound too large to
/// bake, blocks baked between them, then undo across seals. After each
/// change the kept world matches CPU ray casts exactly.
#[test]
fn sealed_edits_render_exactly_under_a_running_world() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let grid = *planet.grid();
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let ground = planet.surface_point(dir, 0.0);
    let block = |p: DVec3| {
        let (cell, _) = grid.locate(p);
        Brush { center: grid.cell_center(cell).to_array(), radius: grid.voxel_size() * 0.5, shape: BrushShape::Cube, op: BrushOp::Add, material: material::BRICK, height: 0.0 }
    };
    let layer = |planet: &mut Planet, height: f64, every: i32| {
        for x in -12i32..12 {
            for z in -12..12 {
                if (x + z).rem_euclid(every) == 0 {
                    planet.apply(block(ground + side * (f64::from(x) * 0.1) + fwd * (f64::from(z) * 0.1) + up * height)).unwrap();
                }
            }
        }
    };
    let eye = ground + up * 9.0 - fwd * 7.0;
    let view = View::new(eye, ((ground - up * 4.0) - eye).normalize().as_vec3(), [320, 180]);
    let mut world = World::new(&gpu, &planet);
    let mut check = |planet: &Planet| {
        world.sync(&gpu, planet);
        let (compared, mismatched) = compare(&gpu, &mut world, planet, &view);
        assert!(compared > 400, "{compared} compared");
        assert_eq!(mismatched, 0, "{} edits", planet.edits().len());
    };
    check(&planet);
    planet.apply(Brush { center: (ground - up).to_array(), radius: 6.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
    layer(&mut planet, -5.0, 3);
    check(&planet);
    planet.apply(Brush { center: (ground - up * 5.5 + side * 2.5).to_array(), radius: 4.0, shape: BrushShape::Cube, op: BrushOp::Add, material: material::BRICK, height: 0.0 }).unwrap();
    layer(&mut planet, -3.5, 2);
    layer(&mut planet, -2.5, 2);
    assert_eq!(planet.edits().large.len(), 2, "both large brushes sealed");
    assert!(!planet.edits().baked.is_empty());
    check(&planet);
    // Undo across seals: their bricks and large brushes come back undone.
    let sealed = planet.edits().sealed_len();
    for _ in 0..300 {
        planet.undo().unwrap();
    }
    assert!(planet.edits().sealed_len() < sealed);
    check(&planet);
}

/// From orbit (300 km up) every pixel is drawn as the planet's shape
/// allows: a hit wherever the ray meets the sphere 20 km below the datum,
/// sky wherever it misses the sphere 20 km above it (no relief reaches
/// either).
#[test]
fn orbital_view_has_complete_coverage() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *planet.grid();
    let dir = grid.direction(2, f64::from(grid.cells()) * 0.3, f64::from(grid.cells()) * 0.7);
    let eye = dir * (grid.radius() + 300_000.0);
    let view = View::new(eye, (-dir + dir.any_orthonormal_vector() * 0.5).normalize().as_vec3(), [256, 144]);
    let hits = World::new(&gpu, &planet).settle(&gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let meets = |d: DVec3, r: f64| {
        let b = eye.dot(d);
        b * b - (eye.length_squared() - r * r) >= 0.0 && b < 0.0
    };
    let (mut ground, mut wrong) = (0, 0);
    for y in 0..view.size[1] {
        for x in 0..view.size[0] {
            let d = view.pixel_dir(view_proj, x, y);
            let hit = hits[(y * view.size[0] + x) as usize].is_some();
            if meets(d, grid.radius() - 20_000.0) {
                ground += 1;
                wrong += usize::from(!hit);
            } else if !meets(d, grid.radius() + 20_000.0) {
                wrong += usize::from(hit);
            }
        }
    }
    assert!(ground > hits.len() / 3, "{ground} ground pixels");
    assert_eq!(wrong, 0);
}

/// Tools ask for the terrain hit under a view point (the editor's brush):
/// the answer, frames later, is the hit drawn there; the sky has none.
#[test]
fn picks_report_the_drawn_hit() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let mut view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    // The upper rows see the sky.
    view.forward = (view.forward + planet.grid().up(view.eye).as_vec3() * 0.6).normalize();
    let mut world = World::new(&gpu, &planet);
    let hits = world.settle(&gpu, &planet, &view);
    let pixels: Vec<[u32; 2]> = (0..8).map(|n| [20 + n * 37, 5 + n * 24]).collect();
    let picks = SharedPicks::default();
    picks.lock().unwrap().requests = pixels.iter().enumerate().map(|(n, p)| PickRequest { id: n as u64, uv: [(p[0] as f32 + 0.5) / 320.0, (p[1] as f32 + 0.5) / 180.0] }).collect();
    for _ in 0..8 {
        world.frame(&gpu, &planet, &view, Some(&picks));
    }
    let results = std::mem::take(&mut picks.lock().unwrap().results);
    assert_eq!(results.len(), pixels.len());
    assert!(results.iter().any(|r| r.hit.is_none()) && results.iter().any(|r| r.hit.is_some()), "sky and ground picked");
    for r in results {
        let [x, y] = pixels[r.id as usize];
        let drawn = hits[(y * 320 + x) as usize];
        assert_eq!(r.hit.map(|h| (h.cell, h.level)), drawn.map(|h| (h.cell, h.level)), "pixel {x},{y}");
    }
}
