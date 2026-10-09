//! GPU agreement tests: the WGSL field, GPU-generated residency and exact
//! traversal against the CPU canonical planet.
mod common;
use common::*;
use glam::{DVec3, Vec3};
use helio_pass_voxel_planet::terrain::{self, material};
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Cell, Planet, PlanetRecipe};
use std::sync::Arc;

#[test]
fn terrain_programs_are_bit_identical_to_cpu() {
    let Some(gpu) = gpu() else { return };
    use helio_pass_voxel_planet::grid::Shape;
    use helio_pass_voxel_planet::TerrainSource;
    use helio_pass_voxel_planet::layers::TerrainLayers;
    let moon = TerrainLayers::moon().source(7);
    let flat = TerrainLayers { soil_depth_m: 2.0, ..TerrainLayers::flat_at(-1.25) }.source(7);
    let desert = TerrainLayers::desert().source(7);
    let mut deep = TerrainLayers::earth();
    deep.caves.depth_m = 900.0;
    let deep = deep.source(7);
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
    ] {
        let planet = Planet::new(PlanetRecipe { shape, voxel_size_m: size, plane_size_m: 5_000.0, terrain: terrain.clone(), ..Default::default() }).unwrap();
        helio_pass_voxel_planet::engine::verify_field(&gpu.device, &gpu.queue, &planet, 20_000)
            .unwrap_or_else(|e| panic!("{} on {shape:?} at {size} m: {e}", terrain.generator));
    }
}

fn settle(gpu: &Gpu, target: &Target, renderer: &mut helio_pass_voxel_planet::engine::PlanetRenderer, frame: &helio_pass_voxel_planet::engine::PlanetFrame, forward: Vec3) -> usize {
    let mut frames = 0;
    for n in 0..2000 {
        target.render(gpu, renderer, frame, forward, n);
        frames += 1;
        if renderer.settled() {
            break;
        }
    }
    // One more frame so published columns are traced.
    target.render(gpu, renderer, frame, forward, frames as u64);
    frames
}

/// Compare GPU primary hits inside the level-0 range with CPU ray casts.
fn compare_near(gpu: &Gpu, planet: &Arc<Planet>, eye: DVec3, forward: Vec3, size: [u32; 2]) -> (usize, usize) {
    let target = Target::new(gpu, size);
    let mut renderer = renderer(gpu, planet.clone(), size);
    compare_view(gpu, &target, &mut renderer, planet, eye, forward, size)
}

/// [`compare_near`] with a renderer kept across views (the eye moves).
fn compare_view(
    gpu: &Gpu,
    target: &Target,
    renderer: &mut helio_pass_voxel_planet::engine::PlanetRenderer,
    planet: &Arc<Planet>,
    eye: DVec3,
    forward: Vec3,
    size: [u32; 2],
) -> (usize, usize) {
    let frame = frame(planet, eye);
    let frames = settle(gpu, target, renderer, &frame, forward);
    let stats = renderer.stats();
    eprintln!("settled after {frames} frames: {stats:?}");
    assert_eq!(stats.failed_jobs, 0);
    let hits = hits(gpu, renderer);
    let up = planet.grid().up(eye).as_vec3();
    let camera = target.camera(forward, if forward.normalize().dot(up).abs() > 0.99 { up.any_orthonormal_vector() } else { up });
    let lod0 = stats.lod0_distance;
    let (mut compared, mut mismatched) = (0, 0);
    let mut bad = [0usize; 4];
    for (index, hit) in hits.iter().enumerate() {
        bad[hit.status as usize] += 1;
        let (x, y) = (index as u32 % size[0], index as u32 / size[0]);
        if x % 3 != 0 || y % 3 != 0 {
            continue;
        }
        let dir = pixel_dir(&target, &camera, x, y);
        let cpu = planet.raycast(eye, dir, lod0 * 0.7);
        match cpu {
            Some(c) if c.distance < lod0 * 0.7 => {
                compared += 1;
                let gpu_cell = Cell::new(hit.face, hit.i, hit.j, hit.k);
                if hit.status != 1 || hit.level != 0 || gpu_cell != c.cell || (f64::from(hit.t) - c.distance).abs() > 1e-3 + c.distance * 1e-5 {
                    mismatched += 1;
                    if mismatched < 8 {
                        eprintln!("pixel {x},{y}: gpu {hit:?} cpu {:?} t {}", c.cell, c.distance);
                    }
                }
            }
            _ => {}
        }
    }
    eprintln!("status counts miss/hit/exhausted/loading = {bad:?}");
    assert_eq!(bad[2], 0, "exhausted rays");
    assert_eq!(bad[3], 0, "loading rays after settling");
    (compared, mismatched)
}

#[test]
fn ground_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&planet, 4, 0.37, 0.61);
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let forward = (up.any_orthonormal_vector() - up * 0.35).normalize().as_vec3();
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [320, 180]);
    eprintln!("compared {compared}, mismatched {mismatched}");
    assert!(compared > 1000);
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// An eye inside a generated cave: an air cell with air around it, at least
/// four cells below its column's heightfield top, in a cave region.
fn find_cave(planet: &Planet) -> Option<(DVec3, Vec3)> {
    let grid = *planet.grid();
    let field = planet.field();
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng
    };
    // Land sites across the faces until one lies in a cave region.
    let sites = (0..6u8).flat_map(|face| [(0.37, 0.61), (0.2, 0.3), (0.7, 0.45)].map(|(a, b)| (face, a, b)));
    for (face, a, b) in sites {
        let dir = land(planet, face, a, b);
        let (base, _) = grid.locate(planet.surface_point(dir, 0.0));
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
                // A tunnel or small chamber: walls, floor and ceiling within the
                // level-0 range that compare_near checks.
                let side = up.cross(forward);
                if [forward, -forward, side, -side, up, -up].iter().all(|d| planet.raycast(eye, *d, 8.0).is_some()) {
                    return Some((eye, forward.as_vec3()));
                }
            }
        }
    }
    None
}

/// Generated caves render exactly: from inside one, GPU primary hits (walls,
/// floor, ceiling) match canonical CPU ray casts.
#[test]
fn cave_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let (eye, forward) = find_cave(&planet).expect("a cave near the test site");
    eprintln!("cave eye {} m below its column top", -planet.ground_height(eye));
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [320, 180]);
    eprintln!("compared {compared}, mismatched {mismatched}");
    assert!(compared > 1000);
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// An eye over leaning ground, looking up the slope: a column of an overhang
/// region whose generated cells the lean moved off the heightfield by at
/// least two cells, on sloping ground.
fn find_overhang(planet: &Planet) -> Option<(DVec3, Vec3)> {
    let grid = *planet.grid();
    let field = planet.field();
    let mut rng = 0x1D8E_4E27_C47D_124Fu64;
    let mut next = || {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng
    };
    let sites = (0..6u8).flat_map(|face| [(0.37, 0.61), (0.2, 0.3), (0.7, 0.45)].map(|(a, b)| (face, a, b)));
    for (face, a, b) in sites {
        let dir = land(planet, face, a, b);
        let (base, _) = grid.locate(planet.surface_point(dir, 0.0));
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
            let changed = (t - above..t + above).filter(|&k| planet.solid(Cell::new(base.face, i, j, k)) != (k < t)).count();
            if changed < 2 {
                continue;
            }
            let eye = planet.surface_point(grid.cell_center(Cell::new(base.face, i, j, t)), 2.0);
            let up = eye.normalize();
            // Up the slope, a little downwards.
            let (di, dj) = if si.abs() >= sj.abs() { (si.signum(), 0) } else { (0, sj.signum()) };
            let along = grid.cell_center(Cell::new(base.face, i + di, j + dj, t)) - grid.cell_center(Cell::new(base.face, i, j, t));
            let forward = ((along - up * along.dot(up)).normalize() - up * 0.3).normalize();
            eprintln!("lean moved {changed} cells of the column; slope {} cells over 40", si.abs().max(sj.abs()));
            return Some((eye, forward.as_vec3()));
        }
    }
    None
}

/// Generated overhangs render exactly: over leaning ground, GPU primary hits
/// match canonical CPU ray casts, so the generation's lean lattice
/// reproduces the field's.
#[test]
fn overhang_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let mut stack = helio_pass_voxel_planet::layers::TerrainLayers::earth();
    stack.caves.enabled = false;
    let planet = Arc::new(Planet::new(PlanetRecipe { terrain: stack.source(7), ..Default::default() }).unwrap());
    let (eye, forward) = find_overhang(&planet).expect("an overhang near the test site");
    eprintln!("overhang eye {} m off its column top", planet.ground_height(eye));
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [320, 180]);
    eprintln!("compared {compared}, mismatched {mismatched}");
    assert!(compared > 1000);
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// Caves 800 m deep, far below what one band holds at the fine levels:
/// from inside one at a random depth, GPU hits match CPU ray casts.
#[test]
fn deep_cave_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let mut stack = helio_pass_voxel_planet::layers::TerrainLayers::earth();
    stack.caves.depth_m = 800.0;
    let planet = Arc::new(Planet::new(PlanetRecipe { terrain: stack.source(7), ..Default::default() }).unwrap());
    let (eye, forward) = find_cave(&planet).expect("a cave near the test site");
    eprintln!("cave eye {} m below its column top", -planet.ground_height(eye));
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [320, 180]);
    eprintln!("compared {compared}, mismatched {mismatched}");
    assert!(compared > 1000);
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// Level-0 column indices past 2^23 (the far third of a 0.1 m Earth face)
/// decode unsigned from their 24-bit keys.
#[test]
fn ground_view_near_a_far_face_edge_matches_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land_within(&planet, 4, 0.86, 0.99);
    let cell = planet.grid().locate(dir * planet.grid().radius()).0;
    assert!(cell.i >> 3 >= 1 << 23 && cell.j >> 3 >= 1 << 23, "{cell:?}");
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let forward = (up.any_orthonormal_vector() - up * 0.35).normalize().as_vec3();
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [320, 180]);
    assert!(compared > 1000);
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// The plane sky bound (horizontal distances) only ends rays that provably
/// miss: renders with and without it show the same surfaces, from the
/// ground, low flight and altitude, looking level, up and down.
#[test]
fn plane_sky_bound_is_conservative() {
    let Some(gpu) = gpu() else { return };
    for shape in [helio_pass_voxel_planet::grid::Shape::Plane, helio_pass_voxel_planet::grid::Shape::InfinitePlane] {
        let planet = Arc::new(Planet::new(PlanetRecipe { shape, plane_size_m: 3_000.0, ..Default::default() }).unwrap());
        let mut saved_total = 0i64;
        for (x, z, height, pitch) in [(12.3, -45.6, 1.7, 0.02), (200.0, 100.0, 1.7, 0.3), (-400.0, 300.0, 40.0, -0.1), (100.0, -700.0, 600.0, -0.4), (0.0, 0.0, 3_000.0, -0.2)] {
            let eye = planet.surface_point(DVec3::new(x, 0.0, z), height);
            let forward = Vec3::new(0.6, pitch, -0.8).normalize();
            let size = [320, 180];
            let target = Target::new(&gpu, size);
            let mut r = renderer(&gpu, planet.clone(), size);
            let f = frame(&planet, eye);
            settle(&gpu, &target, &mut r, &f, forward);
            r.settings_mut().lod_dither = 0.0;
            r.settings_mut().freeze_residency = true;
            r.settings_mut().horizon = false;
            target.render(&gpu, &mut r, &f, forward, 5000);
            let reference = hits(&gpu, &r);
            r.settings_mut().horizon = true;
            target.render(&gpu, &mut r, &f, forward, 5000);
            let bounded = hits(&gpu, &r);
            let voxel = planet.grid().voxel_size() as f32;
            let camera = target.camera(forward, Vec3::Y);
            let mut bad = 0;
            for (index, (a, b)) in reference.iter().zip(&bounded).enumerate() {
                saved_total += i64::from(a.steps) - i64::from(b.steps);
                let rise = pixel_dir(&target, &camera, index as u32 % 320, index as u32 / 320).y.abs() as f32;
                if !skipped_nothing(a, b, rise, voxel) {
                    bad += 1;
                    if bad < 6 {
                        eprintln!("{shape:?} height {height} pixel {} {}: plain {a:?} bounded {b:?}", index % 320, index / 320);
                    }
                }
            }
            assert_eq!(bad, 0, "{shape:?} at {height} m");
        }
        eprintln!("{shape:?}: sky bound saved {saved_total} steps");
        assert!(saved_total > 0);
    }
}

/// Finite and infinite planes: ground and elevated views match exact CPU
/// ray casts and settle without exhausted or loading rays.
#[test]
fn plane_views_match_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    for shape in [helio_pass_voxel_planet::grid::Shape::Plane, helio_pass_voxel_planet::grid::Shape::InfinitePlane] {
        let planet = Arc::new(Planet::new(PlanetRecipe { shape, plane_size_m: 3_000.0, ..Default::default() }).unwrap());
        for (x, z, height, pitch) in [(12.3, -45.6, 1.7, -0.35), (-300.0, 250.0, 8.0, -1.2)] {
            let eye = planet.surface_point(DVec3::new(x, 0.0, z), height);
            let forward = Vec3::new(0.8, pitch, 0.6).normalize();
            let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [320, 180]);
            eprintln!("{shape:?} at {height} m: compared {compared}, mismatched {mismatched}");
            assert!(compared > 1000);
            assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
        }
    }
}

#[test]
fn edits_propagate_to_gpu_generation() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let eye = planet.surface_point(dir, 3.0);
    let up = eye.normalize();
    let side = up.any_orthonormal_vector();
    let ground = planet.surface_point(dir, 0.0) + side * 4.0;
    planet
        .apply(Brush { center: ground.to_array(), radius: 2.5, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 })
        .unwrap();
    planet
        .apply(Brush { center: (ground + side * 3.0 + up * 1.5).to_array(), radius: 0.8, shape: BrushShape::Cube, op: BrushOp::Add, material: material::BRICK })
        .unwrap();
    let planet = Arc::new(planet);
    let forward = (side - up * 0.6).normalize().as_vec3();
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [256, 144]);
    eprintln!("compared {compared}, mismatched {mismatched}");
    assert!(compared > 1000);
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// Destruction to any depth: a 600 m shaft is far taller than a column's
/// 256-brick band at the fine levels, which keep a window around the eye
/// (clipped bands) and use coarser levels beyond it. At the bottom and after
/// climbing 300 m (the windows follow the eye), GPU hits near the eye match
/// canonical CPU ray casts exactly, with no loading or exhausted rays.
#[test]
fn a_deep_shaft_renders_exactly_at_any_depth() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 3, 0.41, 0.57);
    let up = planet.grid().up(planet.surface_point(dir, 0.0));
    let ground = planet.surface_point(dir, 0.0);
    let mut depth = -10.0;
    while depth < 600.0 {
        let center = ground - up * depth;
        planet.apply(Brush { center: center.to_array(), radius: 2.0, shape: BrushShape::Cube, op: BrushOp::Remove, material: 0 }).unwrap();
        depth += 3.0;
    }
    let planet = Arc::new(planet);
    let size = [256, 144];
    let target = Target::new(&gpu, size);
    let mut renderer = renderer(&gpu, planet.clone(), size);
    let side = up.any_orthonormal_vector();
    for (depth, pitch) in [(590.0, -0.6), (290.0, 0.4)] {
        let eye = ground - up * depth + side * 0.7;
        let forward = (side + up * pitch).normalize().as_vec3();
        let (compared, mismatched) = compare_view(&gpu, &target, &mut renderer, &planet, eye, forward, size);
        let stats = renderer.stats();
        eprintln!("{depth} m deep: compared {compared}, mismatched {mismatched}, clipped columns {}", stats.clipped_columns);
        assert!(stats.clipped_columns > 0, "the shaft needs clipped bands");
        assert!(compared > 1000);
        assert_eq!(mismatched, 0);
    }
}

/// Tools ask the pass for the terrain hit under a view point (the editor's
/// brush): the answer, a few frames later, lies within its cell size of the
/// exact CPU hit along that pixel's ray, near and far; the sky has none.
#[test]
fn picks_report_the_terrain_hit_under_the_view() {
    use helio_pass_voxel_planet::engine::{PickRequest, SharedPicks};
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let ground = planet.surface_point(land(&planet, 2, 0.47, 0.53), 0.0);
    let eye = planet.surface_point(ground, 60.0);
    let up = planet.grid().up(eye);
    let forward = (up.any_orthonormal_vector() - up * 0.15).normalize().as_vec3();
    let size = [256, 144];
    let target = Target::new(&gpu, size);
    let mut renderer = renderer(&gpu, planet.clone(), size);
    let picks: SharedPicks = Default::default();
    let mut frame = frame(&planet, eye);
    frame.picks = Some(picks.clone());
    let frames = settle(&gpu, &target, &mut renderer, &frame, forward);
    let camera = target.camera(forward, up.as_vec3());
    // Pixels from just below the horizon to the bottom, and one in the sky.
    let pixels = [(128u32, 2u32), (128, 70), (40, 80), (200, 100), (128, 143)];
    picks.lock().unwrap().requests.extend(pixels.iter().enumerate().map(|(n, &(x, y))| PickRequest {
        id: n as u64,
        uv: [(x as f32 + 0.5) / size[0] as f32, (y as f32 + 0.5) / size[1] as f32],
    }));
    for n in 0..4 {
        target.render(&gpu, &mut renderer, &frame, forward, (frames + n) as u64);
    }
    let results = std::mem::take(&mut picks.lock().unwrap().results);
    assert_eq!(results.len(), pixels.len(), "every request is answered");
    for result in results {
        let (x, y) = pixels[result.id as usize];
        let dir = pixel_dir(&target, &camera, x, y);
        let cpu = planet.raycast(eye, dir, 100_000.0);
        eprintln!("pick {x},{y}: {:?} cpu {:?}", result.hit, cpu.map(|c| c.distance));
        match (result.hit, cpu) {
            (Some(hit), Some(cpu)) => assert!((hit.distance - cpu.distance).abs() <= hit.cell_m * 2.0 + 0.5, "pixel {x},{y}: {hit:?} vs {}", cpu.distance),
            (None, None) => {}
            (hit, cpu) => panic!("pixel {x},{y}: pick {hit:?}, cpu {cpu:?}"),
        }
    }
}

/// Planet-scale brushes: a sphere of 3 km radius (64-bit containment; the
/// old 32-bit test capped brushes at 1.85 km) leaves a crater 3 km deep. On
/// its floor, GPU hits match canonical CPU ray casts.
#[test]
fn a_planet_scale_crater_renders_exactly_on_its_floor() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 5, 0.33, 0.52);
    let ground = planet.surface_point(dir, 0.0);
    let up = planet.grid().up(ground);
    planet.apply(Brush { center: ground.to_array(), radius: 3_000.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 }).unwrap();
    let planet = Arc::new(planet);
    let eye = ground - up * (3_000.0 - 1.7);
    assert!(!planet.solid(planet.grid().locate(eye).0), "the eye is in the crater");
    let side = up.any_orthonormal_vector();
    let forward = (side - up * 0.4).normalize().as_vec3();
    let size = [256, 144];
    let target = Target::new(&gpu, size);
    let mut renderer = renderer(&gpu, planet.clone(), size);
    let (compared, mismatched) = compare_view(&gpu, &target, &mut renderer, &planet, eye, forward, size);
    eprintln!("crater floor: compared {compared}, mismatched {mismatched}, clipped columns {}", renderer.stats().clipped_columns);
    assert!(compared > 1000);
    assert_eq!(mismatched, 0);
}

/// The whole planet is destructible: a ball around the planet's centre
/// (resolved in volume space on every face) hollows out the core of a 3 km
/// world. From inside the hollow, GPU hits on its inner wall match
/// canonical CPU ray casts.
#[test]
fn a_hollowed_core_renders_exactly_from_inside() {
    let Some(gpu) = gpu() else { return };
    let recipe = PlanetRecipe { radius_m: 3_000.0, terrain: helio_pass_voxel_planet::layers::TerrainLayers::earth().heightfield().source(7), ..Default::default() };
    let mut planet = Planet::new(recipe).unwrap();
    planet.apply(Brush { center: [0.0; 3], radius: 1_500.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 }).unwrap();
    let up = DVec3::new(0.3, 0.9, 0.2).normalize();
    let eye = up * (1_500.0 - 1.7);
    assert!(!planet.solid(planet.grid().locate(eye).0), "the eye is in the hollow");
    assert!(planet.solid(planet.grid().locate(up * 1_500.5).0), "the hollow has a wall");
    assert!(!planet.solid(planet.grid().locate(up * 10.0).0), "the core is gone");
    let planet = Arc::new(planet);
    let side = up.any_orthonormal_vector();
    let forward = (side + up * 0.5).normalize().as_vec3();
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [256, 144]);
    eprintln!("hollow core: compared {compared}, mismatched {mismatched}");
    assert!(compared > 1000);
    assert_eq!(mismatched, 0);
}

/// Destruction and construction at scale: a building of single 0.1 m
/// blocks and a field of single-block holes (about 18 000 block edits)
/// render exactly as the CPU world has them, with no failed generation and
/// no loading rays once settled.
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
        Brush { center: grid.cell_center(cell).to_array(), radius: grid.voxel_size() * 0.5, shape: BrushShape::Cube, op, material: if op == BrushOp::Add { material::BRICK } else { 0 } }
    };
    let at = |x: i32, y: i32, z: i32| ground + side * (f64::from(x) * 0.1) + fwd * (f64::from(z) * 0.1 + 6.0) + up * (f64::from(y) * 0.1 + 0.55);
    let started = std::time::Instant::now();
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
    eprintln!("{count} block edits applied in {:?}", started.elapsed());
    assert!(count > 12_000, "{count}");
    let planet = Arc::new(planet);
    let eye = ground - fwd * 6.0 + up * 14.0 - side * 3.0;
    let forward = ((ground + fwd * 6.0) - eye).normalize().as_vec3();
    let started = std::time::Instant::now();
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [320, 180]);
    let elapsed = started.elapsed();
    eprintln!("settled and compared in {elapsed:?}: {mismatched}/{compared} mismatched");
    assert!(compared > 400);
    // CPU ray casts through the dense edits stay fast (they query edits per cell).
    assert!(elapsed.as_secs() < 20, "{elapsed:?}");
    assert_eq!(mismatched, 0);
}

#[test]
fn orbital_view_has_complete_coverage() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let grid = *planet.grid();
    let dir = grid.direction(2, f64::from(grid.cells()) * 0.3, f64::from(grid.cells()) * 0.7);
    let eye = dir * (grid.radius() + 300_000.0);
    let forward = (-dir + dir.any_orthonormal_vector() * 0.5).normalize().as_vec3();
    let size = [256, 144];
    let target = Target::new(&gpu, size);
    let mut r = renderer(&gpu, planet.clone(), size);
    let f = frame(&planet, eye);
    let frames = settle(&gpu, &target, &mut r, &f, forward);
    let h = hits(&gpu, &r);
    let mut counts = [0usize; 4];
    for hit in &h {
        counts[hit.status as usize] += 1;
    }
    eprintln!("orbit settled in {frames} frames, statuses {counts:?}, {:?}", r.stats());
    assert_eq!(counts[2] + counts[3], 0);
    assert!(counts[1] > h.len() / 2);
}

/// The directional sky bound only ends rays that provably miss: every pixel
/// matches a render without it, including views up at distant terrain.
#[test]
fn sky_bound_is_conservative() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    for (face, fi, fj, pitch) in [(2u8, 0.47, 0.53, 0.02), (2, 0.47, 0.53, 0.3), (4, 0.37, 0.61, 0.08), (1, 0.52, 0.48, 0.6), (0, 0.3, 0.7, 0.0)] {
        let dir = land(&planet, face, fi, fj);
        let eye = planet.surface_point(dir, 1.7);
        let up = eye.normalize();
        let forward = (up.any_orthonormal_vector() + up * pitch).normalize().as_vec3();
        let size = [320, 180];
        let target = Target::new(&gpu, size);
        let mut r = renderer(&gpu, planet.clone(), size);
        let f = frame(&planet, eye);
        settle(&gpu, &target, &mut r, &f, forward);
        r.settings_mut().lod_dither = 0.0;
        if std::env::var_os("SKY_DUMP").is_some() {
            target.render(&gpu, &mut r, &f, forward, 5000);
            let table = read_buffer(&gpu, r.horizon_buffer(), 257 * 32 * 4);
            let v: Vec<f32> = table.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            for b in 0..32 {
                let row = &v[b * 256..b * 256 + 256];
                let lo = row.iter().copied().fold(f32::INFINITY, f32::min).to_degrees();
                let hi = row.iter().copied().fold(f32::NEG_INFINITY, f32::max).to_degrees();
                eprintln!("bucket {b:2}: clearing elevation {lo:7.3}..{hi:7.3} deg, all {:7.3}", v[256 * 32 + b].to_degrees());
            }
        }
        {
            r.settings_mut().horizon = false;
            target.render(&gpu, &mut r, &f, forward, 5000);
            let reference = hits(&gpu, &r);
            r.settings_mut().horizon = true;
            target.render(&gpu, &mut r, &f, forward, 5000);
            let bounded = hits(&gpu, &r);
            let (mut bad, mut hits_seen, mut saved) = (0, 0, 0i64);
            for (index, (a, b)) in reference.iter().zip(&bounded).enumerate() {
                hits_seen += usize::from(a.status == 1);
                saved += i64::from(a.steps) - i64::from(b.steps);
                let both_miss = a.status == 0 && b.status == 0;
                if !both_miss && (a.status, a.i, a.j, a.k, a.face, a.level) != (b.status, b.i, b.j, b.k, b.face, b.level) {
                    bad += 1;
                    if bad < 6 {
                        eprintln!("pixel {} {}: reference {a:?} bounded {b:?}", index % 320, index / 320);
                    }
                }
            }
            eprintln!("face {face} pitch {pitch}: {bad} differences, {hits_seen} hits, {saved} steps saved");
            assert_eq!(bad, 0);
        }
    }
}

/// The sky bound stays conservative while the eye moves and residency is
/// incomplete (pending columns, lagging windows): after every unsettled step,
/// two frozen renders with and without the bound must agree per pixel.
#[test]
fn sky_bound_is_conservative_while_moving() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&planet, 2, 0.47, 0.53);
    let mut eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let east = up.any_orthonormal_vector();
    let size = [320, 180];
    let target = Target::new(&gpu, size);
    let mut r = renderer(&gpu, planet.clone(), size);
    r.settings_mut().lod_dither = 0.0;
    settle(&gpu, &target, &mut r, &frame(&planet, eye), (east + up * 0.05).normalize().as_vec3());
    let (mut bad, mut compared) = (0, 0);
    for step in 0..80u64 {
        // Walking pace (few pending columns, exact nearest-pending bound),
        // then vehicle speed with a climb (windows lag, many pending).
        eye = if step < 20 {
            planet.surface_point(eye + east * 1.5, 1.7)
        } else if step < 40 {
            planet.surface_point(eye + east * 15.0, 1.7)
        } else {
            planet.surface_point(eye + east * 60.0, 1.7 + (step - 40) as f64 * 3.0)
        };
        let f = frame(&planet, eye);
        let up = eye.normalize();
        let forward = (east + up * (0.02 + 0.01 * (step % 5) as f64)).normalize().as_vec3();
        target.render(&gpu, &mut r, &f, forward, 6000 + step * 3);
        r.settings_mut().freeze_residency = true;
        r.settings_mut().horizon = false;
        target.render(&gpu, &mut r, &f, forward, 6000 + step * 3 + 1);
        let reference = hits(&gpu, &r);
        r.settings_mut().horizon = true;
        target.render(&gpu, &mut r, &f, forward, 6000 + step * 3 + 2);
        let bounded = hits(&gpu, &r);
        r.settings_mut().freeze_residency = false;
        let voxel = planet.grid().voxel_size() as f32;
        let camera = target.camera(forward, up.as_vec3());
        for (index, (a, b)) in reference.iter().zip(&bounded).enumerate() {
            compared += 1;
            let rise = pixel_dir(&target, &camera, index as u32 % 320, index as u32 / 320).dot(up).abs() as f32;
            if !skipped_nothing(a, b, rise, voxel) {
                bad += 1;
                if bad < 6 {
                    eprintln!("step {step} pixel {} {}: reference {a:?} bounded {b:?}", index % 320, index / 320);
                }
            }
        }
        let stats = r.stats();
        if step % 5 == 0 {
            eprintln!("step {step}: pending {} resident {}", stats.pending_columns, stats.resident_columns);
        }
    }
    eprintln!("{bad} differences in {compared} pixels");
    assert_eq!(bad, 0);
}

/// Published column tops bound every occupied cell of the column, and a
/// complete summary block's maximum bounds its columns' tops.
#[test]
fn published_tops_bound_occupancy() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&planet, 2, 0.47, 0.53);
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let forward = (up.any_orthonormal_vector() - up * 0.2).normalize().as_vec3();
    let size = [320, 180];
    let target = Target::new(&gpu, size);
    let mut r = renderer(&gpu, planet.clone(), size);
    settle(&gpu, &target, &mut r, &frame(&planet, eye), forward);
    let [records, pool, _blocks] = r.residency_buffers();
    let words = |b: &wgpu::Buffer| -> Vec<u32> {
        read_buffer(&gpu, b, b.size()).chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let rec = words(records);
    let pool = words(pool);
    let (mut columns, mut bad) = (0usize, 0usize);
    let (mut volumetric, mut wrong_tops) = (0usize, 0usize);
    // Programs with a surface word store it after the header and relief.
    let surface_units = if planet.field().program().wgsl.contains("fn terrain_surface") { 1 } else { 0 };
    for c in rec.chunks_exact(8) {
        let info = c[3];
        if info & 0xc000_0000 != 0x8000_0000 {
            continue;
        }
        columns += 1;
        let k_lo = c[2] as i32;
        let n_band = (info & 511) as i32;
        let gap = ((info >> 22) & 7) as i32;
        let run = c[4];
        let ext = info & 0x2000_0000 != 0;
        // Wide fractional tops occupy two units after the column header;
        // inline tops retain their authored-cell offset in each packed byte.
        let relief = info & 0x1000_0000 != 0;
        // Bit 26: inline fractional tops (generated volume, bit 12, is
        // always wide).
        let inline = info & 0x0400_0000 != 0;
        let heightfield = info & 0x0200_0000 != 0;
        // Then one unit of surface offsets.
        let header = (if ext { 2 } else { 1 }) + (if relief && !inline { 2 } else { 0 }) + surface_units + 1;
        if relief {
            assert!(n_band < 32 || (n_band == 32 && gap > 0), "fractional tops overflow their packed byte range");
        }
        let published = (k_lo + n_band) * 8 - gap;
        // Highest occupied cell from the brick masks.
        let mut highest = i32::MIN;
        if heightfield {
            // Independently decode all 64 authored tops. These columns store
            // solid-below-top occupancy exactly and have no bitmap payload.
            let level = c[0] >> 27;
            for cell in 0..64u32 {
                let word = pool[(run * 16 + (cell >> 2)) as usize];
                let offset = ((word >> ((cell & 3) * 8)) & 255) as i32;
                let top_offset = if inline {
                    (offset + (1i32 << level) - 1) >> level
                } else {
                    offset
                };
                highest = highest.max(k_lo * 8 + top_offset - 1);
            }
            assert_eq!(highest + 1, published, "packed authored top disagrees with published maximum");
        } else {
            // Generated volumetric columns publish their generated tops
            // (material depth) counting down from the band top.
            if info & 0x1000 != 0 {
                volumetric += 1;
                let (level, face) = (c[0] >> 27, ((c[0] >> 24) & 7) as u8);
                let (ci, cj) = ((c[0] & 0xff_ffff) as i32, c[1] as i32);
                for cell in 0..64u32 {
                    let word = pool[(run * 16 + (cell >> 2)) as usize];
                    let down = ((word >> ((cell & 3) * 8)) & 255) as i32;
                    let (i, j) = (ci * 8 + (cell & 7) as i32, cj * 8 + (cell >> 3) as i32);
                    let expected = terrain::generated_top(planet.grid(), planet.field(), face, i, j, level, planet.column_height(face, i, j, level));
                    let got = (k_lo + n_band) * 8 - down;
                    // With relief, the partial top cell is solid (cut at
                    // the surface), whether the volume changed the surface
                    // or not.
                    let ceil = relief && got == expected + 1;
                    if down < 255 && got != expected && !ceil {
                        wrong_tops += 1;
                        if wrong_tops < 6 {
                            eprintln!("column key {:08x} {:08x} cell {cell}: generated top {} expected {expected}", c[0], c[1], (k_lo + n_band) * 8 - down);
                        }
                    }
                }
            }
            for b in (0..n_band).rev() {
                let (mixed, solid, rank) = if b < 32 {
                    let bit = b as u32;
                    ((c[5] >> bit) & 1 != 0, (c[6] >> bit) & 1 != 0, (c[5] & ((1u32 << bit) - 1)).count_ones())
                } else {
                    let e = ((run + 1) * 16) as usize;
                    let (w, bit) = ((b >> 5) as usize, (b & 31) as u32);
                    let mut rank = 0;
                    for q in 0..w {
                        rank += pool[e + q].count_ones();
                    }
                    rank += (pool[e + w] & ((1u32 << bit) - 1)).count_ones();
                    ((pool[e + w] >> bit) & 1 != 0, (pool[e + 8 + w] >> bit) & 1 != 0, rank)
                };
                if solid {
                    highest = (k_lo + b) * 8 + 7;
                    break;
                }
                if mixed {
                    let unit = ((run + header + rank) * 16) as usize;
                    let z = (0..8).rev().find(|z| pool[unit + 2 * z] | pool[unit + 2 * z + 1] != 0).unwrap_or(0) as i32;
                    highest = (k_lo + b) * 8 + z;
                    break;
                }
            }
            }
        if highest >= published {
            bad += 1;
            if bad < 6 {
                eprintln!("column key {:08x} {:08x}: highest occupied {highest} published top {published} (k_lo {k_lo} band {n_band} gap {gap})", c[0], c[1]);
            }
        }
    }
    eprintln!("{columns} columns, {bad} with occupied cells above the published top");
    eprintln!("{volumetric} volumetric columns, {wrong_tops} generated tops differing from the CPU");
    assert!(columns > 1000);
    assert_eq!(bad, 0);
    assert_eq!(wrong_tops, 0);
}

/// Surface offsets reconstruct the generator's exact height below voxel
/// precision: stored height plus offset is the field height to 1/128 of the
/// stored height's precision, for the smooth shading normals and material
/// height of natural ground. A base cell over a level-0 or relief top; a
/// level cell over the whole-cell tops of columns without relief (here the
/// columns a wide dig touches), whose surface once sank below its
/// neighbours' (dark outlines around every edit).
#[test]
fn surface_offsets_reconstruct_the_field_height() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 4, 0.37, 0.61);
    let ground = planet.surface_point(dir, 0.0);
    planet.apply(Brush { center: (ground + (ground.normalize().any_orthonormal_vector()) * 60.0).to_array(), radius: 40.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 }).unwrap();
    let planet = Arc::new(planet);
    let eye = planet.surface_point(dir, 30.0);
    let up = eye.normalize();
    let forward = (up.any_orthonormal_vector() - up * 0.5).normalize().as_vec3();
    let size = [320, 180];
    let target = Target::new(&gpu, size);
    // Canonical heights (no ridge display), so the CPU field is the reference.
    let settings = helio_pass_voxel_planet::engine::Settings { ridge_display: false, ..Default::default() };
    let mut r = helio_pass_voxel_planet::engine::PlanetRenderer::new(&gpu.device, &gpu.queue, planet.clone(), settings, size);
    settle(&gpu, &target, &mut r, &frame(&planet, eye), forward);
    let [records, pool, _] = r.residency_buffers();
    let words = |b: &wgpu::Buffer| -> Vec<u32> {
        read_buffer(&gpu, b, b.size()).chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let (rec, pool) = (words(records), words(pool));
    let surface_units = if planet.field().program().wgsl.contains("fn terrain_surface") { 1 } else { 0 };
    let layer = planet.grid().layer_mm() as f64;
    let (mut lanes, mut whole, mut worst) = (0usize, 0usize, 0.0f64);
    for c in rec.chunks_exact(8) {
        let info = c[3];
        // Valid columns other than generated volume whose tops fit their
        // byte: level-0 tops, inline relief, and whole-cell tops (no relief).
        let n_band = info & 511;
        let fits = n_band < 32 || (n_band == 32 && (info >> 22) & 7 != 0);
        if info & 0xc000_0000 != 0x8000_0000 || info & 0x1000 != 0 || !fits {
            continue;
        }
        let level = c[0] >> 27;
        let relief = info & 0x1000_0000 != 0;
        let inline = relief && info & 0x0400_0000 != 0;
        if level != 0 && relief && !inline {
            continue;
        }
        // Units of the stored height and the offset, in base cells.
        let unit = if level == 0 || relief { 1.0 } else { f64::from(1u32 << level) };
        let (face, ci, cj, k_lo, run) = (((c[0] >> 24) & 7) as u8, (c[0] & 0xff_ffff) as i32, c[1] as i32, c[2] as i32, c[4]);
        let ext = info & 0x2000_0000 != 0;
        let offsets = run + if ext { 2 } else { 1 } + surface_units;
        for cell in 0..64u32 {
            let byte = |unit: u32| ((pool[(unit * 16 + (cell >> 2)) as usize] >> ((cell & 3) * 8)) & 255) as i32;
            // Stored height (base cells): the top byte above the band base,
            // in base cells (inline relief, level 0) or level cells.
            let stored = if unit == 1.0 { f64::from(((k_lo * 8) << level) + byte(run)) } else { f64::from(k_lo * 8 + byte(run)) * unit };
            let offset = (byte(offsets) - 128) as f64 / 128.0 * unit;
            let (i, j) = (ci * 8 + (cell & 7) as i32, cj * 8 + (cell >> 3) as i32);
            let exact = planet.column_height(face, i, j, level) as f64 / layer;
            worst = worst.max((stored + offset - exact).abs() / unit);
            lanes += 1;
            whole += usize::from(unit > 1.0);
        }
    }
    eprintln!("{lanes} lanes ({whole} over whole-cell tops), largest error {worst:.4} of a stored unit");
    assert!(lanes > 10_000 && whole > 64);
    assert!(worst <= 1.0 / 128.0 + 1e-9, "{worst}");
}

/// Generated volume is stored only where the volume changes a cell: in cave
/// and overhang regions, columns whose cells all keep the heightfield's
/// kinds stay heightfield columns (band, relief, no bitmap bricks), and
/// every generated column changes some cell.
#[test]
fn generated_volume_is_stored_only_where_cells_change() {
    let Some(gpu) = gpu() else { return };
    let mut stack = helio_pass_voxel_planet::layers::TerrainLayers::earth();
    stack.caves.depth_m = 30.0;
    let planet = Arc::new(Planet::new(PlanetRecipe { terrain: stack.source(7), ..Default::default() }).unwrap());
    let size = [320, 180];
    let target = Target::new(&gpu, size);
    let (mut extent_columns, mut generated, mut checked, mut unchanged) = (0usize, 0usize, 0usize, 0usize);
    for (eye, forward) in [find_cave(&planet).expect("a cave"), find_overhang(&planet).expect("an overhang")] {
        let mut r = renderer(&gpu, planet.clone(), size);
        settle(&gpu, &target, &mut r, &frame(&planet, eye), forward);
        let [records, _, _] = r.residency_buffers();
        let rec: Vec<u32> = read_buffer(&gpu, records, records.size()).chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let grid = planet.grid();
        for c in rec.chunks_exact(8) {
            let info = c[3];
            if info & 0xc000_0000 != 0x8000_0000 {
                continue;
            }
            let (level, face) = (c[0] >> 27, ((c[0] >> 24) & 7) as u8);
            let (ci, cj) = ((c[0] & 0xff_ffff) as i32, c[1] as i32);
            let lanes: Vec<_> = (0..64)
                .map(|cell| (ci * 8 + (cell & 7), cj * 8 + (cell >> 3)))
                .map(|(i, j)| (i, j, planet.field().extent(grid.domain_point(face, i, j, level), level)))
                .collect();
            if lanes.iter().all(|(_, _, e)| *e == (0, 0)) {
                continue;
            }
            extent_columns += 1;
            let is_generated = info & 0x1000 != 0;
            generated += usize::from(is_generated);
            // Columns with edits are generated by their brushes, not the volume.
            if !is_generated || info & 0x0800_0000 != 0 || checked >= 32 {
                continue;
            }
            checked += 1;
            let differs = lanes.iter().any(|&(i, j, (below, above))| {
                let height = planet.column_height(face, i, j, level);
                let top = terrain::top_cells(grid, height, level);
                (top - below..top + above).any(|k| terrain::generated_kind(grid, planet.field(), face, i, j, k, level, height) != u32::from(k < top))
            });
            unchanged += usize::from(!differs);
        }
    }
    eprintln!("{extent_columns} columns with a volumetric extent, {generated} generated; {unchanged} of {checked} checked change no cell");
    assert!(extent_columns > 100 && checked > 0);
    assert!(generated < extent_columns, "every extent column stored as generated volume");
    assert_eq!(unchanged, 0, "generated volume stored for columns the volume leaves as the heightfield");
}

/// Sky bound with the default LOD dither, settled and moving: frozen renders
/// with a fixed dither pattern must match the plain traversal.
#[test]
fn sky_bound_is_conservative_with_dither() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&planet, 2, 0.47, 0.53);
    let mut eye = planet.surface_point(dir, 1.7);
    let east = eye.normalize().any_orthonormal_vector();
    let size = [320, 180];
    let target = Target::new(&gpu, size);
    let mut r = renderer(&gpu, planet.clone(), size);
    r.settings_mut().lod_dither = 0.25;
    settle(&gpu, &target, &mut r, &frame(&planet, eye), east.as_vec3());
    let mut bad = 0;
    for step in 0..60u64 {
        if step >= 20 {
            eye = planet.surface_point(eye + east * 8.0, 1.7 + (step % 7) as f64 * 5.0);
        }
        let f = frame(&planet, eye);
        let up = eye.normalize();
        let forward = (east + up * (-0.3 + 0.05 * (step % 9) as f64)).normalize().as_vec3();
        target.render(&gpu, &mut r, &f, forward, 7000 + step * 3);
        r.settings_mut().freeze_residency = true;
        r.settings_mut().frame_override = Some(step as u32 * 37 % 1024);
        r.settings_mut().horizon = false;
        target.render(&gpu, &mut r, &f, forward, 7000 + step * 3 + 1);
        let reference = hits(&gpu, &r);
        r.settings_mut().horizon = true;
        target.render(&gpu, &mut r, &f, forward, 7000 + step * 3 + 2);
        let fast = hits(&gpu, &r);
        r.settings_mut().freeze_residency = false;
        r.settings_mut().frame_override = None;
        let voxel = planet.grid().voxel_size() as f32;
        let camera = target.camera(forward, up.as_vec3());
        for (index, (a, b)) in reference.iter().zip(&fast).enumerate() {
            let rise = pixel_dir(&target, &camera, index as u32 % 320, index as u32 / 320).dot(up).abs() as f32;
            let differs = !skipped_nothing(a, b, rise, voxel);
            if differs {
                bad += 1;
                if bad < 8 {
                    eprintln!("step {step} pixel {} {}: plain {a:?} accelerated {b:?}", index % 320, index / 320);
                }
            }
        }
    }
    eprintln!("{bad} differences");
    assert_eq!(bad, 0);
}

/// The sky span stays conservative while columns stream in during fast
/// climbs and descents (partial summary blocks fall back to coarser levels,
/// which the per-level fallback distances must account for): no hit of a
/// render without it is skipped.
#[test]
fn accelerations_change_nothing_while_streaming() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&planet, 2, 0.47, 0.53);
    let ground = planet.surface_point(dir, 1.7);
    let east = ground.normalize().any_orthonormal_vector();
    let size = [320, 180];
    let target = Target::new(&gpu, size);
    let mut r = renderer(&gpu, planet.clone(), size);
    r.settings_mut().lod_dither = 0.25;
    settle(&gpu, &target, &mut r, &frame(&planet, ground), east.as_vec3());
    let (mut bad, mut compared) = (0usize, 0usize);
    for step in 0..48u64 {
        // Climb to 20 km and back down while residency lags behind.
        let altitude = 1.7 * (20_000.0f64 / 1.7).powf(1.0 - ((step as f64 / 24.0) - 1.0).abs());
        let eye = ground.normalize() * (ground.length() + altitude) + east * step as f64 * 40.0;
        let up = eye.normalize();
        let forward = (east - up * 0.3).normalize().as_vec3();
        let f = frame(&planet, eye);
        target.render(&gpu, &mut r, &f, forward, 10_000 + step * 3);
        r.settings_mut().freeze_residency = true;
        r.settings_mut().frame_override = Some(step as u32 * 7 % 1024);
        r.settings_mut().horizon = false;
        target.render(&gpu, &mut r, &f, forward, 10_000 + step * 3 + 1);
        let reference = hits(&gpu, &r);
        r.settings_mut().horizon = true;
        target.render(&gpu, &mut r, &f, forward, 10_000 + step * 3 + 2);
        let hinted = hits(&gpu, &r);
        r.settings_mut().freeze_residency = false;
        r.settings_mut().frame_override = None;
        let voxel = planet.grid().voxel_size() as f32;
        let camera = target.camera(forward, up.as_vec3());
        for (index, (a, b)) in reference.iter().zip(&hinted).enumerate() {
            compared += 1;
            let rise = pixel_dir(&target, &camera, index as u32 % 320, index as u32 / 320).dot(up).abs() as f32;
            if !skipped_nothing(a, b, rise, voxel) {
                bad += 1;
                if bad < 8 {
                    eprintln!("step {step} pixel {} {}: plain {a:?} accelerated {b:?}", index % 320, index / 320);
                }
            }
        }
        if step % 8 == 0 {
            eprintln!("step {step}: altitude {altitude:.0} m pending {}", r.stats().pending_columns);
        }
    }
    eprintln!("{bad} differences in {compared} pixels");
    assert_eq!(bad, 0);
}

/// The GPU column table equals the CPU table after every frame while the
/// eye moves fast enough that each frame evicts and admits many columns
/// (backward-shift deletion rewrites slots several times per frame; the GPU
/// applies patches in parallel).
#[test]
fn gpu_column_table_matches_cpu_while_moving() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = land(&planet, 2, 0.47, 0.53);
    let ground = planet.surface_point(dir, 30.0);
    let east = ground.normalize().any_orthonormal_vector();
    let size = [320, 180];
    let target = Target::new(&gpu, size);
    let mut r = renderer(&gpu, planet.clone(), size);
    r.settings_mut().table_snapshots = true;
    settle(&gpu, &target, &mut r, &frame(&planet, ground), east.as_vec3());
    let mut worst = 0usize;
    for step in 0..40u64 {
        let eye = planet.surface_point(ground + east * step as f64 * 25.0, 30.0);
        let forward = (east - eye.normalize() * 0.4).normalize().as_vec3();
        target.render(&gpu, &mut r, &frame(&planet, eye), forward, 5_000 + step);
        let (buffer, cpu) = r.column_table();
        assert!(!cpu.is_empty(), "no table snapshot");
        let bytes = read_buffer(&gpu, buffer, (cpu.len() * 4) as u64);
        let gpu_table: &[u32] = bytemuck::cast_slice(&bytes);
        let differ = gpu_table.iter().zip(cpu).filter(|(a, b)| a != b).count();
        worst = worst.max(differ);
        if differ > 0 {
            eprintln!("step {step}: {differ} table slots differ (resident {})", r.stats().resident_columns);
        }
    }
    assert_eq!(worst, 0);
}

/// Diagnostics: cold pipeline compile times (run with
/// `HELIO_VOXEL_PIPELINE_TIMES=1 HELIO_VOXEL_SHADER_SALT=<new number>`).
#[test]
#[ignore = "measurement"]
fn pipeline_compile_times() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let started = std::time::Instant::now();
    let _renderer = renderer(&gpu, planet, [64, 64]);
    eprintln!("renderer built in {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
}

/// Diagnostics: GPU cost of the terrain program per column, for each preset,
/// on the ground (generation evaluates it for every column it streams).
#[test]
#[ignore = "measurement"]
fn terrain_program_cost() {
    let Some(gpu) = gpu() else { return };
    use helio_pass_voxel_planet::layers::TerrainLayers;
    for (name, stack) in [("earth", TerrainLayers::earth()), ("moon", TerrainLayers::moon()), ("desert", TerrainLayers::desert())] {
        let planet = Planet::new(PlanetRecipe { terrain: stack.source(7), ..Default::default() }).unwrap();
        let eye = planet.surface_point(land(&planet, 4, 0.37, 0.61), 2.0);
        let mut runs: Vec<f64> = (0..5).map(|_| helio_pass_voxel_planet::engine::time_field(&gpu.device, &gpu.queue, &planet, eye, 1024, 20)).collect();
        runs.sort_by(f64::total_cmp);
        eprintln!("TERRAIN_COST {name}: {:.2} ns/column (median of 5; min {:.2})", runs[2], runs[0]);
    }
}
