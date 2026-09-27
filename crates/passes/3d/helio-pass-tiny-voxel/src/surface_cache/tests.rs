use super::*;
use crate::world::Edit;

fn assert_samples(brick: &Brick, sample: impl Fn([i32; 3]) -> u32) {
    for z in 0..32 {
        for y in 0..32 {
            for x in 0..32 {
                assert_eq!(
                    brick.material([x, y, z]),
                    sample([x as i32, y as i32, z as i32])
                );
            }
        }
    }
}

#[test]
fn compression_preserves_materials_and_halo_surface_bounds() {
    let air = Brick::from_samples(|_| 0);
    assert_eq!(air.material_bytes(), 4);
    assert_eq!(air.summary, Summary::default());
    let solid = Brick::from_samples(|_| 3);
    assert_eq!(solid.material_bytes(), 4);
    assert_eq!(solid.summary.occupied, 32 * 32 * 32);
    assert_eq!(solid.summary.face_count(), 0);
    let plane = |q: [i32; 3]| u32::from(q[1] < 17) * 2;
    let slab = Brick::from_samples(plane);
    assert_samples(&slab, plane);
    assert_eq!(slab.summary.faces[2][2], 1024);
    assert_eq!(slab.summary.face_count(), 1024);
    assert_eq!(slab.summary.bounds, Some([[0, 17, 0], [32, 17, 32]]));
    assert_eq!(slab.mixed_patterns(), 1);
    assert_eq!(slab.material_bytes(), (1 + 512 + 4) * 4);
    let checker = |q: [i32; 3]| ((q[0] + q[1] + q[2]).rem_euclid(4)) as u32;
    let mixed = Brick::from_samples(checker);
    assert_samples(&mixed, checker);
    assert_eq!(mixed.mixed_patterns(), 1);
    let random = |q: [i32; 3]| {
        let mut h = (q[0] as u32).wrapping_mul(73_856_093)
            ^ (q[1] as u32).wrapping_mul(19_349_663)
            ^ (q[2] as u32).wrapping_mul(83_492_791);
        h ^= h >> 13;
        h = h.wrapping_mul(0x85ebca6b);
        (h >> 16) & 3
    };
    let entropy = Brick::from_samples(random);
    assert_samples(&entropy, random);
    assert!(entropy.material_bytes() <= (1 + 512 + 512 * 4) * 4);
}

#[test]
fn append_undo_and_halo_edits_replace_only_affected_snapshots() {
    let mut world = World::default();
    // Empty space well above the planet; a solid edit fills one whole tile.
    let cell = [-32, 70_000_000, -32];
    let key = Key::containing(cell, 1);
    let low = key.low(1);
    world
        .apply_edit(Edit {
            cell: low.map(|v| v + 16),
            radius: 4.0,
            material: 2,
        })
        .unwrap();
    let old = Arc::new(world);
    let mut cache = Cache::new(64 * 1024, 4);
    cache.set_world(old.clone());
    let a = cache.get(key).unwrap();
    let neighbour = Key([key.0[0] - 8, key.0[1], key.0[2]]);
    let remote = cache.get(neighbour).unwrap();
    assert_eq!(a.material([31, 16, 16]), 2);
    assert_eq!(a.summary.face_count(), 0);
    let mut edited = (*old).clone();
    edited
        .apply_edit(Edit {
            cell: [low[0] + 32, low[1] + 16, low[2] + 16],
            radius: 0.051,
            material: 0,
        })
        .unwrap();
    cache.set_world(Arc::new(edited));
    let b = cache.get(key).unwrap();
    assert!(!Arc::ptr_eq(&a, &b));
    assert_eq!(b.material([31, 16, 16]), 2);
    assert_eq!(b.summary.faces[0][2], 1);
    assert_eq!(a.summary.face_count(), 0);
    assert!(Arc::ptr_eq(&remote, &cache.get(neighbour).unwrap()));
    cache.set_world(old.clone());
    assert_eq!(cache.get(key).unwrap().summary, a.summary);
    assert_eq!(cache.stats().invalidated, 2);
    // A replacement grid invalidates every resident address, even with the
    // same edit journal. The old Arc remains a coherent, readable snapshot.
    let mut grid = (*old).clone();
    grid.set_voxel_size(0.3).unwrap();
    cache.set_world(Arc::new(grid));
    assert_eq!(cache.stats().resident, 0);
    assert_eq!(a.material([31, 16, 16]), 2);
}

#[test]
fn budget_and_authored_grid_do_not_turn_cache_misses_into_air() {
    let mut no_space = Cache::new(0, 0);
    no_space.set_world(Arc::new(World::default()));
    assert!(no_space.get(Key([0, 0, 0])).is_none());
    for step in [1, 3, 10] {
        let mut world = World::default();
        world.set_voxel_size(f64::from(step) * 0.1).unwrap();
        let mut cache = Cache::new(264, 2);
        cache.set_world(Arc::new(world));
        let cell = [-1, 70_000_013, -33];
        let key = Key::containing(cell, step);
        let low = key.low(step);
        for a in 0..3 {
            assert!(cell[a] >= low[a] && cell[a] < low[a] + 32 * step as i32);
        }
        let held = cache.get(key).unwrap();
        assert_eq!(held.material([31, 0, 31]), 0);
        cache.get(Key([key.0[0] + 1, key.0[1], key.0[2]])).unwrap();
        cache.get(Key([key.0[0] + 2, key.0[1], key.0[2]])).unwrap();
        assert_eq!(cache.stats().evicted, 1);
        assert_eq!(cache.stats().resident, 2);
        assert_eq!(cache.stats().logical_bytes, 264);
        assert_eq!(held.material([31, 0, 31]), 0);
    }
}

#[test]
fn planetary_materials_and_surface_bounds_match_canonical_grids() {
    for step in [1, 3, 10] {
        let mut world = World::default();
        world.set_voxel_size(f64::from(step) * 0.1).unwrap();
        let ground = world.ground_spawn(-0.3, -0.3, 0.0);
        let cell = crate::world::cell_of(ground);
        world
            .apply_edit(Edit {
                cell,
                radius: 0.61,
                material: 3,
            })
            .unwrap();
        world
            .apply_edit(Edit {
                cell: [cell[0] + 1, cell[1], cell[2]],
                radius: 0.31,
                material: 0,
            })
            .unwrap();
        let key = Key::containing(cell, step);
        let low = key.low(step);
        let brick = Brick::from_world(&world, key);
        assert_samples(&brick, |q| {
            world.material(std::array::from_fn(|a| low[a] + q[a] * step as i32))
        });
        assert!(brick.summary.face_count() > 0);
        let [lo, hi] = brick.summary.bounds.unwrap();
        assert!((0..3).all(|a| lo[a] <= hi[a] && hi[a] <= 32));
    }
}

mod gpu;

#[test]
#[ignore = "diagnostic source construction and cache reuse measurements"]
fn measure_planetary_cache_construction_and_local_edits() {
    use std::time::Instant;
    for step in [1, 3, 10] {
        let mut world = World::default();
        world.set_voxel_size(f64::from(step) * 0.1).unwrap();
        let ground = world.ground_spawn(-0.3, -0.3, 0.0);
        let cell = crate::world::cell_of(ground);
        let anchor = Key::containing(cell, step);
        // The ground query has a source cache of its own. Reset it so the cold
        // measurement includes source generation, not just repacking warm data.
        world.chunks = Default::default();
        let world = Arc::new(world);
        let mut cache = Cache::new(1024 * 1024, 64);
        cache.set_world(world.clone());
        let keys: Vec<_> = (-1..=1)
            .flat_map(|z| {
                (-1..=1).map(move |x| Key([anchor.0[0] + x, anchor.0[1], anchor.0[2] + z]))
            })
            .collect();
        let start = Instant::now();
        let bricks: Vec<_> = keys.iter().map(|key| cache.get(*key).unwrap()).collect();
        let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
        let materials: usize = bricks.iter().map(|b| b.material_bytes()).sum();
        let faces: u32 = bricks.iter().map(|b| b.summary.face_count()).sum();
        let start = Instant::now();
        for _ in 0..1000 {
            for key in &keys {
                std::hint::black_box(cache.get(*key).unwrap());
            }
        }
        let reuse_ns = start.elapsed().as_secs_f64() * 1e9 / (1000 * keys.len()) as f64;
        let mut edited = (*world).clone();
        edited
            .apply_edit(Edit {
                cell,
                radius: 0.61,
                material: 0,
            })
            .unwrap();
        let start = Instant::now();
        cache.set_world(Arc::new(edited));
        for key in &keys {
            std::hint::black_box(cache.get(*key).unwrap());
        }
        let edit_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("SURFACE_CACHE_CPU step={step} tiles={} cold_ms={cold_ms:.3} reuse_ns={reuse_ns:.1} edit_ms={edit_ms:.3} invalidated={} material_bytes={materials} logical_bytes={} faces={faces}",
            keys.len(),cache.stats().invalidated,cache.stats().logical_bytes);
    }
}
