use super::*;

fn view(eye: DVec3) -> View {
    let forward = DVec3::new(0.0, -0.15, -1.0).normalize();
    let right = forward.cross(DVec3::Y).normalize();
    View {
        eye,
        forward,
        right,
        up: right.cross(forward),
        tan: 0.41421356,
        aspect: 16.0 / 9.0,
        height: 720.0,
    }
}

#[test]
fn cached_classification_matches_fresh_after_append_undo_and_replacement() {
    let mut world = World::default();
    let cell = crate::world::cell_of(world.ground_spawn(0.0, 0.0, 0.0));
    let mut cache = SelectionCache::default();
    let keys: Vec<_> = (0..16)
        .flat_map(|level| {
            let side = 32 << level;
            let low = cell.map(|v| v.div_euclid(side) * side);
            [-1, 0, 1].into_iter().map(move |offset| Key {
                low: [low[0] + offset * side, low[1], low[2]],
                level,
            })
        })
        .collect();
    let original = world.clone();
    for step in 0..4 {
        match step {
            1 => world
                .apply_edit(Edit {
                    cell,
                    radius: 8.0,
                    material: 0,
                })
                .unwrap(),
            2 => world = original.clone(),
            3 => world
                .apply_edit(Edit {
                    cell,
                    radius: 32.0,
                    material: 3,
                })
                .unwrap(),
            _ => {}
        }
        cache.reconcile(&world);
        for &key in &keys {
            assert_eq!(
                cache.classify(&world, key),
                classify(&world, key),
                "step={step} key={key:?}"
            );
        }
        cache.reconcile(&world);
        for &key in &keys {
            cache.classify(&world, key);
        }
        assert_eq!(cache.classified, 0);
        assert_eq!(cache.reused, keys.len());
    }
}

#[test]
fn zoom_and_roll_require_new_selection() {
    let reference = view(DVec3::ZERO);
    let mut zoomed = reference;
    zoomed.tan *= 0.5;
    assert!(reference.changed(zoomed));
    let mut rolled = reference;
    rolled.up = reference.right;
    rolled.right = -reference.up;
    assert!(reference.changed(rolled));
    assert!(!reference.changed(reference));
}

#[test]
#[ignore = "CPU selection benchmark; reports cold and cached identical cuts"]
fn selection_flight_benchmark() {
    let world = Arc::new(World::default());
    let ground = world.ground_spawn(0.0, 0.0, 3.0);
    let mut cache = SelectionCache::default();
    for (index, offset) in [
        DVec3::ZERO,
        DVec3::X * 2.0,
        DVec3::X * 4.0,
        DVec3::Y * 200.0,
        DVec3::Y * 1_000.0,
        DVec3::Y * 300_000.0,
        DVec3::ZERO,
    ]
    .into_iter()
    .enumerate()
    {
        let camera = view(ground + offset);
        let start = std::time::Instant::now();
        let cold = build(
            world.clone(),
            camera,
            31_744,
            &mut SelectionCache::default(),
        );
        let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = std::time::Instant::now();
        let warm = build(world.clone(), camera, 31_744, &mut cache);
        let cached_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(cold.leaves, warm.leaves);
        assert_eq!(cold.nodes.len(), warm.nodes.len());
        for (a, b) in cold.nodes.iter().zip(&warm.nodes) {
            assert_eq!((a.low, a.level, a.child), (b.low, b.level, b.child));
        }
        eprintln!("SELECTION_FLIGHT step={index} cold_ms={cold_ms:.3} cached_ms={cached_ms:.3} nodes={} bricks={} classified={} reused={}",
            warm.nodes.len(), warm.leaves.len(), cache.classified, cache.reused);
    }
}
