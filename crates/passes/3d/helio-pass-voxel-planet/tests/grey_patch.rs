//! Regression for interior radial hits manufactured by a level switch.
mod common;
use common::*;
use glam::{DVec3, Vec3};
use helio_pass_voxel_planet::{Planet, PlanetRecipe};
use std::sync::Arc;

#[test]
fn alpine_snow_does_not_become_grass_when_level_height_rounds_to_zero() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let grid = *planet.grid();
    let dir = grid.direction(1, 69_257_724.5, 67_647_343.5);
    let target = Target::new(&gpu, [128, 72]);
    let mut renderer = renderer(&gpu, planet.clone(), [128, 72]);
    // At 1,000 km, radial level cells exceed the mountain's elevation.
    for height in [64_000.0, 300_000.0, 1_000_000.0] {
        let f = frame(&planet, planet.surface_point(dir, height));
        let forward = -dir.as_vec3();
        for n in 0..2000 {
            target.render(&gpu, &mut renderer, &f, forward, n);
            if renderer.settled() { break; }
        }
        assert!(renderer.settled());
        target.render(&gpu, &mut renderer, &f, forward, 2001);
        let surfaces = read_buffer(&gpu, renderer.surface_buffer(), 128 * 72 * 16);
        let snow = surfaces.chunks_exact(16).filter(|p| {
            let flags = u32::from_le_bytes(p[12..16].try_into().unwrap());
            (flags & 3) == 1 && ((flags >> 8) & 255) == helio_pass_voxel_planet::terrain::material::SNOW
        }).count();
        eprintln!("alpine {height} m: {snow} snow pixels");
        assert!(snow > 0, "alpine feature disappeared at {height} m");
    }
}

#[test]
fn settled_level_transitions_enter_the_surface_not_subsoil() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let target = Target::new(&gpu, [384, 216]);
    let mut renderer = renderer(&gpu, planet.clone(), [384, 216]);
    for height in [1_000., 4_000., 16_000., 64_000.] {
        let frame = frame(&planet, planet.surface_point(DVec3::Y, height));
        let forward = Vec3::new(1., -0.6, 0.).normalize();
        for n in 0..2000 {
            target.render(&gpu, &mut renderer, &frame, forward, n);
            if renderer.settled() { break; }
        }
        assert!(renderer.settled());
        target.render(&gpu, &mut renderer, &frame, forward, 2001);
        let mut tops = 0;
        for hit in hits(&gpu, &renderer) {
            assert!(hit.status < 2, "unresolved ray at {height} m: {hit:?}");
            if hit.status == 1 && hit.normal == 4 {
                tops += 1;
                let top = planet.column_top(hit.face, hit.i, hit.j, hit.level);
                assert_eq!(hit.k, top - 1, "synthetic interior top-face hit at {height} m: {hit:?}");
            }
        }
        assert!(tops > 10_000);
        eprintln!("{height} m: {tops} radial surface hits, no interior entries");
    }
}
