//! Regression for interior radial hits manufactured by a level switch.
mod common;
use common::*;
use glam::{DVec3, Vec3};
use helio_pass_voxel_planet::{Planet, PlanetRecipe};
use std::sync::Arc;

/// Compare the bounded shortcut with full canonical climate queries through
/// the actual GPU shaders, including odd screen edges and alpine transitions.
#[test]
fn fine_grass_preserves_authored_srgb_after_linear_upload() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe {
        shape: helio_pass_voxel_planet::grid::Shape::Plane,
        plane_size_m: 1024.0,
        terrain: helio_pass_voxel_planet::TerrainSource {
            generator: helio_pass_voxel_planet::landform::FLAT_ID.into(), version: helio_pass_voxel_planet::landform::FLAT_VERSION, ..Default::default()
        },
        ..Default::default()
    }).unwrap());
    let mut settings = helio_pass_voxel_planet::engine::Settings::default();
    settings.appearance.grass = [[0.25, 0.5, 0.75, 0.0]; 3];
    settings.appearance.detail = [0.0; 4];
    let size = [96, 64];
    let target = Target::new(&gpu, size);
    let mut renderer = helio_pass_voxel_planet::engine::PlanetRenderer::new(
        &gpu.device, &gpu.queue, planet.clone(), settings, size);
    let f = frame(&planet, DVec3::Y * 1.7);
    for n in 0..2000 {
        target.render(&gpu, &mut renderer, &f, -Vec3::Y, n);
        if renderer.settled() { break; }
    }
    assert!(renderer.settled());
    target.render(&gpu, &mut renderer, &f, -Vec3::Y, 2001);
    let surfaces = read_buffer(&gpu, renderer.surface_buffer(), 96 * 64 * 16);
    let mut checked = 0;
    for (h, s) in hits(&gpu, &renderer).iter().zip(surfaces.chunks_exact(16)) {
        if h.status != 1 || h.level != 0 || h.normal != 4 { continue; }
        let flags = u32::from_le_bytes(s[12..16].try_into().unwrap());
        assert_eq!((flags >> 8) & 255, helio_pass_voxel_planet::terrain::material::GRASS);
        for (actual, expected) in s[4..7].iter().zip([64i16, 128, 191]) {
            assert!((i16::from(*actual) - expected).abs() <= 1, "double/missing gamma conversion: {:?}", &s[4..7]);
        }
        checked += 1;
    }
    assert!(checked > 5000, "only {checked} fine grass hits");
}

#[test]
fn alpine_snow_does_not_become_grass_when_level_height_rounds_to_zero() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let dir = peak(&planet);
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
    // Top faces are checked against the heightfield top: Landform without
    // caves and overhangs (cave_view covers cave floors and overhang tops).
    let terrain = helio_pass_voxel_planet::TerrainSource {
        settings: helio_pass_voxel_planet::landform::HEIGHTFIELD_SETTINGS.into(),
        ..Default::default()
    };
    let planet = Arc::new(Planet::new(PlanetRecipe { terrain, ..Default::default() }).unwrap());
    let target = Target::new(&gpu, [384, 216]);
    let mut renderer = renderer(&gpu, planet.clone(), [384, 216]);
    let grid = *planet.grid();
    assert!(renderer.settings_mut().coarse_relief, "this regression exercises authored fractional tops");
    // Tops are checked against the canonical field below; the ridge-envelope
    // display height (coarse mountains keep their mean mass) is not it.
    renderer.settings_mut().ridge_display = false;
    for height in [1_000., 4_000., 16_000., 64_000.] {
        let frame = frame(&planet, planet.surface_point(DVec3::Y, height));
        let forward = Vec3::new(1., -0.6, 0.).normalize();
        let camera = target.camera(forward, Vec3::Y);
        for n in 0..2000 {
            target.render(&gpu, &mut renderer, &frame, forward, n);
            if renderer.settled() { break; }
        }
        assert!(renderer.settled());
        target.render(&gpu, &mut renderer, &frame, forward, 2001);
        let mut tops = 0;
        for (pixel, hit) in hits(&gpu, &renderer).into_iter().enumerate() {
            assert!(hit.status < 2, "unresolved ray at {height} m: {hit:?}");
            if hit.status == 1 && hit.normal == 4 {
                tops += 1;
                // Query the CPU generator independently of GPU metadata.
                // Fractional coarse tops preserve the authored base layer
                // inside a conservatively ceil-rounded coarse cell, whereas
                // column_top is the legacy whole-cell floor-rounded surface.
                let field_height = planet.field().height(
                    grid.domain_point(hit.face, hit.i, hit.j, hit.level),
                    hit.level + grid.level_offset(),
                );
                let base_top = i64::from(field_height).div_euclid(i64::from(grid.layer_mm()));
                let scale = 1i64 << hit.level;
                let enclosing_top = -(-base_top).div_euclid(scale);

                // A correct cell ID alone cannot certify a real surface entry.
                // Reconstruct the world-space ray in f64 and require its hit
                // radius to match the authored boundary, allowing only f32
                // ray roundoff and the stored 16-bit fraction's quantization.
                let direction = pixel_dir(&target, &camera,
                    pixel as u32 % target.size[0], pixel as u32 / target.size[0]);
                let actual_height = grid.height(frame.eye + direction * f64::from(hit.t));
                let authored_height = base_top as f64 * grid.voxel_size();
                let fraction_quantum = if hit.level <= 16 { 0.0 }
                    else { grid.voxel_size() * scale as f64 / 65536.0 };
                let ray_roundoff = grid.voxel_size() * 0.25 + f64::from(hit.t) * 2.0e-6;
                assert!((actual_height - authored_height).abs() <= fraction_quantum + ray_roundoff,
                    "synthetic interior radial entry at {height} m: {hit:?}, hit height{actual_height}, authored{authored_height}");
                assert_eq!(i64::from(hit.k), enclosing_top - 1,
                    "synthetic interior top-face cell at {height} m: {hit:?}, authored base top{base_top}");
            }
        }
        assert!(tops > 10_000);
        eprintln!("{height} m: {tops} radial surface hits, no interior entries");
    }
}
