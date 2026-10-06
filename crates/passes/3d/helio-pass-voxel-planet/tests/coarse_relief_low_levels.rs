//! Authored radial relief at L1–5; serial GPU fixtures.
mod common;
use common::*;
use glam::{DVec3, Vec3};
use helio_pass_voxel_planet::engine::{PlanetRenderer, Settings};
use helio_pass_voxel_planet::grid::Shape;
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Planet, PlanetRecipe};
use std::sync::Arc;

const RELIEF: u32 = 0x10000000;
const INLINE_RELIEF: u32 = 0x04000000;
fn flat(shape: Shape, height: f64) -> Planet {
    Planet::new(PlanetRecipe {
        shape,
        radius_m: 10000.0,
        plane_size_m: 40000.0,
        terrain: helio_pass_voxel_planet::layers::TerrainLayers::flat_at(height).source(7),
        ..Default::default()
    })
    .unwrap()
}
fn authored_top(planet: &Planet, height: f64) -> f64 {
    // These fixture heights are exact millimetre generator inputs. Integer
    // Euclidean division exercises the negative-height floor explicitly.
    let height_mm = (height * 1000.0).round() as i64;
    height_mm.div_euclid(i64::from(planet.grid().layer_mm())) as f64 * planet.grid().voxel_size()
}
fn lod_for(planet: &Planet, size: [u32; 2], fov: f32, t: f64, level: u32) -> f32 {
    let lod0 = t / (1.5 * f64::from(1u32 << (level - 1)));
    (lod0 * 2.0 * f64::from((fov * 0.5).tan()) / (planet.grid().voxel_size() * f64::from(size[1])))
        as f32
}
fn render(
    gpu: &Gpu,
    planet: Arc<Planet>,
    eye: DVec3,
    forward: Vec3,
    size: [u32; 2],
    fov: f32,
    lod_pixels: f32,
    relief: bool,
) -> (Target, PlanetRenderer, Vec<Hit>) {
    let mut target = Target::new(gpu, size);
    target.fov_y = fov;
    let mut renderer = PlanetRenderer::new(
        &gpu.device,
        &gpu.queue,
        planet.clone(),
        Settings {
            coarse_relief: relief,
            lod_pixels,
            lod_dither: 0.0,
            horizon: false,
            frame_override: Some(37),
            ..Default::default()
        },
        size,
    );
    let frame = frame(&planet, eye);
    let mut settled = false;
    for n in 0..2000 {
        target.render(gpu, &mut renderer, &frame, forward, n);
        if renderer.settled() {
            settled = true;
            break;
        }
    }
    assert!(settled, "low-level fixture did not settle");
    target.render(gpu, &mut renderer, &frame, forward, 2001);
    assert_eq!(renderer.stats().failed_jobs, 0, "publication failed");
    assert!(
        renderer.stats().free_pages >= 0,
        "pool accounting underflow"
    );
    let hits = hits(gpu, &renderer);
    assert!(
        hits.iter().all(|h| h.status < 2),
        "unresolved low-level rays: {hits:?}"
    );
    (target, renderer, hits)
}
fn camera(
    target: &Target,
    planet: &Planet,
    eye: DVec3,
    forward: Vec3,
) -> helio_core::GpuCameraUniforms {
    let up = planet.grid().up(eye).as_vec3();
    target.camera(
        forward,
        if forward.normalize().dot(up).abs() > 0.99 {
            up.any_orthonormal_vector()
        } else {
            up
        },
    )
}
fn hit_columns(gpu: &Gpu, renderer: &PlanetRenderer) -> Vec<[u32; 8]> {
    let count = (renderer.screen_size()[0] * renderer.screen_size()[1]) as usize;
    let hits = read_buffer(gpu, renderer.hit_buffer(), count as u64 * 32);
    let ids: Vec<usize> = hits
        .chunks_exact(32)
        .map(|c| u32::from_le_bytes(c[20..24].try_into().unwrap()) as usize)
        .collect();
    let end = ids.iter().copied().max().unwrap() + 1;
    let records = read_buffer(gpu, renderer.residency_buffers()[0], end as u64 * 32);
    ids.into_iter()
        .map(|i| {
            std::array::from_fn(|k| {
                u32::from_le_bytes(
                    records[i * 32 + k * 4..i * 32 + k * 4 + 4]
                        .try_into()
                        .unwrap(),
                )
            })
        })
        .collect()
}

#[test]
fn l1_to_l5_plane_and_sphere_preserve_authored_radial_tops() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let size = [9, 5];
    let fov = std::f32::consts::FRAC_PI_4;
    for level in 1..=5 {
        for shape in [Shape::Plane, Shape::Sphere] {
            for height in [53.7, -53.7, 71.9] {
                let planet = Arc::new(flat(shape, height));
                let top = authored_top(&planet, height);
                // Negative planes have a datum-bounded outer slab (empty edit_top=0).
                // Keep the slab entry beyond the requested level's lower distance,
                // otherwise the single vertical column legitimately stays at L0.
                let altitude = if shape == Shape::Plane && height < 0.0 {
                    200.0
                } else {
                    20.0
                } * f64::from(1u32 << (level - 1));
                let eye = DVec3::Y * (planet.grid().radius() + top + altitude);
                let lod = lod_for(&planet, size, fov, altitude, level);
                let (target, r, h) =
                    render(&gpu, planet.clone(), eye, -Vec3::Y, size, fov, lod, true);
                let columns = hit_columns(&gpu, &r);
                let cam = camera(&target, &planet, eye, -Vec3::Y);
                let center = (size[0] * size[1] / 2) as usize;
                assert_eq!(
                    h[center].level, level,
                    "fixture did not select intended level"
                );
                for (index, v) in h.iter().enumerate() {
                    assert_eq!(v.status, 1);
                    assert_eq!(
                        v.level, level,
                        "all fixture rays must exercise requested level"
                    );
                    assert_ne!(
                        columns[index][3] & RELIEF,
                        0,
                        "generated column omitted fractions"
                    );
                    let d = pixel_dir(
                        &target,
                        &cam,
                        index as u32 % size[0],
                        index as u32 / size[0],
                    );
                    let position = eye + d * f64::from(v.t);
                    let actual = if shape == Shape::Plane {
                        position.y
                    } else {
                        position.length() - planet.grid().radius()
                    };
                    assert!((actual-top).abs()<0.02,"level={level} shape={shape:?} height={height} pixel={index} actual={actual} top={top} hit={v:?}");
                }
                eprintln!("low radial L{level} {shape:?} height={height}: {} exact tops; free_pages={} failures={}",h.len(),r.stats().free_pages,r.stats().failed_jobs);
            }
        }
    }
}

#[test]
fn l1_to_l5_grazing_spheres_keep_analytic_hits_and_misses() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let size = [9, 5];
    let fov = std::f32::consts::FRAC_PI_4;
    let planet = Arc::new(flat(Shape::Sphere, 53.7));
    let radius = planet.grid().radius() + authored_top(&planet, 53.7);
    let eye = DVec3::Y * (radius + 5.0);
    let critical = (1.0 - (radius / eye.y).powi(2)).sqrt();
    let forward = Vec3::new(
        (1.0 - (critical * 1.1).powi(2)).sqrt() as f32,
        -(critical * 1.1) as f32,
        0.0,
    )
    .normalize();
    let center_d = forward.as_dvec3().normalize();
    let b = eye.dot(center_d);
    let center_t = -b - (b * b - (eye.length_squared() - radius * radius)).sqrt();
    for level in 1..=5 {
        let lod = lod_for(&planet, size, fov, center_t, level);
        let (target, r, h) = render(&gpu, planet.clone(), eye, forward, size, fov, lod, true);
        assert_eq!(
            h[(size[0] * size[1] / 2) as usize].level,
            level,
            "grazing center did not select intended level"
        );
        let cam = camera(&target, &planet, eye, forward);
        let (mut hit_count, mut miss_count) = (0, 0);
        for (index, v) in h.iter().enumerate() {
            let d = pixel_dir(
                &target,
                &cam,
                index as u32 % size[0],
                index as u32 / size[0],
            );
            let b = eye.dot(d);
            let disc = b * b - (eye.length_squared() - radius * radius);
            let expected_hit = b < 0.0 && disc >= 0.0;
            if expected_hit {
                assert_eq!(
                    v.status, 1,
                    "L{level} lost analytic silhouette pixel={index} disc={disc} {v:?}"
                );
                let expected = -b - disc.sqrt();
                let actual_radius = (eye + d * f64::from(v.t)).length();
                assert!(
                    (actual_radius - radius).abs() < 0.02,
                    "L{level} moved radial shell pixel={index} {v:?}"
                );
                let tolerance = 0.02 / (disc.sqrt() / radius).max(0.001) + expected * 2e-6;
                assert!(
                    (f64::from(v.t) - expected).abs() < tolerance,
                    "L{level} wrong analytic depth pixel={index} {v:?}"
                );
                hit_count += 1;
            } else {
                assert_eq!(
                    v.status, 0,
                    "L{level} false analytic silhouette pixel={index} disc={disc} {v:?}"
                );
                miss_count += 1;
            }
        }
        assert!(
            hit_count > 5 && miss_count > 5,
            "grazing fixture must contain both sides of silhouette"
        );
        eprintln!(
            "grazing L{level}: {hit_count}hit {miss_count}miss; free_pages={} failures={}",
            r.stats().free_pages,
            r.stats().failed_jobs
        );
    }
}

#[test]
fn l1_to_l5_adjacent_paint_columns_preserve_authored_geometry() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let size = [9, 9];
    let fov = 0.08f32;
    for level in 1..=5 {
        let original = Arc::new(flat(Shape::Plane, 53.7));
        let q = original.grid().voxel_size() * f64::from(1u32 << level);
        let mut p = original.as_ref().clone();
        p.apply(Brush {
            center: [-4.0 * q, 53.7, -4.0 * q],
            radius: 2.0 * q,
            shape: BrushShape::Cube,
            op: BrushOp::Paint,
            material: 13,
        })
        .unwrap();
        let planet = Arc::new(p);
        let altitude = 20.0 * f64::from(1u32 << (level - 1));
        let eye = DVec3::new(0.0, 53.7 + altitude, -4.0 * q);
        let lod = lod_for(&planet, size, fov, altitude, level);
        let (_, old, old_hits) = render(&gpu, original, eye, -Vec3::Y, size, fov, lod, true);
        let (target, new, new_hits) =
            render(&gpu, planet.clone(), eye, -Vec3::Y, size, fov, lod, true);
        let columns = hit_columns(&gpu, &new);
        let cam = camera(&target, &planet, eye, -Vec3::Y);
        let old_surface = read_buffer(
            &gpu,
            old.surface_buffer(),
            u64::from(size[0] * size[1]) * 16,
        );
        let new_surface = read_buffer(
            &gpu,
            new.surface_buffer(),
            u64::from(size[0] * size[1]) * 16,
        );
        let (mut edited, mut unchanged) = (Vec::new(), Vec::new());
        let (mut painted, mut interior_exact) = (0, 0);
        for (index, (a, b)) in old_hits.iter().zip(&new_hits).enumerate() {
            assert_eq!(a.status, 1);
            assert_eq!(b.status, 1);
            assert_eq!(a.level, level);
            assert_eq!(b.level, level);
            assert_eq!(
                (
                    a.status,
                    a.face,
                    a.i,
                    a.j,
                    a.k,
                    a.level,
                    a.normal,
                    a.t.to_bits()
                ),
                (
                    b.status,
                    b.face,
                    b.i,
                    b.j,
                    b.k,
                    b.level,
                    b.normal,
                    b.t.to_bits()
                ),
                "Paint reshaped geometry L{level} pixel{index}"
            );
            let sa = &old_surface[index * 16..index * 16 + 16];
            let sb = &new_surface[index * 16..index * 16 + 16];
            assert_eq!(&sa[..4], &sb[..4], "Paint changed depth");
            assert_eq!(&sa[8..12], &sb[8..12], "Paint changed normal");
            assert_eq!(sa[7], sb[7], "Paint changed AO");
            let fa = u32::from_le_bytes(sa[12..16].try_into().unwrap());
            let fb = u32::from_le_bytes(sb[12..16].try_into().unwrap());
            assert_eq!(
                fa & !0xff00,
                fb & !0xff00,
                "Paint changed filtering/light flags"
            );
            assert_ne!(
                columns[index][3] & RELIEF,
                0,
                "Paint/neighbor omitted authored relief"
            );
            assert_eq!(columns[index][3] & 0x08000000, 0, "Paint marked topology");
            let d = pixel_dir(
                &target,
                &cam,
                index as u32 % size[0],
                index as u32 / size[0],
            );
            assert!(
                (eye.y + d.y * f64::from(b.t) - authored_top(&planet, 53.7)).abs() < 0.02,
                "Paint changed authored radial top: {b:?}"
            );
            if (fb >> 8) & 255 == 13 {
                assert_ne!(&sa[4..7], &sb[4..7], "Paint failed to change pigment");
                painted += 1;
            } else {
                assert_eq!(sa, sb, "Paint changed untouched surface");
            }
            if columns[index][7] != 0 {
                edited.push(index);
                if b.i & 7 > 0 && b.i & 7 < 7 && b.j & 7 > 0 && b.j & 7 < 7 {
                    interior_exact += 1;
                }
            } else {
                unchanged.push(index);
            }
        }
        assert!(
            !edited.is_empty() && !unchanged.is_empty(),
            "fixture must cover painted and unpainted columns"
        );
        assert!(painted > 0, "no actual paint samples");
        assert!(
            interior_exact > 0,
            "fixture must expose a painted-column interior"
        );
        assert!(
            edited.iter().any(|&a| unchanged.iter().any(|&b| {
                let ha = &new_hits[a];
                let hb = &new_hits[b];
                ((ha.i >> 3) - (hb.i >> 3)).abs() + ((ha.j >> 3) - (hb.j >> 3)).abs() == 1
            })),
            "fixture must retain directly adjacent column ownership"
        );
        eprintln!("Paint boundary L{level}: {} edited-list and {} unedited columns, {painted} pigment changes; all geometry/normals/AO authored",edited.len(),unchanged.len());
    }
}

#[test]
fn l1_to_l5_zero_fraction_matches_legacy_whole_cell_hits() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let size = [9, 5];
    let fov = std::f32::consts::FRAC_PI_4;
    for level in 1..=5 {
        for shape in [Shape::Plane, Shape::Sphere] {
            let q = 0.1 * f64::from(1u32 << level);
            let height = 8.0 * q;
            let planet = Arc::new(flat(shape, height));
            let top = authored_top(&planet, height);
            assert!(
                (top - height).abs() < 1e-8,
                "fixture top must be exactly coarse aligned"
            );
            let altitude = 20.0 * f64::from(1u32 << (level - 1));
            let eye = DVec3::Y * (planet.grid().radius() + top + altitude);
            let lod = lod_for(&planet, size, fov, altitude, level);
            let (_, _old, old_hits) =
                render(&gpu, planet.clone(), eye, -Vec3::Y, size, fov, lod, false);
            let (_, new, new_hits) =
                render(&gpu, planet.clone(), eye, -Vec3::Y, size, fov, lod, true);
            let columns = hit_columns(&gpu, &new);
            let pool = read_buffer(
                &gpu,
                new.residency_buffers()[1],
                columns
                    .iter()
                    .map(|c| u64::from(c[4] + 3) * 64)
                    .max()
                    .unwrap(),
            );
            for (index, (a, b)) in old_hits.iter().zip(&new_hits).enumerate() {
                assert_eq!(a.status, 1);
                assert_eq!(b.status, 1);
                assert_eq!(a.level, level);
                assert_eq!(b.level, level);
                assert_ne!(columns[index][3] & RELIEF, 0);
                let cell = (b.j & 7) as usize * 8 + (b.i & 7) as usize;
                if columns[index][3] & INLINE_RELIEF != 0 {
                    // Inspect the base-layer remainder independently of the
                    // shader decoder. Inline relief must also use a smaller
                    // run than the old four-unit Q16 representation.
                    let at = columns[index][4] as usize * 64 + cell;
                    assert_eq!(u32::from(pool[at]) % (1 << level), 0);
                    assert!((columns[index][3] >> 18) & 15 < 2);
                } else {
                    let at = columns[index][4] as usize * 64 + 64 + cell * 2;
                    assert_eq!(
                        u16::from_le_bytes(pool[at..at + 2].try_into().unwrap()),
                        0,
                        "fixture must use zero-fraction fast path"
                    );
                }
                assert_eq!(
                    (
                        a.status,
                        a.face,
                        a.i,
                        a.j,
                        a.k,
                        a.level,
                        a.normal,
                        a.t.to_bits()
                    ),
                    (
                        b.status,
                        b.face,
                        b.i,
                        b.j,
                        b.k,
                        b.level,
                        b.normal,
                        b.t.to_bits()
                    ),
                    "whole-cell occupancy changed level={level} shape={shape:?} pixel={index}"
                );
            }
            eprintln!(
                "zero fraction L{level} {shape:?}: {} byte-identical hit records",
                new_hits.len()
            );
        }
    }
}
#[test]
fn authored_pixel_filter_hides_coarse_ao_grid_but_keeps_near_edges() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let p = Arc::new(flat(Shape::Plane, 53.7));
    let size = [9, 9];
    let fov = 0.08;
    let altitude = 200.0;
    let eye = DVec3::new(0.0, 53.7 + altitude, 0.0);
    let mut reference: Option<Vec<u8>> = None;
    let mut legacy_grid = 0;
    for level in [1, 5, 8] {
        let lod = lod_for(&p, size, fov, altitude, level);
        let (target, mut r, hs) = render(&gpu, p.clone(), eye, -Vec3::Y, size, fov, lod, true);
        assert!(hs.iter().all(|h| h.status == 1 && h.level == level));
        r.settings_mut().freeze_residency = true;
        r.settings_mut().far_relief = false;
        target.render(&gpu, &mut r, &frame(&p, eye), -Vec3::Y, 2002);
        let before = read_buffer(&gpu, r.surface_buffer(), 81 * 16);
        let primary = read_buffer(&gpu, r.hit_buffer(), 81 * 32);
        r.settings_mut().far_relief = true;
        target.render(&gpu, &mut r, &frame(&p, eye), -Vec3::Y, 2003);
        let after = read_buffer(&gpu, r.surface_buffer(), 81 * 16);
        assert_eq!(primary, read_buffer(&gpu, r.hit_buffer(), 81 * 32));
        for (index, s) in after.chunks_exact(16).enumerate() {
            assert_eq!(s[7], 255, "L{level} pixel{index} exposed a coarse edge/AO grid despite subpixel authored voxels");
            assert_eq!(&s[8..12], &before[index * 16 + 8..index * 16 + 12]);
            if before[index * 16 + 7] < 255 {
                legacy_grid += 1;
            }
            if let Some(ref bytes) = reference {
                assert_eq!(
                    &s[4..12],
                    &bytes[index * 16 + 4..index * 16 + 12],
                    "L{level} changed constant-field pigment/AO/normal at pixel{index}"
                );
            }
        }
        reference.get_or_insert(after);
    }
    assert!(
        legacy_grid > 0,
        "fixture did not expose the removed coarse AO grid"
    );
    // At this footprint authored voxels cover >20px, so crisp near edges
    // must remain byte-identical and visibly present with either shader.
    let p = Arc::new(flat(Shape::Plane, 0.0));
    let near = DVec3::new(0.0, 0.5, 0.0);
    let (target, mut r, hs) = render(&gpu, p.clone(), near, -Vec3::Y, size, fov, 0.125, true);
    assert!(
        hs.iter().all(|h| h.status == 1 && h.level == 0),
        "hinted near rays must expose L0"
    );
    r.settings_mut().residency_hints = false;
    target.render(&gpu, &mut r, &frame(&p, near), -Vec3::Y, 2002);
    let no_hint = hits(&gpu, &r);
    for (hinted, plain) in hs.iter().zip(&no_hint) {
        assert_eq!(
            (
                hinted.status,
                hinted.face,
                hinted.level,
                hinted.i,
                hinted.j,
                hinted.k,
                hinted.normal,
                hinted.t.to_bits()
            ),
            (
                plain.status,
                plain.face,
                plain.level,
                plain.i,
                plain.j,
                plain.k,
                plain.normal,
                plain.t.to_bits()
            ),
            "near block hint changed physical hit"
        );
    }
    r.settings_mut().residency_hints = true;
    r.settings_mut().freeze_residency = true;
    r.settings_mut().far_relief = false;
    target.render(&gpu, &mut r, &frame(&p, near), -Vec3::Y, 2002);
    let before = read_buffer(&gpu, r.surface_buffer(), 81 * 16);
    r.settings_mut().far_relief = true;
    target.render(&gpu, &mut r, &frame(&p, near), -Vec3::Y, 2003);
    let after = read_buffer(&gpu, r.surface_buffer(), 81 * 16);
    assert_eq!(
        before, after,
        "authored pixel filtering changed resolved near voxels"
    );
    assert!(
        after.chunks_exact(16).any(|s| s[7] < 255),
        "near voxel edges were erased"
    );
    eprintln!("AO filter: {legacy_grid} fake coarse-grid samples removed across L1/L5/L8; near L0 edges byte-identical");
}
