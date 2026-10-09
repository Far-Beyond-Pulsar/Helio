//! Analytic fixtures for optional coarse radial relief. Run serially on GPU.
mod common;
use common::*;
use glam::{DVec3, IVec3, Vec3};
use helio_pass_voxel_planet::engine::{PlanetRenderer, Settings};
use helio_pass_voxel_planet::grid::Shape;
use helio_pass_voxel_planet::{
    Brush, BrushOp, BrushShape, GeneratorInfo, Grid, Planet, PlanetRecipe, TerrainField,
    TerrainGenerator, TerrainProgram, TerrainSource,
};
use std::sync::{Arc, Once};

fn flat(shape: Shape, height: f64) -> Planet {
    Planet::new(PlanetRecipe {
        shape,
        radius_m: 10_000.0,
        plane_size_m: 40_000.0,
        terrain: helio_pass_voxel_planet::layers::TerrainLayers::flat_at(height).source(7),
        ..Default::default()
    })
    .unwrap()
}
fn settled_render(
    gpu: &Gpu,
    planet: Arc<Planet>,
    eye: DVec3,
    forward: Vec3,
    size: [u32; 2],
    enabled: bool,
    lod_pixels: f32,
    horizon: bool,
) -> (Target, PlanetRenderer, Vec<Hit>) {
    let target = Target::new(gpu, size);
    let mut renderer = PlanetRenderer::new(
        &gpu.device,
        &gpu.queue,
        planet.clone(),
        Settings {
            coarse_relief: enabled,
            lod_pixels,
            lod_dither: 0.0,
            horizon,
            frame_override: Some(37),
            ..Default::default()
        },
        size,
    );
    let frame = frame(&planet, eye);
    let mut done = false;
    for n in 0..2000 {
        target.render(gpu, &mut renderer, &frame, forward, n);
        if renderer.settled() {
            done = true;
            break;
        }
    }
    assert!(done, "residency failed to settle");
    target.render(gpu, &mut renderer, &frame, forward, 2001);
    assert_eq!(renderer.stats().failed_jobs, 0);
    let h = hits(gpu, &renderer);
    assert!(h.iter().all(|v| v.status < 2), "unresolved hits: {h:?}");
    (target, renderer, h)
}
fn record_info(gpu: &Gpu, renderer: &PlanetRenderer, pixel: usize) -> u32 {
    let raw = read_buffer(gpu, renderer.hit_buffer(), (pixel as u64 + 1) * 32);
    let at = u32::from_le_bytes(raw[pixel * 32 + 20..pixel * 32 + 24].try_into().unwrap()) as usize;
    let data = read_buffer(gpu, renderer.residency_buffers()[0], (at as u64 + 1) * 32);
    u32::from_le_bytes(data[at * 32 + 12..at * 32 + 16].try_into().unwrap())
}

#[test]
fn authored_radial_top_survives_mixed_and_solid_coarse_bricks() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    // At centre L10, ceil(717/102.4)==8: the top brick is entirely solid.
    // 53.7 produces a mixed top brick. Both must clip its fractional surface.
    for height in [53.7, 717.0, -53.7] {
        let planet = Arc::new(flat(Shape::Plane, height));
        let eye = DVec3::new(0.0, height + 4000.0, 0.0);
        let (_, r, h) = settled_render(
            &gpu,
            planet.clone(),
            eye,
            -Vec3::Y,
            [3, 3],
            true,
            16.0,
            false,
        );
        for v in &h {
            assert_eq!(v.status, 1);
            assert!(v.level >= 6);
            assert!((record_info(&gpu, &r, 4) & 0x10000000) != 0);
        }
        let actual = eye.y - f64::from(h[4].t);
        let cell = planet.grid().voxel_size();
        let expected = (height / cell + 1e-9).floor() * cell;
        let error = cell * f64::from(1u32 << h[4].level) / 65536.0 + 0.02;
        assert!(
            (actual - expected).abs() <= error,
            "height={height} actual={actual} expected={expected} hit={:?}",
            h[4]
        );
        eprintln!(
            "coarse top height={height} actual={actual} centre={:?} logical_bytes={}",
            h[4],
            r.stats().logical_bytes
        );
    }
}

#[test]
fn orbital_flat_shell_keeps_silhouette_hit_and_miss() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let planet = Arc::new(flat(Shape::Sphere, 53.7));
    let eye = DVec3::Y * 30_000.0;
    let size = [16, 9];
    for horizon in [false, true] {
        let (target, _, h) = settled_render(
            &gpu,
            planet.clone(),
            eye,
            -Vec3::Y,
            size,
            true,
            1.0,
            horizon,
        );
        let camera = target.camera(-Vec3::Y, Vec3::X);
        let radius = planet.grid().radius() + 53.7;
        let (mut nhit, mut nmiss) = (0, 0);
        for (index, v) in h.iter().enumerate() {
            let d = pixel_dir(
                &target,
                &camera,
                index as u32 % size[0],
                index as u32 / size[0],
            );
            let b = eye.dot(d);
            let disc = b * b - (eye.length_squared() - radius * radius);
            if disc < 0.0 {
                assert_eq!(v.status, 0, "false silhouette hit at {index}: {v:?}");
                nmiss += 1;
            } else {
                let expected = -b - disc.sqrt();
                assert_eq!(v.status, 1, "lost silhouette at {index}: {v:?}");
                let radial_error =
                    planet.grid().voxel_size() * f64::from(1u32 << v.level) / 65536.0;
                let incidence = disc.sqrt() / radius;
                let error = radial_error / incidence.max(0.001) + expected * 2e-6 + 0.05;
                assert!(
                    (f64::from(v.t) - expected).abs() <= error,
                    "index={index} expected={expected} error={error} {v:?}"
                );
                nhit += 1;
            }
        }
        assert!(nhit > 10 && nmiss > 10);
        eprintln!("orbital shell horizon={horizon}: {nhit}hit {nmiss}miss");
    }
}

#[derive(Debug)]
struct Hill;
impl TerrainField for Hill {
    fn height(&self, p: IVec3, _: u32) -> i32 {
        53_700 + (400_000 - (p.x >> 2).abs() * 10).max(0)
    }
    fn ground_material(&self, _: IVec3, _: u32, _: i32, _: i32, _: i32, _: i32) -> u32 {
        3
    }
    fn height_range(&self) -> (i32, i32) {
        (53_700, 453_700)
    }
    fn bound_margins(&self) -> [i32; 24] {
        [4; 24]
    }
    fn program(&self) -> TerrainProgram {
        TerrainProgram {
            key: "test.coarse-relief-hill/1".into(),
            wgsl: r#"
        struct TerrainConstants {dummy:vec4<i32>}
        fn terrain_height(p:vec3<i32>,level:u32)->i32{return 53700+max(400000-abs(p.x>>2u)*10,0);}
        fn ground_material(p:vec3<i32>,surface:u32, top:i32,depth:i32,slope:i32,layer:i32)->u32{return 3u;}
    "#
            .into(),
            constants: vec![0; 16],
        }
    }
}
impl TerrainGenerator for Hill {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: "test.coarse-relief-hill".into(),
            version: 1,
            name: "hill".into(),
            description: "fixture".into(),
            settings_component: None,
        }
    }
    fn build(&self, _: &Grid, _: u64, _: &str) -> Result<Arc<dyn TerrainField>, String> {
        Ok(Arc::new(Hill))
    }
}
#[test]
fn broad_hill_has_radial_relief_inside_coarse_cells() {
    static INIT: Once = Once::new();
    INIT.call_once(|| helio_pass_voxel_planet::terrain::register(Arc::new(Hill)).unwrap());
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let planet = Arc::new(
        Planet::new(PlanetRecipe {
            shape: Shape::Plane,
            plane_size_m: 40_000.0,
            terrain: TerrainSource {
                generator: "test.coarse-relief-hill".into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    );
    let eye = DVec3::new(0.0, 4453.7, 0.0);
    let size = [5, 3];
    let (target, _, h) =
        settled_render(&gpu, planet.clone(), eye, -Vec3::Y, size, true, 16.0, false);
    let camera = target.camera(-Vec3::Y, Vec3::X);
    let mut radial = 0;
    for (n, v) in h.iter().enumerate() {
        assert_eq!(v.status, 1);
        assert!(v.level >= 6);
        let d = pixel_dir(&target, &camera, n as u32 % size[0], n as u32 / size[0]);
        let actual = eye.y + d.y * f64::from(v.t);
        let p = planet.grid().domain_point(v.face, v.i, v.j, v.level);
        let height = planet
            .field()
            .height(p, v.level + planet.grid().level_offset());
        let base = height / planet.grid().layer_mm() as i32;
        let expected = f64::from(base) * planet.grid().voxel_size();
        assert!(
            actual <= expected + 0.05,
            "above hill {n} actual={actual} expected={expected} {v:?}"
        );
        if v.normal == 4 {
            assert!((actual - expected).abs() < 0.05);
            radial += 1;
        }
    }
    assert!(radial >= 3);
    let centre = &h[7];
    let centre_height = eye.y - f64::from(centre.t);
    assert!(centre_height > 300.0, "hill vanished: {centre:?}");
}

#[test]
fn paint_preserves_unpainted_fractional_surface() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let original = Arc::new(flat(Shape::Plane, 53.7));
    let mut planet = original.as_ref().clone();
    planet
        .apply(Brush {
            center: [0.0, 40.0, 0.0],
            radius: 1500.0,
            shape: BrushShape::Cube,
            op: BrushOp::Paint,
            material: 13,
            height: 0.0,
        })
        .unwrap();
    let planet = Arc::new(planet);
    let eye = DVec3::new(0.0, 4053.7, 0.0);
    let (_, a, ha) = settled_render(&gpu, original, eye, -Vec3::Y, [3, 3], true, 16.0, false);
    let (_, b, hb) = settled_render(
        &gpu,
        planet.clone(),
        eye,
        -Vec3::Y,
        [3, 3],
        true,
        16.0,
        false,
    );
    assert_ne!(
        record_info(&gpu, &b, 4) & 0x10000000,
        0,
        "Paint removed fractional relief"
    );
    assert_eq!(
        record_info(&gpu, &b, 4) & 0x08000000,
        0,
        "Paint marked topology"
    );
    let sa = read_buffer(&gpu, a.surface_buffer(), 9 * 16);
    let sb = read_buffer(&gpu, b.surface_buffer(), 9 * 16);
    let mut painted = 0;
    for (index, (a, b)) in ha.iter().zip(&hb).enumerate() {
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
            "Paint reshaped fractional surface"
        );
        let x = &sa[index * 16..index * 16 + 16];
        let y = &sb[index * 16..index * 16 + 16];
        assert_eq!(&x[..4], &y[..4]);
        assert_eq!(&x[8..12], &y[8..12]);
        assert_eq!(x[7], y[7]);
        let fa = u32::from_le_bytes(x[12..16].try_into().unwrap());
        let fb = u32::from_le_bytes(y[12..16].try_into().unwrap());
        assert_eq!(fa & !0xff00, fb & !0xff00);
        if (fb >> 8) & 255 == 13 {
            assert_ne!(&x[4..7], &y[4..7]);
            painted += 1;
        } else {
            assert_eq!(x, y);
        }
    }
    assert!(painted > 0, "fixture did not expose pigment changes");
    let actual = eye.y - f64::from(hb[4].t);
    let expected = (53.7 / planet.grid().voxel_size() + 1e-9).floor() * planet.grid().voxel_size();
    assert!(
        (actual - expected).abs() < 0.02,
        "Paint flattened authored top {actual} vs{expected}"
    );
}

#[derive(Debug)]
struct Steep;
impl TerrainField for Steep {
    fn height(&self, p: IVec3, _: u32) -> i32 {
        if (p.x >> 2) < 1600 {
            0
        } else {
            5_000_000
        }
    }
    fn ground_material(&self, _: IVec3, _: u32, _: i32, _: i32, _: i32, _: i32) -> u32 {
        3
    }
    fn height_range(&self) -> (i32, i32) {
        (0, 5_000_000)
    }
    fn bound_margins(&self) -> [i32; 24] {
        [1_000_000; 24]
    }
    fn program(&self) -> TerrainProgram {
        TerrainProgram {
            key: "test.coarse-relief-steep/1".into(),
            wgsl: r#"
        struct TerrainConstants {dummy:vec4<i32>}
        fn terrain_height(p:vec3<i32>,level:u32)->i32{return select(5000000,0,(p.x>>2u)<1600);}
        fn ground_material(p:vec3<i32>,surface:u32, top:i32,depth:i32,slope:i32,layer:i32)->u32{return 3u;}
    "#
            .into(),
            constants: vec![0; 16],
        }
    }
}
impl TerrainGenerator for Steep {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: "test.coarse-relief-steep".into(),
            version: 1,
            name: "steep".into(),
            description: "fixture".into(),
            settings_component: None,
        }
    }
    fn build(&self, _: &Grid, _: u64, _: &str) -> Result<Arc<dyn TerrainField>, String> {
        Ok(Arc::new(Steep))
    }
}
#[test]
fn unsupported_tall_band_preserves_legacy_brick_occupancy() {
    static INIT: Once = Once::new();
    INIT.call_once(|| helio_pass_voxel_planet::terrain::register(Arc::new(Steep)).unwrap());
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let planet = Arc::new(
        Planet::new(PlanetRecipe {
            shape: Shape::Plane,
            plane_size_m: 4000.0,
            terrain: TerrainSource {
                generator: "test.coarse-relief-steep".into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    );
    let eye = DVec3::new(90.0, 5015.0, 90.0);
    let (_, r, h) = settled_render(
        &gpu,
        planet.clone(),
        eye,
        -Vec3::Y,
        [3, 3],
        true,
        1.0,
        false,
    );
    let c = &h[4];
    assert_eq!(c.status, 1);
    assert_eq!(c.level, 6, "fixture must exercise L6 steep column");
    let info = record_info(&gpu, &r, 4);
    assert!(
        (info & 511) > 32,
        "fixture did not produce truncated byte tops: info={info:08x}"
    );
    assert_eq!(
        info & 0x10000000,
        0,
        "unsupported byte tops advertised fractional relief"
    );
    let size = planet.grid().voxel_size() * f64::from(1u32 << c.level);
    let expected = (5000.0 / size).floor() * size;
    let actual = eye.y - f64::from(c.t);
    assert!(
        (actual - expected).abs() < 0.05,
        "fallback lost legacy occupancy: actual={actual} expected={expected} {c:?}"
    );
}

#[derive(Debug)]
struct SteepBoundary;
impl TerrainField for SteepBoundary {
    fn height(&self, p: IVec3, _: u32) -> i32 {
        if (p.x >> 2) < 1600 {
            6400
        } else {
            1_638_400
        }
    }
    fn ground_material(&self, _: IVec3, _: u32, _: i32, _: i32, _: i32, _: i32) -> u32 {
        3
    }
    fn height_range(&self) -> (i32, i32) {
        (6400, 1_638_400)
    }
    fn bound_margins(&self) -> [i32; 24] {
        [1_000_000; 24]
    }
    fn program(&self) -> TerrainProgram {
        TerrainProgram {
            key: "test.coarse-relief-boundary/1".into(),
            wgsl: r#"
        struct TerrainConstants {dummy:vec4<i32>}
        fn terrain_height(p:vec3<i32>,level:u32)->i32{return select(1638400,6400,(p.x>>2u)<1600);}
        fn ground_material(p:vec3<i32>,surface:u32, top:i32,depth:i32,slope:i32,layer:i32)->u32{return 3u;}
    "#
            .into(),
            constants: vec![0; 16],
        }
    }
}
impl TerrainGenerator for SteepBoundary {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: "test.coarse-relief-boundary".into(),
            version: 1,
            name: "boundary".into(),
            description: "fixture".into(),
            settings_component: None,
        }
    }
    fn build(&self, _: &Grid, _: u64, _: &str) -> Result<Arc<dyn TerrainField>, String> {
        Ok(Arc::new(SteepBoundary))
    }
}
#[test]
fn exact_32_brick_boundary_does_not_advertise_truncated_top() {
    static INIT: Once = Once::new();
    INIT.call_once(|| helio_pass_voxel_planet::terrain::register(Arc::new(SteepBoundary)).unwrap());
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let planet = Arc::new(
        Planet::new(PlanetRecipe {
            shape: Shape::Plane,
            plane_size_m: 4000.0,
            terrain: TerrainSource {
                generator: "test.coarse-relief-boundary".into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    );
    let eye = DVec3::new(90.0, 1653.4, 90.0);
    let (_, r, h) = settled_render(
        &gpu,
        planet.clone(),
        eye,
        -Vec3::Y,
        [3, 3],
        true,
        1.0,
        false,
    );
    let c = &h[4];
    assert_eq!(c.status, 1);
    assert_eq!(c.level, 6);
    let info = record_info(&gpu, &r, 4);
    assert_eq!(info & 511, 32, "fixture must span exactly32bricks");
    assert_eq!((info >> 22) & 7, 0, "top must lie on the256-cell boundary");
    assert_eq!(info & 0x10000000, 0, "relative top256 cannot fit one byte");
    let actual = eye.y - f64::from(c.t);
    assert!(
        (actual - 1638.4).abs() < 0.05,
        "legacy occupancy truncated: {actual} {c:?}"
    );
}
