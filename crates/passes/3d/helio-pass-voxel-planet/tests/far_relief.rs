//! Perceptual relief must survive radial quantization without changing hits.
mod common;
use common::*;
use glam::{DVec3, IVec3, Vec3};
use helio_pass_voxel_planet::{
    engine::{PlanetRenderer, Settings},
    grid::{Grid, Shape},
    terrain::{self, material, GeneratorInfo, TerrainField, TerrainGenerator, TerrainProgram},
    Brush, BrushOp, BrushShape, Planet, PlanetRecipe, TerrainSource,
};
use std::{
    borrow::Cow,
    sync::{Arc, Once},
};

const GENERATOR: &str = "qualification.broad-relief";
struct BroadRelief;
impl TerrainGenerator for BroadRelief {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: GENERATOR.into(),
            version: 1,
            name: "Broad authored relief".into(),
            description: "Shallow ramp clamped outside the test region".into(),
            settings_component: None,
        }
    }
    fn build(&self, _: &Grid, _: u64, _: &str) -> Result<Arc<dyn TerrainField>, String> {
        Ok(Arc::new(BroadRelief))
    }
}
impl TerrainField for BroadRelief {
    fn height(&self, p: IVec3, _: u32) -> i32 {
        (1_000_000 + (p.x >> 2) * 3 / 2 + (p.z >> 2)).clamp(-4_000_000, 4_000_000)
    }
    fn ground_material(&self, _: IVec3, _: u32, _: i32, _: i32, _: i32, _: i32) -> u32 {
        material::GRASS
    }
    fn height_range(&self) -> (i32, i32) {
        (-4_000_000, 4_000_000)
    }
    fn bound_margins(&self) -> [i32; 24] {
        [10_000; 24]
    }
    fn program(&self) -> TerrainProgram {
        TerrainProgram {
            key: Cow::Borrowed("qualification.broad-relief/1"),
            constants: vec![0; 16],
            wgsl: Cow::Borrowed(
                r#"
struct TerrainConstants { pad: vec4<i32>, }
fn terrain_height(p: vec3<i32>, level: u32) -> i32 {
    return clamp(1000000 + (p.x >> 2u) * 3 / 2 + (p.z >> 2u), -4000000, 4000000);
}
fn ground_material(p: vec3<i32>, surface:u32, top: i32, depth: i32, slope: i32, layer: i32) -> u32 { return M_GRASS; }
"#,
            ),
        }
    }
}
fn world(size: f64) -> Arc<Planet> {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| terrain::register(Arc::new(BroadRelief)).unwrap());
    Arc::new(
        Planet::new(PlanetRecipe {
            shape: Shape::InfinitePlane,
            voxel_size_m: size,
            terrain: TerrainSource {
                generator: GENERATOR.into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    )
}
// A separate material oracle leaves the all-grass normal field unchanged.
struct CutDepth(u32);
impl TerrainGenerator for CutDepth {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: "qualification.cut-relief-depth".into(),
            version: 1,
            name: "Depth-sensitive cut relief".into(),
            description: "Physical subsoil below two metres".into(),
            settings_component: None,
        }
    }
    fn build(&self, grid: &Grid, _: u64, _: &str) -> Result<Arc<dyn TerrainField>, String> {
        Ok(Arc::new(CutDepth(grid.layer_mm())))
    }
}
impl TerrainField for CutDepth {
    fn height(&self, p: IVec3, level: u32) -> i32 {
        BroadRelief.height(p, level)
    }
    fn ground_material(&self, _: IVec3, _: u32, _: i32, depth: i32, _: i32, _: i32) -> u32 {
        if i64::from(depth) * i64::from(self.0) > 2000 {
            material::STONE
        } else {
            material::GRASS
        }
    }
    fn height_range(&self) -> (i32, i32) {
        BroadRelief.height_range()
    }
    fn bound_margins(&self) -> [i32; 24] {
        BroadRelief.bound_margins()
    }
    fn program(&self) -> TerrainProgram {
        let mut constants = (self.0 as i32).to_le_bytes().to_vec();
        constants.resize(16, 0);
        TerrainProgram {
            key: Cow::Borrowed("qualification.cut-relief-depth/1"),
            constants,
            wgsl: Cow::Borrowed(
                r#"
struct TerrainConstants { params: vec4<i32>, }
fn terrain_height(p: vec3<i32>, level: u32) -> i32 { return clamp(1000000 + (p.x>>2u)*3/2 + (p.z>>2u), -4000000, 4000000); }
fn ground_material(p: vec3<i32>, surface:u32, top:i32, depth:i32, slope:i32, layer:i32) -> u32 {
    return select(M_GRASS, M_STONE, depth * terrain.params.x > 2000);
}
"#,
            ),
        }
    }
}
fn cut_depth_world(size: f64) -> Arc<Planet> {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| terrain::register(Arc::new(CutDepth(0))).unwrap());
    Arc::new(
        Planet::new(PlanetRecipe {
            shape: Shape::InfinitePlane,
            voxel_size_m: size,
            terrain: TerrainSource {
                generator: "qualification.cut-relief-depth".into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    )
}
fn authored_normal(grid: &Grid) -> Vec3 {
    let scale = f64::from(grid.domain_scale()) / 16_777_216.0;
    Vec3::new(
        (-0.003 * scale / grid.voxel_size()) as f32,
        1.0,
        (-0.002 * scale / grid.voxel_size()) as f32,
    )
    .normalize()
}
fn unpack_normal(word: u32) -> Vec3 {
    let x = ((word as u16 as i16) as f32 / 32767.0).max(-1.0);
    let y = (((word >> 16) as u16 as i16) as f32 / 32767.0).max(-1.0);
    let z = 1.0 - x.abs() - y.abs();
    if z < 0.0 {
        Vec3::new(
            (1.0 - y.abs()) * if x >= 0.0 { 1.0 } else { -1.0 },
            (1.0 - x.abs()) * if y >= 0.0 { 1.0 } else { -1.0 },
            z,
        )
        .normalize()
    } else {
        Vec3::new(x, y, z).normalize()
    }
}

#[test]
fn broad_relief_has_the_authored_slope_across_grid_sizes() {
    for size in [0.1, 0.3, 1.0] {
        let p = world(size);
        let g = p.grid();
        let origin = g.cells() / 2;
        let dx = g.domain_point(2, origin + 1000, origin, 0)
            - g.domain_point(2, origin - 1000, origin, 0);
        let dh = BroadRelief.height(dx, 0) - BroadRelief.height(IVec3::ZERO, 0);
        let slope = f64::from(dh) * 0.001 / (2000.0 * g.voxel_size());
        let normal = authored_normal(g);
        assert!((slope + f64::from(normal.x / normal.y)).abs() < 1e-6);
        assert!(normal.dot(Vec3::Y) < 0.9998);
    }
}

#[test]
fn paint_preserves_authored_relief_geometry_and_normals() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let target = Target::new(&gpu, [96, 54]);
    for size in [0.1, 0.3, 1.0] {
        let original = world(size);
        let mut painted = original.as_ref().clone();
        painted
            .apply(Brush {
                center: [0.0, 1000.0, 0.0],
                radius: 1850.0,
                shape: BrushShape::Cube,
                op: BrushOp::Paint,
                material: material::BRICK,
            })
            .unwrap();
        let painted = Arc::new(painted);
        let eye = DVec3::new(12.5, 64_000.0, 15.5);
        let render = |p: &Arc<Planet>| {
            let mut r = PlanetRenderer::new(
                &gpu.device,
                &gpu.queue,
                p.clone(),
                Settings {
                    coarse_relief: true,
                    frame_override: Some(37),
                    ..Default::default()
                },
                target.size,
            );
            let f = frame(p, eye);
            for n in 0..2000 {
                target.render(&gpu, &mut r, &f, -Vec3::Y, n);
                if r.settled() {
                    break;
                }
            }
            assert!(
                r.settled(),
                "paint oracle failed to settle: {:?}",
                r.stats()
            );
            target.render(&gpu, &mut r, &f, -Vec3::Y, 2001);
            r
        };
        let before = render(&original);
        let after = render(&painted);
        let ah = hits(&gpu, &before);
        let bh = hits(&gpu, &after);
        let sa = read_buffer(&gpu, before.surface_buffer(), 96 * 54 * 16);
        let sb = read_buffer(&gpu, after.surface_buffer(), 96 * 54 * 16);
        let raw_hits = read_buffer(&gpu, after.hit_buffer(), 96 * 54 * 32);
        let columns = read_buffer(
            &gpu,
            after.residency_buffers()[0],
            after.residency_buffers()[0].size(),
        );
        let mut painted_samples = 0;
        let mut paint_columns = 0;
        for (index, (a, b)) in ah.iter().zip(&bh).enumerate() {
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
                "Paint reshaped authored relief at {size}m pixel{index}"
            );
            if a.status != 1 {
                continue;
            }
            let x = &sa[index * 16..index * 16 + 16];
            let y = &sb[index * 16..index * 16 + 16];
            assert_eq!(&x[..4], &y[..4], "Paint changed depth");
            assert_eq!(&x[8..12], &y[8..12], "Paint changed authored normal");
            assert_eq!(x[7], y[7], "Paint changed occupancy AO");
            let fa = u32::from_le_bytes(x[12..16].try_into().unwrap());
            let fb = u32::from_le_bytes(y[12..16].try_into().unwrap());
            assert_eq!(
                fa & !0xff00,
                fb & !0xff00,
                "Paint changed filtering or light origin"
            );
            let record = u32::from_le_bytes(
                raw_hits[index * 32 + 20..index * 32 + 24]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let info = u32::from_le_bytes(
                columns[record * 32 + 12..record * 32 + 16]
                    .try_into()
                    .unwrap(),
            );
            let edits = u32::from_le_bytes(
                columns[record * 32 + 28..record * 32 + 32]
                    .try_into()
                    .unwrap(),
            );
            if edits != 0 && b.level > 0 {
                assert_ne!(info & 0x10000000, 0, "Paint discarded fractional heights");
                assert_eq!(info & 0x08000000, 0, "Paint advertised geometry edits");
                paint_columns += 1;
            }
            if (fb >> 8) & 255 == material::BRICK {
                assert_ne!(&x[4..7], &y[4..7], "Paint failed to alter pigment");
                painted_samples += 1;
            } else {
                assert_eq!(x, y, "Paint changed an unpainted surface");
            }
        }
        assert!(
            painted_samples > 0 && paint_columns > 0,
            "fixture did not expose paint for {size}m"
        );
        eprintln!("Paint {size}m: {painted_samples} pigment changes, {paint_columns} edited-list relief samples; hit/depth/normal parity");
    }
}

#[test]
fn stored_coarse_heights_recover_authored_normal_without_raw_climate_queries() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let target = Target::new(&gpu, [96, 54]);
    for size in [0.1, 0.3, 1.0] {
        let p = world(size);
        let f = frame(&p, DVec3::new(12.5, 64_000.0, 15.5));
        let mut r = PlanetRenderer::new(
            &gpu.device,
            &gpu.queue,
            p.clone(),
            Settings {
                lod_pixels: 4.0,
                coarse_relief: true,
                frame_override: Some(37),
                ..Default::default()
            },
            target.size,
        );
        for n in 0..2000 {
            target.render(&gpu, &mut r, &f, -Vec3::Y, n);
            if r.settled() {
                break;
            }
        }
        assert!(r.settled(), "stored relief did not settle: {:?}", r.stats());
        r.settings_mut().freeze_residency = true;
        target.render(&gpu, &mut r, &f, -Vec3::Y, 2001);
        let hs = hits(&gpu, &r);
        let surfaces = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
        let expected = authored_normal(p.grid());
        let mut checked = 0;
        for y in 18..36 {
            for x in 34..62 {
                let index = (x + y * 96) as usize;
                let h = hs[index];
                if h.status != 1 || h.level < 6 {
                    continue;
                }
                let pixel = h.t * 2.0 * (target.fov_y * 0.5).tan() / 54.0;
                let cell = size as f32 * (1u32 << h.level) as f32;
                let smooth = ((2.5 - cell / pixel) / 1.5).clamp(0.0, 1.0);
                // The existing footprint blend deliberately retains visible faces.
                // Qualify the authored macro normal only where its full weight applies.
                if smooth < 0.99999 {
                    continue;
                }
                let word = u32::from_le_bytes(
                    surfaces[index * 16 + 8..index * 16 + 12]
                        .try_into()
                        .unwrap(),
                );
                let actual = unpack_normal(word);
                let error = actual.dot(expected).clamp(-1.0, 1.0).acos();
                assert!(error < 0.004, "stored normal error {error} size{size} xy({x},{y}) actual{actual:?} expected{expected:?}");
                checked += 1;
            }
        }
        assert!(
            checked >= 24,
            "insufficient full-weight stored relief pixels: size{size} checked{checked}"
        );
        eprintln!("stored coarse normal size{size}: {checked} authored slopes; raw climate relief disabled");
    }
}

// This regression qualifies edit publication scope rather than edited topology:
// sub-voxel brushes remain in the journal but the edit shaders ignore them at the
// displayed coarse level. They must not flatten unrelated or same-column relief.

#[test]
fn pending_paint_preserves_unaffected_raw_relief_and_undo() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let target = Target::new(&gpu, [96, 54]);
    for size in [0.1, 0.3, 1.0] {
        let mut p = world(size);
        let eye = DVec3::new(12.5, 64000.0, 15.5);
        let mut r = PlanetRenderer::new(
            &gpu.device,
            &gpu.queue,
            p.clone(),
            Settings {
                coarse_relief: true,
                job_budget: 256,
                frame_override: Some(37),
                ..Default::default()
            },
            target.size,
        );
        let mut number = 0;
        let settle = |r: &mut PlanetRenderer, p: &Arc<Planet>, number: &mut u64| {
            for _ in 0..2000 {
                target.render(&gpu, r, &frame(p, eye), -Vec3::Y, *number);
                *number += 1;
                if r.settled() {
                    return;
                }
            }
            panic!("paint publication failed to settle: {:?}", r.stats());
        };
        settle(&mut r, &p, &mut number);
        let original = hits(&gpu, &r);
        let baseline = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
        let expected = authored_normal(p.grid());
        let mut selected = Vec::new();
        for y in 18..36 {
            for x in 34..62 {
                let index = (x + y * 96) as usize;
                let h = original[index];
                if h.status != 1 || h.level <= 4 {
                    continue;
                }
                let s = &baseline[index * 16..index * 16 + 16];
                let n = unpack_normal(u32::from_le_bytes(s[8..12].try_into().unwrap()));
                if n.dot(expected).clamp(-1.0, 1.0).acos() < 0.004 {
                    selected.push(index);
                }
            }
        }
        assert!(
            selected.len() >= 24,
            "no authored normal exposure: {size}m {}",
            selected.len()
        );
        let check = |r: &PlanetRenderer, phase: &str| {
            let current = hits(&gpu, r);
            let bytes = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
            let mut unchanged = 0;
            for &index in &selected {
                let a = original[index];
                let b = current[index];
                if (
                    a.status,
                    a.face,
                    a.i,
                    a.j,
                    a.k,
                    a.level,
                    a.normal,
                    a.t.to_bits(),
                ) != (
                    b.status,
                    b.face,
                    b.i,
                    b.j,
                    b.k,
                    b.level,
                    b.normal,
                    b.t.to_bits(),
                ) {
                    continue;
                }
                let s = &bytes[index * 16..index * 16 + 16];
                let flags = u32::from_le_bytes(s[12..16].try_into().unwrap());
                if (flags >> 8) & 255 == material::BRICK {
                    continue;
                }
                let n = unpack_normal(u32::from_le_bytes(s[8..12].try_into().unwrap()));
                let error = n.dot(expected).clamp(-1.0, 1.0).acos();
                assert!(
                    error < 0.004,
                    "{phase} flattened unrelated {size}m normal: index{index} {error}rad"
                );
                assert_eq!(
                    s,
                    &baseline[index * 16..index * 16 + 16],
                    "{phase} changed unaffected surface"
                );
                unchanged += 1;
            }
            assert!(
                unchanged >= 24,
                "{phase} insufficient unaffected matches: {unchanged}"
            );
            eprintln!("pending Paint {size}m {phase}: {unchanged} unchanged authored surfaces");
        };
        // This radius is active in visible coarse levels, unlike a tiny brush
        // discarded by the native CPU edit query. It must issue real edit jobs.
        Arc::make_mut(&mut p)
            .apply(Brush {
                center: [0.0, 1000.0, 0.0],
                radius: 1850.0,
                shape: BrushShape::Cube,
                op: BrushOp::Paint,
                material: material::BRICK,
            })
            .unwrap();
        // Residency plans run a frame ahead: the edit's jobs reach the GPU
        // within a few frames (more when a plan is late), not the first.
        for _ in 0..30 {
            target.render(&gpu, &mut r, &frame(&p, eye), -Vec3::Y, number);
            number += 1;
            if r.stats().jobs > 0 {
                break;
            }
        }
        assert!(
            r.stats().jobs > 0 && !r.settled(),
            "fixture did not issue a pending edit publication: {:?}",
            r.stats()
        );
        check(&r, "publishing");
        settle(&mut r, &p, &mut number);
        check(&r, "published");
        Arc::make_mut(&mut p).undo().expect("brush available");
        settle(&mut r, &p, &mut number);
        check(&r, "undo");
    }
}

#[test]
fn distant_add_remove_cut_faces_keep_native_normals() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let mut target = Target::new(&gpu, [96, 54]);
    // A crop of the native-resolution footprint (~95m/pixel at64km),
    // so legal1850m brushes expose multiple cut faces at actual distance.
    target.fov_y = 0.08;
    for size in [0.1, 0.3, 1.0] {
        for op in [BrushOp::Add, BrushOp::Remove] {
            let original = if op == BrushOp::Remove {
                cut_depth_world(size)
            } else {
                world(size)
            };
            let mut p = original.as_ref().clone();
            let (center, radius, eye, aim) = match op {
                BrushOp::Add => (
                    [0.0, 6000.0, 0.0],
                    1850.0,
                    DVec3::new(-64000.0, 2000.0, 0.0),
                    DVec3::new(0.0, 5500.0, 0.0),
                ),
                _ => (
                    [0.0, 0.0, 0.0],
                    1850.0,
                    DVec3::new(-64000.0, 16000.0, 0.0),
                    DVec3::ZERO,
                ),
            };
            p.apply(Brush {
                center,
                radius,
                shape: BrushShape::Cube,
                op,
                material: material::BRICK,
            })
            .unwrap();
            let p = Arc::new(p);
            let forward = (aim - eye).as_vec3().normalize();
            let mut r = PlanetRenderer::new(
                &gpu.device,
                &gpu.queue,
                p.clone(),
                Settings {
                    coarse_relief: true,
                    lod_pixels: if op == BrushOp::Remove { 0.5 } else { 0.125 },
                    lod_dither: 0.0,
                    horizon: false,
                    // Cut faces and their corner AO; sky visibility (deep
                    // pits are occluded) has its own test.
                    sky_occlusion: false,
                    frame_override: Some(37),
                    ..Default::default()
                },
                target.size,
            );
            let f = frame(&p, eye);
            for n in 0..2000 {
                target.render(&gpu, &mut r, &f, forward, n);
                if r.settled() {
                    break;
                }
            }
            assert!(r.settled(), "cut fixture failed to settle {:?}", r.stats());
            target.render(&gpu, &mut r, &f, forward, 2001);
            let hs = hits(&gpu, &r);
            let bytes = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
            let raw = read_buffer(&gpu, r.hit_buffer(), 96 * 54 * 32);
            let cols = read_buffer(
                &gpu,
                r.residency_buffers()[0],
                r.residency_buffers()[0].size(),
            );
            let mut cuts = 0;
            let mut underside = 0;
            let mut subsoil = 0;
            let mut subsoil_walls = 0;
            let mut untouched_risers = 0;
            for (index, h) in hs.iter().enumerate() {
                assert!(h.status < 2, "unresolved cut ray {index}: {h:?}");
                if h.status != 1
                    || h.level == 0
                    || (h.normal == 4 && op == BrushOp::Add)
                    || h.normal >= 6
                {
                    continue;
                }
                let s = &bytes[index * 16..index * 16 + 16];
                let flags = u32::from_le_bytes(s[12..16].try_into().unwrap());
                let mat = (flags >> 8) & 255;
                if op == BrushOp::Add && mat != material::BRICK {
                    continue;
                }
                if op == BrushOp::Remove {
                    let mut air = [h.i, h.j, h.k];
                    let axis = (h.normal >> 1) as usize;
                    air[axis] += if h.normal & 1 == 0 { 1 } else { -1 };
                    // A true cut face borders a cell the Remove turned from
                    // base-field solid into air; the entered hit remains solid.

                    if original
                        .sample_kind(h.level, h.face, air[0], air[1], air[2])
                        .0
                        == 0
                        || p.sample_kind(h.level, h.face, air[0], air[1], air[2]).0 != 0
                    {
                        // An untouched natural riser in the same active column
                        // must keep surface pigment, not acquire a grey band.
                        if original
                            .sample_kind(h.level, h.face, air[0], air[1], air[2])
                            .0
                            == 0
                        {
                            let record = u32::from_le_bytes(
                                raw[index * 32 + 20..index * 32 + 24].try_into().unwrap(),
                            ) as usize;
                            let info = u32::from_le_bytes(
                                cols[record * 32 + 12..record * 32 + 16].try_into().unwrap(),
                            );
                            if info & 0x08000000 != 0 && h.normal < 4 {
                                assert_eq!(
                                    mat,
                                    material::GRASS,
                                    "untouched natural riser acquired subsoil {h:?}"
                                );
                                untouched_risers += 1;
                            }
                        }
                        continue;
                    }
                    let own_top = original.column_top(h.face, h.i, h.j, h.level);
                    let depth = (own_top - 1 - h.k).max(0) << h.level;
                    let expected_material =
                        original
                            .field()
                            .ground_material(IVec3::ZERO, 0, 0, depth, 0, 0);
                    assert_eq!(
                        mat, expected_material,
                        "Remove material depth oracle at {h:?}, depth{depth}"
                    );
                    if expected_material == material::STONE {
                        subsoil += 1;
                        if h.normal < 4 {
                            subsoil_walls += 1;
                        }
                    }
                    assert_eq!(
                        p.sample_kind(h.level, h.face, h.i, h.j, h.k).0,
                        1,
                        "Remove ray entered CPU-air cell"
                    );
                }
                let record =
                    u32::from_le_bytes(raw[index * 32 + 20..index * 32 + 24].try_into().unwrap())
                        as usize;
                let info = u32::from_le_bytes(
                    cols[record * 32 + 12..record * 32 + 16].try_into().unwrap(),
                );
                assert_ne!(info & 0x08000000, 0, "cut omitted topology metadata");
                assert_eq!(info & 0x10000000, 0, "cut incorrectly retained base relief");
                let [n, a, b] =
                    helio_pass_voxel_planet::grid::face_axes(h.face).map(|v| v.as_vec3());
                let expected = match h.normal {
                    0 => a,
                    1 => -a,
                    2 => b,
                    3 => -b,
                    4 => n,
                    5 => -n,
                    _ => unreachable!(),
                };
                let actual = unpack_normal(u32::from_le_bytes(s[8..12].try_into().unwrap()));
                let error = actual.dot(expected).clamp(-1.0, 1.0).acos();
                assert!(error<0.001,"{op:?} {size}m cut normal pixel{index}: actual{actual:?} expected{expected:?} error{error}");
                assert_eq!(s[7], 255, "subpixel authored cells exposed coarse AO grid: {op:?} {size}m {h:?}");
                assert_eq!(
                    (flags >> 21) & 7,
                    0,
                    "cut incorrectly used base-field shadow smoothing"
                );
                cuts += 1;
                if h.normal == 5 {
                    underside += 1;
                }
            }
            assert!(
                cuts >= 16,
                "no meaningful native cut-face exposure for {op:?} {size}m: {cuts}"
            );
            if op == BrushOp::Add {
                assert!(
                    underside > 0,
                    "floating cube must expose its downward ceiling"
                );
            }
            if op == BrushOp::Remove {
                assert!(
                    subsoil_walls >= 16,
                    "Remove failed to expose deep side walls: {subsoil_walls}"
                );
                assert!(untouched_risers >= 16, "Remove must expose untouched natural risers in an active topology column: {untouched_risers}");
                assert!(
                    subsoil >= 16,
                    "Remove failed to expose deep subsoil: {subsoil}"
                );
            }
            eprintln!("{op:?} {size}m: {cuts} native cut normals, {underside} undersides, {subsoil} depth-sensitive subsoil ({subsoil_walls} walls), {untouched_risers} untouched natural risers; no coarse AO grid");
        }
    }
}
