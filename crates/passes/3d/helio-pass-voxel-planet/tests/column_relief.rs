//! Stored fractional heights must preserve slope-driven material classes.
mod common;
use common::*;
use glam::{DVec3, IVec3, Vec3};
use helio_pass_voxel_planet::{
    engine::{PlanetRenderer, Settings},
    grid::{Grid, Shape},
    terrain::{self, material, GeneratorInfo, TerrainField, TerrainGenerator, TerrainProgram},
    Planet, PlanetRecipe, TerrainSource,
};
use std::{
    borrow::Cow,
    sync::{Arc, Once},
};
const GENERATOR: &str = "qualification.slope-threshold";
struct SlopeThreshold {
    layer_mm: i32,
}
impl TerrainGenerator for SlopeThreshold {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: GENERATOR.into(),
            version: 1,
            name: "Slope threshold".into(),
            description: "Authored slope with a material boundary at two eighths".into(),
            settings_component: None,
        }
    }
    fn build(&self, grid: &Grid, _: u64, _: &str) -> Result<Arc<dyn TerrainField>, String> {
        Ok(Arc::new(Self {
            layer_mm: grid.layer_mm() as i32,
        }))
    }
}
impl TerrainField for SlopeThreshold {
    fn height(&self, p: IVec3, _: u32) -> i32 {
        (1_000_000 + (p.x >> 2) * 12).clamp(-4_000_000, 4_000_000)
    }
    fn ground_material(&self, _: IVec3, _: i32, _: i32, slope: i32, _: i32) -> u32 {
        if slope >= 2 {
            material::BRICK
        } else {
            material::GRASS
        }
    }
    fn height_range(&self) -> (i32, i32) {
        (-4_000_000, 4_000_000)
    }
    fn bound_margins(&self) -> [i32; 24] {
        std::array::from_fn(|level| {
            (i64::from(self.layer_mm) * (1i64 << level)).clamp(10_000, 8_000_000) as i32
        })
    }
    fn program(&self) -> TerrainProgram {
        TerrainProgram {
            key: Cow::Borrowed("qualification.slope-threshold/1"),
            constants: vec![0; 16],
            wgsl: Cow::Borrowed(
                r#"
struct TerrainConstants { pad: vec4<i32>, }
fn terrain_height(p:vec3<i32>,level:u32)->i32 { return clamp(1000000+(p.x>>2u)*12,-4000000,4000000); }
fn ground_material(p:vec3<i32>,top:i32,depth:i32,slope:i32,layer:i32)->u32 { return select(M_GRASS,M_BRICK,slope>=2); }
"#,
            ),
        }
    }
}
fn world(size: f64, shape: Shape) -> Arc<Planet> {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| terrain::register(Arc::new(SlopeThreshold { layer_mm: 100 })).unwrap());
    Arc::new(
        Planet::new(PlanetRecipe {
            shape,
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
// Oracle queries actual authored heights at the four production block endpoints;
// it does not decode GPU fractions or reproduce the shader correction helper.
fn slopes(p: &Planet, h: Hit) -> (i32, i32) {
    let g = p.grid();
    let ci = h.i & !7;
    let cj = h.j & !7;
    let base = |i, j| {
        i64::from(
            p.field()
                .height(g.domain_point(h.face, i, j, h.level), h.level),
        )
        .div_euclid(i64::from(g.layer_mm()))
    };
    let tx0 = base(ci, h.j);
    let tx7 = base(ci + 7, h.j);
    let ty0 = base(h.i, cj);
    let ty7 = base(h.i, cj + 7);
    let scale = 1i64 << h.level;
    let authored = ((tx7 - tx0).abs().max((ty7 - ty0).abs()) * 8 / (7 * scale)) as i32;
    let ceil = |v: i64| -(-v).div_euclid(scale);
    let rounded = ((ceil(tx7) - ceil(tx0))
        .abs()
        .max((ceil(ty7) - ceil(ty0)).abs())
        * 8
        / 7) as i32;
    (authored, rounded)
}
#[test]
fn stored_material_slope_matches_field_and_exposes_rounded_classification() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let target = Target::new(&gpu, [96, 54]);
    for size in [0.1, 0.3, 1.0] {
        let p = world(size, Shape::InfinitePlane);
        let f = frame(&p, DVec3::new(12.5, 64_000.0, 15.5));
        let mut r = PlanetRenderer::new(
            &gpu.device,
            &gpu.queue,
            p.clone(),
            Settings {
                coarse_relief: true,
                far_relief: false,
                climate_height_reuse: true,
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
        assert!(
            r.settled(),
            "slope fixture failed to settle: {:?}",
            r.stats()
        );
        r.settings_mut().freeze_residency = true;
        target.render(&gpu, &mut r, &f, -Vec3::Y, 2001);
        let hs = hits(&gpu, &r);
        let ss = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
        let mut checked = 0;
        let mut wrong_if_rounded = 0;
        let mut diagnostics = std::collections::BTreeMap::new();
        for y in 18..36 {
            for x in 36..60 {
                let index = (x + y * 96) as usize;
                let h = hs[index];
                if h.status != 1 || !(6..=16).contains(&h.level) {
                    continue;
                }
                let (slope, rounded) = slopes(&p, h);
                *diagnostics
                    .entry((h.level, slope, rounded))
                    .or_insert(0usize) += 1;
                let expected = if slope >= 2 {
                    material::BRICK
                } else {
                    material::GRASS
                };
                let flags =
                    u32::from_le_bytes(ss[index * 16 + 12..index * 16 + 16].try_into().unwrap());
                let actual = (flags >> 8) & 255;
                assert_eq!(actual,expected,"material slope mismatch size{size} xy({x},{y}) cell{h:?} authored{slope} rounded{rounded}");
                checked += 1;
                if (slope >= 2) != (rounded >= 2) {
                    wrong_if_rounded += 1
                }
            }
        }
        eprintln!("material diagnostics size{size}: {diagnostics:?}");
        assert!(
            checked >= 24,
            "not enough material samples {checked} size{size}"
        );
        assert!(
            wrong_if_rounded >= 12,
            "fixture fails to expose quantization classification {wrong_if_rounded} size{size}"
        );
        eprintln!("stored material slope {size}m: {checked} field matches; {wrong_if_rounded} would change class from ceil tops");
    }
}
#[test]
fn authored_slope_oracle_distinguishes_coarse_rounding_without_gpu() {
    for size in [0.1, 0.3, 1.0] {
        let p = world(size, Shape::InfinitePlane);
        let origin = p.grid().cells() / 2;
        let mut differences = 0;
        for level in [6, 10, 14] {
            for x in 0..256 {
                let h = Hit {
                    t: 0.0,
                    i: (origin >> level) + x,
                    j: origin >> level,
                    k: 0,
                    status: 1,
                    face: 2,
                    level,
                    normal: 0,
                    steps: 0,
                };
                let (a, b) = slopes(&p, h);
                if (a >= 2) != (b >= 2) {
                    differences += 1
                }
            }
        }
        assert!(
            differences >= 24,
            "oracle doesn't expose enough threshold differences size{size} count{differences}"
        );
    }
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
fn authored_macro_normal(p: &Planet, h: Hit, eye: DVec3, dir: Vec3) -> Vec3 {
    let g = p.grid();
    let x = h.i & 7;
    let y = h.j & 7;
    let ci = h.i & !7;
    let cj = h.j & !7;
    let x0 = (x - 1).max(0);
    let x1 = (x + 1).min(7);
    let y0 = (y - 1).max(0);
    let y1 = (y + 1).min(7);
    let base = |i, j| {
        i64::from(
            p.field()
                .height(g.domain_point(h.face, i, j, h.level), h.level),
        )
        .div_euclid(i64::from(g.layer_mm()))
    };
    let step = (1u32 << h.level) as f64;
    let gi = (base(ci + x1, h.j) - base(ci + x0, h.j)) as f64 / (step * f64::from(x1 - x0));
    let gj = (base(h.i, cj + y1) - base(h.i, cj + y0)) as f64 / (step * f64::from(y1 - y0));
    let [n, a, b] = helio_pass_voxel_planet::grid::face_axes(h.face);
    let ai = g.angle(f64::from(h.i << h.level));
    let aj = g.angle(f64::from(h.j << h.level));
    let mi = a * ai.cos() - n * ai.sin();
    let mj = b * aj.cos() - n * aj.sin();
    let up = (eye + dir.as_dvec3() * f64::from(h.t)).normalize();
    (up - gi * mi - gj * mj).normalize().as_vec3()
}
#[test]
fn stored_sphere_normals_match_authored_macro_slopes_and_ignore_reuse_hint() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let target = Target::new(&gpu, [96, 54]);
    for size in [0.1, 0.3, 1.0] {
        let p = world(size, Shape::Sphere);
        for axis in [DVec3::Y, DVec3::new(1.0, 1.0, 0.23).normalize()] {
            let eye = axis * (p.grid().radius() + 64_000.0);
            let f = frame(&p, eye);
            let forward = -axis.as_vec3();
            let up = axis.as_vec3();
            let camera = target.camera(forward, up.any_orthonormal_vector());
            let inv = glam::Mat4::from_cols_array(&camera.inv_view_proj);
            let mut r = PlanetRenderer::new(
                &gpu.device,
                &gpu.queue,
                p.clone(),
                Settings {
                    coarse_relief: true,
                    lod_pixels: 4.0,
                    far_relief: false,
                    climate_height_reuse: true,
                    frame_override: Some(37),
                    ..Default::default()
                },
                target.size,
            );
            for n in 0..2000 {
                target.render(&gpu, &mut r, &f, forward, n);
                if r.settled() {
                    break;
                }
            }
            assert!(
                r.settled(),
                "sphere did not settle size{size} axis{axis:?}: {:?}",
                r.stats()
            );
            r.settings_mut().freeze_residency = true;
            target.render(&gpu, &mut r, &f, forward, 2001);
            let hs = hits(&gpu, &r);
            let hit_bytes = read_buffer(&gpu, r.hit_buffer(), 96 * 54 * 32);
            let before = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
            let mut checked = 0;
            let mut tilted = 0;
            let mut sphere_diag = std::collections::BTreeMap::new();
            for y in 18..36 {
                for x in 34..62 {
                    let index = (x + y * 96) as usize;
                    let h = hs[index];
                    if h.status != 1 || !(6..=16).contains(&h.level) {
                        continue;
                    }
                    *sphere_diag.entry((h.status, h.level)).or_insert(0usize) += 1;
                    let pixel = h.t * 2.0 * (target.fov_y * 0.5).tan() / 54.0;
                    let cell = size as f32 * (1u32 << h.level) as f32;
                    if ((2.5 - cell / pixel) / 1.5).clamp(0.0, 1.0) < 0.99999 {
                        continue;
                    }
                    let q = inv
                        * glam::Vec4::new(
                            (x as f32 + 0.5) / 96.0 * 2.0 - 1.0,
                            1.0 - (y as f32 + 0.5) / 54.0 * 2.0,
                            0.5,
                            1.0,
                        );
                    let dir = (q.truncate() / q.w).normalize();
                    let expected = authored_macro_normal(&p, h, eye, dir);
                    let actual = unpack_normal(u32::from_le_bytes(
                        before[index * 16 + 8..index * 16 + 12].try_into().unwrap(),
                    ));
                    let error = actual.dot(expected).clamp(-1.0, 1.0).acos();
                    assert!(error<0.004,"sphere normal error{error} size{size} axis{axis:?} hit{h:?} actual{actual:?} expected{expected:?}");
                    checked += 1;
                    let localup = (eye + dir.as_dvec3() * f64::from(h.t))
                        .normalize()
                        .as_vec3();
                    if expected.dot(localup) < 0.999 {
                        tilted += 1
                    }
                }
            }
            eprintln!("sphere diagnostics size{size}: {sphere_diag:?}");
            assert!(
                checked >= 24,
                "insufficient sphere macro normals size{size} axis{axis:?} checked{checked}"
            );
            if axis == DVec3::Y {
                assert!(
                    tilted >= 12,
                    "sphere fixture never recovered meaningful tilted relief: {tilted}"
                )
            }
            r.settings_mut().climate_height_reuse = false;
            target.render(&gpu, &mut r, &f, forward, 2002);
            assert_eq!(
                hit_bytes,
                read_buffer(&gpu, r.hit_buffer(), 96 * 54 * 32),
                "climate hint changed primary"
            );
            let after = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
            for (index, h) in hs.iter().enumerate() {
                if h.status == 1 {
                    assert_eq!(
                        &before[index * 16 + 8..index * 16 + 12],
                        &after[index * 16 + 8..index * 16 + 12],
                        "climate hint changed stored normal"
                    )
                }
            }
            eprintln!("sphere stored normals size{size} axis{axis:?}: {checked} CPU field matches, {tilted} tilted, reuse-hint byte parity");
        }
    }
}
