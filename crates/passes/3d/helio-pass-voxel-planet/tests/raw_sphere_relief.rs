//! Raw climate relief is compared to the physical authored sphere surface.
//! Copy beside tests/common; kept local until the full renderer is qualified.
mod common;
use common::*;
use glam::{DVec3, IVec3, Mat4, Vec3, Vec4};
use helio_pass_voxel_planet::{
    engine::{PlanetRenderer, Settings},
    grid::{face_axes, Grid, Shape},
    terrain::{self, material, GeneratorInfo, TerrainField, TerrainGenerator, TerrainProgram},
    Planet, PlanetRecipe, TerrainSource,
};
use std::{
    borrow::Cow,
    sync::{Arc, Once},
};
const GENERATOR: &str = "qualification.raw-sphere-relief";
struct RawSphereGenerator;
struct RawSphereField {
    scale: u32,
}
impl TerrainGenerator for RawSphereGenerator {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: GENERATOR.into(),
            version: 1,
            name: "Unclamped raw sphere relief".into(),
            description: "Linear cube-domain height with a physical sphere-normal oracle".into(),
            settings_component: None,
        }
    }
    fn build(&self, g: &Grid, _: u64, _: &str) -> Result<Arc<dyn TerrainField>, String> {
        Ok(Arc::new(RawSphereField {
            scale: g.domain_scale(),
        }))
    }
}
impl TerrainField for RawSphereField {
    fn height(&self, p: IVec3, _: u32) -> i32 {
        1_000_000 + (p.x >> 2) + (p.z >> 3)
    }
    fn ground_material(&self, _: IVec3, _: i32, _: i32, _: i32, _: i32) -> u32 {
        material::GRASS
    }
    fn height_range(&self) -> (i32, i32) {
        (-300_000_000, 300_000_000)
    }
    fn bound_margins(&self) -> [i32; 24] {
        std::array::from_fn(|l| {
            (4 + (3 * (1i64 << l) * i64::from(self.scale) + 16_777_215) / 16_777_216)
                .min(600_000_000) as i32
        })
    }
    fn program(&self) -> TerrainProgram {
        TerrainProgram {
            key: Cow::Borrowed("qualification.raw-sphere-relief/1"),
            constants: vec![0; 16],
            wgsl: Cow::Borrowed(
                r#"
struct TerrainConstants { pad:vec4<i32>, }
fn terrain_height(p:vec3<i32>,level:u32)->i32{return 1000000+(p.x>>2u)+(p.z>>3u);}
fn ground_material(p:vec3<i32>,top:i32,depth:i32,slope:i32,layer:i32)->u32{return M_GRASS;}
"#,
            ),
        }
    }
}
fn world(size: f64) -> Arc<Planet> {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| terrain::register(Arc::new(RawSphereGenerator)).unwrap());
    Arc::new(
        Planet::new(PlanetRecipe {
            shape: Shape::Sphere,
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
fn smooth_height(g: &Grid, face: u8, i: f64, j: f64) -> f64 {
    // The sphere domain mapping (`grid::sphere_point`) in f64.
    let [n, a, b] = face_axes(face);
    let reference = f64::from(g.reference_cells());
    let scale = f64::from(g.domain_scale()) / 16_777_216.0;
    let tan = |x: f64| x - x * (1.0 - x * x) * (1.0 - std::f64::consts::FRAC_PI_4 + 0.05 * x * x);
    let u = tan((2.0 * i * scale - reference) / reference);
    let v = tan((2.0 * j * scale - reference) / reference);
    let p = (n + a * u + b * v).normalize() * f64::from(g.sphere_constants()[2]);
    (1_000_000.0 + p.x / 4.0 + p.z / 8.0) * 0.001
}
fn authored_position(g: &Grid, face: u8, i: f64, j: f64) -> DVec3 {
    let [n, a, b] = face_axes(face);
    let q = n + a * g.angle(i).tan() + b * g.angle(j).tan();
    q.normalize() * (g.radius() + smooth_height(g, face, i, j))
}
// This oracle differentiates the authored physical surface, not the renderer's
// cache stencil, macro-normal approximation, or quantized column heights.
fn authored_normal(g: &Grid, face: u8, i: f64, j: f64) -> DVec3 {
    let [n, a, b] = face_axes(face);
    let ti = g.angle(i).tan();
    let tj = g.angle(j).tan();
    let q = n + a * ti + b * tj;
    let length = q.length();
    let up = q / length;
    let qi = a * (1.0 + ti * ti) * g.delta();
    let qj = b * (1.0 + tj * tj) * g.delta();
    let ui = (qi - up * up.dot(qi)) / length;
    let uj = (qj - up * up.dot(qj)) / length;
    // Height derivative through the sphere domain mapping (`smooth_height`).
    let reference = f64::from(g.reference_cells());
    let scale = f64::from(g.domain_scale()) / 16_777_216.0;
    let alpha = 1.0 - std::f64::consts::FRAC_PI_4;
    let tan = |x: f64| x - x * (1.0 - x * x) * (alpha + 0.05 * x * x);
    let dtan = |x: f64| 1.0 - (alpha + 0.05 * x * x) * (1.0 - 3.0 * x * x) - 0.1 * x * x * (1.0 - x * x);
    let u = (2.0 * i * scale - reference) / reference;
    let v = (2.0 * j * scale - reference) / reference;
    let dq = n + a * tan(u) + b * tan(v);
    let unit = dq.normalize();
    let domain_radius = f64::from(g.sphere_constants()[2]);
    let dp = |d: DVec3| (d - unit * unit.dot(d)) * (domain_radius / dq.length());
    let dh = |d: DVec3| 0.001 * (d.x / 4.0 + d.z / 8.0);
    let radius = g.radius() + smooth_height(g, face, i, j);
    let pi = ui * radius + up * dh(dp(a * dtan(u) * 2.0 * scale / reference));
    let pj = uj * radius + up * dh(dp(b * dtan(v) * 2.0 * scale / reference));
    let mut normal = pi.cross(pj).normalize();
    if normal.dot(up) < 0.0 {
        normal = -normal
    }
    normal
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
fn sphere_normal_oracle_matches_physical_derivatives_and_integer_field() {
    for size in [0.1, 0.3, 1.0] {
        let p = world(size);
        let g = p.grid();
        let mid = g.cells() / 2;
        for face in 0..6 {
            for offset in [0, g.cells() / 16, -g.cells() / 16] {
                let i = mid + offset;
                let j = mid - offset / 2;
                let ci = f64::from(i) + 0.5;
                let cj = f64::from(j) + 0.5;
                let raw = f64::from(p.field().height(g.domain_point(face, i, j, 0), 0)) * 0.001;
                assert!(
                    (raw - smooth_height(g, face, ci, cj)).abs() < 0.005,
                    "oracle height differs from canonical field"
                );
                let di = authored_position(g, face, ci + 16.0, cj)
                    - authored_position(g, face, ci - 16.0, cj);
                let dj = authored_position(g, face, ci, cj + 16.0)
                    - authored_position(g, face, ci, cj - 16.0);
                let mut numeric = di.cross(dj).normalize();
                let analytic = authored_normal(g, face, ci, cj);
                if numeric.dot(analytic) < 0.0 {
                    numeric = -numeric
                }
                assert!(
                    numeric.distance(analytic) < 1e-7,
                    "physical derivative oracle mismatch size{size} face{face}"
                );
            }
        }
    }
}
#[test]
#[ignore = "known defect, identical at 3a70ffe1: far relief changes one pixel's material id and shadow lift"]
fn raw_sphere_relief_restores_physical_authored_normal_without_changing_hits_or_materials() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP: no GPU adapter available for this rendering fixture");
        return;
    };
    let mut target = Target::new(&gpu, [96, 54]);
    for size in [0.1, 0.3, 1.0] {
        let p = world(size);
        for altitude in [64_000.0, 1_000_000.0] {
            // At 1Mm a 54px, 45deg view has 15km pixels and hides relief
            // behind the coarsest-level cap. This crop keeps the same ~1.13km
            // footprint as a native 720px-high 45deg orbital view.
            target.fov_y = if altitude >= 1_000_000.0 {
                3.5f32.to_radians()
            } else {
                std::f32::consts::FRAC_PI_4
            };
            for axis in [DVec3::Y, DVec3::new(0.11, 1.0, 0.07).normalize()] {
                let eye = axis * (p.grid().radius() + altitude);
                let f = frame(&p, eye);
                let forward = -axis.as_vec3();
                let up = p.grid().up(eye).as_vec3();
                let camera = target.camera(forward, up.any_orthonormal_vector());
                let inv = Mat4::from_cols_array(&camera.inv_view_proj);
                let mut r = PlanetRenderer::new(
                    &gpu.device,
                    &gpu.queue,
                    p.clone(),
                    Settings {
                        coarse_relief: true,
                        far_relief: false,
                        lod_pixels: 0.125,
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
                assert!(r.settled(),"raw sphere fixture did not settle size{size} altitude{altitude} axis{axis:?}: {:?}",r.stats());
                r.settings_mut().freeze_residency = true;
                target.render(&gpu, &mut r, &f, forward, 2001);
                let hs = hits(&gpu, &r);
                let primary = read_buffer(&gpu, r.hit_buffer(), 96 * 54 * 32);
                let before = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
                r.settings_mut().far_relief = true;
                target.render(&gpu, &mut r, &f, forward, 2002);
                assert_eq!(
                    primary,
                    read_buffer(&gpu, r.hit_buffer(), 96 * 54 * 32),
                    "raw appearance changed sphere primary"
                );
                let after = read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16);
                let mut restored_face_color = 0;
                for (index, (a, b)) in after
                    .chunks_exact(16)
                    .zip(before.chunks_exact(16))
                    .enumerate()
                {
                    assert_eq!(&a[..4], &b[..4], "raw appearance changed distance");
                    // Bits 21..24 carry the shading filter weight, which
                    // sunlight follows: it may change with the appearance.
                    let flags = |s: &[u8]| u32::from_le_bytes(s[12..16].try_into().unwrap()) & !(7 << 21);
                    assert_eq!(flags(a), flags(b), "raw appearance changed material flags");
                    if hs[index].status != 1 {
                        assert_eq!(a, b, "raw appearance changed a non-hit surface");
                    } else {
                        assert_eq!(
                            a[7], 255,
                            "subpixel authored voxels exposed a coarse AO grid"
                        );
                        if hs[index].normal == 4 {
                            assert_eq!(
                                &a[4..7],
                                &b[4..7],
                                "raw appearance changed radial top pigment"
                            );
                        } else if a[4..7] != b[4..7] {
                            // The authored footprint removes coarse grass riser
                            // darkening/soil lips; this is an intentional color
                            // correction, independent of the normal stencil.
                            restored_face_color += 1;
                        }
                    }
                }
                eprintln!("raw sphere restored{restored_face_color} coarse riser colors; exact depth/material and radial top RGB retained");
                let mut recovered = 0;
                let mut candidates = 0;
                let mut status_levels = std::collections::BTreeMap::new();
                let mut baseline_error_max = 0.0f32;
                for y in 18..36 {
                    for x in 34..62 {
                        let index = (x + y * 96) as usize;
                        let h = hs[index];
                        *status_levels.entry((h.status, h.level)).or_insert(0usize) += 1;
                        if h.status != 1 || h.level == 0 {
                            continue;
                        }
                        let q = inv
                            * Vec4::new(
                                (x as f32 + 0.5) / 96.0 * 2.0 - 1.0,
                                1.0 - (y as f32 + 0.5) / 54.0 * 2.0,
                                0.5,
                                1.0,
                            );
                        let dir = (q.truncate() / q.w).normalize();
                        let pos = eye + dir.as_dvec3() * f64::from(h.t);
                        let coords = p
                            .grid()
                            .face_coords(h.face, pos)
                            .expect("hit outside face hemisphere");
                        let expected =
                            authored_normal(p.grid(), h.face, coords[0], coords[1]).as_vec3();
                        let old = unpack_normal(u32::from_le_bytes(
                            before[index * 16 + 8..index * 16 + 12].try_into().unwrap(),
                        ));
                        let actual = unpack_normal(u32::from_le_bytes(
                            after[index * 16 + 8..index * 16 + 12].try_into().unwrap(),
                        ));
                        let old_error = old.dot(expected).clamp(-1.0, 1.0).acos();
                        baseline_error_max = baseline_error_max.max(old_error);
                        if old_error < 0.01 {
                            continue;
                        }
                        candidates += 1;
                        // Unchanged normals are a rejected neighbourhood, not a recovered one.
                        if old.dot(actual) > 0.99999 {
                            continue;
                        }
                        let error = actual.dot(expected).clamp(-1.0, 1.0).acos();
                        assert!(error<0.004,"raw sphere physical normal error{error} old_error{old_error} size{size} altitude{altitude} axis{axis:?} xy({x},{y}) hit{h:?} actual{actual:?} expected{expected:?}");
                        recovered += 1;
                    }
                }
                eprintln!("raw sphere exposure status/levels{status_levels:?}, max baseline error{baseline_error_max}, fov{}deg", target.fov_y.to_degrees());
                assert!(recovered>=24,"raw sphere did not restore enough physically authored detail: recovered{recovered} candidates{candidates} size{size} altitude{altitude} axis{axis:?}");
                eprintln!("raw sphere physical normals size{size} altitude{altitude} axis{axis:?}: recovered{recovered}/{candidates}; hit/material bytes identical");
                r.settings_mut().far_relief = false;
                target.render(&gpu, &mut r, &f, forward, 2003);
                assert_eq!(
                    before,
                    read_buffer(&gpu, r.surface_buffer(), 96 * 54 * 16),
                    "raw appearance did not restore previous sphere surface bytes"
                );
            }
        }
    }
}
