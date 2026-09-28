//! GPU agreement tests: the WGSL field, GPU-generated residency and exact
//! traversal against the CPU canonical planet.
mod common;
use common::*;
use glam::{DVec3, IVec4, Vec3};
use helio_pass_voxel_planet::field::{self, FieldConstants};
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Cell, Planet, PlanetRecipe};
use std::sync::Arc;
use wgpu::util::DeviceExt;

fn field_kernel(gpu: &Gpu, consts: &FieldConstants, inputs: &[IVec4], extra: &[IVec4]) -> Vec<[i32; 4]> {
    let src = format!(
        "{}\n@group(0) @binding(1) var<uniform> field: FieldConstants;
@group(0) @binding(2) var<storage, read> inputs: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> extra: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read_write> outputs: array<vec4<i32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= arrayLength(&inputs) {{ return; }}
    let a = inputs[id.x];
    let e = extra[id.x];
    let p = domain_point(u32(a.x), a.y, a.z, u32(a.w));
    let h = terrain_height(p, u32(a.w));
    let m = ground_material(p, e.x, e.y, e.z, e.w);
    outputs[id.x] = vec4<i32>(h, i32(m), p.x ^ p.y ^ p.z, noise(p, 7u, 99u) ^ noise(p, 19u + (u32(a.w) & 7u), 5u));
}}",
        include_str!("../shaders/field.wgsl")
    );
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let uniform = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(consts),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let ins = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ext = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(extra),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (inputs.len() * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ins.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ext.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let data = read_buffer(gpu, &out, (inputs.len() * 16) as u64);
    bytemuck::cast_slice::<u8, [i32; 4]>(&data).to_vec()
}

#[test]
fn wgsl_field_is_bit_identical_to_cpu() {
    let Some(gpu) = gpu() else { return };
    for size in [0.1, 0.3, 1.0] {
        let planet = Planet::new(PlanetRecipe { voxel_size_m: size, ..Default::default() }).unwrap();
        let grid = *planet.grid();
        let n = grid.cells();
        let mut inputs = Vec::new();
        let mut extra = Vec::new();
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for s in 0..20_000 {
            let level = (next() % u64::from(grid.levels())) as u32;
            let cells = n >> level;
            let face = (next() % 6) as i32;
            let (i, j) = if s % 4 == 0 {
                // Face edges and corners.
                ((next() % 2) as i32 * (cells - 1), (next() % u64::from(cells as u32)) as i32)
            } else {
                ((next() % u64::from(cells as u32)) as i32, (next() % u64::from(cells as u32)) as i32)
            };
            inputs.push(IVec4::new(face, i, j, level as i32));
            extra.push(IVec4::new(
                (next() % 8_000_000) as i32 - 3_000_000,
                (next() % 40) as i32,
                (next() % 4) as i32,
                (next() % 200_000) as i32 - 100_000,
            ));
        }
        let out = field_kernel(&gpu, planet.field(), &inputs, &extra);
        for ((input, e), gpu_out) in inputs.iter().zip(&extra).zip(out) {
            let p = grid.domain_point(input.x as u8, input.y, input.z, input.w as u32);
            let h = field::height(planet.field(), p, input.w as u32);
            let m = field::ground_material(planet.field(), p, e.x, e.y, e.z, e.w);
            assert_eq!(gpu_out, [h, m as i32, p.x ^ p.y ^ p.z, field::noise(p, 7, 99) ^ field::noise(p, 19 + (input.w as u32 & 7), 5)], "{input} {e}");
        }
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
    let frame = frame(planet, eye);
    let frames = settle(gpu, &target, &mut renderer, &frame, forward);
    let stats = renderer.stats();
    eprintln!("settled after {frames} frames: {stats:?}");
    assert_eq!(stats.failed_jobs, 0);
    let hits = hits(gpu, &renderer);
    let up = eye.normalize().as_vec3();
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
    let grid = *planet.grid();
    let dir = land(&planet, 4, 0.37, 0.61);
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let forward = (up.any_orthonormal_vector() - up * 0.35).normalize().as_vec3();
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [320, 180]);
    eprintln!("compared {compared}, mismatched {mismatched}");
    assert!(compared > 1000);
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

#[test]
fn edits_propagate_to_gpu_generation() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *planet.grid();
    let dir = land(&planet, 1, 0.52, 0.48);
    let eye = planet.surface_point(dir, 3.0);
    let up = eye.normalize();
    let side = up.any_orthonormal_vector();
    let ground = planet.surface_point(dir, 0.0) + side * 4.0;
    planet
        .apply(Brush { center: ground.to_array(), radius: 2.5, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0 })
        .unwrap();
    planet
        .apply(Brush { center: (ground + side * 3.0 + up * 1.5).to_array(), radius: 0.8, shape: BrushShape::Cube, op: BrushOp::Add, material: field::material::BRICK })
        .unwrap();
    let planet = Arc::new(planet);
    let forward = (side - up * 0.6).normalize().as_vec3();
    let (compared, mismatched) = compare_near(&gpu, &planet, eye, forward, [256, 144]);
    eprintln!("compared {compared}, mismatched {mismatched}");
    assert!(compared > 1000);
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
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
        for (index, (a, b)) in reference.iter().zip(&bounded).enumerate() {
            compared += 1;
            let both_miss = a.status == 0 && b.status == 0;
            if !both_miss && (a.status, a.i, a.j, a.k, a.face, a.level) != (b.status, b.i, b.j, b.k, b.face, b.level) {
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
        let header = if ext { 2 } else { 1 };
        let published = (k_lo + n_band) * 8 - gap;
        // Highest occupied cell from the brick masks.
        let mut highest = i32::MIN;
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
        if highest >= published {
            bad += 1;
            if bad < 6 {
                eprintln!("column key {:08x} {:08x}: highest occupied {highest} published top {published} (k_lo {k_lo} band {n_band} gap {gap})", c[0], c[1]);
            }
        }
    }
    eprintln!("{columns} columns, {bad} with occupied cells above the published top");
    assert!(columns > 1000);
    assert_eq!(bad, 0);
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
            // Vertical component of the pixel ray: a height difference of
            // one cell moves a grazing hit far along the ray.
            let rise = pixel_dir(&target, &camera, index as u32 % 320, index as u32 / 320).dot(up).abs().max(1e-3) as f32;
            // A different start can pick a different dithered level for the
            // same surface; the geometry must still agree.
            // Skipped geometry shows as a farther hit or a wrong miss. A ray
            // that starts later may choose another level for the same surface
            // (the dither, or a partial block falling back): its hit may then
            // differ by up to a coarse cell in height.
            let differs = if a.status != b.status {
                true
            } else if a.status != 1 {
                false
            } else if a.level == b.level {
                // Rays grazing a voxel edge within float precision may
                // resolve to a neighbour at the same distance.
                (a.i, a.j, a.k, a.face) != (b.i, b.j, b.k, b.face)
                    && (a.t - b.t).abs() > 0.25 * voxel * (1u32 << a.level) as f32
            } else {
                (b.t - a.t) * rise > 2.0 * voxel * (1u32 << a.level.max(b.level)) as f32
            };
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
/// which the per-level fallback distances must account for): hits match a
/// render without it. (Distances may differ in the last bits where a ray
/// starts later, so cells are compared.)
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
        for (a, b) in reference.iter().zip(&hinted) {
            compared += 1;
            let both_miss = a.status == 0 && b.status == 0;
            if !both_miss && (a.status, a.i, a.j, a.k, a.face, a.level) != (b.status, b.i, b.j, b.k, b.face, b.level) {
                bad += 1;
            }
        }
        if step % 8 == 0 {
            eprintln!("step {step}: altitude {altitude:.0} m pending {}", r.stats().pending_columns);
        }
    }
    eprintln!("{bad} differences in {compared} pixels");
    assert_eq!(bad, 0);
}
