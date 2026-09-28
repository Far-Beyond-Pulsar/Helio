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

#[test]
fn beam_start_is_conservative() {
    let Some(gpu) = gpu() else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    for (face, fi, fj, pitch) in [(4u8, 0.37, 0.61, -0.35), (2, 0.47, 0.53, -0.1), (1, 0.52, 0.48, 0.05)] {
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
        r.settings_mut().beam = false;
        target.render(&gpu, &mut r, &f, forward, 5000);
        let reference = hits(&gpu, &r);
        r.settings_mut().beam = true;
        target.render(&gpu, &mut r, &f, forward, 5000);
        let beamed = hits(&gpu, &r);
        let mut bad = 0;
        for (index, (a, b)) in reference.iter().zip(&beamed).enumerate() {
            let both_miss = a.status == 0 && b.status == 0;
            if !both_miss && (a.status, a.i, a.j, a.k, a.face, a.level) != (b.status, b.i, b.j, b.k, b.face, b.level) {
                bad += 1;
                if bad < 6 {
                    eprintln!("pixel {} {}: no beam {a:?} beam {b:?}", index % 320, index / 320);
                }
            }
        }
        eprintln!("face {face}: {bad} beam differences");
        assert_eq!(bad, 0);
    }
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
            let v: Vec<i32> = table.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
            let eye_layer = ((eye.length() - planet.grid().radius()) / planet.grid().voxel_size()) as i64;
            for b in 0..32 {
                let row = &v[b * 256..b * 256 + 256];
                eprintln!("bucket {b:2}: min {:9} max {:9} all {:9} (rel eye m: {:.0} {:.0})", row.iter().min().unwrap(), row.iter().max().unwrap(), v[256 * 32 + b],
                    (*row.iter().min().unwrap() as i64 - eye_layer) as f64 * planet.grid().voxel_size(), (*row.iter().max().unwrap() as i64 - eye_layer) as f64 * planet.grid().voxel_size());
            }
        }
        for beam in [false, true] {
            r.settings_mut().beam = beam;
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
            eprintln!("face {face} pitch {pitch} beam {beam}: {bad} differences, {hits_seen} hits, {saved} steps saved");
            assert_eq!(bad, 0);
        }
    }
}
