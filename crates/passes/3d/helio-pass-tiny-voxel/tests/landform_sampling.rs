//! Qualification of immutable data and integer sampling, not the final terrain recipe.
use helio_pass_tiny_voxel::landforms::{
    GlobalTopology, LandformRecipe, LandformSnapshot, SAMPLING_SHADER,
};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use wgpu::util::DeviceExt;

const INVALID: [u32; 8] = [u32::MAX, 0, 0, 0, 0, 0, 0, 0];

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn save(root: &Path, name: &str, bytes: &[u8]) -> serde_json::Value {
    std::fs::write(root.join(name), bytes).unwrap();
    serde_json::json!({"file":name,"bytes":bytes.len(),"sha256":hash(bytes)})
}

fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^ (h >> 16)
}

fn random(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_add(0x9e3779b9);
    mix(*seed)
}

fn recipe(backend_revision: u32) -> LandformRecipe {
    LandformRecipe {
        revision: 1,
        seed: 419,
        iterations: 32,
        backend_revision,
        rain_scale: 1.0,
        erosion_rate: 0.0005,
    }
}

// Independent oracle: i128/u128 direct division and weighted interpolation,
// rather than the library's i64 delta/shift or GPU's emulated wide integers.
fn oracle(snapshot: &LandformSnapshot, cell: [i32; 3]) -> [u32; 8] {
    if cell.iter().any(|c| i64::from(*c).abs() > 100_000_000) {
        return INVALID;
    }
    let p = cell.map(|c| i128::from(c) * 2 + 1);
    let a = p.map(i128::abs);
    let axis = if a[0] >= a[1] && a[0] >= a[2] {
        0
    } else if a[1] >= a[2] {
        1
    } else {
        2
    };
    let face = 2 * axis + usize::from(p[axis] < 0);
    let components = match axis {
        0 => [p[1], p[2]],
        1 => [p[0], p[2]],
        _ => [p[0], p[1]],
    };
    let n = i128::from(snapshot.resolution());
    let denominator = a[axis] * 2;
    let uv = components.map(|c| {
        let numerator = (c + a[axis]) * n;
        let integer = (numerator / denominator).min(n - 1);
        [
            integer,
            ((numerator - integer * denominator) * 65536) / denominator,
        ]
    });
    let side = snapshot.resolution() as usize + 1;
    let i = face * side * side + uv[1][0] as usize * side + uv[0][0] as usize;
    let h = snapshot.height_atlas();
    let lerp = |a: i128, b: i128, t: i128| (a * (65536 - t) + b * t).div_euclid(65536);
    let row0 = lerp(i128::from(h[i]), i128::from(h[i + 1]), uv[0][1]);
    let row1 = lerp(
        i128::from(h[i + side]),
        i128::from(h[i + side + 1]),
        uv[0][1],
    );
    let height = lerp(row0, row1, uv[1][1]);
    let radius = (127420000 + height) as u128;
    let squared = p.iter().map(|x| x.unsigned_abs().pow(2)).sum::<u128>();
    [
        face as u32,
        uv[0][0] as u32,
        uv[1][0] as u32,
        uv[0][1] as u32,
        uv[1][1] as u32,
        height as i32 as u32,
        u32::from(squared <= radius * radius),
        1,
    ]
}

fn public_cpu(snapshot: &LandformSnapshot, c: [i32; 3]) -> [u32; 8] {
    let Some(s) = snapshot.sample_cell(c) else {
        return INVALID;
    };
    [
        s.address.face,
        s.address.u,
        s.address.v,
        s.address.fraction[0],
        s.address.fraction[1],
        s.height_units as u32,
        u32::from(s.solid),
        1,
    ]
}

fn positions(snapshot: &LandformSnapshot) -> Vec<[i32; 4]> {
    let mut result = Vec::with_capacity(65536);
    // Exhaust every sign of face/corner ties, including cell centers +/-1.
    for k in [0, 1, 15, 63, 255, 65535, 36_782_985, 63_709_999, 99_999_999] {
        for mask in 0..8 {
            let mut c = [k; 3];
            for axis in 0..3 {
                if mask & (1 << axis) != 0 {
                    c[axis] = -c[axis] - 1;
                }
            }
            result.push([c[0], c[1], c[2], 0]);
        }
    }
    for axis in 0..3 {
        for k in [
            i32::MIN,
            -100_000_001,
            -100_000_000,
            100_000_000,
            100_000_001,
            i32::MAX,
        ] {
            let mut c = [0; 4];
            c[axis] = k;
            result.push(c);
        }
    }
    let mut seed = 0x517e419;
    while result.len() < 65536 {
        let mode = result.len() % 16;
        let mut c = [0; 3];
        if mode < 8 {
            let mut d = [0.0; 3];
            for x in &mut d {
                *x = f64::from(random(&mut seed) % 200001) - 100000.0;
            }
            if d == [0.0; 3] {
                d[0] = 1.0;
            }
            let length = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            d = d.map(|x| x / length);
            let mut radius = 127420000.0;
            // Position generation alone uses floats; the oracle never does.
            for _ in 0..8 {
                c = d.map(|x| ((x * radius - 1.0) * 0.5).floor() as i32);
                radius = 127420000.0 + f64::from(oracle(snapshot, c)[5] as i32);
            }
            let offset = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0][mode];
            c = d.map(|x| ((x * (radius + offset) - 1.0) * 0.5).floor() as i32);
        } else if mode < 12 {
            c = c.map(|_| (random(&mut seed) % 200000001) as i32 - 100000000);
        } else if mode < 14 {
            let k = (random(&mut seed) % 99999999) as i32;
            c = [k, k, k];
            for x in &mut c {
                if random(&mut seed) & 1 != 0 {
                    *x = -*x - 1;
                }
            }
            let axis = (random(&mut seed) % 3) as usize;
            c[axis] += (random(&mut seed) % 3) as i32 - 1;
        } else if mode == 14 {
            c = c.map(|_| match random(&mut seed) % 6 {
                0 => -100000000,
                1 => 100000000,
                2 => -1,
                3 => 0,
                4 => -63710000,
                _ => 63709999,
            });
        } else {
            // Probe immediately around projection grid boundaries on long axes.
            let dominant = i64::from((random(&mut seed) % 30000000 + 60000000) as i32) * 2 + 1;
            let n = i64::from(snapshot.resolution());
            let projected = |r: u32| {
                let t = i64::from(r % (snapshot.resolution() + 1));
                let center = dominant * (2 * t - n) / n;
                ((center - 1).div_euclid(2)).clamp(-100000000, 100000000) as i32
            };
            c = [
                ((dominant - 1) / 2) as i32,
                projected(random(&mut seed)),
                projected(random(&mut seed)),
            ];
            c[1] += (random(&mut seed) % 3) as i32 - 1;
            c.rotate_left((random(&mut seed) % 3) as usize);
            for x in &mut c {
                if random(&mut seed) & 1 != 0 {
                    *x = -*x - 1;
                }
            }
        }
        result.push([c[0], c[1], c[2], 0]);
    }
    result
}

struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}
impl Device {
    fn new() -> (Self, String) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let info = format!("{:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let source = format!(
            "{}\n{}",
            SAMPLING_SHADER,
            r#"
@group(0) @binding(2) var<storage,read> sample_cells: array<vec4<i32>>;
@group(0) @binding(3) var<storage,read_write> samples: array<LFCellSample>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x<arrayLength(&sample_cells) { samples[id.x]=lf_sample_cell(sample_cells[id.x].xyz); }
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exact landform sampler"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("exact landform sampler"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (
            Self {
                device,
                queue,
                pipeline,
            },
            info,
        )
    }

    fn run(&self, snapshot: &LandformSnapshot, cells: &[[i32; 4]]) -> Vec<[u32; 8]> {
        let d = &self.device;
        let settings = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[snapshot.resolution(), 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let heights = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(snapshot.height_atlas()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let input = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(cells),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (cells.len() * 32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let buffers = [&settings, &heights, &input, &output];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = d.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((cells.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let values = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
        readback.unmap();
        values
    }
}

fn publication(snapshot: &LandformSnapshot, root: &Path, name: &str) -> serde_json::Value {
    let directory = root.join("published");
    let paths = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| snapshot.publish(&directory).unwrap()))
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(paths.iter().all(|p| p == &paths[0]));
    let loaded = LandformSnapshot::load(&paths[0], snapshot.id()).unwrap();
    assert_eq!(loaded.encode(), snapshot.encode());
    assert!(std::fs::read_dir(&directory).unwrap().all(|p| !p
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
    let bad_directory = root.join(format!("corrupt-{name}"));
    std::fs::create_dir(&bad_directory).unwrap();
    let corrupt_path = bad_directory.join(format!("{}.hlfm", snapshot.id()));
    let bad = b"corrupt existing identifier must not be replaced";
    std::fs::write(&corrupt_path, bad).unwrap();
    assert!(snapshot.publish(&bad_directory).is_err());
    assert_eq!(std::fs::read(&corrupt_path).unwrap(), bad);
    assert_eq!(std::fs::read_dir(&bad_directory).unwrap().count(), 1);
    serde_json::json!({"id":snapshot.id().to_string(),"file":paths[0],"bytes":snapshot.encode().len(),
        "concurrent_publishers":8,"roundtrip":true,"corrupt_existing_rejected_and_preserved":true})
}

#[test]
#[ignore = "GPU publication/sampling qualification; requires frozen HELIO_LANDFORM_SAMPLING protocol"]
fn immutable_landforms_exact_gpu_cell_sampling() {
    let root = PathBuf::from(
        std::env::var_os("HELIO_LANDFORM_SAMPLING").expect("HELIO_LANDFORM_SAMPLING"),
    );
    assert!(root.join("protocol.json").is_file());
    let data = root.join("data");
    std::fs::create_dir(&data).expect("fresh evidence directory required");
    let started = Instant::now();
    let (device, info) = Device::new();
    let mut report = serde_json::json!({"device":info,"device_setup_ms":started.elapsed().as_secs_f64()*1000.0,
        "scope":"Immutable radial base data and exact sampled cell decisions only; no renderer performance or visual qualification.","cases":[]});
    for (n, real) in [
        (1, false),
        (16, false),
        (64, false),
        (256, false),
        (64, true),
    ] {
        let topology = GlobalTopology::new(n).unwrap();
        let name = format!("{n}-{}", if real { "eroded" } else { "synthetic" });
        let mut generation_input = serde_json::Value::Null;
        let snapshot = if real {
            let path = root
                .parent()
                .unwrap()
                .join("global-landforms/data/64-erosion-gpu.bin");
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(bytes.len(), topology.nodes().len() * 8);
            generation_input =
                serde_json::json!({"path":path,"sha256":hash(&bytes),"bytes":bytes.len()});
            let state: Vec<[f32; 2]> = bytes
                .chunks_exact(8)
                .map(|s| {
                    [
                        f32::from_le_bytes(s[..4].try_into().unwrap()),
                        f32::from_le_bytes(s[4..].try_into().unwrap()),
                    ]
                })
                .collect();
            LandformSnapshot::from_state(&topology, recipe(2), &state).unwrap()
        } else {
            let mut heights: Vec<_> = topology
                .keys()
                .iter()
                .map(|k| {
                    let key = (k.0[0] as u32).wrapping_mul(0x8da6b343)
                        ^ (k.0[1] as u32).wrapping_mul(0xd8163841)
                        ^ (k.0[2] as u32).wrapping_mul(0xcb1ab31f)
                        ^ 419;
                    (mix(key) % 80000001) as i32 - 40000000
                })
                .collect();
            heights[0] = -40000000;
            heights[1] = 40000000;
            let streams: Vec<_> = (0..heights.len())
                .map(|i| (mix(i as u32) & 65535) as f32 * 0.125)
                .collect();
            LandformSnapshot::from_units(&topology, recipe(1), &heights, &streams).unwrap()
        };
        let published = publication(&snapshot, &data, &name);
        let cells = positions(&snapshot);
        let reference: Vec<_> = cells
            .iter()
            .map(|c| oracle(&snapshot, [c[0], c[1], c[2]]))
            .collect();
        let cpu: Vec<_> = cells
            .iter()
            .map(|c| public_cpu(&snapshot, [c[0], c[1], c[2]]))
            .collect();
        let started = Instant::now();
        let gpu = device.run(&snapshot, &cells);
        let diagnostic_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cpu_errors = reference.iter().zip(&cpu).filter(|(a, b)| a != b).count();
        let gpu_errors = reference.iter().zip(&gpu).filter(|(a, b)| a != b).count();
        let mut errors_by_word = [0usize; 8];
        for (a, b) in reference.iter().zip(&gpu) {
            for i in 0..8 {
                errors_by_word[i] += usize::from(a[i] != b[i]);
            }
        }
        let examples:Vec<_> = reference.iter().zip(&gpu).enumerate().filter(|(_, (a,b))|a!=b).take(8)
            .map(|(i,(a,b))|serde_json::json!({"index":i,"cell":cells[i],"expected":a,"actual":b})).collect();
        let artifacts: Vec<_> = [
            ("input", bytemuck::cast_slice(&cells)),
            ("oracle", bytemuck::cast_slice(&reference)),
            ("cpu", bytemuck::cast_slice(&cpu)),
            ("gpu", bytemuck::cast_slice(&gpu)),
        ]
        .into_iter()
        .map(|(kind, bytes)| save(&data, &format!("{name}-{kind}.bin"), bytes))
        .collect();
        let mut faces = [0usize; 6];
        for r in &reference {
            if r[7] != 0 {
                faces[r[0] as usize] += 1;
            }
        }
        let solid = reference.iter().filter(|r| r[6] != 0).count();
        let invalid = reference.iter().filter(|r| r[7] == 0).count();
        let result = serde_json::json!({"name":name,"resolution":n,"records":cells.len(),"published":published,
            "generation_input":generation_input,"cpu_errors":cpu_errors,"gpu_errors":gpu_errors,
            "gpu_errors_by_word":errors_by_word,"mismatch_examples":examples,"face_counts":faces,
            "solid":solid,"air":cells.len()-invalid-solid,"invalid":invalid,"artifacts":artifacts,
            "diagnostic_upload_dispatch_readback_wall_ms":diagnostic_wall_ms});
        println!("SAMPLING_CASE {name}: records={} cpu_errors={cpu_errors} gpu_errors={gpu_errors} solid={solid} invalid={invalid}",cells.len());
        report["cases"].as_array_mut().unwrap().push(result);
        std::fs::write(
            root.join("results.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        assert_eq!(
            cpu_errors, 0,
            "CPU disagrees with independent integer oracle"
        );
        assert_eq!(
            gpu_errors, 0,
            "GPU disagrees with independent integer oracle"
        );
        assert!(faces.iter().all(|n| *n > 1000));
        assert!(solid > 1000 && cells.len() - invalid - solid > 1000 && invalid > 0);
    }
    report["completed"] = true.into();
    std::fs::write(
        root.join("results.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
