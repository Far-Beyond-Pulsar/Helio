//! Certificates for the new radial field; no old-generator equivalence gate.
use helio_pass_tiny_voxel::landforms::{
    BoundedLandforms, LandformSnapshot, SnapshotId, BOUNDS_SHADER, SAMPLING_SHADER,
};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Region {
    minimum: [i32; 4],
    maximum: [i32; 4],
}
impl Region {
    fn low(self) -> [i32; 3] {
        [self.minimum[0], self.minimum[1], self.minimum[2]]
    }
    fn high(self) -> [i32; 3] {
        [self.maximum[0], self.maximum[1], self.maximum[2]]
    }
    fn valid(self) -> bool {
        (0..3).all(|a| {
            self.minimum[a] >= -100000000
                && self.maximum[a] <= 100000000
                && self.minimum[a] <= self.maximum[a]
        })
    }
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn random(state: &mut u32) -> u32 {
    *state = state.wrapping_add(0x9e3779b9);
    let mut h = *state;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^ (h >> 16)
}
fn around(c: [i32; 3], edge: i32) -> Region {
    let mut minimum = [0; 4];
    let mut maximum = [0; 4];
    for a in 0..3 {
        minimum[a] = (i64::from(c[a]) - i64::from(edge / 2))
            .clamp(-100000000, 100000001 - i64::from(edge)) as i32;
        maximum[a] = minimum[a] + edge - 1;
    }
    Region { minimum, maximum }
}

fn regions(cells: &[[i32; 4]]) -> Vec<Region> {
    assert_eq!(cells.len(), 65536);
    let mut result = Vec::with_capacity(8192);
    for i in 0..4096 {
        let c = cells[(i * 13) % cells.len()];
        result.push(around([c[0], c[1], c[2]], 1 << (i % 4)));
    }
    for i in 0..3072 {
        let c = cells[(i * 17 + 991) % cells.len()];
        result.push(around([c[0], c[1], c[2]], 1 << (4 + i % 21)));
    }
    let mut seed = 0x41419419;
    for i in 0..768 {
        let k = [0, 1, 36_782_985, 63_709_999, 99_999_999][i % 5];
        let mut c = [k; 3];
        for a in 0..3 {
            if random(&mut seed) & 1 != 0 {
                c[a] = -c[a] - 1;
            }
        }
        if i % 3 == 0 {
            c[i % 3] = 0;
        }
        if i % 3 == 1 {
            c[(i + 1) % 3] += 1;
        }
        result.push(around(c, [1, 2, 4, 8, 16, 256, 4096, 65536][i % 8]));
    }
    for i in 0..256 {
        let mut r = around([0; 3], 4);
        let a = i % 3;
        match i % 4 {
            0 => {
                r.minimum[a] = -100000001;
                r.maximum[a] = -100000000;
            }
            1 => {
                r.minimum[a] = 100000000;
                r.maximum[a] = 100000001;
            }
            2 => {
                r.minimum[a] = 3;
                r.maximum[a] = 2;
            }
            _ => {
                r.minimum[a] = i32::MIN;
                r.maximum[a] = i32::MAX;
            }
        }
        result.push(r);
    }
    assert_eq!(result.len(), 8192);
    result
}

fn cpu(field: &BoundedLandforms, r: Region) -> [u32; 4] {
    match field.classify(r.low(), r.high()) {
        None => [0; 4],
        Some(b) => [
            b.height_units[0] as u32,
            b.height_units[1] as u32,
            b.classification as u32,
            1,
        ],
    }
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
            "{SAMPLING_SHADER}\n{BOUNDS_SHADER}\n{}",
            r#"
struct Region { minimum:vec4<i32>, maximum:vec4<i32> }
@group(0) @binding(2) var<storage,read> regions:array<Region>;
@group(0) @binding(3) var<storage,read_write> certificates:array<LFRegionBounds>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x<arrayLength(&regions) {
        certificates[id.x]=lf_classify_region(regions[id.x].minimum.xyz,regions[id.x].maximum.xyz);
    }
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("integer region bounds"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("integer region bounds"),
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
    fn run(&self, field: &BoundedLandforms, regions: &[Region]) -> Vec<[u32; 4]> {
        let d = &self.device;
        let settings = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[field.snapshot().resolution(), 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let heights = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(field.snapshot().height_atlas()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let input = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(regions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let hierarchy = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(field.hierarchy()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (regions.len() * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let buffers = [&settings, &heights, &input, &output, &hierarchy];
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
            pass.dispatch_workgroups((regions.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let result = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
        readback.unmap();
        result
    }
}

fn save(root: &Path, name: &str, bytes: &[u8]) -> serde_json::Value {
    std::fs::write(root.join(name), bytes).unwrap();
    serde_json::json!({"file":name,"bytes":bytes.len(),"sha256":hash(bytes)})
}

#[test]
#[ignore = "new-field integer region certificates; frozen HELIO_LANDFORM_BOUNDS protocol required"]
fn conservative_integer_regions_contain_canonical_cells() {
    let root =
        PathBuf::from(std::env::var_os("HELIO_LANDFORM_BOUNDS").expect("HELIO_LANDFORM_BOUNDS"));
    assert!(root.join("protocol.json").is_file());
    let data = root.join("data");
    std::fs::create_dir(&data).expect("fresh evidence directory required");
    let previous = root.parent().unwrap().join("landform-sampling");
    let old: serde_json::Value =
        serde_json::from_slice(&std::fs::read(previous.join("results.json")).unwrap()).unwrap();
    assert_eq!(old["completed"], true);
    let started = Instant::now();
    let (device, info) = Device::new();
    let mut report = serde_json::json!({"device":info,"setup_ms":started.elapsed().as_secs_f64()*1000.0,
        "scope":"Integer cell-region certificates for new radial base field, without fine detail or edits; no old-generator equivalence, FPS or visual gate.","cases":[]});
    for source in old["cases"].as_array().unwrap() {
        let name = source["name"].as_str().unwrap();
        let identity = source["published"]["id"].as_str().unwrap();
        assert_eq!(identity.len(), 64);
        let id = SnapshotId(std::array::from_fn(|i| {
            u8::from_str_radix(&identity[2 * i..2 * i + 2], 16).unwrap()
        }));
        let path = previous
            .join("data/published")
            .join(format!("{identity}.hlfm"));
        let snapshot = LandformSnapshot::load(&path, id).unwrap();
        let input_artifact = source["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["file"].as_str().unwrap().ends_with("-input.bin"))
            .unwrap();
        let input_bytes = std::fs::read(
            previous
                .join("data")
                .join(input_artifact["file"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(
            hash(&input_bytes),
            input_artifact["sha256"].as_str().unwrap()
        );
        let cells: Vec<[i32; 4]> = input_bytes
            .chunks_exact(16)
            .map(|b| {
                std::array::from_fn(|i| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
            })
            .collect();
        let regions = regions(&cells);
        let started = Instant::now();
        let field = BoundedLandforms::new(snapshot);
        let hierarchy_ms = started.elapsed().as_secs_f64() * 1000.0;
        let reference: Vec<_> = regions.iter().map(|r| cpu(&field, *r)).collect();
        let started = Instant::now();
        let gpu = device.run(&field, &regions);
        let diagnostic_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cpu_gpu_errors = reference.iter().zip(&gpu).filter(|(a, b)| a != b).count();
        let mut errors_by_word = [0usize; 4];
        for (a, b) in reference.iter().zip(&gpu) {
            for i in 0..4 {
                errors_by_word[i] += usize::from(a[i] != b[i]);
            }
        }
        let mut distribution = [0usize; 3];
        let mut invalid = 0;
        let mut invalid_errors = 0;
        let mut checked = 0u64;
        let mut exhaustive = 0u64;
        let mut height_errors = 0u64;
        let mut occupancy_errors = 0u64;
        let mut examples = Vec::new();
        let mut seed = 0x77179419;
        for (i, (r, b)) in regions.iter().zip(&gpu).enumerate() {
            if reference[i] != *b && examples.len() < 8 {
                examples.push(serde_json::json!({"kind":"cpu_gpu","region":i,"min":r.low(),"max":r.high(),"cpu":reference[i],"gpu":b}));
            }
            if !r.valid() {
                invalid += 1;
                invalid_errors += usize::from(*b != [0; 4]);
                continue;
            }
            if b[3] != 1 || b[2] > 2 {
                invalid_errors += 1;
                continue;
            }
            distribution[b[2] as usize] += 1;
            let side = r.maximum[0] - r.minimum[0] + 1;
            let mut check = |c: [i32; 3]| {
                checked += 1;
                let sample = field.snapshot().sample_cell(c).unwrap();
                let height_bad =
                    sample.height_units < (b[0] as i32) || sample.height_units > (b[1] as i32);
                let occupied_bad = (b[2] == 1 && sample.solid) || (b[2] == 2 && !sample.solid);
                height_errors += u64::from(height_bad);
                occupancy_errors += u64::from(occupied_bad);
                if (height_bad || occupied_bad) && examples.len() < 8 {
                    examples.push(serde_json::json!({"kind":"false_certificate","region":i,
                    "cell":c,"height":sample.height_units,"solid":sample.solid,"certificate":b}));
                }
            };
            if side <= 8 {
                for z in r.minimum[2]..=r.maximum[2] {
                    for y in r.minimum[1]..=r.maximum[1] {
                        for x in r.minimum[0]..=r.maximum[0] {
                            check([x, y, z]);
                            exhaustive += 1;
                        }
                    }
                }
            } else {
                for mask in 0..8 {
                    check(std::array::from_fn(|a| {
                        if mask & (1 << a) == 0 {
                            r.minimum[a]
                        } else {
                            r.maximum[a]
                        }
                    }));
                }
                for _ in 0..64 {
                    check(std::array::from_fn(|a| {
                        r.minimum[a] + (random(&mut seed) % side as u32) as i32
                    }));
                }
            }
        }
        let artifacts: Vec<_> = [
            ("regions", bytemuck::cast_slice(&regions)),
            ("cpu", bytemuck::cast_slice(&reference)),
            ("gpu", bytemuck::cast_slice(&gpu)),
            ("hierarchy", bytemuck::cast_slice(field.hierarchy())),
        ]
        .into_iter()
        .map(|(kind, bytes)| save(&data, &format!("{name}-{kind}.bin"), bytes))
        .collect();
        let result = serde_json::json!({"name":name,"snapshot_id":identity,"resolution":field.snapshot().resolution(),
            "regions":regions.len(),"hierarchy_bytes":field.hierarchy().len()*8,"hierarchy_ms":hierarchy_ms,
            "cpu_gpu_errors":cpu_gpu_errors,"errors_by_word":errors_by_word,"invalid_regions":invalid,"invalid_errors":invalid_errors,
            "classification_counts_mixed_air_solid":distribution,"checked_cells":checked,"exhaustively_checked_cells":exhaustive,
            "height_bound_errors":height_errors,"occupancy_certificate_errors":occupancy_errors,"mismatch_examples":examples,
            "diagnostic_upload_dispatch_readback_wall_ms":diagnostic_wall_ms,"artifacts":artifacts});
        println!("BOUNDS_CASE {name}: regions={} cpu_gpu_errors={cpu_gpu_errors} height_errors={height_errors} occupancy_errors={occupancy_errors} checked_cells={checked} classes={distribution:?}",regions.len());
        report["cases"].as_array_mut().unwrap().push(result);
        std::fs::write(
            root.join("results.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        assert_eq!(cpu_gpu_errors, 0, "CPU/GPU bounds differ");
        assert_eq!(invalid_errors, 0, "invalid region received a certificate");
        assert_eq!(invalid, 256);
        assert_eq!(height_errors, 0, "height escaped interval");
        assert_eq!(occupancy_errors, 0, "false region certificate");
        assert!(
            distribution[0] >= 32 && distribution[1] >= 500 && distribution[2] >= 500,
            "insufficient classification coverage"
        );
    }
    report["completed"] = true.into();
    std::fs::write(
        root.join("results.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
