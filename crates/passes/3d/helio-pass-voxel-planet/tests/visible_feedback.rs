//! Bounded visible-block feedback uses production WGSL and the real primary path.
mod common;
use common::*;
use glam::{DVec3, Vec3};
use helio_pass_voxel_planet::engine::{PlanetRenderer, Settings};
use helio_pass_voxel_planet::residency::{key0, slot_hash, Capacity};
use helio_pass_voxel_planet::{Planet, PlanetRecipe, TerrainSource};
use std::collections::HashSet;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const BYTES: u64 = 18_448;

fn fixture(gpu: &Gpu, inputs: &[[u32; 4]], epoch: u32, enabled: bool, hint: bool) -> Vec<u8> {
    let common = include_str!("../shaders/common.wgsl");
    let keys = &common
        [common.find("fn column_key0(").unwrap()..common.find("// Table edge log2").unwrap()];
    let source = format!(
        "{}\nstruct Frame {{ hints: vec4<u32> }}\n@group(0) @binding(0) var<uniform> frame: Frame;\n{}\n{}\n{}",
        include_str!("../shaders/noise.wgsl"), keys,
        include_str!("../shaders/visible_feedback.wgsl"),
        r#"@group(0) @binding(22) var<storage, read> inputs: array<vec4<u32>>;
@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&inputs) { return; }
    let v = inputs[id.x];
    begin_visible_feedback_pixel(vec2<u32>(id.x & 255u, id.x >> 8u));
    request_visible_block(v.x, v.y, bitcast<i32>(v.z), bitcast<i32>(v.w));
    // A second hop from the same invocation must not issue a second request.
    request_visible_block((v.x + 1u) % 6u, v.y, bitcast<i32>(v.z) + 32, bitcast<i32>(v.w));
}"#
    );
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production visible feedback test"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("check"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("VISIBLE_FEEDBACK", if enabled { 1.0 } else { 0.0 })],
                ..Default::default()
            },
            cache: None,
        });
    let frame = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0u32, 0, 0, (epoch << 8) | if hint { 16 } else { 0 }]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let probes = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frame.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 21,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 22,
                resource: probes.as_entire_binding(),
            },
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
    read_buffer(gpu, &output, BYTES)
}
fn word(data: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap())
}
fn requests(data: &[u8]) -> HashSet<(u32, u32)> {
    (0..word(data, 0).min(256) as usize)
        .map(|i| (word(data, 4 + i * 2), word(data, 5 + i * 2)))
        .collect()
}
fn full_key(v: [u32; 4]) -> (u32, u32) {
    (
        key0(v[0] as u8, v[1], (v[2] as i32) & !3),
        (v[3] as i32 & !3) as u32,
    )
}
fn bit(key: (u32, u32), epoch: u32) -> u32 {
    slot_hash(key.0, key.1 ^ epoch.wrapping_mul(0x9e3779b9)) & 131071
}

#[test]
fn visible_requests_bound_emission_preserve_keys_and_retry_collisions() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP visible feedback: no GPU adapter");
        return;
    };
    let inputs: Vec<_> = (0..256 * 64)
        .map(|i| {
            [
                (i % 6) as u32,
                (i % 17) as u32,
                (i * 7 + 3) as u32,
                (-(i as i32) * 5 - 1) as u32,
            ]
        })
        .collect();
    for epoch in [0, 9, 63] {
        let data = fixture(&gpu, &inputs, epoch, true, true);
        let expected: HashSet<_> = inputs
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                (i % 256) & 7 == (epoch & 7) as usize
                    && (i / 256) & 7 == ((epoch >> 3) & 7) as usize
            })
            .map(|(_, v)| full_key(*v))
            .collect();
        assert_eq!(
            word(&data, 2),
            256,
            "exactly one attempt per selected pixel"
        );
        let actual = requests(&data);
        let occupied_bits: HashSet<_> = expected.iter().map(|k| bit(*k, epoch)).collect();
        assert_eq!(
            actual.len(),
            occupied_bits.len(),
            "one publication per production hash bit"
        );
        assert!(
            actual.is_subset(&expected),
            "unaligned, wrong signed index or second-hop key"
        );
    }
    // Disable through the runtime hint independently of specialization.
    let data = fixture(&gpu, &inputs, 7, true, false);
    assert_eq!([word(&data, 0), word(&data, 1), word(&data, 2)], [0; 3]);
    // Specialization removes feedback bindings entirely. Check that branch
    // with a shader-only constructor separately rather than imposing a layout.
    let common = include_str!("../shaders/common.wgsl");
    let keys = &common
        [common.find("fn column_key0(").unwrap()..common.find("// Table edge log2").unwrap()];
    let source=format!("{}\nstruct Frame {{hints:vec4<u32>}} @group(0) @binding(0) var<uniform> frame:Frame;\n{}\n{}\n@compute @workgroup_size(1) fn off() {{begin_visible_feedback_pixel(vec2<u32>(0));request_visible_block(0,0,0,0);}}",include_str!("../shaders/noise.wgsl"),keys,include_str!("../shaders/visible_feedback.wgsl"));
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    gpu.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("off"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Overflow is explicit and output stays in-bounds, with at most one
    // attempt for each of 1024 selected pixels in this 256x256 dispatch.
    let big: Vec<_> = (0..256 * 256).map(|i| [0, 3, (i * 4) as u32, 0]).collect();
    let data = fixture(&gpu, &big, 0, true, true);
    assert_eq!(word(&data, 2), 1024);
    assert_ne!(word(&data, 1), 0);
    assert!(word(&data, 0) >= 256);
    assert_eq!(requests(&data).len(), 256);
    let mut by_bit = std::collections::HashMap::new();
    let (a, b) = (0..10000)
        .find_map(|i| {
            let k = full_key([2, 4, i * 4, (-24i32) as u32]);
            by_bit
                .insert(bit(k, 0), k)
                .filter(|old| *old != k)
                .map(|old| (old, k))
        })
        .expect("deliberate production hash collision");
    let epoch = (1..64).find(|e| bit(a, *e) != bit(b, *e)).unwrap();
    let mut pair = vec![[0; 4]; 256 * 16];
    let at = |e: u32, n: usize| ((e >> 3) & 7) as usize * 256 + (e & 7) as usize + n * 8;
    let decode = |k: (u32, u32)| [(k.0 >> 24) & 7, k.0 >> 27, k.0 & 0xffffff, k.1];
    // Every sampled invocation requests one of these two keys.
    for (i, v) in pair.iter_mut().enumerate() {
        *v = decode(if i & 8 == 0 { a } else { b });
    }
    assert_eq!(requests(&fixture(&gpu, &pair, 0, true, true)).len(), 1);
    pair[at(epoch, 0)] = decode(a);
    pair[at(epoch, 1)] = decode(b);
    assert_eq!(
        requests(&fixture(&gpu, &pair, epoch, true, true)),
        HashSet::from([a, b]),
        "next epoch retries suppressed collisions"
    );
}

#[test]
fn real_primary_emits_missing_hinted_blocks_and_stops_when_resident() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP real primary feedback: no GPU adapter");
        return;
    };
    let planet = Arc::new(
        Planet::new(PlanetRecipe {
            shape: helio_pass_voxel_planet::grid::Shape::Plane,
            plane_size_m: 64.0,
            voxel_size_m: 1.0,
            terrain: TerrainSource {
                generator: helio_pass_voxel_planet::landform::FLAT_ID.into(),
                settings: r#"{"height_m":0.0}"#.into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap(),
    );
    let target = Target::new(&gpu, [32, 32]);
    let mut renderer = PlanetRenderer::new(
        &gpu.device,
        &gpu.queue,
        planet.clone(),
        Settings {
            visible_feedback: true,
            job_budget: 256,
            capacity: Capacity {
                table_bits: 14,
                records: 16_384,
                pool_units: 1 << 16,
                scratch_units: 1 << 14,
                edit_words: 1 << 16,
                max_jobs: 256,
                max_evictions: 4096,
            },
            ..Default::default()
        },
        target.size,
    );
    let frame = frame(&planet, DVec3::new(0.23, 2.0, 0.19));
    let mut seen_missing = false;
    for n in 0..240 {
        target.render(&gpu, &mut renderer, &frame, -Vec3::Y, n);
        let stats = renderer.stats();
        assert!(stats.visible_request_attempts <= 16);
        assert!(!stats.visible_request_overflow);
        seen_missing |= stats.visible_request_blocks > 0;
        if renderer.settled() && n >= 12 {
            break;
        }
    }
    assert!(
        seen_missing,
        "actual primary failed to request cold hint-rejected blocks"
    );
    assert!(renderer.settled(), "bounded plane did not converge");
    for n in 240..252 {
        target.render(&gpu, &mut renderer, &frame, -Vec3::Y, n);
    }
    assert_eq!(
        renderer.stats().visible_request_attempts,
        0,
        "resident rays must not request blocks"
    );
    let before = read_buffer(&gpu, renderer.hit_buffer(), 32 * 32 * 32);
    renderer.settings_mut().visible_feedback = false;
    renderer.settings_mut().freeze_residency = true;
    target.render(&gpu, &mut renderer, &frame, -Vec3::Y, 253);
    assert_eq!(
        read_buffer(&gpu, renderer.hit_buffer(), 32 * 32 * 32),
        before,
        "feedback changed resident geometry/counters"
    );
}
