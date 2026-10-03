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

const BYTES: u64 = 18_456;
const URGENT_CAPACITY: usize = 192;
const ORDINARY_CAPACITY: usize = 64;
const FIXTURE_GLOBALS: &str = r#"
const ST_HIT:u32=1u;
struct Hit { t:f32, info:u32 }
struct Frame { hints:vec4<u32>, screen:vec4<f32>, layer:vec4<f32> }
@group(0) @binding(0) var<uniform> frame:Frame;
struct Camera { proj:mat4x4<f32> }
var<private> camera:Camera;
fn level_for(t:f32)->u32 { return frame.hints.z; }
"#;

struct FixtureCase {
    prefer: bool,
    finish: bool,
    mixed: bool,
    status: u32,
    level: u32,
    wanted: u32,
    distance: f32,
    voxel: f32,
}

impl Default for FixtureCase {
    fn default() -> Self {
        Self { prefer:false, finish:true, mixed:false, status:3, level:4,
            wanted:0, distance:50.0, voxel:0.1 }
    }
}

fn fixture(gpu: &Gpu, inputs: &[[u32; 4]], epoch: u32, enabled: bool, hint: bool) -> Vec<u8> {
    fixture_mode(gpu, inputs, epoch, enabled, hint, false, true)
}

fn fixture_mode(gpu: &Gpu, inputs: &[[u32; 4]], epoch: u32, enabled: bool, hint: bool,
    prefer: bool, finish: bool) -> Vec<u8> {
    fixture_case(gpu, inputs, epoch, enabled, hint, FixtureCase {
        prefer, finish, status:if prefer { 1 } else { 3 }, ..Default::default()
    })
}

fn fixture_case(gpu: &Gpu, inputs: &[[u32; 4]], epoch: u32, enabled: bool, hint: bool,
    case: FixtureCase) -> Vec<u8> {
    let common = include_str!("../shaders/common.wgsl");
    let keys = &common
        [common.find("fn column_key0(").unwrap()..common.find("// Table edge log2").unwrap()];
    // This isolated fixture tests requests; the complete primary path below
    // also composes the separate Hit sampling code and its frame bindings.
    let requests_shader = include_str!("../shaders/visible_feedback.wgsl")
        .split("// Four independent").next().unwrap();
    let source = format!(
        "{}\n{}\n{}\n{}\n{}",
        include_str!("../shaders/noise.wgsl"), FIXTURE_GLOBALS, keys,
        requests_shader,
        r#"@group(0) @binding(22) var<storage, read> inputs: array<vec4<u32>>;
override PREFER_FINAL: bool = false;
override MIXED_BANKS: bool = false;
@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&inputs) { return; }
    let v = inputs[id.x];
    camera.proj[1][1] = 1.0;
    begin_visible_feedback_pixel(vec2<u32>(id.x & 255u, id.x >> 8u));
    request_visible_block(v.x, v.y, bitcast<i32>(v.z), bitcast<i32>(v.w));
    // Another hop (or sky retry) retains the first candidate without emitting.
    request_visible_block((v.x + 1u) % 6u, v.y, bitcast<i32>(v.z) + 32, bitcast<i32>(v.w));
    if PREFER_FINAL && (!MIXED_BANKS || (id.x & 8u) == 0u) {
        prefer_visible_block((v.x + 1u) % 6u, v.y, bitcast<i32>(v.z) + 32, bitcast<i32>(v.w));
    }
    if frame.hints.y != 0u {
        let hit = Hit(frame.screen.z, u32(frame.screen.x) | (u32(frame.layer.x) << 5u));
        finish_visible_feedback_pixel(hit);
        // Finalization itself must not be able to consume another attempt.
        finish_visible_feedback_pixel(hit);
    }
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
                constants: &[
                    ("VISIBLE_FEEDBACK", if enabled { 1.0 } else { 0.0 }),
                    ("PREFER_FINAL", if case.prefer { 1.0 } else { 0.0 }),
                    ("MIXED_BANKS", if case.mixed { 1.0 } else { 0.0 }),
                ],
                ..Default::default()
            },
            cache: None,
        });
    let frame = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                [0u32, u32::from(case.finish), case.wanted, (epoch << 8) | if hint { 16 } else { 0 }],
                [(case.status as f32).to_bits(), 729.0f32.to_bits(), case.distance.to_bits(), 0],
                [(case.level as f32).to_bits(), case.voxel.to_bits(), 0, 0],
            ]),
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
    bank_requests(data, true).union(&bank_requests(data, false)).copied().collect()
}
fn bank_requests(data: &[u8], urgent: bool) -> HashSet<(u32, u32)> {
    let (count, offset) = if urgent {
        (word(data, 4).min(URGENT_CAPACITY as u32) as usize, 0)
    } else { (word(data, 0).min(ORDINARY_CAPACITY as u32) as usize, URGENT_CAPACITY) };
    (offset..offset + count).map(|i| (word(data, 6 + i * 2), word(data, 7 + i * 2))).collect()
}
fn full_key(v: [u32; 4]) -> (u32, u32) {
    (
        key0(v[0] as u8, v[1], (v[2] as i32) & !3),
        (v[3] as i32 & !3) as u32,
    )
}
fn bit(key: (u32, u32), epoch: u32) -> u32 {
    slot_hash(key.0, key.1 ^ epoch.wrapping_mul(0x9e3779b9)) & 65535
}

#[test]
fn deferred_request_prefers_final_hit_and_emits_once_after_retry() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP deferred feedback: no GPU adapter");
        return;
    };
    let input = [[2, 4, 13, (-9i32) as u32]];
    let deferred = fixture_mode(&gpu, &input, 0, true, true, false, false);
    assert_eq!([word(&deferred, 0), word(&deferred, 1), word(&deferred, 2)], [0; 3],
        "trace hops must save candidates without publishing before finalization");

    // With no usable final Hit preference, another hop/sky retry cannot
    // replace the first missing block or consume another request attempt.
    let unresolved = fixture(&gpu, &input, 0, true, true);
    assert_eq!(word(&unresolved, 2), 1);
    assert_eq!(requests(&unresolved), HashSet::from([full_key(input[0])]));

    let preferred = fixture_mode(&gpu, &input, 0, true, true, true, true);
    assert_eq!([word(&preferred, 0), word(&preferred, 4)], [0, 1]);
    assert_eq!(word(&preferred, 2), 1, "preference and repeated finalization share one attempt");
    assert_eq!(requests(&preferred), HashSet::from([full_key([3, 4, 45, (-9i32) as u32])]),
        "the final missing Hit footprint must replace the earlier missing air block");
}

#[test]
fn urgent_bank_requires_final_missing_surface_and_strict_coarse_threshold() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP urgent feedback classification: no GPU adapter");
        return;
    };
    let input = [[2, 0, 12, (-12i32) as u32]];
    for (prefer, status, level, wanted, width, urgent) in [
        (false, 1, 1, 0, 8.0, false),
        (true, 1, 1, 0, 3.9, false),
        (true, 1, 1, 0, 4.0, false),
        (true, 1, 1, 0, 4.1, true),
        (true, 1, 1, 1, 8.0, false),
        (true, 1, 0, 0, 8.0, false),
        (true, 0, 1, 0, 8.0, false),
        (true, 2, 1, 0, 8.0, false),
        (true, 3, 1, 0, 8.0, false),
    ] {
        // distance=height/2 and projection=1 produce an exact one-metre
        // pixel footprint; exactly4px therefore has no threshold ambiguity.
        let data = fixture_case(&gpu, &input, 0, true, true, FixtureCase {
            prefer, status, level, wanted, distance:364.5,
            voxel:width / (1u32 << level) as f32, ..Default::default()
        });
        assert_eq!(word(&data, 2), 1);
        assert_eq!([word(&data, 0), word(&data, 4)],
            if urgent { [0,1] } else { [1,0] },
            "prefer={prefer},status={status},level={level},wanted={wanted},width={width}");
        assert_eq!(requests(&data).len(), 1);
    }
}

#[test]
fn ordinary_request_cannot_suppress_urgent_duplicate() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP feedback bank isolation: no GPU adapter");
        return;
    };
    let inputs: Vec<_> = (0..256 * 8).map(|i| {
        if i & 8 == 0 { [2, 4, 12, (-12i32) as u32] }
        else { [3, 4, 44, (-12i32) as u32] }
    }).collect();
    let data = fixture_case(&gpu, &inputs, 0, true, true, FixtureCase {
        prefer:true, mixed:true, status:1, ..Default::default()
    });
    let key = full_key([3, 4, 44, (-12i32) as u32]);
    assert_eq!(word(&data, 2), 32);
    assert_eq!([word(&data, 0), word(&data, 4)], [1,1]);
    assert_eq!(bank_requests(&data, true), HashSet::from([key]));
    assert_eq!(bank_requests(&data, false), HashSet::from([key]));
    assert_eq!(word(&data, 1), 0);
}

#[test]
fn visible_requests_bound_emission_preserve_keys_and_retry_collisions() {
    let Some(gpu) = gpu() else {
        eprintln!("SKIP visible feedback: no GPU adapter");
        return;
    };
    let inputs: Vec<_> = (0..256 * 16)
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
            64,
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
    let requests_shader = include_str!("../shaders/visible_feedback.wgsl")
        .split("// Four independent").next().unwrap();
    let source=format!("{}\n{}\n{}\n{}\n@compute @workgroup_size(1) fn off() {{begin_visible_feedback_pixel(vec2<u32>(0));request_visible_block(0,0,0,0);prefer_visible_block(0,0,4,4);finish_visible_feedback_pixel(Hit(50.0,1u));}}",include_str!("../shaders/noise.wgsl"),FIXTURE_GLOBALS,keys,requests_shader);
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
    assert!(word(&data, 0) >= ORDINARY_CAPACITY as u32);
    assert_eq!(requests(&data).len(), ORDINARY_CAPACITY);
    assert_eq!(word(&data, 4), 0);
    let urgent = fixture_mode(&gpu, &big, 0, true, true, true, true);
    assert_eq!(word(&urgent, 2), 1024);
    assert_ne!(word(&urgent, 1), 0);
    assert!(word(&urgent, 4) >= URGENT_CAPACITY as u32);
    assert_eq!(bank_requests(&urgent, true).len(), URGENT_CAPACITY);
    assert_eq!(word(&urgent, 0), 0);
    let mixed = fixture_case(&gpu, &big, 0, true, true, FixtureCase {
        prefer:true, mixed:true, status:1, ..Default::default()
    });
    assert_eq!(word(&mixed, 2), 1024);
    assert_ne!(word(&mixed, 1), 0);
    assert_eq!(bank_requests(&mixed, true).len(), URGENT_CAPACITY);
    assert_eq!(bank_requests(&mixed, false).len(), ORDINARY_CAPACITY);
    assert_eq!(requests(&mixed).len(), 256);
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
