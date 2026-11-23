//! The radiance cache's WGSL against its Rust mirror on a real device: keys,
//! levels, buckets and checksums bit for bit (near the world's zero, at an
//! Earth's radius and past 2^32 cells of the finest level); a mark claims a
//! cell with the hit's point and normal and its ancestor; a read finds a
//! value a publisher wrote, answers from a coarser cell when its own has
//! none, keeps sides and spaces apart, and finds nothing in the stand-in.
//! Skipped without a device.

use helio_frame_api::rays::radiance_cache::{
    self, RadianceCacheEntry, RadianceCacheFrame, RadianceCacheKey, RadianceCacheSlot, ANCESTOR,
    SNIPPET,
};
use helio_frame_api::rays::stream::{self, SNIPPET as STREAM};

const KERNEL: &str = r#"//!use frame_ray_stream
//!use frame_radiance_cache
struct Query {
    p: vec4<f32>,
    n: vec4<f32>,
}
@group(0) @binding(0) var<uniform> radiance_cache_frame: RadianceCacheFrame;
@group(0) @binding(1) var<storage, read_write> radiance_cache_slots: array<RadianceCacheSlot>;
@group(0) @binding(2) var<storage, read_write> radiance_cache_entries: array<RadianceCacheEntry>;
@group(0) @binding(3) var<storage, read> queries: array<Query>;
@group(0) @binding(4) var<storage, read_write> out: array<u32>;
@group(0) @binding(5) var<uniform> count: vec4<u32>;

@compute @workgroup_size(64)
fn keys(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= count.x { return; }
    let q = queries[id.x];
    let level = radiance_cache_level(q.p.xyz, q.p.w);
    let k = radiance_cache_key(q.p.xyz, q.n.xyz, level, u32(q.n.w));
    let at = id.x * 12u;
    out[at] = level;
    out[at + 1u] = k.tag;
    out[at + 2u] = k.x_lo;
    out[at + 3u] = k.x_hi;
    out[at + 4u] = k.y_lo;
    out[at + 5u] = k.y_hi;
    out[at + 6u] = k.z_lo;
    out[at + 7u] = k.z_hi;
    out[at + 8u] = radiance_cache_bucket(k);
    out[at + 9u] = radiance_cache_checksum(k);
    out[at + 10u] = radiance_cache_pack_point(q.p.xyz, level).x;
}

@compute @workgroup_size(64)
fn mark_hits(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= count.x { return; }
    let q = queries[id.x];
    radiance_cache_mark(q.p.xyz, q.n.xyz, q.p.w, u32(q.n.w), RADIANCE_CACHE_FAR);
}

@compute @workgroup_size(64)
fn read_hits(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= count.x { return; }
    let q = queries[id.x];
    let e = radiance_cache_read(q.p.xyz, q.n.xyz, q.p.w, u32(q.n.w), RADIANCE_CACHE_LEVELS, RADIANCE_CACHE_FAR);
    let at = id.x * 12u;
    out[at] = select(0u, 1u, e.found);
    out[at + 1u] = e.level;
    out[at + 2u] = bitcast<u32>(e.direct.x);
    out[at + 3u] = bitcast<u32>(e.indirect.y);
}

// Each slot's entry's point back in the frame's space.
@compute @workgroup_size(64)
fn points(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&radiance_cache_entries) { return; }
    let p = radiance_cache_entry_point(radiance_cache_entries[id.x]);
    out[id.x * 3u] = bitcast<u32>(p.x);
    out[id.x * 3u + 1u] = bitcast<u32>(p.y);
    out[id.x * 3u + 2u] = bitcast<u32>(p.z);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Query {
    p: [f32; 4],
    n: [f32; 4],
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 4],
}

fn gpu() -> Option<Gpu> {
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        adapter.request_device(&Default::default()).await.ok()
    })?;
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform(0),
            storage(1, false),
            storage(2, false),
            storage(3, true),
            storage(4, false),
            uniform(5),
        ],
    });
    let source = helio_core::shader::resolve_with(KERNEL, &[STREAM, SNIPPET]);
    #[allow(clippy::disallowed_methods)] // a test's one-off module, not a pass's
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipelines = ["keys", "mark_hits", "read_hits", "points"].map(|entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    Some(Gpu {
        device,
        queue,
        layout,
        pipelines,
    })
}

/// A cache of `slots` slots (0: the stand-in's frame) on the GPU.
struct Cache {
    frame: wgpu::Buffer,
    slots: wgpu::Buffer,
    entries: wgpu::Buffer,
}

impl Gpu {
    fn buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn cache(&self, frame: &RadianceCacheFrame) -> Cache {
        let slots = u64::from(frame.slots().max(1));
        let cache = Cache {
            frame: self.buffer(RadianceCacheFrame::BYTES, wgpu::BufferUsages::UNIFORM),
            // The hash's slots and the new cells' queue after them.
            slots: self.buffer(
                radiance_cache::slot_rows(frame.slots()).max(1) * RadianceCacheSlot::BYTES,
                wgpu::BufferUsages::STORAGE,
            ),
            entries: self.buffer(
                slots * RadianceCacheEntry::BYTES,
                wgpu::BufferUsages::STORAGE,
            ),
        };
        self.queue
            .write_buffer(&cache.frame, 0, bytemuck::bytes_of(frame));
        cache
    }

    /// Runs entry `which` (0 keys, 1 mark, 2 read, 3 points) over `queries`;
    /// the output words.
    fn run(&self, cache: &Cache, which: usize, queries: &[Query], groups: u32) -> Vec<u32> {
        let input = self.buffer(
            (std::mem::size_of_val(queries) as u64).max(32),
            wgpu::BufferUsages::STORAGE,
        );
        self.queue
            .write_buffer(&input, 0, bytemuck::cast_slice(queries));
        let out_size = (queries.len() as u64 * 12 * 4).max(cache.entries.size() / 64 * 12);
        let out = self.buffer(out_size, wgpu::BufferUsages::STORAGE);
        let count = self.buffer(16, wgpu::BufferUsages::UNIFORM);
        self.queue.write_buffer(
            &count,
            0,
            bytemuck::cast_slice(&[queries.len() as u32, 0, 0, 0]),
        );
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                entry(0, &cache.frame),
                entry(1, &cache.slots),
                entry(2, &cache.entries),
                entry(3, &input),
                entry(4, &out),
                entry(5, &count),
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[which]);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        self.read(&out)
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let words = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
        words
    }
}

fn h(x: u32) -> f32 {
    let mut v = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    v = ((v >> ((v >> 28) + 4)) ^ v).wrapping_mul(277_803_737);
    ((v >> 22) ^ v) as f32 / u32::MAX as f32
}

/// Points about the eye within `reach` metres (frame space), footprints
/// from a millimetre to tens of metres, every side, spaces 0 to 3.
fn queries(count: u32, reach: f32) -> Vec<Query> {
    (0..count)
        .map(|i| {
            let r = i * 16;
            let p = [0, 1, 2].map(|k| (h(r + k) * 2.0 - 1.0) * reach);
            let footprint = (h(r + 3) * 16.0 - 10.0).exp2();
            let axis = (h(r + 4) * 3.0) as usize % 3;
            let mut n = [0.2, -0.3, 0.1];
            n[axis] = if h(r + 5) < 0.5 { -1.0 } else { 1.0 };
            let l = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            let n = n.map(|c: f32| c / l.sqrt());
            let space = (h(r + 6) * 4.0) as u32 % 4;
            Query {
                p: [p[0], p[1], p[2], footprint],
                n: [n[0], n[1], n[2], space as f32],
            }
        })
        .collect()
}

/// Whether the level of `q` sits off a rounding boundary, so the CPU's and
/// the GPU's log2 agree on it.
fn clear_of_rounding(frame: &RadianceCacheFrame, q: &Query) -> bool {
    let p = [q.p[0], q.p[1], q.p[2]];
    let d = (0..3)
        .map(|k| (p[k] - frame.eye[k]).powi(2))
        .sum::<f32>()
        .sqrt();
    let l = q.p[3].max(frame.offset[3] * d).log2();
    (l - l.floor() - 0.5).abs() > 1e-3
}

/// Keys from the WGSL are the Rust mirror's, bit for bit, wherever the
/// frame's origin is: the world's zero, an Earth radius away in every
/// axis's sign, and 10^12 m (past 2^32 finest cells, so the high words
/// carry), and with an origin off the 1024 m step.
#[test]
fn keys_match_the_rust_mirror_at_every_scale() {
    let Some(gpu) = gpu() else { return };
    for origin in [
        [0.0f64, 0.0, 0.0],
        [
            6_371_000.0 - 6_371_000.0 % 1024.0,
            -1024.0 * 3.0,
            -6_370_304.0,
        ],
        [1.0e12 - 1.0e12 % 1024.0, 2048.0, -1.0e12 + 1.0e12 % 1024.0],
        [1000.0, 2.0, -500.0],
    ] {
        let frame = RadianceCacheFrame::new(origin, [1.5, -0.25, 3.0], 0.04, 1 << 12, 9);
        let cache = gpu.cache(&frame);
        let qs = queries(4096, 900.0);
        let out = gpu.run(&cache, 0, &qs, 64);
        let mut checked = 0;
        for (i, q) in qs.iter().enumerate() {
            if !clear_of_rounding(&frame, q) {
                continue;
            }
            let w = &out[i * 12..i * 12 + 12];
            let p = [q.p[0], q.p[1], q.p[2]];
            let level = frame.level(p, q.p[3]);
            let key = frame.key(p, [q.n[0], q.n[1], q.n[2]], level, q.n[3] as u32);
            let gpu_key = RadianceCacheKey {
                tag: w[1],
                x_lo: w[2],
                x_hi: w[3],
                y_lo: w[4],
                y_hi: w[5],
                z_lo: w[6],
                z_hi: w[7],
            };
            assert_eq!(w[0], level, "{origin:?} query {i}: level");
            assert_eq!(gpu_key, key, "{origin:?} query {i}");
            assert_eq!(
                w[8],
                key.bucket(frame.slots()),
                "{origin:?} query {i}: bucket"
            );
            assert_eq!(w[9], key.checksum(), "{origin:?} query {i}: checksum");
            // The world cell holds the world point (f64), at the key's level.
            let size = f64::from(radiance_cache::cell_size(level));
            let world: [f64; 3] = std::array::from_fn(|k| origin[k] + f64::from(p[k]));
            let cell = key.cell();
            for k in 0..3 {
                let lo = cell[k] as f64 * size;
                assert!(
                    world[k] >= lo - 1e-3 * size.max(1.0)
                        && world[k] < lo + size + 1e-3 * size.max(1.0),
                    "{origin:?} query {i}: axis {k} {} not in [{lo}, {})",
                    world[k],
                    lo + size
                );
            }
            checked += 1;
        }
        assert!(checked > 4000, "{checked} queries clear of rounding");
    }
}

/// Marks claim each cell once, with the hit's point and normal, and the
/// cell ANCESTOR levels coarser; a mark of a cell in use only marks it. A
/// read finds no value until a publisher writes one, then finds it; a cell
/// without a value answers from its ancestor; sides and spaces are apart;
/// the stand-in finds nothing and writes nothing.
#[test]
fn marks_claim_cells_and_reads_find_what_a_publisher_wrote() {
    let Some(gpu) = gpu() else { return };
    let origin = [6_371_000.0 - 6_371_000.0 % 1024.0, 0.0, -1024.0 * 7.0];
    let frame = RadianceCacheFrame::new(origin, [0.0; 3], 0.001, 1 << 12, 5);
    let cache = gpu.cache(&frame);
    // A floor's worth of hits seen from above, a metre down, 2 cm
    // footprints (wider than the eye angle there: the finest level), in
    // space 1.
    let up = [0.0, 1.0, 0.0, 1.0];
    let floor: Vec<Query> = (0..40)
        .map(|i| Query {
            p: [
                (i % 8) as f32 * 0.37 - 1.3,
                -1.0,
                (i / 8) as f32 * 0.41 - 1.0,
                0.02,
            ],
            n: up,
        })
        .collect();
    let groups = 1;
    gpu.run(&cache, 1, &floor, groups);
    // Twice: the second marks only.
    gpu.run(&cache, 1, &floor, groups);
    let slots: Vec<RadianceCacheSlot> = bytemuck::cast_slice(&gpu.read(&cache.slots)).to_vec();
    let mut entries: Vec<RadianceCacheEntry> =
        bytemuck::cast_slice(&gpu.read(&cache.entries)).to_vec();
    let n = frame.slots() as usize;
    let occupied: Vec<usize> = (0..n).filter(|&s| slots[s].state != 0).collect();
    // The new cells' queue after the slots: each cell claimed once, a place
    // each; a place whose mark claimed nothing holds none.
    let queued = slots[n].state as usize;
    let mut placed: Vec<usize> = slots[n + 1..n + 1 + queued]
        .iter()
        .filter(|q| q.state != u32::MAX)
        .map(|q| q.state as usize)
        .collect();
    placed.sort_unstable();
    assert_eq!(placed, occupied, "the queue holds the cells claimed");
    let mut want = std::collections::HashSet::new();
    for q in &floor {
        let p = [q.p[0], q.p[1], q.p[2]];
        let level = frame.level(p, q.p[3]);
        want.insert(frame.key(p, [0.0, 1.0, 0.0], level, 1));
        want.insert(frame.key(p, [0.0, 1.0, 0.0], level + ANCESTOR, 1));
    }
    assert_eq!(occupied.len(), want.len(), "one slot a cell");
    for &s in &occupied {
        let e = entries[s];
        assert!(want.contains(&e.key), "slot {s}: {:?}", e.key);
        assert_eq!(slots[s].state, e.key.checksum());
        assert_eq!(slots[s].used, 5);
        assert!(!e.has_value());
        let bucket = e.key.bucket(frame.slots());
        assert!((bucket..bucket + radiance_cache::BUCKET).contains(&(s as u32)));
        let n = stream::unpack_unit(e.normal);
        assert!((n[1] - 1.0).abs() < 1e-4, "{n:?}");
    }
    // Every claimed point is one of the hits, back in the frame within a
    // 32768th of its cell (and f32's rounding there).
    let points = gpu.run(&cache, 3, &floor, frame.slots() / 64);
    for &s in &occupied {
        let p = [0, 1, 2].map(|k| f32::from_bits(points[s * 3 + k]));
        let size = radiance_cache::cell_size(entries[s].key.level());
        let near = floor
            .iter()
            .any(|q| (0..3).all(|k| (q.p[k] - p[k]).abs() <= size / 32768.0 + 1e-5));
        assert!(near, "slot {s}: {p:?}");
    }
    // Nothing found before a value.
    let out = gpu.run(&cache, 2, &floor, groups);
    assert!(out.chunks(12).take(floor.len()).all(|w| w[0] == 0));
    // A publisher's values: the fine cells' direct 1 + their slot, the
    // ancestors' indirect 6.5 and bounced 0.5 (a read's indirect light is
    // their sum, 7); the first half of the fine cells only.
    let fine_level = frame.level([0.0, -1.0, 0.0], 0.02);
    let mut fine: Vec<usize> = occupied
        .iter()
        .copied()
        .filter(|&s| entries[s].key.level() == fine_level)
        .collect();
    fine.sort_unstable();
    let lit: std::collections::HashSet<usize> = fine[..fine.len() / 2].iter().copied().collect();
    for &s in &occupied {
        if entries[s].key.level() == fine_level && !lit.contains(&s) {
            continue;
        }
        entries[s].history = (entries[s].history & 0xffff_0000) | 0x0101;
        entries[s].direct = [1.0 + s as f32, 0.0, 0.0];
        entries[s].indirect = [0.0, 6.5, 0.0];
        entries[s].bounced = [0.0, 0.5, 0.0];
        // Its rays found every way open (f16 65504, the snippet's
        // RADIANCE_CACHE_FAR, in each half).
        entries[s].clear = [0x7bff_7bff; 4];
    }
    gpu.queue
        .write_buffer(&cache.entries, 0, bytemuck::cast_slice(&entries));
    let out = gpu.run(&cache, 2, &floor, groups);
    for (i, q) in floor.iter().enumerate() {
        let w = &out[i * 12..i * 12 + 4];
        let p = [q.p[0], q.p[1], q.p[2]];
        let key = frame.key(p, [0.0, 1.0, 0.0], fine_level, 1);
        let slot = occupied
            .iter()
            .copied()
            .find(|&s| entries[s].key == key)
            .unwrap();
        assert_eq!(w[0], 1, "query {i} found nothing");
        if lit.contains(&slot) {
            assert_eq!(
                (w[1], f32::from_bits(w[2])),
                (fine_level, 1.0 + slot as f32)
            );
        } else {
            // Its own cell has no value: the ancestor's.
            assert_eq!(w[1], fine_level + ANCESTOR, "query {i}");
            assert_eq!(f32::from_bits(w[3]), 7.0);
        }
    }
    // Where a cell's light holds: a point a third of a cell from a lit fine
    // cell's point reads it while its rays found its ways open, and the
    // ancestor once they found something a millimetre away every way (an
    // occluder between): the cell's light is its point's side's.
    let (i, q) = floor
        .iter()
        .enumerate()
        .find(|(_, q)| {
            let key = frame.key([q.p[0], q.p[1], q.p[2]], [0.0, 1.0, 0.0], fine_level, 1);
            occupied
                .iter()
                .any(|&s| entries[s].key == key && lit.contains(&s))
        })
        .unwrap();
    let size = radiance_cache::cell_size(fine_level);
    let point = entries[occupied
        .iter()
        .copied()
        .find(|&s| {
            entries[s].key == frame.key([q.p[0], q.p[1], q.p[2]], [0.0, 1.0, 0.0], fine_level, 1)
        })
        .unwrap()]
    .point_in_cell();
    // A third of a cell along x, staying in the cell.
    let dx = if point[0] < 0.5 {
        size / 3.0
    } else {
        -size / 3.0
    };
    let aside = [Query {
        p: [q.p[0] + dx, q.p[1], q.p[2], q.p[3]],
        n: q.n,
    }];
    let out = gpu.run(&cache, 2, &aside, 1);
    assert_eq!((out[0], out[1]), (1, fine_level), "query {i}: open ways");
    let s = occupied
        .iter()
        .copied()
        .find(|&s| {
            entries[s].key == frame.key([q.p[0], q.p[1], q.p[2]], [0.0, 1.0, 0.0], fine_level, 1)
        })
        .unwrap();
    // f16 0.001 (0x1419) in each half.
    entries[s].clear = [0x1419_1419; 4];
    gpu.queue
        .write_buffer(&cache.entries, 0, bytemuck::cast_slice(&entries));
    let out = gpu.run(&cache, 2, &aside, 1);
    assert_eq!(
        (out[0], out[1], f32::from_bits(out[3])),
        (1, fine_level + ANCESTOR, 7.0),
        "query {i}: blocked ways"
    );
    entries[s].clear = [0x7bff_7bff; 4];
    gpu.queue
        .write_buffer(&cache.entries, 0, bytemuck::cast_slice(&entries));
    // The other side, another space: other cells, nothing found.
    let flipped: Vec<Query> = floor
        .iter()
        .map(|q| Query {
            p: q.p,
            n: [0.0, -1.0, 0.0, 1.0],
        })
        .chain(floor.iter().map(|q| Query {
            p: q.p,
            n: [0.0, 1.0, 0.0, 2.0],
        }))
        .collect();
    let out = gpu.run(&cache, 2, &flipped, 2);
    assert!(out.chunks(12).take(flipped.len()).all(|w| w[0] == 0));
    // The stand-in finds nothing; a frame without slots writes nothing.
    let stand_in = radiance_cache::stand_in(&gpu.device);
    let none = Cache {
        frame: stand_in.frame,
        slots: stand_in.slots,
        entries: stand_in.entries,
    };
    gpu.run(&none, 1, &floor, groups);
    let out = gpu.run(&none, 2, &floor, groups);
    assert!(out.chunks(12).take(floor.len()).all(|w| w[0] == 0));
    let empty = gpu.cache(&RadianceCacheFrame::default());
    gpu.run(&empty, 1, &floor, groups);
    assert!(gpu.read(&empty.slots).iter().all(|&w| w == 0));
    assert!(gpu.read(&empty.entries).iter().all(|&w| w == 0));
}
