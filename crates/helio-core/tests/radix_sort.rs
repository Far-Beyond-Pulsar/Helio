//! `helio_core::radix_sort` against a CPU stable sort: counts that do and do
//! not fill their last block, keys with many duplicates (stability: equal
//! keys keep their input order, which the index payload shows), full-range
//! keys, and no elements at all.

use helio_core::cmd::ComputeCmds;
use helio_core::radix_sort::{RadixSort, RADIX_SORT_BLOCK};
use wgpu::util::DeviceExt;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        adapter.request_device(&Default::default()).await.ok()
    })
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn sort_on_gpu(device: &wgpu::Device, queue: &wgpu::Queue, sort: &RadixSort, keys: &[u32], capacity: u32) -> (Vec<u32>, Vec<u32>) {
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let mut padded = keys.to_vec();
    padded.resize(capacity as usize, 0xdead_beef);
    let indices: Vec<u32> = (0..capacity).collect();
    let buffer = |label, data: &[u32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytemuck::cast_slice(data), usage });
    let keys_a = buffer("keys a", &padded);
    let keys_b = buffer("keys b", &vec![0; capacity as usize]);
    let indices_a = buffer("indices a", &indices);
    let indices_b = buffer("indices b", &vec![0; capacity as usize]);
    let count = keys.len() as u32;
    let blocks = count.div_ceil(RADIX_SORT_BLOCK).max(1);
    let frame = buffer("frame", &[count, blocks]);
    let dispatch = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dispatch"),
        contents: bytemuck::cast_slice(&[blocks, 1, 1]),
        usage: wgpu::BufferUsages::INDIRECT,
    });
    let binds = sort.bind(device, &frame, [&keys_a, &keys_b], [&indices_a, &indices_b], capacity);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = ComputeCmds::from_wgpu(encoder.begin_compute_pass(&Default::default()));
        sort.record(&mut pass, &binds, &dispatch, 0);
    }
    let bytes = u64::from(count.max(1)) * 4;
    let read = |label| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let (out_keys, out_indices) = (read("out keys"), read("out indices"));
    encoder.copy_buffer_to_buffer(&keys_a, 0, &out_keys, 0, bytes);
    encoder.copy_buffer_to_buffer(&indices_a, 0, &out_indices, 0, bytes);
    queue.submit([encoder.finish()]);
    let fetch = |b: &wgpu::Buffer| {
        b.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let v: Vec<u32> = bytemuck::cast_slice(&b.slice(..).get_mapped_range().unwrap()).to_vec();
        v[..keys.len()].to_vec()
    };
    (fetch(&out_keys), fetch(&out_indices))
}

#[test]
fn sorts_stably_like_the_cpu() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let sort = RadixSort::new(&device);
    let mut rng = Rng(7);
    let cases: Vec<(&str, Vec<u32>, u32)> = vec![
        ("empty", vec![], 256),
        ("one", vec![42], 256),
        ("partial block, few distinct keys", (0..1000).map(|_| rng.next() % 7).collect(), 1024),
        ("full blocks, duplicates in every digit", (0..4096).map(|_| (rng.next() % 64) * 0x0101_0101).collect(), 4096),
        ("full-range keys", (0..70_001).map(|_| rng.next() ^ (rng.next() << 16)).collect(), 70_400),
        ("all equal", vec![0xabcd_ef01; 3000], 3072),
    ];
    for (name, keys, capacity) in cases {
        let (got_keys, got_indices) = sort_on_gpu(&device, &queue, &sort, &keys, capacity);
        let mut expected: Vec<(u32, u32)> = keys.iter().copied().zip(0..).collect();
        expected.sort_by_key(|p| p.0); // stable
        let (want_keys, want_indices): (Vec<u32>, Vec<u32>) = expected.into_iter().unzip();
        assert_eq!(got_keys, want_keys, "{name}: keys");
        assert_eq!(got_indices, want_indices, "{name}: input order of equal keys");
    }
}
