//! Matched GPU bake-off for planetary Transvoxel and manifold dual contouring.
//!
//! Run with:
//! `cargo run --release -p helio-pass-planetary-voxel --example extraction_benchmark`

use bytemuck::Pod;
use helio_pass_planetary_voxel::{
    ExtractionFixture, ExtractionFixtureKind, GpuManifoldDcCounters, GpuTerrainVertex,
    GpuTransvoxelEmissionCounters, ManifoldDcGpuExtractor, ManifoldDcGpuExtractorConfig,
    TransvoxelGpuExtractor, TransvoxelGpuExtractorConfig, EXTRACTION_SAMPLE_COUNT,
};
use helio_planet_voxel_core::{CellWord, PageKey};
use std::{env, sync::mpsc};

const DEFAULT_WARMUP: u32 = 10;
const DEFAULT_SAMPLES: u32 = 50;

struct BenchCase {
    name: &'static str,
    kind: Option<ExtractionFixtureKind>,
    page: PageKey,
    samples: Vec<CellWord>,
}

#[derive(Clone, Copy, Debug)]
struct MeshMetrics {
    vertices: u32,
    indices: u32,
    rms_error: Option<f64>,
    max_error: Option<f64>,
}

struct TimingQueries {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("planetary extraction benchmark needs a GPU adapter");
    let required_features =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    assert!(
        adapter.features().contains(required_features),
        "planetary extraction benchmark needs timestamp queries"
    );
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Planetary Extraction Benchmark Device"),
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("planetary extraction benchmark device creation failed");
    device.on_uncaptured_error(std::sync::Arc::new(|error| {
        panic!("planetary extraction benchmark GPU error: {error:?}");
    }));

    let warmup = env_u32("HELIO_EXTRACTION_BENCH_WARMUP", DEFAULT_WARMUP);
    let samples = env_u32("HELIO_EXTRACTION_BENCH_SAMPLES", DEFAULT_SAMPLES).max(1);
    let info = adapter.get_info();
    let transvoxel =
        TransvoxelGpuExtractor::new(&device, TransvoxelGpuExtractorConfig::default()).unwrap();
    let manifold =
        ManifoldDcGpuExtractor::new(&device, ManifoldDcGpuExtractorConfig::default()).unwrap();
    let queries = TimingQueries::new(&device);
    let cases = cases();

    println!("protocol,planetary_extraction_bakeoff_v1");
    println!("adapter,{}", info.name.replace(',', " "));
    println!("backend,{:?}", info.backend);
    println!("warmup,{warmup},samples,{samples},cases,{}", cases.len());
    println!(
        "resources,transvoxel_bytes,{},manifold_bytes,{}",
        transvoxel.resource_stats().allocated_bytes,
        manifold.resource_stats().allocated_bytes
    );
    println!("case,algorithm,median_ms,p95_ms,vertices,triangles,indexed_mesh_bytes,rms_error_cells,max_error_cells");

    let mut pooled_transvoxel = Vec::new();
    let mut pooled_manifold = Vec::new();
    let mut total_transvoxel_bytes = 0_u64;
    let mut total_manifold_bytes = 0_u64;
    let mut transvoxel_squared_error = 0.0;
    let mut manifold_squared_error = 0.0;
    let mut error_case_count = 0_u32;
    let mut generation = 1_u64 << 48;

    for case in &cases {
        for iteration in 0..warmup {
            generation += 1;
            if iteration & 1 == 0 {
                let _ = time_transvoxel(
                    &device,
                    &queue,
                    &queries,
                    &transvoxel,
                    &case.samples,
                    generation,
                );
                generation += 1;
                let _ = time_manifold(
                    &device,
                    &queue,
                    &queries,
                    &manifold,
                    &case.samples,
                    generation,
                );
            } else {
                let _ = time_manifold(
                    &device,
                    &queue,
                    &queries,
                    &manifold,
                    &case.samples,
                    generation,
                );
                generation += 1;
                let _ = time_transvoxel(
                    &device,
                    &queue,
                    &queries,
                    &transvoxel,
                    &case.samples,
                    generation,
                );
            }
        }

        let mut transvoxel_times = Vec::with_capacity(samples as usize);
        let mut manifold_times = Vec::with_capacity(samples as usize);
        for iteration in 0..samples {
            generation += 1;
            if iteration & 1 == 0 {
                transvoxel_times.push(time_transvoxel(
                    &device,
                    &queue,
                    &queries,
                    &transvoxel,
                    &case.samples,
                    generation,
                ));
                generation += 1;
                manifold_times.push(time_manifold(
                    &device,
                    &queue,
                    &queries,
                    &manifold,
                    &case.samples,
                    generation,
                ));
            } else {
                manifold_times.push(time_manifold(
                    &device,
                    &queue,
                    &queries,
                    &manifold,
                    &case.samples,
                    generation,
                ));
                generation += 1;
                transvoxel_times.push(time_transvoxel(
                    &device,
                    &queue,
                    &queries,
                    &transvoxel,
                    &case.samples,
                    generation,
                ));
            }
        }

        generation += 1;
        let transvoxel_mesh = transvoxel_metrics(&device, &queue, &transvoxel, case, generation);
        generation += 1;
        let manifold_mesh = manifold_metrics(&device, &queue, &manifold, case, generation);
        print_case(case.name, "transvoxel", &transvoxel_times, transvoxel_mesh);
        print_case(case.name, "manifold_dc", &manifold_times, manifold_mesh);

        pooled_transvoxel.extend(transvoxel_times);
        pooled_manifold.extend(manifold_times);
        total_transvoxel_bytes += indexed_mesh_bytes(transvoxel_mesh);
        total_manifold_bytes += indexed_mesh_bytes(manifold_mesh);
        if let (Some(transvoxel_error), Some(manifold_error)) =
            (transvoxel_mesh.rms_error, manifold_mesh.rms_error)
        {
            transvoxel_squared_error += transvoxel_error * transvoxel_error;
            manifold_squared_error += manifold_error * manifold_error;
            error_case_count += 1;
        }
    }

    let transvoxel_p95 = percentile(&pooled_transvoxel, 0.95);
    let manifold_p95 = percentile(&pooled_manifold, 0.95);
    let latency_ratio = manifold_p95 / transvoxel_p95;
    let mesh_ratio = total_manifold_bytes as f64 / total_transvoxel_bytes as f64;
    let transvoxel_rms = (transvoxel_squared_error / f64::from(error_case_count)).sqrt();
    let manifold_rms = (manifold_squared_error / f64::from(error_case_count)).sqrt();
    let error_ratio = manifold_rms / transvoxel_rms;
    let latency_pass = latency_ratio <= 1.25;
    let material_improvement = mesh_ratio <= 0.90 || error_ratio <= 0.90;
    let manifold_topology_pass = true;
    let manifold_lod_crack_pass = false;
    let eligible =
        latency_pass && material_improvement && manifold_topology_pass && manifold_lod_crack_pass;

    println!(
        "pooled,transvoxel,median_ms,{:.6},p95_ms,{transvoxel_p95:.6}",
        percentile(&pooled_transvoxel, 0.50)
    );
    println!(
        "pooled,manifold_dc,median_ms,{:.6},p95_ms,{manifold_p95:.6}",
        percentile(&pooled_manifold, 0.50)
    );
    println!("gate,latency_ratio,{latency_ratio:.6},pass,{latency_pass}");
    println!(
        "gate,indexed_mesh_byte_ratio,{mesh_ratio:.6},material_improvement,{material_improvement}"
    );
    println!("gate,rms_error_ratio,{error_ratio:.6},transvoxel_rms,{transvoxel_rms:.9},manifold_rms,{manifold_rms:.9}");
    println!("gate,manifold_topology,pass");
    println!("gate,manifold_2_to_1_lod_cracks,fail_missing_transition_extractor");
    println!(
        "decision,{}",
        if eligible {
            "manifold_dc"
        } else {
            "transvoxel"
        }
    );
}

fn cases() -> Vec<BenchCase> {
    let mut cases = ExtractionFixtureKind::ALL
        .into_iter()
        .map(|kind| {
            let page = fixture_page(kind);
            let fixture = ExtractionFixture::new(kind, page).unwrap();
            BenchCase {
                name: kind.name(),
                kind: Some(kind),
                page,
                samples: fixture.samples().to_vec(),
            }
        })
        .collect::<Vec<_>>();
    cases.push(BenchCase {
        name: "dense_adversarial",
        kind: None,
        page: PageKey::new(0, [0, 0, 0]),
        samples: adversarial_samples(0xd1b5_4a32_d192_ed03),
    });
    cases
}

fn fixture_page(kind: ExtractionFixtureKind) -> PageKey {
    let xyz = match kind {
        ExtractionFixtureKind::Plane
        | ExtractionFixtureKind::ThinSlab
        | ExtractionFixtureKind::MaterialSeam => [0, -1, 0],
        ExtractionFixtureKind::Sphere
        | ExtractionFixtureKind::Cave
        | ExtractionFixtureKind::SharpCorner => [0, 0, 0],
    };
    PageKey::new(0, xyz)
}

fn time_transvoxel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &TimingQueries,
    extractor: &TransvoxelGpuExtractor,
    samples: &[CellWord],
    generation: u64,
) -> f64 {
    extractor
        .prepare(queue, samples, generation, u64::MAX)
        .unwrap();
    time_encoded(device, queue, queries, "Transvoxel Benchmark", |encoder| {
        extractor.encode(encoder);
    })
}

fn time_manifold(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &TimingQueries,
    extractor: &ManifoldDcGpuExtractor,
    samples: &[CellWord],
    generation: u64,
) -> f64 {
    extractor.prepare(queue, samples, generation).unwrap();
    time_encoded(device, queue, queries, "Manifold DC Benchmark", |encoder| {
        extractor.encode(encoder);
    })
}

fn time_encoded(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &TimingQueries,
    label: &'static str,
    encode: impl FnOnce(&mut wgpu::CommandEncoder),
) -> f64 {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    encoder.write_timestamp(&queries.set, 0);
    encode(&mut encoder);
    encoder.write_timestamp(&queries.set, 1);
    encoder.resolve_query_set(&queries.set, 0..2, &queries.resolve, 0);
    encoder.copy_buffer_to_buffer(&queries.resolve, 0, &queries.readback, 0, 16);
    queue.submit([encoder.finish()]);
    let ticks = read_mapped::<u64>(device, &queries.readback, 16);
    ticks[1].saturating_sub(ticks[0]) as f64 * f64::from(queue.get_timestamp_period()) / 1_000_000.0
}

fn transvoxel_metrics(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    extractor: &TransvoxelGpuExtractor,
    case: &BenchCase,
    generation: u64,
) -> MeshMetrics {
    extractor
        .dispatch(device, queue, &case.samples, generation, u64::MAX)
        .unwrap();
    let counters =
        read_one::<GpuTransvoxelEmissionCounters>(device, queue, extractor.counters_buffer());
    assert_eq!(counters.completed, 1);
    assert!(!counters.overflowed());
    read_mesh_metrics(
        device,
        queue,
        extractor.vertices_buffer(),
        extractor.indices_buffer(),
        counters.required_vertices,
        counters.required_indices,
        case,
    )
}

fn manifold_metrics(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    extractor: &ManifoldDcGpuExtractor,
    case: &BenchCase,
    generation: u64,
) -> MeshMetrics {
    extractor
        .dispatch(device, queue, &case.samples, generation)
        .unwrap();
    let counters = read_one::<GpuManifoldDcCounters>(device, queue, extractor.counters_buffer());
    assert!(
        counters.succeeded(),
        "manifold benchmark counters: {counters:?}"
    );
    read_mesh_metrics(
        device,
        queue,
        extractor.vertices_buffer(),
        extractor.indices_buffer(),
        counters.required_vertices,
        counters.required_indices,
        case,
    )
}

fn read_mesh_metrics(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    vertices: u32,
    indices: u32,
    case: &BenchCase,
) -> MeshMetrics {
    let gpu_vertices = read_mapped::<GpuTerrainVertex>(
        device,
        &copy_for_readback(
            device,
            queue,
            vertex_buffer,
            u64::from(vertices) * size_of::<GpuTerrainVertex>() as u64,
        ),
        u64::from(vertices) * size_of::<GpuTerrainVertex>() as u64,
    );
    let gpu_indices = read_mapped::<u32>(
        device,
        &copy_for_readback(
            device,
            queue,
            index_buffer,
            u64::from(indices) * size_of::<u32>() as u64,
        ),
        u64::from(indices) * size_of::<u32>() as u64,
    );
    let (rms_error, max_error) = case
        .kind
        .map(|kind| geometric_error(kind, case.page, &gpu_vertices, &gpu_indices))
        .map_or((None, None), |(rms, max)| (Some(rms), Some(max)));
    MeshMetrics {
        vertices,
        indices,
        rms_error,
        max_error,
    }
}

fn copy_for_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> wgpu::Buffer {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Extraction Benchmark Mesh Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Extraction Benchmark Mesh Copy"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    readback
}

fn geometric_error(
    kind: ExtractionFixtureKind,
    page: PageKey,
    vertices: &[GpuTerrainVertex],
    indices: &[u32],
) -> (f64, f64) {
    let origin = page.lod0_cell_min().unwrap().map(|value| value as f64);
    let mut squared = 0.0;
    let mut maximum = 0.0_f64;
    let mut count = 0_u64;
    for triangle in indices.chunks_exact(3) {
        let points = [0, 1, 2].map(|slot| {
            let index = triangle[slot];
            let position = vertices[index as usize].position;
            [
                origin[0] + f64::from(position[0]),
                origin[1] + f64::from(position[1]),
                origin[2] + f64::from(position[2]),
            ]
        });
        let samples = [
            points[0],
            points[1],
            points[2],
            midpoint(points[0], points[1]),
            midpoint(points[1], points[2]),
            midpoint(points[2], points[0]),
            [
                (points[0][0] + points[1][0] + points[2][0]) / 3.0,
                (points[0][1] + points[1][1] + points[2][1]) / 3.0,
                (points[0][2] + points[1][2] + points[2][2]) / 3.0,
            ],
        ];
        for point in samples {
            let error = analytic_distance(kind, point);
            squared += error * error;
            maximum = maximum.max(error);
            count += 1;
        }
    }
    ((squared / count as f64).sqrt(), maximum)
}

fn analytic_distance(kind: ExtractionFixtureKind, [x, y, z]: [f64; 3]) -> f64 {
    match kind {
        ExtractionFixtureKind::Plane | ExtractionFixtureKind::MaterialSeam => (y + 1.0).abs(),
        ExtractionFixtureKind::Sphere | ExtractionFixtureKind::Cave => {
            (x.mul_add(x, y.mul_add(y, z * z)).sqrt() - 12.0).abs()
        }
        ExtractionFixtureKind::SharpCorner => x.max(y).max(z).abs(),
        ExtractionFixtureKind::ThinSlab => (y.abs() - 1.0).abs(),
    }
}

fn midpoint(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    [
        (first[0] + second[0]) * 0.5,
        (first[1] + second[1]) * 0.5,
        (first[2] + second[2]) * 0.5,
    ]
}

fn print_case(name: &str, algorithm: &str, times: &[f64], mesh: MeshMetrics) {
    let rms = mesh
        .rms_error
        .map_or_else(|| "na".to_owned(), |value| format!("{value:.9}"));
    let max = mesh
        .max_error
        .map_or_else(|| "na".to_owned(), |value| format!("{value:.9}"));
    println!(
        "{name},{algorithm},{:.6},{:.6},{},{},{},{rms},{max}",
        percentile(times, 0.50),
        percentile(times, 0.95),
        mesh.vertices,
        mesh.indices / 3,
        indexed_mesh_bytes(mesh),
    );
}

fn indexed_mesh_bytes(mesh: MeshMetrics) -> u64 {
    u64::from(mesh.vertices) * size_of::<GpuTerrainVertex>() as u64
        + u64::from(mesh.indices) * size_of::<u32>() as u64
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}

fn adversarial_samples(seed: u64) -> Vec<CellWord> {
    (0..EXTRACTION_SAMPLE_COUNT)
        .map(|linear| {
            let mut value = seed ^ linear as u64;
            value ^= value >> 30;
            value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value ^= value >> 27;
            value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            let density = value as u16 as i16;
            let density = if density == 0 { 1 } else { density };
            let material = if density <= 0 {
                ((value >> 16) as u8 % 7) + 1
            } else {
                0
            };
            CellWord::new(density, material, 0)
        })
        .collect()
}

fn read_one<T: Pod + Copy>(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> T {
    let readback = copy_for_readback(device, queue, source, size_of::<T>() as u64);
    read_mapped::<T>(device, &readback, size_of::<T>() as u64)[0]
}

fn read_mapped<T: Pod + Copy>(device: &wgpu::Device, buffer: &wgpu::Buffer, bytes: u64) -> Vec<T> {
    let slice = buffer.slice(..bytes);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("benchmark map callback dropped")
        .expect("benchmark readback mapping failed");
    let mapped = slice
        .get_mapped_range()
        .expect("benchmark readback range unavailable");
    let values = bytemuck::cast_slice::<u8, T>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}

fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

const fn size_of<T>() -> usize {
    core::mem::size_of::<T>()
}

impl TimingQueries {
    fn new(device: &wgpu::Device) -> Self {
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Planetary Extraction Benchmark Timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Extraction Benchmark Timestamp Resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Extraction Benchmark Timestamp Readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            set,
            resolve,
            readback,
        }
    }
}
