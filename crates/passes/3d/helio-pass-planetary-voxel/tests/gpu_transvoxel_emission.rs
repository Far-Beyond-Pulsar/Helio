use bytemuck::Pod;
use helio_pass_planetary_voxel::{
    regular_case_from_fixture, ExtractionFixture, ExtractionFixtureKind, GpuTerrainVertex,
    GpuTransvoxelCellOffset, GpuTransvoxelEmissionCounters, GpuTransvoxelScanBlock,
    TransvoxelGpuExtractor, TransvoxelGpuExtractorConfig, TRANSVOXEL_REGULAR_CORNERS,
    TRANSVOXEL_SCAN_WORKGROUP_SIZE,
};
use helio_planet_voxel_core::{PageKey, PAGE_CELL_COUNT, PAGE_EDGE};
use std::sync::mpsc;

#[test]
fn headless_transvoxel_emission_matches_cpu_geometry_and_overflow_contract() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!(
                "GPU_VALIDATION_SKIPPED_NO_ADAPTER: no primary or fallback adapter available"
            );
            return;
        };
        let adapter_info = adapter.get_info();
        eprintln!(
            "GPU_VALIDATION_ADAPTER: name={:?} backend={:?} device_type={:?}",
            adapter_info.name, adapter_info.backend, adapter_info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Planetary Transvoxel Emission Validation Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("planetary Transvoxel emission GPU validation error: {error:?}");
        }));

        let mut extractor =
            TransvoxelGpuExtractor::new(&device, TransvoxelGpuExtractorConfig::default()).unwrap();
        assert_eq!(extractor.resource_stats().buffers, 12);
        assert_eq!(extractor.resource_stats().allocated_bytes, 15_771_248);
        let resources = extractor.resource_stats();
        extractor.resize(3840, 2160);
        assert_eq!(extractor.resource_stats(), resources);

        for (fixture_index, kind) in ExtractionFixtureKind::ALL.into_iter().enumerate() {
            let generation = 1_000 + fixture_index as u64;
            let fixture = fixture(kind);
            let expected = expected_mesh(&fixture, u64::MAX);
            extractor
                .dispatch(&device, &queue, fixture.samples(), generation, u64::MAX)
                .unwrap();
            let counters = read_one::<GpuTransvoxelEmissionCounters>(
                &device,
                &queue,
                extractor.counters_buffer(),
            );
            assert_eq!(counters.completed, 1, "{} completed", kind.name());
            assert!(!counters.overflowed(), "{} overflow", kind.name());
            assert_eq!(
                counters.required_vertices as usize,
                expected.vertices.len(),
                "{} vertex count",
                kind.name()
            );
            assert_eq!(
                counters.required_indices as usize,
                expected.indices.len(),
                "{} index count",
                kind.name()
            );
            assert_eq!(counters.emitted_vertices, counters.required_vertices);
            assert_eq!(counters.emitted_indices, counters.required_indices);

            let actual_vertices: Vec<GpuTerrainVertex> = read_buffer(
                &device,
                &queue,
                extractor.vertices_buffer(),
                counters.required_vertices as u64 * size_of::<GpuTerrainVertex>() as u64,
            );
            let actual_indices: Vec<u32> = read_buffer(
                &device,
                &queue,
                extractor.indices_buffer(),
                counters.required_indices as u64 * size_of::<u32>() as u64,
            );
            assert_vertices_close(kind, &actual_vertices, &expected.vertices);
            assert_eq!(actual_indices, expected.indices, "{} indices", kind.name());

            let offsets: Vec<GpuTransvoxelCellOffset> = read_buffer(
                &device,
                &queue,
                extractor.offsets_buffer(),
                PAGE_CELL_COUNT as u64 * size_of::<GpuTransvoxelCellOffset>() as u64,
            );
            let blocks: Vec<GpuTransvoxelScanBlock> = read_buffer(
                &device,
                &queue,
                extractor.blocks_buffer(),
                128 * size_of::<GpuTransvoxelScanBlock>() as u64,
            );
            for (linear, offset) in offsets.iter().enumerate() {
                assert_eq!(
                    offset.generation(),
                    generation,
                    "{} cell {linear}",
                    kind.name()
                );
                let block = blocks[linear / TRANSVOXEL_SCAN_WORKGROUP_SIZE as usize];
                assert_eq!(
                    block.first_vertex + offset.first_vertex,
                    expected.cell_ranges[linear].unwrap().0,
                    "{} vertex range cell {linear}",
                    kind.name()
                );
                assert_eq!(
                    block.first_index + offset.first_index,
                    expected.cell_ranges[linear].unwrap().1,
                    "{} index range cell {linear}",
                    kind.name()
                );
            }

            extractor
                .dispatch(&device, &queue, fixture.samples(), generation, u64::MAX)
                .unwrap();
            let repeated_vertices: Vec<GpuTerrainVertex> = read_buffer(
                &device,
                &queue,
                extractor.vertices_buffer(),
                counters.required_vertices as u64 * size_of::<GpuTerrainVertex>() as u64,
            );
            let repeated_indices: Vec<u32> = read_buffer(
                &device,
                &queue,
                extractor.indices_buffer(),
                counters.required_indices as u64 * size_of::<u32>() as u64,
            );
            assert_eq!(
                bytemuck::cast_slice::<GpuTerrainVertex, u8>(&repeated_vertices),
                bytemuck::cast_slice::<GpuTerrainVertex, u8>(&actual_vertices),
                "{} deterministic vertices",
                kind.name()
            );
            assert_eq!(
                repeated_indices,
                actual_indices,
                "{} deterministic indices",
                kind.name()
            );
        }

        let fixture = fixture(ExtractionFixtureKind::Plane);
        let dirty_generation = 1_500;
        let dirty_microbricks = 1_u64 << 12;
        let dirty_expected = expected_mesh(&fixture, dirty_microbricks);
        extractor
            .dispatch(
                &device,
                &queue,
                fixture.samples(),
                dirty_generation,
                dirty_microbricks,
            )
            .unwrap();
        let dirty_counters =
            read_one::<GpuTransvoxelEmissionCounters>(&device, &queue, extractor.counters_buffer());
        assert!(!dirty_counters.overflowed());
        assert_eq!(
            dirty_counters.emitted_vertices as usize,
            dirty_expected.vertices.len()
        );
        assert_eq!(
            dirty_counters.emitted_indices as usize,
            dirty_expected.indices.len()
        );
        assert!(dirty_counters.emitted_vertices > 0);
        let dirty_vertices: Vec<GpuTerrainVertex> = read_buffer(
            &device,
            &queue,
            extractor.vertices_buffer(),
            u64::from(dirty_counters.emitted_vertices) * size_of::<GpuTerrainVertex>() as u64,
        );
        let dirty_indices: Vec<u32> = read_buffer(
            &device,
            &queue,
            extractor.indices_buffer(),
            u64::from(dirty_counters.emitted_indices) * size_of::<u32>() as u64,
        );
        assert_vertices_close(
            ExtractionFixtureKind::Plane,
            &dirty_vertices,
            &dirty_expected.vertices,
        );
        assert_eq!(dirty_indices, dirty_expected.indices);
        let dirty_offsets: Vec<GpuTransvoxelCellOffset> = read_buffer(
            &device,
            &queue,
            extractor.offsets_buffer(),
            PAGE_CELL_COUNT as u64 * size_of::<GpuTransvoxelCellOffset>() as u64,
        );
        for (linear, (offset, expected_range)) in dirty_offsets
            .iter()
            .zip(&dirty_expected.cell_ranges)
            .enumerate()
        {
            if expected_range.is_some() {
                assert_eq!(offset.generation(), dirty_generation, "dirty cell {linear}");
            } else {
                assert_ne!(offset.generation(), dirty_generation, "stale cell {linear}");
            }
        }

        let tiny =
            TransvoxelGpuExtractor::new(&device, TransvoxelGpuExtractorConfig::new(1, 1).unwrap())
                .unwrap();
        tiny.dispatch(&device, &queue, fixture.samples(), 2_000, u64::MAX)
            .unwrap();
        let counters =
            read_one::<GpuTransvoxelEmissionCounters>(&device, &queue, tiny.counters_buffer());
        assert_eq!(counters.completed, 1);
        assert!(counters.vertex_overflow != 0);
        assert!(counters.index_overflow != 0);
        assert_eq!(counters.emitted_vertices, 0);
        assert_eq!(counters.emitted_indices, 0);
        assert!(counters.required_vertices > 1);
        assert!(counters.required_indices > 1);
    });
}

struct ExpectedMesh {
    vertices: Vec<GpuTerrainVertex>,
    indices: Vec<u32>,
    cell_ranges: Vec<Option<(u32, u32)>>,
}

fn expected_mesh(fixture: &ExtractionFixture, dirty_microbricks: u64) -> ExpectedMesh {
    let mut mesh = ExpectedMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
        cell_ranges: vec![None; PAGE_CELL_COUNT],
    };
    for z in 0..PAGE_EDGE as u8 {
        for y in 0..PAGE_EDGE as u8 {
            for x in 0..PAGE_EDGE as u8 {
                let linear = usize::from(x)
                    + usize::from(y) * PAGE_EDGE
                    + usize::from(z) * PAGE_EDGE * PAGE_EDGE;
                let microbrick = u32::from(x / 8) + u32::from(y / 8) * 4 + u32::from(z / 8) * 16;
                if dirty_microbricks & (1_u64 << microbrick) == 0 {
                    continue;
                }
                let first_vertex = mesh.vertices.len() as u32;
                let first_index = mesh.indices.len() as u32;
                mesh.cell_ranges[linear] = Some((first_vertex, first_index));
                let fixture_case = fixture.cell_case([x, y, z]).unwrap();
                let topology = regular_case_from_fixture(fixture_case);
                for vertex_index in 0..topology.vertex_count() {
                    let code = topology.vertex(vertex_index).unwrap();
                    let [first_corner, second_corner] = code.endpoints();
                    let cell = [i32::from(x), i32::from(y), i32::from(z)];
                    let first = add(cell, TRANSVOXEL_REGULAR_CORNERS[usize::from(first_corner)]);
                    let second = add(cell, TRANSVOXEL_REGULAR_CORNERS[usize::from(second_corner)]);
                    let first_word = fixture.sample(first).unwrap();
                    let second_word = fixture.sample(second).unwrap();
                    let first_density = f32::from(first_word.density());
                    let second_density = f32::from(second_word.density());
                    let denominator = first_density - second_density;
                    let interpolation = if denominator.abs() > 1.0e-12 {
                        (first_density / denominator).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let first_position = first.map(|axis| axis as f32);
                    let second_position = second.map(|axis| axis as f32);
                    let position = mix(first_position, second_position, interpolation);
                    let normal = normalize_or_up(mix(
                        gradient(fixture, first),
                        gradient(fixture, second),
                        interpolation,
                    ));
                    let material = if first_density <= 0.0 {
                        first_word.material()
                    } else {
                        second_word.material()
                    };
                    mesh.vertices.push(GpuTerrainVertex {
                        position,
                        material: u32::from(material),
                        normal,
                        flags: 0,
                    });
                }
                for triangle in 0..topology.triangle_count() {
                    for local_vertex in topology.triangle(triangle).unwrap() {
                        mesh.indices.push(first_vertex + u32::from(local_vertex));
                    }
                }
            }
        }
    }
    mesh
}

fn gradient(fixture: &ExtractionFixture, position: [i32; 3]) -> [f32; 3] {
    let mut result = [0.0; 3];
    for axis in 0..3 {
        let mut lower = position;
        let mut upper = position;
        result[axis] = if position[axis] <= -1 {
            upper[axis] += 1;
            density_at(fixture, upper) - density_at(fixture, position)
        } else if position[axis] >= PAGE_EDGE as i32 {
            lower[axis] -= 1;
            density_at(fixture, position) - density_at(fixture, lower)
        } else {
            lower[axis] -= 1;
            upper[axis] += 1;
            (density_at(fixture, upper) - density_at(fixture, lower)) * 0.5
        };
    }
    result
}

fn density_at(fixture: &ExtractionFixture, position: [i32; 3]) -> f32 {
    f32::from(fixture.sample(position).unwrap().density())
}

fn add(left: [i32; 3], right: [u8; 3]) -> [i32; 3] {
    [
        left[0] + i32::from(right[0]),
        left[1] + i32::from(right[1]),
        left[2] + i32::from(right[2]),
    ]
}

fn mix(first: [f32; 3], second: [f32; 3], amount: f32) -> [f32; 3] {
    [
        first[0] + (second[0] - first[0]) * amount,
        first[1] + (second[1] - first[1]) * amount,
        first[2] + (second[2] - first[2]) * amount,
    ]
}

fn normalize_or_up(value: [f32; 3]) -> [f32; 3] {
    let squared_length = value.into_iter().map(|axis| axis * axis).sum::<f32>();
    if squared_length > 1.0e-12 {
        let inverse_length = squared_length.sqrt().recip();
        value.map(|axis| axis * inverse_length)
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn assert_vertices_close(
    kind: ExtractionFixtureKind,
    actual: &[GpuTerrainVertex],
    expected: &[GpuTerrainVertex],
) {
    assert_eq!(actual.len(), expected.len(), "{} vertices", kind.name());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.material,
            expected.material,
            "{} material {index}",
            kind.name()
        );
        assert_eq!(
            actual.flags,
            expected.flags,
            "{} flags {index}",
            kind.name()
        );
        for axis in 0..3 {
            assert!(
                (actual.position[axis] - expected.position[axis]).abs() <= 1.0e-5,
                "{} position {index}:{axis}: {:?} != {:?}",
                kind.name(),
                actual.position,
                expected.position
            );
            assert!(
                (actual.normal[axis] - expected.normal[axis]).abs() <= 2.0e-4,
                "{} normal {index}:{axis}: {:?} != {:?}",
                kind.name(),
                actual.normal,
                expected.normal
            );
        }
    }
}

fn fixture(kind: ExtractionFixtureKind) -> ExtractionFixture {
    let page_xyz = match kind {
        ExtractionFixtureKind::Plane
        | ExtractionFixtureKind::ThinSlab
        | ExtractionFixtureKind::MaterialSeam => [0, -1, 0],
        ExtractionFixtureKind::Sphere
        | ExtractionFixtureKind::Cave
        | ExtractionFixtureKind::SharpCorner => [0, 0, 0],
    };
    ExtractionFixture::new(kind, PageKey::new(0, page_xyz)).unwrap()
}

async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force_fallback_adapter in [false, true] {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            return Some(adapter);
        }
    }
    None
}

fn read_one<T: Pod + Copy>(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> T {
    read_buffer(device, queue, source, size_of::<T>() as u64)[0]
}

fn read_buffer<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> Vec<T> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Transvoxel Emission Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Transvoxel Emission Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("GPU readback callback must run")
        .expect("GPU readback mapping must succeed");
    let mapped = slice
        .get_mapped_range()
        .expect("GPU readback range must be available");
    let values = bytemuck::cast_slice::<u8, T>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}

const fn size_of<T>() -> usize {
    core::mem::size_of::<T>()
}
