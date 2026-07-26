use bytemuck::Pod;
use helio_pass_planetary_voxel::{
    regular_case_from_fixture, ExtractionFixture, ExtractionFixtureKind, GpuTransvoxelCell,
    GpuTransvoxelClassifyCounters, TransvoxelGpuClassifier, TransvoxelGpuError,
};
use helio_planet_voxel_core::{PageKey, PAGE_CELL_COUNT, PAGE_EDGE};
use std::sync::mpsc;

#[test]
fn headless_transvoxel_classification_matches_every_fixture_cell() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = request_test_adapter(&instance).await;
        let Some(adapter) = adapter else {
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
                label: Some("Planetary Transvoxel Validation Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("planetary Transvoxel GPU validation error: {error:?}");
        }));

        let mut classifier = TransvoxelGpuClassifier::new(&device).unwrap();
        assert_eq!(classifier.resource_stats().buffers, 6);
        assert_eq!(classifier.resource_stats().allocated_bytes, 682_640);
        let resources = classifier.resource_stats();
        classifier.resize(3840, 2160);
        assert_eq!(classifier.resource_stats(), resources);

        for (fixture_index, kind) in ExtractionFixtureKind::ALL.into_iter().enumerate() {
            let generation = 10 + fixture_index as u64;
            let fixture = fixture(kind);
            let expected = expected_classification(&fixture, generation, u64::MAX);
            classifier
                .dispatch(&device, &queue, fixture.samples(), generation, u64::MAX)
                .unwrap();
            let actual_cells: Vec<GpuTransvoxelCell> = read_buffer(
                &device,
                &queue,
                classifier.output_buffer(),
                PAGE_CELL_COUNT as u64 * size_of::<GpuTransvoxelCell>() as u64,
            );
            let actual_counters: Vec<GpuTransvoxelClassifyCounters> = read_buffer(
                &device,
                &queue,
                classifier.counters_buffer(),
                size_of::<GpuTransvoxelClassifyCounters>() as u64,
            );
            assert_eq!(actual_cells, expected.0, "{} cell parity", kind.name());
            assert_eq!(
                actual_counters,
                vec![expected.1],
                "{} counters",
                kind.name()
            );

            classifier
                .dispatch(&device, &queue, fixture.samples(), generation, u64::MAX)
                .unwrap();
            let repeated: Vec<GpuTransvoxelCell> = read_buffer(
                &device,
                &queue,
                classifier.output_buffer(),
                PAGE_CELL_COUNT as u64 * size_of::<GpuTransvoxelCell>() as u64,
            );
            assert_eq!(repeated, actual_cells, "{} deterministic", kind.name());
        }

        let fixture = fixture(ExtractionFixtureKind::Plane);
        let generation = 100;
        let dirty_microbricks = 1_u64 << 12;
        let expected = expected_classification(&fixture, generation, dirty_microbricks);
        classifier
            .dispatch(
                &device,
                &queue,
                fixture.samples(),
                generation,
                dirty_microbricks,
            )
            .unwrap();
        let cells: Vec<GpuTransvoxelCell> = read_buffer(
            &device,
            &queue,
            classifier.output_buffer(),
            PAGE_CELL_COUNT as u64 * size_of::<GpuTransvoxelCell>() as u64,
        );
        let counters: Vec<GpuTransvoxelClassifyCounters> = read_buffer(
            &device,
            &queue,
            classifier.counters_buffer(),
            size_of::<GpuTransvoxelClassifyCounters>() as u64,
        );
        for (linear, (cell, expected_cell)) in cells.iter().zip(&expected.0).enumerate() {
            let x = linear % PAGE_EDGE;
            let y = (linear / PAGE_EDGE) % PAGE_EDGE;
            let z = linear / (PAGE_EDGE * PAGE_EDGE);
            let microbrick = x / 8 + (y / 8) * 4 + (z / 8) * 16;
            if microbrick == 12 {
                assert_eq!(cell, expected_cell, "dirty cell {linear}");
                assert!(cell.is_valid_for(generation));
            } else {
                assert!(!cell.is_valid_for(generation), "stale cell {linear}");
            }
        }
        assert_eq!(counters, vec![expected.1]);
        assert_eq!(counters[0].visited_cells, 8 * 8 * 8);
        assert!(counters[0].active_cells > 0);

        assert!(matches!(
            classifier.dispatch(
                &device,
                &queue,
                &fixture.samples()[..fixture.samples().len() - 1],
                generation,
                u64::MAX,
            ),
            Err(TransvoxelGpuError::SampleCount { .. })
        ));
    });
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

fn expected_classification(
    fixture: &ExtractionFixture,
    generation: u64,
    dirty_microbricks: u64,
) -> (Vec<GpuTransvoxelCell>, GpuTransvoxelClassifyCounters) {
    let mut cells = vec![GpuTransvoxelCell::default(); PAGE_CELL_COUNT];
    let mut counters = GpuTransvoxelClassifyCounters::default();
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
                counters.visited_cells += 1;
                let fixture_case = fixture.cell_case([x, y, z]).unwrap();
                let topology = regular_case_from_fixture(fixture_case);
                cells[linear] = GpuTransvoxelCell::new(
                    topology.case_index() as u8,
                    topology.class_index(),
                    topology.vertex_count() as u8,
                    topology.triangle_count() as u8,
                    generation,
                );
                if topology.vertex_count() != 0 {
                    counters.active_cells += 1;
                    counters.vertices += topology.vertex_count() as u32;
                    counters.triangles += topology.triangle_count() as u32;
                }
            }
        }
    }
    (cells, counters)
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

fn read_buffer<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> Vec<T> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Transvoxel Validation Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Transvoxel Validation Readback Encoder"),
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
