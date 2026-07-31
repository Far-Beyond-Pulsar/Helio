use bytemuck::Pod;
use helio_pass_planetary_voxel::{
    transition_case, ExtractionFixtureKind, GpuTerrainVertex, GpuTransvoxelCellOffset,
    GpuTransvoxelScanBlock, GpuTransvoxelTransitionCell, GpuTransvoxelTransitionCounters,
    TransitionFace, TransvoxelGpuTransitionExtractor, TransvoxelGpuTransitionExtractorConfig,
    TransvoxelTransitionFaceFixture, TransvoxelTransitionGpuError,
    TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT, TRANSITION_FACE_SLAB_EDGE,
    TRANSITION_FACE_SLAB_SAMPLE_COUNT, TRANSITION_FULL_SAMPLE_UV,
    TRANSVOXEL_TRANSITION_CASE_WEIGHTS, TRANSVOXEL_TRANSITION_CELLS_PER_FACE,
    TRANSVOXEL_TRANSITION_CELL_COUNT, TRANSVOXEL_TRANSITION_SCAN_BLOCKS,
};
use helio_planet_voxel_core::{CellWord, PageKey};
use std::{mem::size_of, sync::mpsc};

#[test]
fn headless_gpu_transition_emission_matches_cpu_on_all_six_faces() {
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
                label: Some("Planetary Transvoxel Transition Validation Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("planetary Transvoxel transition GPU validation error: {error:?}");
        }));

        let mut extractor = TransvoxelGpuTransitionExtractor::new(
            &device,
            TransvoxelGpuTransitionExtractorConfig::default(),
        )
        .unwrap();
        assert_eq!(extractor.resource_stats().buffers, 9);
        assert_eq!(extractor.resource_stats().allocated_bytes, 3_799_224);
        let resources = extractor.resource_stats();
        extractor.resize(3840, 2160);
        assert_eq!(extractor.resource_stats(), resources);

        let cases = [
            (
                ExtractionFixtureKind::Plane,
                PageKey::new(1, [0, -1, 0]),
                0x3f,
            ),
            (
                ExtractionFixtureKind::Sphere,
                PageKey::new(1, [-1, -1, -1]),
                0x2a,
            ),
            (
                ExtractionFixtureKind::Sphere,
                PageKey::new(1, [0, 0, 0]),
                0x15,
            ),
        ];

        for (case_number, (kind, page, mask)) in cases.into_iter().enumerate() {
            let generation = 10_000 + case_number as u64;
            let (slabs, fixtures) = fixtures_and_slabs(kind, page);
            let expected = expected_mesh(&fixtures, mask);
            extractor
                .dispatch(&device, &queue, &slabs, mask, generation)
                .unwrap();
            let counters = read_one::<GpuTransvoxelTransitionCounters>(
                &device,
                &queue,
                extractor.counters_buffer(),
            );
            assert_eq!(counters.completed, 1, "case {case_number}");
            assert!(!counters.overflowed(), "case {case_number}");
            assert_eq!(counters.active_faces, mask.count_ones());
            assert_eq!(counters.active_cells, expected.active_cells);
            assert_eq!(counters.required_vertices as usize, expected.vertices.len());
            assert_eq!(counters.required_indices as usize, expected.indices.len());
            assert_eq!(counters.emitted_vertices, counters.required_vertices);
            assert_eq!(counters.emitted_indices, counters.required_indices);

            let actual_vertices: Vec<GpuTerrainVertex> = read_buffer(
                &device,
                &queue,
                extractor.vertices_buffer(),
                u64::from(counters.required_vertices) * size_of::<GpuTerrainVertex>() as u64,
            );
            let actual_indices: Vec<u32> = read_buffer(
                &device,
                &queue,
                extractor.indices_buffer(),
                u64::from(counters.required_indices) * size_of::<u32>() as u64,
            );
            assert_vertices_close(&actual_vertices, &expected.vertices, case_number);
            assert_eq!(
                actual_indices, expected.indices,
                "case {case_number} indices"
            );

            let actual_cells: Vec<GpuTransvoxelTransitionCell> = read_buffer(
                &device,
                &queue,
                extractor.cells_buffer(),
                u64::from(TRANSVOXEL_TRANSITION_CELL_COUNT)
                    * size_of::<GpuTransvoxelTransitionCell>() as u64,
            );
            let offsets: Vec<GpuTransvoxelCellOffset> = read_buffer(
                &device,
                &queue,
                extractor.offsets_buffer(),
                u64::from(TRANSVOXEL_TRANSITION_CELL_COUNT)
                    * size_of::<GpuTransvoxelCellOffset>() as u64,
            );
            let blocks: Vec<GpuTransvoxelScanBlock> = read_buffer(
                &device,
                &queue,
                extractor.blocks_buffer(),
                u64::from(TRANSVOXEL_TRANSITION_SCAN_BLOCKS)
                    * size_of::<GpuTransvoxelScanBlock>() as u64,
            );
            for (linear, actual) in actual_cells.iter().copied().enumerate() {
                let face_index = linear / TRANSVOXEL_TRANSITION_CELLS_PER_FACE as usize;
                let active_face = mask & (1 << face_index) != 0;
                if !active_face {
                    assert!(!actual.is_valid_for(generation), "inactive cell {linear}");
                    continue;
                }
                let face_cell = linear % TRANSVOXEL_TRANSITION_CELLS_PER_FACE as usize;
                let u = (face_cell % 32) as u8;
                let v = (face_cell / 32) as u8;
                let cpu_cell = fixtures[face_index].cell([u, v]).unwrap();
                let topology = cpu_cell.topology();
                assert!(actual.is_valid_for(generation), "cell {linear}");
                assert_eq!(actual.case_index(), cpu_cell.case_index(), "cell {linear}");
                assert_eq!(
                    actual.class_index(),
                    topology.class_index(),
                    "cell {linear}"
                );
                assert_eq!(
                    actual.reverse_winding(),
                    topology.reverse_winding(),
                    "cell {linear}"
                );
                assert_eq!(
                    usize::from(actual.vertex_count()),
                    topology.vertex_count(),
                    "cell {linear}"
                );
                assert_eq!(
                    usize::from(actual.triangle_count()),
                    topology.triangle_count(),
                    "cell {linear}"
                );
                let (expected_vertex, expected_index) = expected.cell_offsets[linear].unwrap();
                let block = blocks[linear / 256];
                assert_eq!(
                    block.first_vertex + offsets[linear].first_vertex,
                    expected_vertex,
                    "cell {linear} vertex offset"
                );
                assert_eq!(
                    block.first_index + offsets[linear].first_index,
                    expected_index,
                    "cell {linear} index offset"
                );
                assert_eq!(offsets[linear].generation(), generation, "cell {linear}");
            }

            extractor
                .dispatch(&device, &queue, &slabs, mask, generation)
                .unwrap();
            let repeated_vertices: Vec<GpuTerrainVertex> = read_buffer(
                &device,
                &queue,
                extractor.vertices_buffer(),
                u64::from(counters.required_vertices) * size_of::<GpuTerrainVertex>() as u64,
            );
            let repeated_indices: Vec<u32> = read_buffer(
                &device,
                &queue,
                extractor.indices_buffer(),
                u64::from(counters.required_indices) * size_of::<u32>() as u64,
            );
            assert_eq!(
                bytemuck::cast_slice::<GpuTerrainVertex, u8>(&repeated_vertices),
                bytemuck::cast_slice::<GpuTerrainVertex, u8>(&actual_vertices),
                "case {case_number} deterministic vertices"
            );
            assert_eq!(
                repeated_indices, actual_indices,
                "case {case_number} deterministic indices"
            );
        }

        let sweep_generation = 15_000;
        let sweep_slabs = exhaustive_case_slabs();
        extractor
            .dispatch(
                &device,
                &queue,
                &sweep_slabs,
                TransitionFace::NegativeX.bit() | TransitionFace::PositiveX.bit(),
                sweep_generation,
            )
            .unwrap();
        let sweep_cells: Vec<GpuTransvoxelTransitionCell> = read_buffer(
            &device,
            &queue,
            extractor.cells_buffer(),
            u64::from(TRANSVOXEL_TRANSITION_CELL_COUNT)
                * size_of::<GpuTransvoxelTransitionCell>() as u64,
        );
        for case_index in 0..512_u16 {
            let (face, u, v) = sweep_case_location(case_index);
            let linear = usize::from(face) * TRANSVOXEL_TRANSITION_CELLS_PER_FACE as usize
                + usize::from(u)
                + usize::from(v) * 32;
            let actual = sweep_cells[linear];
            let topology = transition_case(case_index).unwrap();
            assert!(
                actual.is_valid_for(sweep_generation),
                "sweep case {case_index}"
            );
            assert_eq!(actual.case_index(), case_index, "sweep case {case_index}");
            assert_eq!(
                actual.class_index(),
                topology.class_index(),
                "sweep case {case_index} class"
            );
            assert_eq!(
                actual.reverse_winding(),
                topology.reverse_winding(),
                "sweep case {case_index} winding"
            );
            assert_eq!(
                usize::from(actual.vertex_count()),
                topology.vertex_count(),
                "sweep case {case_index} vertices"
            );
            assert_eq!(
                usize::from(actual.triangle_count()),
                topology.triangle_count(),
                "sweep case {case_index} triangles"
            );
        }

        extractor
            .dispatch(&device, &queue, &sweep_slabs, 0, sweep_generation)
            .unwrap();
        let empty = read_one::<GpuTransvoxelTransitionCounters>(
            &device,
            &queue,
            extractor.counters_buffer(),
        );
        assert_eq!(empty.completed, 1);
        assert_eq!(empty.active_faces, 0);
        assert_eq!(empty.active_cells, 0);
        assert_eq!(empty.required_vertices, 0);
        assert_eq!(empty.required_indices, 0);
        assert_eq!(empty.emitted_vertices, 0);
        assert_eq!(empty.emitted_indices, 0);
        let invalidated_cells: Vec<GpuTransvoxelTransitionCell> = read_buffer(
            &device,
            &queue,
            extractor.cells_buffer(),
            u64::from(TRANSVOXEL_TRANSITION_CELL_COUNT)
                * size_of::<GpuTransvoxelTransitionCell>() as u64,
        );
        assert!(invalidated_cells
            .iter()
            .all(|cell| !cell.is_valid_for(sweep_generation)));

        let (slabs, _) =
            fixtures_and_slabs(ExtractionFixtureKind::Plane, PageKey::new(1, [0, -1, 0]));
        let tiny = TransvoxelGpuTransitionExtractor::new(
            &device,
            TransvoxelGpuTransitionExtractorConfig::new(1, 1).unwrap(),
        )
        .unwrap();
        tiny.dispatch(&device, &queue, &slabs, 0x3f, 20_000)
            .unwrap();
        let counters =
            read_one::<GpuTransvoxelTransitionCounters>(&device, &queue, tiny.counters_buffer());
        assert_eq!(counters.completed, 1);
        assert!(counters.vertex_overflow != 0);
        assert!(counters.index_overflow != 0);
        assert_eq!(counters.emitted_vertices, 0);
        assert_eq!(counters.emitted_indices, 0);
        assert!(counters.required_vertices > 1);
        assert!(counters.required_indices > 1);

        assert!(matches!(
            extractor.dispatch(&device, &queue, &slabs[..3], 1, 1),
            Err(TransvoxelTransitionGpuError::SampleCount { .. })
        ));
        assert!(matches!(
            extractor.dispatch(&device, &queue, &slabs, 0x80, 1),
            Err(TransvoxelTransitionGpuError::TransitionMask(0x80))
        ));
    });
}

fn exhaustive_case_slabs() -> Vec<CellWord> {
    let air = CellWord::new(1, 0, 0);
    let solid = CellWord::new(-1, 1, 0);
    let mut slabs = vec![air; TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT];
    for case_index in 0..512_u16 {
        let (face, cell_u, cell_v) = sweep_case_location(case_index);
        for (sample, uv) in TRANSITION_FULL_SAMPLE_UV.into_iter().enumerate() {
            let slab_u = usize::from(cell_u) * 2 + usize::from(uv[0]) + 1;
            let slab_v = usize::from(cell_v) * 2 + usize::from(uv[1]) + 1;
            let index = usize::from(face) * TRANSITION_FACE_SLAB_SAMPLE_COUNT
                + slab_u
                + slab_v * TRANSITION_FACE_SLAB_EDGE
                + TRANSITION_FACE_SLAB_EDGE * TRANSITION_FACE_SLAB_EDGE;
            slabs[index] = if case_index & TRANSVOXEL_TRANSITION_CASE_WEIGHTS[sample] != 0 {
                solid
            } else {
                air
            };
        }
    }
    slabs
}

fn sweep_case_location(case_index: u16) -> (u8, u8, u8) {
    let face = (case_index / 256) as u8;
    let slot = (case_index % 256) as u8;
    (face, (slot % 16) * 2, (slot / 16) * 2)
}

struct ExpectedMesh {
    vertices: Vec<GpuTerrainVertex>,
    indices: Vec<u32>,
    cell_offsets: Vec<Option<(u32, u32)>>,
    active_cells: u32,
}

fn expected_mesh(fixtures: &[TransvoxelTransitionFaceFixture; 6], mask: u8) -> ExpectedMesh {
    let mut expected = ExpectedMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
        cell_offsets: vec![None; TRANSVOXEL_TRANSITION_CELL_COUNT as usize],
        active_cells: 0,
    };
    for face in TransitionFace::ALL {
        if mask & face.bit() == 0 {
            continue;
        }
        let fixture = fixtures[usize::from(face.index())];
        let face_mesh = fixture.extract();
        let base_vertex = expected.vertices.len() as u32;
        let base_index = expected.indices.len() as u32;
        for (face_cell, range) in face_mesh.cell_ranges.into_iter().enumerate() {
            let range = range.unwrap();
            let linear = usize::from(face.index()) * TRANSVOXEL_TRANSITION_CELLS_PER_FACE as usize
                + face_cell;
            expected.cell_offsets[linear] = Some((
                base_vertex + range.first_vertex,
                base_index + range.first_index,
            ));
            expected.active_cells += u32::from(range.vertex_count != 0);
        }
        expected
            .vertices
            .extend(face_mesh.vertices.into_iter().map(|vertex| vertex.vertex));
        expected.indices.extend(
            face_mesh
                .indices
                .into_iter()
                .map(|index| base_vertex + index),
        );
    }
    expected
}

fn fixtures_and_slabs(
    kind: ExtractionFixtureKind,
    page: PageKey,
) -> (Vec<CellWord>, [TransvoxelTransitionFaceFixture; 6]) {
    let fixtures = TransitionFace::ALL
        .map(|face| TransvoxelTransitionFaceFixture::new(kind, page, face).unwrap());
    let mut slabs = vec![CellWord::AIR; TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT];
    for face in TransitionFace::ALL {
        let face_index = usize::from(face.index());
        let start = face_index * TRANSITION_FACE_SLAB_SAMPLE_COUNT;
        slabs[start..start + TRANSITION_FACE_SLAB_SAMPLE_COUNT]
            .copy_from_slice(&fixtures[face_index].slab_samples());
    }
    (slabs, fixtures)
}

fn assert_vertices_close(actual: &[GpuTerrainVertex], expected: &[GpuTerrainVertex], case: usize) {
    assert_eq!(actual.len(), expected.len(), "case {case} vertices");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.material, expected.material,
            "vertex {index} material"
        );
        assert_eq!(actual.flags, expected.flags, "vertex {index} flags");
        for axis in 0..3 {
            assert!(
                (actual.position[axis] - expected.position[axis]).abs() <= 1.0e-5,
                "case {case} vertex {index}:{axis} position {:?} != {:?}",
                actual.position,
                expected.position
            );
            assert!(
                (actual.normal[axis] - expected.normal[axis]).abs() <= 2.0e-4,
                "case {case} vertex {index}:{axis} normal {:?} != {:?}",
                actual.normal,
                expected.normal
            );
        }
    }
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
        label: Some("Planetary Transvoxel Transition Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Transvoxel Transition Readback Encoder"),
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
