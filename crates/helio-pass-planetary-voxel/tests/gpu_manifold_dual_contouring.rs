use bytemuck::Pod;
use helio_pass_planetary_voxel::{
    audit_manifold_dc_mesh, extract_manifold_dc_page, manifold_dc_cell_topology, ExtractionFixture,
    ExtractionFixtureKind, GpuManifoldDcCell, GpuManifoldDcCounters, GpuManifoldDcOwner,
    GpuManifoldDcQuad, GpuTerrainVertex, GpuTransvoxelCellOffset, ManifoldDcGpuError,
    ManifoldDcGpuExtractor, ManifoldDcGpuExtractorConfig, ManifoldDcMesh, ManifoldDcQuad,
    ManifoldDcVertex, EXTRACTION_SAMPLE_COUNT, MANIFOLD_DC_GPU_CELL_COUNT,
    MANIFOLD_DC_GPU_OWNER_COUNT,
};
use helio_planet_voxel_core::{CellWord, PageKey};
use std::{collections::BTreeMap, sync::mpsc};

#[test]
fn headless_manifold_dc_matches_cpu_and_suppresses_overflow() {
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
                label: Some("Planetary Manifold DC Validation Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("planetary manifold DC GPU validation error: {error:?}");
        }));

        let mut extractor =
            ManifoldDcGpuExtractor::new(&device, ManifoldDcGpuExtractorConfig::default()).unwrap();
        assert_eq!(extractor.resource_stats().buffers, 13);
        assert!(extractor.resource_stats().allocated_bytes > 0);
        let resources = extractor.resource_stats();
        extractor.resize(3840, 2160);
        assert_eq!(extractor.resource_stats(), resources);
        assert!(matches!(
            extractor.dispatch(&device, &queue, &[], 9_999),
            Err(ManifoldDcGpuError::SampleCount { .. })
        ));

        for (fixture_index, kind) in ExtractionFixtureKind::ALL.into_iter().enumerate() {
            let fixture = fixture(kind);
            let cpu_mesh = extract_manifold_dc_page(fixture.samples()).unwrap();
            let audit = audit_manifold_dc_mesh(&cpu_mesh);
            assert!(
                audit.is_two_manifold(),
                "{} CPU topology: {audit:?}",
                kind.name()
            );
            let expected = cpu_mesh.gpu_mesh().unwrap();
            let generation = (1_u64 << 40) + 10_000 + fixture_index as u64;

            extractor
                .dispatch(&device, &queue, fixture.samples(), generation)
                .unwrap();
            let counters =
                read_one::<GpuManifoldDcCounters>(&device, &queue, extractor.counters_buffer());
            assert_eq!(counters.completed, 1, "{} completed", kind.name());
            assert!(
                counters.succeeded(),
                "{} counters: {counters:?}",
                kind.name()
            );
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
            assert_eq!(counters.active_edges as usize, cpu_mesh.quads.len());

            if fixture_index <= 1 {
                let cells: Vec<GpuManifoldDcCell> = read_buffer(
                    &device,
                    &queue,
                    extractor.cells_buffer(),
                    u64::from(MANIFOLD_DC_GPU_CELL_COUNT) * size_of::<GpuManifoldDcCell>() as u64,
                );
                assert!(cells.iter().all(|cell| cell.is_valid_for(generation)));
                if fixture_index == 1 {
                    assert!(cells
                        .iter()
                        .all(|cell| !cell.is_valid_for((1_u64 << 40) + 10_000)));
                }
                let owners: Vec<GpuManifoldDcOwner> = read_buffer(
                    &device,
                    &queue,
                    extractor.owners_buffer(),
                    u64::from(MANIFOLD_DC_GPU_OWNER_COUNT) * size_of::<GpuManifoldDcOwner>() as u64,
                );
                assert!(owners.iter().all(|owner| owner.is_valid_for(generation)));
                if fixture_index == 1 {
                    assert!(owners
                        .iter()
                        .all(|owner| !owner.is_valid_for((1_u64 << 40) + 10_000)));
                }
                let cell_offsets: Vec<GpuTransvoxelCellOffset> = read_buffer(
                    &device,
                    &queue,
                    extractor.cell_offsets_buffer(),
                    u64::from(MANIFOLD_DC_GPU_CELL_COUNT)
                        * size_of::<GpuTransvoxelCellOffset>() as u64,
                );
                assert!(cell_offsets
                    .iter()
                    .all(|offset| offset.generation() == generation));
                let owner_offsets: Vec<GpuTransvoxelCellOffset> = read_buffer(
                    &device,
                    &queue,
                    extractor.owner_offsets_buffer(),
                    u64::from(MANIFOLD_DC_GPU_OWNER_COUNT)
                        * size_of::<GpuTransvoxelCellOffset>() as u64,
                );
                assert!(owner_offsets
                    .iter()
                    .all(|offset| offset.generation() == generation));
            }

            let vertices: Vec<GpuTerrainVertex> = read_buffer(
                &device,
                &queue,
                extractor.vertices_buffer(),
                u64::from(counters.required_vertices) * size_of::<GpuTerrainVertex>() as u64,
            );
            let indices: Vec<u32> = read_buffer(
                &device,
                &queue,
                extractor.indices_buffer(),
                u64::from(counters.required_indices) * size_of::<u32>() as u64,
            );
            let quads: Vec<GpuManifoldDcQuad> = read_buffer(
                &device,
                &queue,
                extractor.quads_buffer(),
                u64::from(counters.required_indices / 24) * size_of::<GpuManifoldDcQuad>() as u64,
            );
            assert_quad_records(kind.name(), generation, &quads, &indices, &cpu_mesh.quads);
            assert_meshes_equivalent(
                kind.name(),
                &vertices,
                &indices,
                &expected.vertices,
                &expected.indices,
            );

            extractor
                .dispatch(&device, &queue, fixture.samples(), generation)
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
                bytemuck::cast_slice::<GpuTerrainVertex, u8>(&vertices),
                "{} deterministic vertices",
                kind.name()
            );
            assert_eq!(
                repeated_indices,
                indices,
                "{} deterministic indices",
                kind.name()
            );
        }

        let samples = adversarial_samples(0xd1b5_4a32_d192_ed03);
        let cpu_mesh = extract_manifold_dc_page(&samples).unwrap();
        let audit = audit_manifold_dc_mesh(&cpu_mesh);
        assert!(
            audit.is_two_manifold(),
            "adversarial CPU topology: {audit:?}"
        );
        let expected = cpu_mesh.gpu_mesh().unwrap();
        let adversarial_generation = (1_u64 << 41) + 19_000;
        extractor
            .dispatch(&device, &queue, &samples, adversarial_generation)
            .unwrap();
        let counters =
            read_one::<GpuManifoldDcCounters>(&device, &queue, extractor.counters_buffer());
        assert!(counters.succeeded(), "adversarial counters: {counters:?}");
        assert!(!counters.overflowed());
        assert_eq!(counters.required_vertices as usize, expected.vertices.len());
        assert_eq!(counters.required_indices as usize, expected.indices.len());
        assert_eq!(counters.active_edges as usize, cpu_mesh.quads.len());
        let cells: Vec<GpuManifoldDcCell> = read_buffer(
            &device,
            &queue,
            extractor.cells_buffer(),
            u64::from(MANIFOLD_DC_GPU_CELL_COUNT) * size_of::<GpuManifoldDcCell>() as u64,
        );
        let owners: Vec<GpuManifoldDcOwner> = read_buffer(
            &device,
            &queue,
            extractor.owners_buffer(),
            u64::from(MANIFOLD_DC_GPU_OWNER_COUNT) * size_of::<GpuManifoldDcOwner>() as u64,
        );
        assert_classification_sums(adversarial_generation, counters, &cells, &owners);
        let vertices: Vec<GpuTerrainVertex> = read_buffer(
            &device,
            &queue,
            extractor.vertices_buffer(),
            u64::from(counters.required_vertices) * size_of::<GpuTerrainVertex>() as u64,
        );
        let indices: Vec<u32> = read_buffer(
            &device,
            &queue,
            extractor.indices_buffer(),
            u64::from(counters.required_indices) * size_of::<u32>() as u64,
        );
        let quads: Vec<GpuManifoldDcQuad> = read_buffer(
            &device,
            &queue,
            extractor.quads_buffer(),
            u64::from(counters.required_indices / 24) * size_of::<GpuManifoldDcQuad>() as u64,
        );
        assert_quad_records(
            "adversarial",
            adversarial_generation,
            &quads,
            &indices,
            &cpu_mesh.quads,
        );
        assert_meshes_equivalent(
            "adversarial",
            &vertices,
            &indices,
            &expected.vertices,
            &expected.indices,
        );

        for axis in 0..3 {
            let mut negative_page = [0, 0, 0];
            negative_page[axis] = -1;
            let mut positive_page = negative_page;
            positive_page[axis] += 1;
            let negative_fixture = ExtractionFixture::new(
                ExtractionFixtureKind::Sphere,
                PageKey::new(0, negative_page),
            )
            .unwrap();
            let positive_fixture = ExtractionFixture::new(
                ExtractionFixtureKind::Sphere,
                PageKey::new(0, positive_page),
            )
            .unwrap();
            let negative_generation = (1_u64 << 42) + 30_000 + axis as u64 * 2;
            extractor
                .dispatch(
                    &device,
                    &queue,
                    negative_fixture.samples(),
                    negative_generation,
                )
                .unwrap();
            let negative = read_qef_components(
                &device,
                &queue,
                &extractor,
                negative_fixture.page(),
                negative_generation,
            );
            let positive_generation = negative_generation + 1;
            extractor
                .dispatch(
                    &device,
                    &queue,
                    positive_fixture.samples(),
                    positive_generation,
                )
                .unwrap();
            let positive = read_qef_components(
                &device,
                &queue,
                &extractor,
                positive_fixture.page(),
                positive_generation,
            );
            let shared_coordinate = negative_fixture.page().lod0_cell_min().unwrap()[axis] + 31;
            let mut compared = 0;
            for (key, negative_vertex) in negative
                .iter()
                .filter(|((cell, _), _)| cell[axis] == shared_coordinate)
            {
                let Some(positive_vertex) = positive.get(key) else {
                    continue;
                };
                compared += 1;
                for coordinate in 0..3 {
                    assert!(
                        (negative_vertex.position[coordinate]
                            - positive_vertex.position[coordinate])
                            .abs()
                            <= 1.0e-5,
                        "axis {axis} shared QEF position {key:?}"
                    );
                    assert!(
                        (negative_vertex.normal[coordinate] - positive_vertex.normal[coordinate])
                            .abs()
                            <= 1.0e-5,
                        "axis {axis} shared QEF normal {key:?}"
                    );
                }
            }
            assert!(compared > 0, "axis {axis} shared GPU QEF vertices");
        }

        let fixture = fixture(ExtractionFixtureKind::Plane);
        let tiny =
            ManifoldDcGpuExtractor::new(&device, ManifoldDcGpuExtractorConfig::new(1, 1).unwrap())
                .unwrap();
        tiny.dispatch(&device, &queue, fixture.samples(), (1_u64 << 43) + 20_000)
            .unwrap();
        let counters = read_one::<GpuManifoldDcCounters>(&device, &queue, tiny.counters_buffer());
        assert_eq!(counters.completed, 1);
        assert!(counters.vertex_overflow != 0);
        assert!(counters.index_overflow != 0);
        assert_eq!(counters.emitted_vertices, 0);
        assert_eq!(counters.emitted_indices, 0);
        assert!(counters.required_vertices > 1);
        assert!(counters.required_indices > 1);
    });
}

fn read_qef_components(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    extractor: &ManifoldDcGpuExtractor,
    page: PageKey,
    generation: u64,
) -> BTreeMap<([i64; 3], u8), GpuTerrainVertex> {
    let counters = read_one::<GpuManifoldDcCounters>(device, queue, extractor.counters_buffer());
    assert!(counters.succeeded(), "QEF readback counters: {counters:?}");
    let cells: Vec<GpuManifoldDcCell> = read_buffer(
        device,
        queue,
        extractor.cells_buffer(),
        u64::from(MANIFOLD_DC_GPU_CELL_COUNT) * size_of::<GpuManifoldDcCell>() as u64,
    );
    let offsets: Vec<GpuTransvoxelCellOffset> = read_buffer(
        device,
        queue,
        extractor.cell_offsets_buffer(),
        u64::from(MANIFOLD_DC_GPU_CELL_COUNT) * size_of::<GpuTransvoxelCellOffset>() as u64,
    );
    let vertices: Vec<GpuTerrainVertex> = read_buffer(
        device,
        queue,
        extractor.vertices_buffer(),
        u64::from(counters.required_qef_vertices) * size_of::<GpuTerrainVertex>() as u64,
    );
    let page_min = page.lod0_cell_min().unwrap();
    let mut output = BTreeMap::new();
    for (linear, (cell, offset)) in cells.iter().zip(&offsets).enumerate() {
        assert!(cell.is_valid_for(generation));
        assert_eq!(offset.generation(), generation);
        let local = cell_coordinates(linear as u32);
        let canonical = [
            page_min[0] + i64::from(local[0]),
            page_min[1] + i64::from(local[1]),
            page_min[2] + i64::from(local[2]),
        ];
        let topology = manifold_dc_cell_topology(cell.fixture_case());
        let mut fan_offset = offset.first_vertex;
        for fan in 0..usize::from(cell.fan_count()) {
            let component = (0..12)
                .find(|edge| cell.edge_fan(*edge) == Some(fan as u8))
                .and_then(|edge| topology.component_for_edge(edge))
                .expect("every GPU fan maps to a source component");
            let mut vertex = vertices[fan_offset as usize];
            for (coordinate, origin) in vertex.position.iter_mut().zip(page_min) {
                *coordinate += origin as f32;
            }
            if let Some(existing) = output.insert((canonical, component), vertex) {
                for axis in 0..3 {
                    assert!((existing.position[axis] - vertex.position[axis]).abs() <= 1.0e-5);
                    assert!((existing.normal[axis] - vertex.normal[axis]).abs() <= 1.0e-5);
                }
            }
            fan_offset += u32::from(cell.fan_material_count(fan));
        }
    }
    output
}

fn cell_coordinates(linear: u32) -> [i32; 3] {
    let z = linear / (33 * 33);
    let remainder = linear - z * 33 * 33;
    let y = remainder / 33;
    let x = remainder - y * 33;
    [x as i32 - 1, y as i32 - 1, z as i32 - 1]
}

fn assert_classification_sums(
    generation: u64,
    counters: GpuManifoldDcCounters,
    cells: &[GpuManifoldDcCell],
    owners: &[GpuManifoldDcOwner],
) {
    let mut qef_vertices = 0_u32;
    for cell in cells {
        assert!(cell.is_valid_for(generation));
        assert!(cell.fan_count() <= 12);
        assert!(cell.vertex_count() <= 12);
        let material_vertices: u32 = (0..usize::from(cell.fan_count()))
            .map(|fan| u32::from(cell.fan_material_count(fan)))
            .sum();
        assert_eq!(material_vertices, u32::from(cell.vertex_count()));
        for edge in 0..12 {
            if let Some(fan) = cell.edge_fan(edge) {
                assert!(fan < cell.fan_count());
            }
        }
        qef_vertices += u32::from(cell.vertex_count());
    }
    let topology_vertices: u32 = owners
        .iter()
        .map(|owner| {
            assert!(owner.is_valid_for(generation));
            assert!(owner.topology_vertex_count <= 27);
            owner.topology_vertex_count
        })
        .sum();
    assert_eq!(qef_vertices, counters.required_qef_vertices);
    assert_eq!(topology_vertices, counters.required_topology_vertices);
    assert_eq!(qef_vertices + topology_vertices, counters.required_vertices);
    assert_eq!(counters.active_edges * 24, counters.required_indices);
}

fn assert_quad_records(
    label: &str,
    generation: u64,
    quads: &[GpuManifoldDcQuad],
    indices: &[u32],
    expected: &[ManifoldDcQuad],
) {
    assert_eq!(quads.len(), expected.len(), "{label} quad count");
    for (quad_index, ((quad, expected), emitted)) in quads
        .iter()
        .zip(expected)
        .zip(indices.chunks_exact(24))
        .enumerate()
    {
        assert!(quad.is_valid_for(generation), "{label} quad {quad_index}");
        assert_eq!(
            quad.material as u8, expected.material,
            "{label} quad material"
        );
        for side in 0..4 {
            let first = quad.qef_vertices[side];
            let midpoint = quad.midpoint_vertices[side];
            let second = quad.qef_vertices[(side + 1) % 4];
            let center = quad.center_vertex;
            assert_eq!(
                &emitted[side * 6..side * 6 + 6],
                &[first, midpoint, center, midpoint, second, center],
                "{label} quad {quad_index} side {side}"
            );
        }
    }
}

fn assert_meshes_equivalent(
    label: &str,
    actual_vertices: &[GpuTerrainVertex],
    actual_indices: &[u32],
    expected_vertices: &[GpuTerrainVertex],
    expected_indices: &[u32],
) {
    assert_eq!(
        actual_vertices.len(),
        expected_vertices.len(),
        "{label} vertices"
    );
    assert!(actual_vertices.iter().all(|vertex| {
        vertex.position.iter().all(|value| value.is_finite())
            && vertex.normal.iter().all(|value| value.is_finite())
    }));

    assert_eq!(
        actual_indices.len(),
        expected_indices.len(),
        "{label} indices"
    );
    let mut max_position_error = 0.0_f32;
    let mut max_normal_error = 0.0_f32;
    for (slot, (&actual, &expected)) in actual_indices.iter().zip(expected_indices).enumerate() {
        let actual = actual_vertices[actual as usize];
        let expected = expected_vertices[expected as usize];
        assert_eq!(
            actual.material,
            expected.material,
            "{label} material {slot}: actual_index={} expected_index={} actual_position={:?} expected_position={:?}",
            actual_indices[slot],
            expected_indices[slot],
            actual.position,
            expected.position,
        );
        assert_eq!(actual.flags, expected.flags, "{label} flags {slot}");
        for axis in 0..3 {
            max_position_error =
                max_position_error.max((actual.position[axis] - expected.position[axis]).abs());
            max_normal_error =
                max_normal_error.max((actual.normal[axis] - expected.normal[axis]).abs());
        }
    }
    eprintln!(
        "GPU_MANIFOLD_PARITY: fixture={label} max_position_error={max_position_error:.9} max_normal_error={max_normal_error:.9}"
    );
    assert!(
        max_position_error <= 1.0 / 1_024.0,
        "{label} max position error {max_position_error}"
    );
    assert!(
        max_normal_error <= 2.0e-3,
        "{label} max normal error {max_normal_error}"
    );

    let audit_mesh = ManifoldDcMesh {
        vertices: actual_vertices
            .iter()
            .copied()
            .map(|gpu| ManifoldDcVertex {
                gpu,
                qef_error: 0.0,
                cell_min: [0; 3],
                component: u8::MAX,
                hermite_count: 0,
            })
            .collect(),
        quads: (0..actual_indices.len() / 24)
            .map(|_| ManifoldDcQuad {
                vertices: [0; 4],
                material: 0,
                axis: 0,
                edge_min: [0; 3],
            })
            .collect(),
        indices: actual_indices.to_vec(),
    };
    let audit = audit_manifold_dc_mesh(&audit_mesh);
    assert!(audit.is_two_manifold(), "{label} GPU topology: {audit:?}");
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
        label: Some("Planetary Manifold DC Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Manifold DC Readback Encoder"),
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
