use bytemuck::Pod;
use helio_pass_planetary_voxel::{
    FrameUpdateOutcome, GpuLookupKey, GpuLookupQuery, GpuLookupResult, GpuResidencyCounters,
    GpuResidencyError, GpuUploadOutcome, PlanetaryVoxelGpuConfig, PlanetaryVoxelResidency,
    RESIDENCY_WGSL,
};
use helio_planet_voxel_core::{
    CellWord, EvictOutcome, GpuPageMeta, PageEvict, PageKey, PageUpload, PlanetFrameUniform,
    PlanetId, PlanetPageKey, PlanetPosition, UploadOutcome, VisiblePage, VisiblePageSet,
    PAGE_CELL_BYTES, PAGE_CELL_COUNT,
};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[test]
fn headless_residency_round_trips_cells_metadata_lookup_and_rebuild() {
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
                label: Some("Planetary Voxel Residency Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("planetary voxel GPU validation error: {error:?}");
        }));

        let config = PlanetaryVoxelGpuConfig::new(4, 16, 16, 4, 8, 4).unwrap();
        let mut residency = PlanetaryVoxelResidency::new(&device, &queue, config).unwrap();
        let initial_resources = residency.resource_stats();
        assert_eq!(
            initial_resources.buffers,
            initial_resources.atlas_shards + 5
        );

        let planet_a = PlanetId([0x11; 16]);
        let planet_b = PlanetId([0x82; 16]);
        let key_a = PlanetPageKey::new(planet_a, PageKey::new(0, [-2, 1, -3]));
        let key_b = PlanetPageKey::new(planet_b, PageKey::new(2, [3, -4, 5]));
        residency
            .set_planet_frame(
                &queue,
                PlanetFrameUniform::from_camera(
                    planet_a,
                    PlanetPosition::from_meters([0.25, 1.0, 2.0]).unwrap(),
                    1,
                ),
            )
            .unwrap();
        residency
            .set_planet_frame(
                &queue,
                PlanetFrameUniform::from_camera(
                    planet_b,
                    PlanetPosition::from_meters([2.0, 3.0, 1.0]).unwrap(),
                    1,
                ),
            )
            .unwrap();

        let cell_a = CellWord::new(-120, 7, 3);
        let cell_b = CellWord::new(-8, 19, 1);
        let outcomes = residency
            .apply_upload_batch(
                &device,
                &queue,
                vec![upload(key_a, 9, cell_a), upload(key_b, 11, cell_b)],
            )
            .unwrap();
        assert!(matches!(
            outcomes.as_slice(),
            [
                GpuUploadOutcome::Residency(UploadOutcome::Inserted { slot: 0, .. }),
                GpuUploadOutcome::Residency(UploadOutcome::Inserted { slot: 1, .. })
            ]
        ));

        assert!(matches!(
            residency
                .apply_upload_batch(&device, &queue, vec![upload(key_a, 8, CellWord::AIR)])
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(UploadOutcome::Stale {
                newest_generation: 9
            })]
        ));
        assert_eq!(
            residency
                .apply_evict_batch(
                    &device,
                    &queue,
                    vec![PageEvict {
                        key: key_a,
                        generation: 8,
                    }],
                )
                .unwrap(),
            vec![EvictOutcome::Recorded { removed: None }]
        );
        assert_eq!(residency.cache().resident(key_a).unwrap().generation, 9);

        assert!(matches!(
            residency.apply_upload_batch(
                &device,
                &queue,
                (0..5)
                    .map(|index| {
                        upload(
                            PlanetPageKey::new(planet_a, PageKey::new(0, [index, 0, 0])),
                            1,
                            CellWord::AIR,
                        )
                    })
                    .collect(),
            ),
            Err(GpuResidencyError::BatchCapacity {
                actual: 5,
                maximum: 4
            })
        ));
        assert_eq!(residency.cache().counters().resident_pages, 2);

        assert!(matches!(
            residency.set_planet_frame(
                &queue,
                PlanetFrameUniform::from_camera(
                    planet_a,
                    PlanetPosition::from_meters([9.0, 0.0, 0.0]).unwrap(),
                    1,
                ),
            ),
            Ok(FrameUpdateOutcome::FrameConflict)
        ));
        let outside_origin = [i64::from(i32::MAX).div_euclid(32) * 32 + 32, 0, 0];
        assert!(matches!(
            residency.set_planet_frame(
                &queue,
                PlanetFrameUniform::from_camera(
                    planet_a,
                    PlanetPosition::from_lod0_cell(outside_origin),
                    2,
                ),
            ),
            Err(GpuResidencyError::Address(_))
        ));
        assert!(matches!(
            residency.set_planet_frame(
                &queue,
                PlanetFrameUniform::from_camera(
                    planet_a,
                    PlanetPosition::from_lod0_cell([32, 0, 0]),
                    2,
                ),
            ),
            Ok(FrameUpdateOutcome::Applied {
                previous_frame: Some(1)
            })
        ));
        residency
            .apply_visible_set(
                &queue,
                VisiblePageSet {
                    frame_index: 3,
                    pages: vec![VisiblePage {
                        key: key_a,
                        generation: 9,
                        transition_mask: 0b10_0101,
                    }],
                },
            )
            .unwrap();

        residency.resize(1920, 1080);
        assert_eq!(residency.resource_stats(), initial_resources);
        residency.recreate_gpu_resources(&device, &queue).unwrap();
        assert_eq!(residency.resource_stats(), initial_resources);

        let frame_a = [32_i64, 0, 0];
        let frame_b = [0_i64; 3];
        let missing = PlanetPageKey::new(planet_a, PageKey::new(1, [-99, 4, 2]));
        let queries = [
            GpuLookupQuery::from(GpuLookupKey::from_planet_page(key_a, frame_a).unwrap()),
            GpuLookupQuery::from(GpuLookupKey::from_planet_page(key_b, frame_b).unwrap()),
            GpuLookupQuery::from(GpuLookupKey::from_planet_page(missing, frame_a).unwrap()),
        ];
        let results = dispatch_lookup(&device, &queue, &residency, &queries);
        assert!(results[0].found());
        assert_eq!(results[0].slot, 0);
        assert_eq!(results[0].generation(), 9);
        assert!(results[1].found());
        assert_eq!(results[1].slot, 1);
        assert_eq!(results[1].generation(), 11);
        assert!(!results[2].found());

        let first_cell: Vec<CellWord> = read_buffer_range(
            &device,
            &queue,
            residency.atlas_buffers().next().unwrap(),
            0,
            size_of::<CellWord>() as u64,
        );
        assert_eq!(first_cell, vec![cell_a]);
        let metadata: Vec<GpuPageMeta> = read_buffer_range(
            &device,
            &queue,
            residency.metadata_buffer(),
            0,
            size_of::<GpuPageMeta>() as u64,
        );
        assert_eq!(metadata[0].slot, 0);
        assert_eq!(metadata[0].generation(), 9);
        assert_eq!(metadata[0].relative_lod0_cell_min, [-96, 32, -96]);
        assert_eq!(metadata[0].transition_mask, 0b10_0101);
        let counters: Vec<GpuResidencyCounters> = read_buffer_range(
            &device,
            &queue,
            residency.counters_buffer(),
            0,
            size_of::<GpuResidencyCounters>() as u64,
        );
        assert_eq!(counters[0].resident_pages, 2);
        assert_eq!(counters[0].peak_resident_pages, 2);
        assert_eq!(counters[0].device_rebuilds, 1);
        assert_eq!(counters[0].uploads_published, 2);
        assert_eq!(counters[0].batches_submitted, 2);
        assert_eq!(
            u64::from(counters[0].cell_bytes_uploaded_low)
                | (u64::from(counters[0].cell_bytes_uploaded_high) << 32),
            4 * PAGE_CELL_BYTES as u64
        );
        assert_eq!(counters[0].resource_buffers, initial_resources.buffers);
        assert_eq!(counters[0].atlas_shards, initial_resources.atlas_shards);
        assert_eq!(
            u64::from(counters[0].resident_cell_bytes_low)
                | (u64::from(counters[0].resident_cell_bytes_high) << 32),
            2 * PAGE_CELL_BYTES as u64
        );
        assert_eq!(
            u64::from(counters[0].allocated_gpu_bytes_low)
                | (u64::from(counters[0].allocated_gpu_bytes_high) << 32),
            initial_resources.allocated_bytes
        );

        assert!(matches!(
            residency
                .apply_evict_batch(
                    &device,
                    &queue,
                    vec![PageEvict {
                        key: key_a,
                        generation: 9,
                    }],
                )
                .unwrap()
                .as_slice(),
            [EvictOutcome::Recorded { removed: Some(_) }]
        ));
        let result = dispatch_lookup(&device, &queue, &residency, &queries[..1]);
        assert!(!result[0].found());
        let cleared_cell: Vec<CellWord> = read_buffer_range(
            &device,
            &queue,
            residency.atlas_buffers().next().unwrap(),
            0,
            size_of::<CellWord>() as u64,
        );
        assert_eq!(cleared_cell, vec![CellWord::AIR]);

        let replacement = CellWord::new(-512, 33, 5);
        assert!(matches!(
            residency
                .apply_upload_batch(&device, &queue, vec![upload(key_a, 9, cell_a)])
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(UploadOutcome::Stale {
                newest_generation: 9
            })]
        ));
        assert!(matches!(
            residency
                .apply_upload_batch(&device, &queue, vec![upload(key_a, 10, replacement)])
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(UploadOutcome::Inserted {
                slot: 0,
                ..
            })]
        ));
        assert!(matches!(
            residency
                .apply_upload_batch(&device, &queue, vec![upload(key_a, 10, cell_a)])
                .unwrap()
                .as_slice(),
            [GpuUploadOutcome::Residency(
                UploadOutcome::GenerationConflict { slot: 0 }
            )]
        ));
        let replacement_result = dispatch_lookup(&device, &queue, &residency, &queries[..1]);
        assert!(replacement_result[0].found());
        assert_eq!(replacement_result[0].generation(), 10);
        let replacement_cell: Vec<CellWord> = read_buffer_range(
            &device,
            &queue,
            residency.atlas_buffers().next().unwrap(),
            0,
            size_of::<CellWord>() as u64,
        );
        assert_eq!(replacement_cell, vec![replacement]);

        validate_table_probe_backpressure(&device, &queue, planet_a);
    });
}

fn validate_table_probe_backpressure(device: &wgpu::Device, queue: &wgpu::Queue, planet: PlanetId) {
    let mut residency = PlanetaryVoxelResidency::new(
        device,
        queue,
        PlanetaryVoxelGpuConfig::new(3, 8, 1, 3, 4, 2).unwrap(),
    )
    .unwrap();
    residency
        .set_planet_frame(
            queue,
            PlanetFrameUniform::from_camera(planet, PlanetPosition::from_lod0_cell([0; 3]), 1),
        )
        .unwrap();
    let mut collisions = Vec::new();
    for x in -10_000..10_000 {
        let key = PlanetPageKey::new(planet, PageKey::new(0, [x, 0, 0]));
        let lookup = GpuLookupKey::from_planet_page(key, [0; 3]).unwrap();
        if lookup.hash() & 7 == 0 {
            collisions.push(key);
            if collisions.len() == 2 {
                break;
            }
        }
    }
    assert_eq!(collisions.len(), 2);
    residency
        .apply_upload_batch(device, queue, vec![upload(collisions[0], 1, CellWord::AIR)])
        .unwrap();
    assert_eq!(
        residency
            .apply_upload_batch(device, queue, vec![upload(collisions[1], 1, CellWord::AIR)])
            .unwrap(),
        vec![GpuUploadOutcome::PageTableBackpressure]
    );
    assert!(residency.cache().resident(collisions[1]).is_none());
    assert_eq!(residency.counters().table_saturation_events, 1);
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

fn upload(key: PlanetPageKey, generation: u64, cell: CellWord) -> PageUpload {
    PageUpload::new(key, generation, vec![cell; PAGE_CELL_COUNT]).unwrap()
}

fn dispatch_lookup(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    residency: &PlanetaryVoxelResidency,
    queries: &[GpuLookupQuery],
) -> Vec<GpuLookupResult> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Planetary Voxel Lookup Validation Shader"),
        source: wgpu::ShaderSource::Wgsl(RESIDENCY_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Planetary Voxel Lookup Validation Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("validate_lookup"),
        compilation_options: Default::default(),
        cache: None,
    });
    let query_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Planetary Voxel Lookup Queries"),
        contents: bytemuck::cast_slice(queries),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_bytes = std::mem::size_of_val(queries) as u64 / size_of::<GpuLookupQuery>() as u64
        * size_of::<GpuLookupResult>() as u64;
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Voxel Lookup Results"),
        size: result_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Planetary Voxel Lookup Validation Bind Group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: residency.page_table_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: residency.residency_uniform_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: query_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Voxel Lookup Validation Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Planetary Voxel Lookup Validation Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((queries.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    read_buffer_range(device, queue, &result_buffer, 0, result_bytes)
}

fn read_buffer_range<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Vec<T> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Voxel Validation Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Voxel Validation Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(source, offset, &readback, 0, size);
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

fn size_of<T>() -> usize {
    core::mem::size_of::<T>()
}
