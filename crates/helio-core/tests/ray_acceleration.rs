use helio_core::acceleration::stat;
use helio_core::{
    AccelerationError, BlasGeometry, BlasManager, BlasPolicy, BlasSegment, BlasWork, FrameStats,
    Movability, TlasInstanceInput, TlasManager, TlasPolicy,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

struct Gpu {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
}

impl Gpu {
    fn new() -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..helio_core::wgpu_instance_descriptor()
            });
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            eprintln!("RT acceleration adapter: {:?}", adapter.get_info());
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                    required_limits: adapter.limits(),
                    // These opt-in tests deliberately exercise wgpu's experimental API.
                    experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                    ..Default::default()
                })
                .await
                .expect("Vulkan ray-query device required for this explicit GPU test");
            Self {
                device: Arc::new(device),
                queue,
            }
        })
    }

    fn buffer(&self, values: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU-resident triangle positions"),
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn trace(&self, encoder: &mut wgpu::CommandEncoder, tlas: &wgpu::Tlas) -> wgpu::Buffer {
        self.trace_opacity(encoder, tlas, true)
    }
    fn trace_opacity(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tlas: &wgpu::Tlas,
        force_opaque: bool,
    ) -> wgpu::Buffer {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("known world-space ray segments"),
            source: wgpu::ShaderSource::Wgsl(r#"
enable wgpu_ray_query;
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> hits: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var query: ray_query;
    rayQueryInitialize(&query, scene,
        RayDesc(0x05u, 0xffu, 0.001, 10.0, vec3<f32>(f32(id.x)*2.0, 0.0, 2.0), vec3<f32>(0.0, 0.0, -1.0)));
    while rayQueryProceed(&query) {}
    hits[id.x] = select(1u, 0u, rayQueryGetCommittedIntersection(&query).kind == RAY_QUERY_INTERSECTION_NONE);
}
"#.replace("0x05u", if force_opaque { "0x05u" } else { "0x04u" }).into()),
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("known segments"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let results = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray results"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ray inputs"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: results.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(4, 1, 1);
        }
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ray readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&results, 0, &readback, 0, 16);
        readback
    }

    fn read(&self, encoder: wgpu::CommandEncoder, readback: &wgpu::Buffer) -> Vec<u32> {
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let values =
            bytemuck::cast_slice::<u8, u32>(&readback.slice(..).get_mapped_range().unwrap())
                .to_vec();
        readback.unmap();
        values
    }
}

fn triangle(x: f32) -> [f32; 9] {
    [x - 0.75, -0.75, 0.0, x + 0.75, -0.75, 0.0, x, 0.75, 0.0]
}

fn geometry(vertices: &wgpu::Buffer, revision: u64) -> BlasGeometry<'_> {
    BlasGeometry {
        positions_revision: revision,
        topology_revision: 0,
        vertices,
        first_vertex: 0,
        vertex_count: 3,
        vertex_stride: 12,
        indices: None,
        first_index: 0,
        index_count: 0,
    }
}

/// One opaque static segment as BLAS 7.
fn static_build(
    blas: &mut BlasManager,
    encoder: &mut wgpu::CommandEncoder,
    geometry: BlasGeometry<'_>,
) -> BlasWork {
    build(blas, 7, encoder, geometry, Movability::Static).unwrap()
}

/// One opaque segment of `movability`.
fn build(
    blas: &mut BlasManager,
    id: u64,
    encoder: &mut wgpu::CommandEncoder,
    geometry: BlasGeometry<'_>,
    movability: Movability,
) -> Result<BlasWork, AccelerationError> {
    blas.build(
        id,
        encoder,
        vec![BlasSegment {
            geometry,
            opaque: true,
        }],
        movability,
    )
}

fn instance(geometry_id: u64, x: f32) -> TlasInstanceInput {
    TlasInstanceInput {
        geometry_id,
        custom_data: 0,
        transform: [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn versioned_gpu_geometry_changes_actual_ray_visibility() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let vertices = gpu.buffer(&triangle(0.0));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert_eq!(
        static_build(&mut blas, &mut encoder, geometry(&vertices, 0)),
        BlasWork::Built
    );
    assert_eq!(
        static_build(&mut blas, &mut encoder, geometry(&vertices, 0)),
        BlasWork::Unchanged
    );
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [1, 0, 0, 0]);

    // In-place GPU data replacement with identical allocation/counts.
    gpu.queue
        .write_buffer(&vertices, 0, bytemuck::cast_slice(&triangle(2.0)));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert_eq!(
        static_build(&mut blas, &mut encoder, geometry(&vertices, 1)),
        BlasWork::Built
    );
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 1, 0, 0]);

    // Reallocation must invalidate the cache even when the revision is unchanged.
    let replacement = gpu.buffer(&triangle(4.0));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert_eq!(
        static_build(&mut blas, &mut encoder, geometry(&replacement, 1)),
        BlasWork::Built
    );
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 0, 1, 0]);
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn tlas_growth_removal_empty_and_missing_mesh_never_leave_stale_casters() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let vertices = gpu.buffer(&triangle(0.0));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    build(
        &mut blas,
        7,
        &mut encoder,
        geometry(&vertices, 0),
        Movability::Static,
    )
    .unwrap();
    tlas.build(
        &mut encoder,
        &[instance(7, 0.0), instance(7, 2.0), instance(7, 4.0)],
        &blas,
    )
    .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(
        gpu.read(encoder, &readback),
        [1, 1, 1, 0],
        "initial capacity must not truncate instances"
    );

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    tlas.build(&mut encoder, &[instance(7, 6.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 0, 0, 1]);

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert_eq!(
        tlas.build(&mut encoder, &[instance(999, 0.0)], &blas),
        Err(AccelerationError::MissingBlas(999))
    );
    assert!(tlas.tlas().is_none());
    tlas.build(&mut encoder, &[], &blas).unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 0, 0, 0]);
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn indexed_suballocations_and_invalid_replacements() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let vertices = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("COPY_SRC pool"),
            contents: bytemuck::cast_slice(&[triangle(0.0), triangle(2.0)].concat()),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let indices = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("suballocated indices"),
            contents: bytemuck::cast_slice(&[0u32, 0, 0, 0, 1, 2]),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut mesh = geometry(&vertices, 0);
    mesh.first_vertex = 3;
    mesh.indices = Some(&indices);
    mesh.first_index = 3;
    mesh.index_count = 3;
    build(&mut blas, 7, &mut encoder, mesh, Movability::Static).unwrap();
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 1, 0, 0]);

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut invalid = geometry(&vertices, 1);
    invalid.first_vertex = u32::MAX;
    assert!(matches!(
        build(&mut blas, 7, &mut encoder, invalid, Movability::Static),
        Err(AccelerationError::InvalidGeometry(_))
    ));
    assert!(blas.get_blas(7).is_none());
    assert_eq!(
        tlas.build(&mut encoder, &[instance(7, 0.0)], &blas),
        Err(AccelerationError::MissingBlas(7))
    );
    assert!(tlas.tlas().is_none());
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn padded_pool_vertices_copy_the_complete_stride() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let mut padded = Vec::new();
    for position in triangle(0.0).chunks_exact(3) {
        padded.extend_from_slice(position);
        padded.extend_from_slice(&[0.0; 7]);
    }
    let vertices = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("40-byte pool vertices"),
            contents: bytemuck::cast_slice(&padded),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut mesh = geometry(&vertices, 0);
    mesh.vertex_stride = 40;
    build(&mut blas, 7, &mut encoder, mesh, Movability::Static).unwrap();
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [1, 0, 0, 0]);
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn changing_blas_opacity_rebuilds_and_changes_hardware_candidate_handling() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let vertices = gpu.buffer(&triangle(0.0));
    for (opaque, rebuild) in [(true, true), (false, true), (false, false), (true, true)] {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let work = blas
            .build(
                1,
                &mut encoder,
                vec![BlasSegment {
                    geometry: geometry(&vertices, 0),
                    opaque,
                }],
                Movability::Static,
            )
            .unwrap();
        assert_eq!(work.recorded(), rebuild);
        tlas.build(&mut encoder, &[instance(1, 0.0)], &blas)
            .unwrap();
        let results = gpu.trace_opacity(&mut encoder, tlas.tlas().unwrap(), false);
        assert_eq!(
            gpu.read(encoder, &results),
            vec![u32::from(opaque), 0, 0, 0]
        );
    }
}

/// An indexed grid of `cells`^2 quads in the plane z = 0 over
/// [x - 0.75, x + 0.75] x [-0.75, 0.75]: many triangles over ray 0's footprint
/// (at x = 0), for compaction to have something to save.
fn grid(cells: u32, x: f32) -> (Vec<f32>, Vec<u32>) {
    let side = cells + 1;
    let mut vertices = Vec::with_capacity((side * side * 3) as usize);
    for j in 0..side {
        for i in 0..side {
            let u = i as f32 / cells as f32 * 1.5 - 0.75;
            let v = j as f32 / cells as f32 * 1.5 - 0.75;
            vertices.extend_from_slice(&[x + u, v, 0.0]);
        }
    }
    let mut indices = Vec::with_capacity((cells * cells * 6) as usize);
    for j in 0..cells {
        for i in 0..cells {
            let a = j * side + i;
            indices.extend_from_slice(&[a, a + 1, a + side, a + 1, a + side + 1, a + side]);
        }
    }
    (vertices, indices)
}

impl Gpu {
    fn indices(&self, values: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU-resident indices"),
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            })
    }

    /// Submit and wait (tests only: the managers themselves never wait).
    fn finish(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
}

fn indexed<'a>(
    vertices: &'a wgpu::Buffer,
    indices: &'a wgpu::Buffer,
    vertex_count: u32,
    index_count: u32,
    revision: u64,
) -> BlasGeometry<'a> {
    BlasGeometry {
        positions_revision: revision,
        topology_revision: 0,
        vertices,
        first_vertex: 0,
        vertex_count,
        vertex_stride: 12,
        indices: Some(indices),
        first_index: 0,
        index_count,
    }
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn static_blas_is_compacted_without_waiting_and_the_tlas_follows_the_swap() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let (positions, triangles) = grid(64, 0.0);
    let vertices = gpu.buffer(&positions);
    let indices = gpu.indices(&triangles);
    let mesh = || {
        indexed(
            &vertices,
            &indices,
            positions.len() as u32 / 3,
            triangles.len() as u32,
            0,
        )
    };
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert_eq!(
        build(&mut blas, 1, &mut encoder, mesh(), Movability::Static).unwrap(),
        BlasWork::Built
    );
    tlas.build(&mut encoder, &[instance(1, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [1, 0, 0, 0]);
    let built = blas.memory();
    assert_eq!(built.count, 1);
    assert_eq!(built.bytes, built.bytes_before_compaction);
    assert_eq!(built.bytes, blas.get_blas(1).unwrap().sizes().structure);
    let original = blas.get_blas(1).unwrap().clone();

    // Prepared one maintain after the build's frame, compacted once the
    // device reports it ready: a later maintain, never a wait inside one.
    blas.maintain(&gpu.queue);
    blas.maintain(&gpu.queue);
    let revision = blas.revision();
    let mut compacted = false;
    for _ in 0..100 {
        gpu.queue.submit([]);
        gpu.device.poll(wgpu::PollType::Poll).unwrap();
        blas.maintain(&gpu.queue);
        if blas.counters().compactions == 1 {
            compacted = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(compacted, "the static BLAS was never compacted");
    assert_ne!(blas.revision(), revision);
    assert_ne!(blas.get_blas(1).unwrap(), &original);
    drop(original);
    let after = blas.memory();
    assert!(after.bytes < built.bytes, "{after:?} vs {built:?}");
    assert_eq!(after.bytes_before_compaction, built.bytes);
    assert_eq!(after.bytes, blas.get_blas(1).unwrap().sizes().structure);
    eprintln!(
        "compaction: {} triangles, {} -> {} bytes ({:.0}%)",
        triangles.len() / 3,
        built.bytes,
        after.bytes,
        100.0 * after.bytes as f64 / built.bytes as f64
    );

    // The compacted copy lands in the queue's pending writes, submitted ahead
    // of this encoder: the TLAS built in it traces the compacted structure.
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    tlas.build(&mut encoder, &[instance(1, 2.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 1, 0, 0]);
    // An unchanged geometry is not built again, nor compacted twice.
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert_eq!(
        build(&mut blas, 1, &mut encoder, mesh(), Movability::Static).unwrap(),
        BlasWork::Unchanged
    );
    for _ in 0..4 {
        blas.maintain(&gpu.queue);
    }
    assert_eq!(blas.counters().compactions, 1);
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 1, 0, 0]);
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn deforming_blas_refits_to_moved_vertices_and_rebuilds_on_policy() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    blas.set_policy(BlasPolicy {
        refits_per_rebuild: 3,
        ..BlasPolicy::default()
    });
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let vertices = gpu.buffer(&triangle(0.0));
    let mut works = Vec::new();
    for frame in 0..10u64 {
        let slot = frame % 4;
        gpu.queue.write_buffer(
            &vertices,
            0,
            bytemuck::cast_slice(&triangle(slot as f32 * 2.0)),
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        if frame == 6 {
            blas.request_rebuild(3);
        }
        let work = build(
            &mut blas,
            3,
            &mut encoder,
            geometry(&vertices, frame),
            Movability::Dynamic,
        )
        .unwrap();
        works.push(work);
        tlas.build(&mut encoder, &[instance(3, 0.0)], &blas)
            .unwrap();
        let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
        let mut expected = [0; 4];
        expected[slot as usize] = 1;
        assert_eq!(gpu.read(encoder, &readback), expected, "frame {frame}");
    }
    use BlasWork::{Built, Refit};
    // The first deforming BLAS rebuilds after 3 refits; the rebuild
    // requested on frame 6 restarts the count.
    assert_eq!(
        works,
        [Built, Refit, Refit, Refit, Built, Refit, Built, Refit, Refit, Refit]
    );
    let counters = blas.counters();
    assert_eq!(counters.builds, 3);
    assert_eq!(counters.policy_rebuilds, 1);
    assert_eq!(counters.refits, 7);
    // Deforming geometry is never compacted, and one structure is reused.
    for _ in 0..4 {
        blas.maintain(&gpu.queue);
    }
    assert_eq!(blas.counters().compactions, 0);
    assert_eq!(blas.memory().count, 1);
    // Unchanged positions record nothing.
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert_eq!(
        build(
            &mut blas,
            3,
            &mut encoder,
            geometry(&vertices, 9),
            Movability::Dynamic
        )
        .unwrap(),
        BlasWork::Unchanged
    );
    // A topology change is a new structure, whatever the policy.
    let mut changed = geometry(&vertices, 9);
    changed.topology_revision = 1;
    assert_eq!(
        build(&mut blas, 3, &mut encoder, changed, Movability::Dynamic).unwrap(),
        BlasWork::Built
    );
    gpu.finish(encoder);
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn deforming_blas_rebuilds_are_staggered() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    blas.set_policy(BlasPolicy {
        refits_per_rebuild: 4,
        ..BlasPolicy::default()
    });
    let buffers: Vec<_> = (0..8).map(|_| gpu.buffer(&triangle(0.0))).collect();
    let mut rebuilds_per_frame = Vec::new();
    for frame in 0..9u64 {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut rebuilds = 0;
        for (id, vertices) in buffers.iter().enumerate() {
            let work = build(
                &mut blas,
                id as u64,
                &mut encoder,
                geometry(vertices, frame),
                Movability::Dynamic,
            )
            .unwrap();
            rebuilds += u32::from(work == BlasWork::Built);
        }
        gpu.finish(encoder);
        rebuilds_per_frame.push(rebuilds);
    }
    // All eight are built on frame 0; afterwards their rebuilds spread over
    // the 5 frames of a cycle (4 refits, a rebuild): 8 in 5 frames, at most
    // two a frame, never all at once.
    assert_eq!(rebuilds_per_frame, [8, 1, 1, 2, 2, 2, 1, 1, 2]);
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn tlas_refits_on_moves_and_rebuilds_on_membership() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), 4);
    // No policy rebuilds: membership alone decides.
    tlas.set_policy(TlasPolicy {
        max_refits: u32::MAX,
        moved_fraction: f32::INFINITY,
    });
    let vertices = gpu.buffer(&triangle(0.0));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    for id in [1, 2] {
        build(
            &mut blas,
            id,
            &mut encoder,
            geometry(&vertices, 0),
            Movability::Static,
        )
        .unwrap();
    }
    gpu.finish(encoder);
    let step = |tlas: &mut TlasManager,
                instances: &[TlasInstanceInput],
                masks: &[u8],
                expected: [u32; 4]| {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        tlas.build_masked(&mut encoder, instances, &blas, masks)
            .unwrap();
        let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
        assert_eq!(gpu.read(encoder, &readback), expected);
        let c = tlas.counters();
        (c.builds, c.refits)
    };
    let two = |a: TlasInstanceInput, b: TlasInstanceInput| [a, b];
    assert_eq!(
        step(
            &mut tlas,
            &two(instance(1, 0.0), instance(2, 4.0)),
            &[],
            [1, 0, 1, 0]
        ),
        (1, 0)
    );
    // Moved: a refit that traces the new place.
    assert_eq!(
        step(
            &mut tlas,
            &two(instance(1, 2.0), instance(2, 4.0)),
            &[],
            [0, 1, 1, 0]
        ),
        (1, 1)
    );
    // A mask change: a refit; the ray mask (0xff) still sees bit 0.
    assert_eq!(
        step(
            &mut tlas,
            &two(instance(1, 2.0), instance(2, 4.0)),
            &[0x01, 0xFF],
            [0, 1, 1, 0]
        ),
        (1, 2)
    );
    // Another instance in slot 0 at the same count: rebuilt.
    assert_eq!(
        step(
            &mut tlas,
            &two(instance(2, 6.0), instance(2, 4.0)),
            &[],
            [0, 0, 1, 1]
        ),
        (2, 2)
    );
    // Same membership, moved back: a refit again.
    assert_eq!(
        step(
            &mut tlas,
            &two(instance(2, 0.0), instance(2, 4.0)),
            &[],
            [1, 0, 1, 0]
        ),
        (2, 3)
    );
    // A changed count: rebuilt.
    assert_eq!(
        step(&mut tlas, &[instance(2, 0.0)], &[], [1, 0, 0, 0]),
        (3, 3)
    );
    // Requested: rebuilt.
    tlas.request_rebuild();
    assert_eq!(
        step(&mut tlas, &[instance(2, 6.0)], &[], [0, 0, 0, 1]),
        (4, 3)
    );
    assert_eq!(tlas.counters().policy_rebuilds, 0);

    // The policy: moves since the last build adding up to half the instances
    // rebuild.
    tlas.set_policy(TlasPolicy {
        max_refits: 100,
        moved_fraction: 0.5,
    });
    let four: Vec<_> = (0..4).map(|i| instance(1, i as f32 * 2.0)).collect();
    step(&mut tlas, &four, &[], [1, 1, 1, 1]);
    let start = tlas.counters();
    let mut frames = Vec::new();
    for frame in 0..6 {
        // One of four instances moves a little each frame (1/4 of the
        // instances, below 1/2, but they add up); the last frame, three.
        let mut moved = four.clone();
        moved[0].transform[7] = 0.01 * (frame + 1) as f32;
        if frame == 5 {
            moved[1].transform[7] = 0.01;
            moved[2].transform[7] = 0.01;
        }
        step(&mut tlas, &moved, &[], [1, 1, 1, 1]);
        let c = tlas.counters();
        frames.push((
            c.builds - start.builds,
            c.policy_rebuilds - start.policy_rebuilds,
        ));
    }
    // Frame 0: refit (1/4 moved); frame 1: 2/4 reached, rebuilt; frame 2:
    // refit; frame 3: rebuilt; frame 4: refit; frame 5: 3/4 moved, rebuilt.
    assert_eq!(frames, [(0, 0), (1, 1), (1, 1), (2, 2), (2, 2), (3, 3)]);

    // Refits are capped.
    tlas.set_policy(TlasPolicy {
        max_refits: 2,
        moved_fraction: f32::INFINITY,
    });
    let start = tlas.counters();
    for frame in 0..6 {
        let mut moved = four.clone();
        moved[0].transform[7] = 0.1 + 0.01 * frame as f32;
        step(&mut tlas, &moved, &[], [1, 1, 1, 1]);
    }
    let c = tlas.counters();
    // Refit, refit, rebuild, refit, refit, rebuild.
    assert_eq!(
        (
            c.refits - start.refits,
            c.policy_rebuilds - start.policy_rebuilds
        ),
        (4, 2)
    );

    // Only a BLAS the TLAS does not hold changed: nothing is recorded.
    let before = tlas.counters();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    build(
        &mut blas,
        9,
        &mut encoder,
        geometry(&vertices, 0),
        Movability::Static,
    )
    .unwrap();
    let mut last = four.clone();
    last[0].transform[7] = 0.1 + 0.01 * 5.0;
    tlas.build(&mut encoder, &last, &blas).unwrap();
    assert_eq!(tlas.counters(), before);
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [1, 1, 1, 1]);
}

/// Deterministic pseudo-random numbers.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }
}

/// One object of the churn test: its own triangle, seen by ray
/// `slot + offset` (its instance slot plus its vertices' offset).
struct Object {
    id: u64,
    vertices: wgpu::Buffer,
    movability: Movability,
    revision: u64,
    slot: u32,
    offset: u32,
}

#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn churn_never_traces_stale_geometry_and_memory_adds_up() {
    let gpu = Gpu::new();
    let stats = FrameStats::new(&gpu.device);
    let mut blas = BlasManager::new(gpu.device.clone());
    // One compaction a frame, rebuilds every other refit: every path, often.
    blas.set_policy(BlasPolicy {
        refits_per_rebuild: 2,
        compaction_bytes_per_frame: 1,
    });
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let mut rng = Lcg(7);
    let mut objects: Vec<Object> = Vec::new();
    let mut next_id = 0;
    for frame in 0..80 {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        stats.cycle(&mut encoder);
        blas.maintain(&gpu.queue);
        for _ in 0..1 + rng.next(3) {
            match rng.next(5) {
                // Add.
                0 | 1 if objects.len() < 6 => {
                    let offset = rng.next(2) as u32;
                    objects.push(Object {
                        id: next_id,
                        vertices: gpu.buffer(&triangle(offset as f32 * 2.0)),
                        movability: if rng.next(2) == 0 {
                            Movability::Static
                        } else {
                            Movability::Dynamic
                        },
                        revision: 0,
                        slot: rng.next(3) as u32,
                        offset,
                    });
                    next_id += 1;
                }
                // Remove.
                2 if !objects.is_empty() => {
                    let index = rng.next(objects.len() as u64) as usize;
                    let gone = objects.swap_remove(index);
                    blas.remove_blas(gone.id);
                }
                // Move.
                3 if !objects.is_empty() => {
                    let index = rng.next(objects.len() as u64) as usize;
                    objects[index].slot = rng.next(3) as u32;
                }
                // Deform: static ones too (they are built again).
                _ if !objects.is_empty() => {
                    let index = rng.next(objects.len() as u64) as usize;
                    let object = &mut objects[index];
                    object.offset = 1 - object.offset;
                    object.revision += 1;
                    gpu.queue.write_buffer(
                        &object.vertices,
                        0,
                        bytemuck::cast_slice(&triangle(object.offset as f32 * 2.0)),
                    );
                }
                _ => {}
            }
        }
        for object in &objects {
            build(
                &mut blas,
                object.id,
                &mut encoder,
                geometry(&object.vertices, object.revision),
                object.movability,
            )
            .unwrap();
        }
        let instances: Vec<_> = objects
            .iter()
            .map(|o| instance(o.id, o.slot as f32 * 2.0))
            .collect();
        tlas.build(&mut encoder, &instances, &blas).unwrap();
        let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
        blas.report(&stats);
        tlas.report(&stats);
        let mut expected = [0u32; 4];
        for object in &objects {
            expected[(object.slot + object.offset) as usize] = 1;
        }
        assert_eq!(gpu.read(encoder, &readback), expected, "frame {frame}");
        stats.submitted();

        // Memory adds up from the structures themselves.
        let memory = blas.memory();
        assert_eq!(memory.count, objects.len() as u64);
        let bytes: u64 = objects
            .iter()
            .map(|o| blas.get_blas(o.id).unwrap().sizes().structure)
            .sum();
        assert_eq!(memory.bytes, bytes, "frame {frame}");
        assert!(memory.bytes <= memory.bytes_before_compaction);
        let sizes = tlas.tlas().unwrap().sizes();
        assert_eq!(tlas.memory(), sizes.structure + sizes.instances);
    }
    let (b, t) = (blas.counters(), tlas.counters());
    eprintln!("churn: {b:?} {t:?}");
    assert!(b.compactions > 0 && b.refits > 0 && b.policy_rebuilds > 0);
    assert!(t.refits > 0 && t.builds > 0);

    // The last frame reported is what the managers hold.
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    stats.cycle(&mut encoder);
    gpu.finish(encoder);
    stats.submitted();
    gpu.finish(gpu.device.create_command_encoder(&Default::default()));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    stats.cycle(&mut encoder);
    gpu.finish(encoder);
    let latest = stats.latest();
    assert_eq!(latest.get(stat::BLAS_BYTES), Some(blas.memory().bytes));
    assert_eq!(
        latest.get(stat::BLAS_BYTES_BEFORE_COMPACTION),
        Some(blas.memory().bytes_before_compaction)
    );
    assert_eq!(latest.get(stat::TLAS_BYTES), Some(tlas.memory()));
    assert!(latest.get(stat::SCRATCH_BYTES).is_some());
}

/// Measured, not asserted: a full build against a refit of a large
/// deforming mesh (submit to idle, so dominated by the GPU work), and what
/// compaction saves on its static twin.
#[test]
#[ignore = "requires Vulkan hardware ray queries"]
fn refit_and_compaction_costs() {
    let gpu = Gpu::new();
    let mut blas = BlasManager::new(gpu.device.clone());
    blas.set_policy(BlasPolicy {
        refits_per_rebuild: u32::MAX,
        ..BlasPolicy::default()
    });
    let (positions, triangles) = grid(512, 0.0);
    let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("deforming grid"),
        size: (positions.len() * 4) as u64,
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let indices = gpu.indices(&triangles);
    let (vertex_count, index_count) = (positions.len() as u32 / 3, triangles.len() as u32);
    let time = |blas: &mut BlasManager, revision: u64, movability| {
        let wave: Vec<f32> = positions
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], 0.05 * (p[0] * 9.0 + revision as f32).sin()])
            .collect();
        gpu.queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&wave));
        gpu.finish(gpu.device.create_command_encoder(&Default::default()));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let start = std::time::Instant::now();
        let work = build(
            blas,
            1,
            &mut encoder,
            indexed(&vertices, &indices, vertex_count, index_count, revision),
            movability,
        )
        .unwrap();
        gpu.finish(encoder);
        (work, start.elapsed().as_secs_f64() * 1e3)
    };
    let mut builds = Vec::new();
    let mut refits = Vec::new();
    for revision in 0..14 {
        if revision % 2 == 0 {
            blas.request_rebuild(1);
        }
        let (work, ms) = time(&mut blas, revision, Movability::Dynamic);
        if revision >= 2 {
            match work {
                BlasWork::Built => builds.push(ms),
                BlasWork::Refit => refits.push(ms),
                BlasWork::Unchanged => unreachable!(),
            }
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    eprintln!(
        "{} triangles: build {:.2} ms, refit {:.2} ms (submit to idle, median of {})",
        index_count / 3,
        median(&mut builds),
        median(&mut refits),
        refits.len()
    );
    let deforming = blas.memory().bytes;
    let (work, _) = time(&mut blas, 100, Movability::Static);
    assert_eq!(work, BlasWork::Built);
    let built = blas.memory().bytes;
    for _ in 0..200 {
        blas.maintain(&gpu.queue);
        if blas.counters().compactions > 0 {
            break;
        }
        gpu.finish(gpu.device.create_command_encoder(&Default::default()));
    }
    eprintln!(
        "{} triangles: fast-build+update {} bytes, fast-trace {} bytes, compacted {} bytes",
        index_count / 3,
        deforming,
        built,
        blas.memory().bytes
    );
    assert_eq!(blas.counters().compactions, 1);
}
