use helio_core::{AccelerationError, BlasGeometry, BlasManager, TlasInstanceInput, TlasManager};
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
                ..wgpu::InstanceDescriptor::new_without_display_handle()
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
"#.into()),
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
        revision,
        vertices,
        first_vertex: 0,
        vertex_count: 3,
        vertex_stride: 12,
        indices: None,
        first_index: 0,
        index_count: 0,
    }
}

fn instance(mesh_id: u64, x: f32) -> TlasInstanceInput {
    TlasInstanceInput {
        mesh_id,
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
    assert!(blas
        .build_from_buffers(7, &mut encoder, geometry(&vertices, 0))
        .unwrap());
    assert!(!blas
        .build_from_buffers(7, &mut encoder, geometry(&vertices, 0))
        .unwrap());
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [1, 0, 0, 0]);

    // In-place GPU data replacement with identical allocation/counts.
    gpu.queue
        .write_buffer(&vertices, 0, bytemuck::cast_slice(&triangle(2.0)));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert!(blas
        .build_from_buffers(7, &mut encoder, geometry(&vertices, 1))
        .unwrap());
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 1, 0, 0]);

    // Reallocation must invalidate the cache even when the revision is unchanged.
    let replacement = gpu.buffer(&triangle(4.0));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    assert!(blas
        .build_from_buffers(7, &mut encoder, geometry(&replacement, 1))
        .unwrap());
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
    blas.build_from_buffers(7, &mut encoder, geometry(&vertices, 0))
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
    blas.build_from_buffers(7, &mut encoder, mesh).unwrap();
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [0, 1, 0, 0]);

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut invalid = geometry(&vertices, 1);
    invalid.first_vertex = u32::MAX;
    assert!(matches!(
        blas.build_from_buffers(7, &mut encoder, invalid),
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
    blas.build_from_buffers(7, &mut encoder, mesh).unwrap();
    tlas.build(&mut encoder, &[instance(7, 0.0)], &blas)
        .unwrap();
    let readback = gpu.trace(&mut encoder, tlas.tlas().unwrap());
    assert_eq!(gpu.read(encoder, &readback), [1, 0, 0, 0]);
}
