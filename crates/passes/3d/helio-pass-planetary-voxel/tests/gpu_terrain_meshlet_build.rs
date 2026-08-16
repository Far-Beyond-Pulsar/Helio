use bytemuck::{Pod, Zeroable};
use helio_pass_planetary_voxel::{
    build_terrain_meshlets, GpuTerrainMeshlet, GpuTerrainMeshletBounds, GpuTerrainVertex,
    GpuTransvoxelEmissionCounters, TERRAIN_MESHLET_BUILD_WGSL,
};
use helio_planet_voxel_core::GpuPageMeta;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceJob {
    slot: u32,
    resident_slot: u32,
    transition_mask: u32,
    generation_low: u32,
    generation_high: u32,
    revision: u32,
    regular_first_vertex: u32,
    regular_first_index: u32,
    regular_first_meshlet: u32,
    transition_first_vertex: u32,
    transition_first_index: u32,
    transition_first_meshlet: u32,
    _pad: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceState {
    generation_low: u32,
    generation_high: u32,
    valid: u32,
    revision: u32,
    regular_first_vertex: u32,
    regular_vertex_count: u32,
    regular_first_index: u32,
    regular_index_count: u32,
    regular_first_meshlet: u32,
    regular_meshlet_count: u32,
    transition_first_vertex: u32,
    transition_vertex_count: u32,
    transition_first_index: u32,
    transition_index_count: u32,
    transition_first_meshlet: u32,
    transition_meshlet_count: u32,
    transition_mask: u32,
    _pad: [u32; 3],
}

#[test]
fn gpu_regular_builder_matches_cpu_descriptors_and_conservative_bounds() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: terrain meshlet build");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Terrain Meshlet Build Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a terrain meshlet validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("terrain meshlet GPU validation error: {error:?}");
        }));

        let generation = 0x0123_4567_89ab_cdef_u64;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for triangle in 0..50_u32 {
            let base = vertices.len() as u32;
            let x = triangle as f32 * 0.25;
            vertices.extend([
                vertex([x, 0.0, 0.0], triangle),
                vertex([x, 1.0, 0.0], triangle),
                vertex([x, 0.0, 1.0], triangle),
            ]);
            indices.extend([base, base + 1, base + 2]);
        }
        let expected = build_terrain_meshlets(&vertices, &indices, 0, 0, 0, generation, 0).unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Meshlet Build Validation Shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_MESHLET_BUILD_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Meshlet Build Validation Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("build_regular"),
            compilation_options: Default::default(),
            cache: None,
        });

        let job = GpuSurfaceJob {
            slot: 0,
            resident_slot: 3,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            ..Default::default()
        };
        let page = GpuPageMeta {
            slot: job.resident_slot,
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            ..Default::default()
        };
        let state = GpuSurfaceState {
            valid: 1,
            ..Default::default()
        };
        let counters = GpuTransvoxelEmissionCounters {
            required_vertices: vertices.len() as u32,
            required_indices: indices.len() as u32,
            emitted_vertices: vertices.len() as u32,
            emitted_indices: indices.len() as u32,
            completed: 1,
            ..Default::default()
        };

        let job_buffer = initialized(&device, "Meshlet Job", bytemuck::bytes_of(&job), true);
        let mut pages = [GpuPageMeta::default(); 4];
        pages[job.resident_slot as usize] = page;
        let page_buffer = initialized(&device, "Meshlet Page", bytemuck::cast_slice(&pages), false);
        let state_buffer = initialized(&device, "Meshlet State", bytemuck::bytes_of(&state), false);
        let counter_buffer = initialized(
            &device,
            "Meshlet Counters",
            bytemuck::bytes_of(&counters),
            false,
        );
        let vertex_buffer = initialized(
            &device,
            "Meshlet Vertices",
            bytemuck::cast_slice(&vertices),
            false,
        );
        let index_buffer = initialized(
            &device,
            "Meshlet Indices",
            bytemuck::cast_slice(&indices),
            false,
        );
        let meshlet_buffer =
            output_buffer::<GpuTerrainMeshlet>(&device, "Meshlet Descriptors", expected.len());
        let bounds_buffer =
            output_buffer::<GpuTerrainMeshletBounds>(&device, "Meshlet Bounds", expected.len());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Meshlet Build Validation Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &job_buffer),
                entry(1, &page_buffer),
                entry(2, &state_buffer),
                entry(3, &counter_buffer),
                entry(4, &vertex_buffer),
                entry(5, &index_buffer),
                entry(6, &meshlet_buffer),
                entry(7, &bounds_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Terrain Meshlet Build Validation Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Terrain Meshlet Build Validation Dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);

        let descriptors =
            read_vec::<GpuTerrainMeshlet>(&device, &queue, &meshlet_buffer, expected.len());
        let bounds =
            read_vec::<GpuTerrainMeshletBounds>(&device, &queue, &bounds_buffer, expected.len());
        for (index, expected) in expected.iter().enumerate() {
            assert_eq!(descriptors[index], expected.descriptor);
            assert_vec3_close(bounds[index].center, expected.bounds.center, 1.0e-5);
            assert_close(bounds[index].radius, expected.bounds.radius, 1.0e-5);
            assert_vec3_close(bounds[index].cone_apex, expected.bounds.cone_apex, 1.0e-4);
            assert_close(
                bounds[index].cone_cutoff,
                expected.bounds.cone_cutoff,
                1.0e-4,
            );
            assert_vec3_close(bounds[index].cone_axis, expected.bounds.cone_axis, 1.0e-5);
        }
    });
}

fn vertex(position: [f32; 3], material: u32) -> GpuTerrainVertex {
    GpuTerrainVertex {
        position,
        material,
        normal: [1.0, 0.0, 0.0],
        flags: material,
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

fn initialized(device: &wgpu::Device, label: &str, contents: &[u8], uniform: bool) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: if uniform {
            wgpu::BufferUsages::UNIFORM
        } else {
            wgpu::BufferUsages::STORAGE
        },
    })
}

fn output_buffer<T>(device: &wgpu::Device, label: &str, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (core::mem::size_of::<T>() * count) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn read_vec<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let bytes = (core::mem::size_of::<T>() * count) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Terrain Meshlet Build Readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Terrain Meshlet Build Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, bytes);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver
        .recv()
        .expect("terrain meshlet readback callback must run")
        .expect("terrain meshlet readback must map");
    let mapped = slice
        .get_mapped_range()
        .expect("terrain meshlet mapped range must be available");
    let values = bytemuck::cast_slice::<u8, T>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} tolerance={tolerance}"
    );
}

fn assert_vec3_close(actual: [f32; 3], expected: [f32; 3], tolerance: f32) {
    for axis in 0..3 {
        assert_close(actual[axis], expected[axis], tolerance);
    }
}
