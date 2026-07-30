use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use helio_core::GpuCameraUniforms;
use helio_pass_planetary_voxel::{
    GpuTerrainCullCounters, GpuTerrainCullUniforms, GpuTerrainDraw, GpuTerrainMeshlet,
    GpuTerrainMeshletBounds, TERRAIN_MESHLET_CULL_WGSL,
};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuSurfaceState {
    generation_low: u32,
    generation_high: u32,
    active_bank: u32,
    valid: u32,
    regular_vertex_count: u32,
    regular_index_count: u32,
    transition_vertex_count: u32,
    transition_index_count: u32,
    regular_meshlet_count: u32,
    transition_meshlet_count: u32,
    _pad: [u32; 2],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuDrawPage {
    relative_lod0_cell_min: [i32; 3],
    lod: u32,
    camera_relative_m: [f32; 3],
    lod0_cell_size_m: f32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    visible: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[test]
#[allow(deprecated)]
fn gpu_cull_is_generation_safe_conservative_and_capacity_bounded() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: terrain meshlet cull");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Terrain Meshlet Cull Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a terrain cull device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("terrain meshlet cull GPU validation error: {error:?}");
        }));

        let camera = GpuCameraUniforms::new(
            Mat4::IDENTITY,
            Mat4::perspective_rh(core::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0),
            Vec3::ZERO,
            0.1,
            100.0,
            1,
            [0.0; 2],
            Mat4::IDENTITY,
        );
        let uniforms = GpuTerrainCullUniforms {
            max_meshlets_per_bank: 5,
            draw_capacity: 1,
            surface_kind: 0,
            _pad: 0,
        };
        let state = GpuSurfaceState {
            generation_low: 7,
            active_bank: 0,
            valid: 1,
            regular_meshlet_count: 5,
            ..Default::default()
        };
        let page = GpuDrawPage {
            lod0_cell_size_m: 1.0,
            generation_low: 7,
            visible: 1,
            ..Default::default()
        };
        let mut meshlets = vec![GpuTerrainMeshlet::default(); 10];
        for (index, meshlet) in meshlets[..5].iter_mut().enumerate() {
            *meshlet = GpuTerrainMeshlet {
                first_index: 10 + index as u32 * 3,
                index_count: 3,
                first_vertex: 20,
                vertex_count: 3,
                bounds_offset: index as u32,
                generation_low: 7,
                generation_high: 0,
                _pad: 0,
            };
        }
        meshlets[3].generation_low = 8;
        let visible = GpuTerrainMeshletBounds {
            center: [0.0, 0.0, -5.0],
            radius: 0.25,
            cone_apex: [0.0, 0.0, -5.0],
            cone_cutoff: 1.0,
            cone_axis: [0.0; 3],
            _pad: 0.0,
        };
        let mut bounds = vec![visible; 10];
        bounds[2].center = [100.0, 0.0, -5.0];
        bounds[2].cone_apex = bounds[2].center;
        bounds[4] = GpuTerrainMeshletBounds {
            cone_cutoff: 0.5,
            cone_axis: [0.0, 0.0, -1.0],
            ..visible
        };

        let camera_buffer = initialized(&device, "Cull Camera", bytemuck::bytes_of(&camera), true);
        let uniform_buffer =
            initialized(&device, "Cull Uniform", bytemuck::bytes_of(&uniforms), true);
        let state_buffer = initialized(&device, "Cull State", bytemuck::bytes_of(&state), false);
        let page_buffer = initialized(&device, "Cull Page", bytemuck::bytes_of(&page), false);
        let meshlet_buffer = initialized(
            &device,
            "Cull Meshlets",
            bytemuck::cast_slice(&meshlets),
            false,
        );
        let bounds_buffer =
            initialized(&device, "Cull Bounds", bytemuck::cast_slice(&bounds), false);
        let indirect_buffer = output_buffer::<DrawIndexedIndirectArgs>(&device, "Cull Indirect", 1);
        let draw_buffer = output_buffer::<GpuTerrainDraw>(&device, "Cull Draws", 1);
        let counter_buffer = output_buffer::<GpuTerrainCullCounters>(&device, "Cull Counters", 1);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Meshlet Cull Test Shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_MESHLET_CULL_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Meshlet Cull Test Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("cull_meshlets"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Meshlet Cull Test Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &camera_buffer),
                entry(1, &uniform_buffer),
                entry(2, &state_buffer),
                entry(3, &page_buffer),
                entry(4, &meshlet_buffer),
                entry(5, &bounds_buffer),
                entry(6, &indirect_buffer),
                entry(7, &draw_buffer),
                entry(8, &counter_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Terrain Meshlet Cull Test Encoder"),
        });
        encoder.clear_buffer(&counter_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Terrain Meshlet Cull Test"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);

        let counters = read_one::<GpuTerrainCullCounters>(&device, &queue, &counter_buffer);
        assert_eq!(counters.regular_draws, 2);
        assert_eq!(counters.transition_draws, 0);
        assert_eq!(counters.overflow, 1);
        assert_eq!(counters.stale, 1);
        assert_eq!(counters.frustum_rejects, 1);
        assert_eq!(counters.cone_rejects, 1);
        assert_eq!(counters.invalid_candidates, 0);

        let indirect = read_one::<DrawIndexedIndirectArgs>(&device, &queue, &indirect_buffer);
        assert_eq!(indirect.index_count, 3);
        assert_eq!(indirect.instance_count, 1);
        assert_eq!(indirect.base_vertex, 20);
        assert_eq!(indirect.first_instance, 0);
        assert!(matches!(indirect.first_index, 10 | 13));
        let draw = read_one::<GpuTerrainDraw>(&device, &queue, &draw_buffer);
        assert_eq!(draw.page_slot, 0);
        assert_eq!(draw.surface_kind, 0);
        assert!(matches!(draw.meshlet_index, 0 | 1));

        // Randomized in-frustum bounds exercise CPU/GPU perspective-cone
        // parity without letting frustum classification mask a mismatch.
        const RANDOM_MESHLETS: u32 = 256;
        let random_uniforms = GpuTerrainCullUniforms {
            max_meshlets_per_bank: RANDOM_MESHLETS,
            draw_capacity: RANDOM_MESHLETS,
            surface_kind: 0,
            _pad: 0,
        };
        let random_state = GpuSurfaceState {
            generation_low: 91,
            active_bank: 0,
            valid: 1,
            regular_meshlet_count: RANDOM_MESHLETS,
            ..Default::default()
        };
        let random_page = GpuDrawPage {
            lod0_cell_size_m: 1.0,
            generation_low: 91,
            visible: 1,
            ..Default::default()
        };
        let mut rng = 0x4d59_5df4_d0f3_3173_u64;
        let mut random_meshlets = vec![GpuTerrainMeshlet::default(); RANDOM_MESHLETS as usize * 2];
        let mut random_bounds =
            vec![GpuTerrainMeshletBounds::default(); RANDOM_MESHLETS as usize * 2];
        let mut expected_visible = Vec::new();
        for index in 0..RANDOM_MESHLETS {
            let depth = 5.0 + random_unit(&mut rng) * 45.0;
            let center = [
                (random_unit(&mut rng) * 2.0 - 1.0) * depth * 0.2,
                (random_unit(&mut rng) * 2.0 - 1.0) * depth * 0.2,
                -depth,
            ];
            let radius = 0.02 + random_unit(&mut rng) * 0.25;
            let cone_axis = if next_random(&mut rng) & 1 == 0 {
                [0.0, 0.0, -1.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let bounds = GpuTerrainMeshletBounds {
                center,
                radius,
                cone_apex: center,
                cone_cutoff: 0.25,
                cone_axis,
                _pad: 0.0,
            };
            random_meshlets[index as usize] = GpuTerrainMeshlet {
                first_index: index * 3,
                index_count: 3,
                first_vertex: index * 3,
                vertex_count: 3,
                bounds_offset: index,
                generation_low: 91,
                generation_high: 0,
                _pad: 0,
            };
            random_bounds[index as usize] = bounds;
            if !conservative_cone_reject(&bounds, [0.0; 3], 1.0) {
                expected_visible.push(index);
            }
        }

        let random_uniform_buffer = initialized(
            &device,
            "Random Cull Uniform",
            bytemuck::bytes_of(&random_uniforms),
            true,
        );
        let random_state_buffer = initialized(
            &device,
            "Random Cull State",
            bytemuck::bytes_of(&random_state),
            false,
        );
        let random_page_buffer = initialized(
            &device,
            "Random Cull Page",
            bytemuck::bytes_of(&random_page),
            false,
        );
        let random_meshlet_buffer = initialized(
            &device,
            "Random Cull Meshlets",
            bytemuck::cast_slice(&random_meshlets),
            false,
        );
        let random_bounds_buffer = initialized(
            &device,
            "Random Cull Bounds",
            bytemuck::cast_slice(&random_bounds),
            false,
        );
        let random_indirect = output_buffer::<DrawIndexedIndirectArgs>(
            &device,
            "Random Cull Indirect",
            RANDOM_MESHLETS as usize,
        );
        let random_draws =
            output_buffer::<GpuTerrainDraw>(&device, "Random Cull Draws", RANDOM_MESHLETS as usize);
        let random_counters =
            output_buffer::<GpuTerrainCullCounters>(&device, "Random Cull Counters", 1);
        let random_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Random Terrain Meshlet Cull Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &camera_buffer),
                entry(1, &random_uniform_buffer),
                entry(2, &random_state_buffer),
                entry(3, &random_page_buffer),
                entry(4, &random_meshlet_buffer),
                entry(5, &random_bounds_buffer),
                entry(6, &random_indirect),
                entry(7, &random_draws),
                entry(8, &random_counters),
            ],
        });
        let mut random_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Random Terrain Meshlet Cull Encoder"),
        });
        random_encoder.clear_buffer(&random_counters, 0, None);
        {
            let mut pass = random_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Random Terrain Meshlet Cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &random_bind_group, &[]);
            pass.dispatch_workgroups(RANDOM_MESHLETS.div_ceil(64), 1, 1);
        }
        queue.submit([random_encoder.finish()]);

        let random_counter_values =
            read_one::<GpuTerrainCullCounters>(&device, &queue, &random_counters);
        assert_eq!(
            random_counter_values.regular_draws,
            expected_visible.len() as u32
        );
        assert_eq!(random_counter_values.overflow, 0);
        assert_eq!(random_counter_values.stale, 0);
        assert_eq!(random_counter_values.frustum_rejects, 0);
        assert_eq!(
            random_counter_values.cone_rejects,
            RANDOM_MESHLETS - expected_visible.len() as u32
        );
        let mut actual_visible =
            read_vec::<GpuTerrainDraw>(&device, &queue, &random_draws, expected_visible.len())
                .into_iter()
                .map(|draw| draw.meshlet_index)
                .collect::<Vec<_>>();
        actual_visible.sort_unstable();
        assert_eq!(actual_visible, expected_visible);
    });
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_unit(state: &mut u64) -> f32 {
    (next_random(state) as u32) as f32 / u32::MAX as f32
}

fn conservative_cone_reject(
    bounds: &GpuTerrainMeshletBounds,
    camera: [f32; 3],
    world_scale: f32,
) -> bool {
    let center_delta = [
        bounds.center[0] * world_scale - camera[0],
        bounds.center[1] * world_scale - camera[1],
        bounds.center[2] * world_scale - camera[2],
    ];
    let guard_radius = bounds.radius * world_scale * 1.5;
    if squared_length(center_delta) <= guard_radius * guard_radius {
        return false;
    }
    let apex_delta = [
        bounds.cone_apex[0] * world_scale - camera[0],
        bounds.cone_apex[1] * world_scale - camera[1],
        bounds.cone_apex[2] * world_scale - camera[2],
    ];
    let apex_length_squared = squared_length(apex_delta);
    if bounds.cone_cutoff > 1.0 || apex_length_squared <= 1.0e-12 {
        return false;
    }
    let inverse_length = apex_length_squared.sqrt().recip();
    let view = [
        apex_delta[0] * inverse_length,
        apex_delta[1] * inverse_length,
        apex_delta[2] * inverse_length,
    ];
    view[0] * bounds.cone_axis[0] + view[1] * bounds.cone_axis[1] + view[2] * bounds.cone_axis[2]
        >= bounds.cone_cutoff
}

fn squared_length(value: [f32; 3]) -> f32 {
    value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
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
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn read_one<T: Pod + Copy>(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> T {
    let bytes = core::mem::size_of::<T>() as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Terrain Meshlet Cull Readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Terrain Meshlet Cull Readback Encoder"),
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
        .expect("terrain cull readback callback must run")
        .expect("terrain cull readback must map");
    let mapped = slice
        .get_mapped_range()
        .expect("terrain cull mapped range must be available");
    let value = *bytemuck::from_bytes::<T>(&mapped);
    drop(mapped);
    readback.unmap();
    value
}

fn read_vec<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let bytes = (core::mem::size_of::<T>() * count) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Terrain Meshlet Cull Vector Readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Terrain Meshlet Cull Vector Readback Encoder"),
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
        .expect("terrain cull vector readback callback must run")
        .expect("terrain cull vector readback must map");
    let mapped = slice
        .get_mapped_range()
        .expect("terrain cull vector mapped range must be available");
    let values = bytemuck::cast_slice::<u8, T>(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}
