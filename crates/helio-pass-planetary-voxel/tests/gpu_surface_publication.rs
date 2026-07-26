use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat4, Quat, Vec3};
use helio_core::GpuCameraUniforms;
use helio_pass_planetary_voxel::{
    GpuTransvoxelEmissionCounters, GpuTransvoxelTransitionCounters, SURFACE_PUBLISH_WGSL,
};
use helio_planet_voxel_core::{GpuPageMeta, PageKey};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceJob {
    slot: u32,
    transition_mask: u32,
    generation_low: u32,
    generation_high: u32,
    regular_max_vertices: u32,
    regular_max_indices: u32,
    transition_max_vertices: u32,
    transition_max_indices: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceState {
    generation_low: u32,
    generation_high: u32,
    active_bank: u32,
    valid: u32,
    regular_vertex_count: u32,
    regular_index_count: u32,
    transition_vertex_count: u32,
    transition_index_count: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceFeedback {
    submitted_jobs: u32,
    published_jobs: u32,
    stale_rejections: u32,
    overflow_rejections: u32,
    incomplete_rejections: u32,
    _pad: [u32; 3],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
struct GpuCameraProbe {
    combined: [f32; 4],
    split: [f32; 4],
}

#[test]
#[allow(deprecated)]
fn uploaded_camera_view_projection_matches_split_gpu_multiplication() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!(
                "GPU_VALIDATION_SKIPPED_NO_ADAPTER: no primary or fallback adapter available"
            );
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Camera Matrix Contract Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("camera matrix GPU validation error: {error:?}");
        }));

        let orientation =
            Quat::from_euler(EulerRot::YXZ, -core::f32::consts::FRAC_PI_2, -0.55, 0.0);
        let view = Mat4::look_at_rh(Vec3::ZERO, orientation * -Vec3::Z, orientation * Vec3::Y);
        let proj = Mat4::perspective_rh(core::f32::consts::FRAC_PI_3, 16.0 / 9.0, 0.01, 2_000.0);
        let camera = GpuCameraUniforms::new(
            view,
            proj,
            Vec3::ZERO,
            0.01,
            2_000.0,
            1,
            [0.0; 2],
            Mat4::IDENTITY,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Camera Matrix Contract Shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}
struct Probe { combined: vec4<f32>, split: vec4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read_write> probe: Probe;
@compute @workgroup_size(1)
fn main() {
    let world = vec4<f32>(2.6, -1.55, 0.0, 1.0);
    probe.combined = camera.view_proj * world;
    probe.split = camera.proj * (camera.view * world);
}
"#
                .into(),
            ),
        });
        let pipeline = compute_pipeline(&device, &shader, "main");
        let camera_buffer = initialized_buffer(
            &device,
            "Camera Matrix Contract Uniform",
            bytemuck::bytes_of(&camera),
            wgpu::BufferUsages::UNIFORM,
        );
        let probe_buffer = initialized_buffer(
            &device,
            "Camera Matrix Contract Output",
            bytemuck::bytes_of(&GpuCameraProbe::default()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Matrix Contract Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[entry(0, &camera_buffer), entry(1, &probe_buffer)],
        });
        dispatch(&device, &queue, &pipeline, &bind_group);
        let probe = read_one::<GpuCameraProbe>(&device, &queue, &probe_buffer);
        for axis in 0..4 {
            assert!(
                (probe.combined[axis] - probe.split[axis]).abs() <= 1.0e-5,
                "camera matrix mismatch on axis {axis}: combined={:?}, split={:?}",
                probe.combined,
                probe.split,
            );
        }
    });
}

#[test]
fn publication_is_atomic_generation_safe_and_visibility_gated() {
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
                label: Some("Planetary Surface Publication Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("planetary surface GPU validation error: {error:?}");
        }));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Surface Publication Validation Shader"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_PUBLISH_WGSL.into()),
        });
        let publish_pipeline = compute_pipeline(&device, &shader, "publish_surface");
        let visibility_pipeline = compute_pipeline(&device, &shader, "refresh_visibility");

        let job = GpuSurfaceJob {
            slot: 0,
            generation_low: 43,
            regular_max_vertices: 32,
            regular_max_indices: 64,
            transition_max_vertices: 16,
            transition_max_indices: 48,
            ..Default::default()
        };
        let metadata = GpuPageMeta::new(PageKey::new(0, [0, 0, 0]), [0, 0, 0], 0, 42, 0)
            .expect("validation metadata is valid");
        let regular_success = GpuTransvoxelEmissionCounters {
            emitted_vertices: 19,
            emitted_indices: 27,
            completed: 1,
            ..Default::default()
        };
        let transition_success = GpuTransvoxelTransitionCounters {
            emitted_vertices: 13,
            emitted_indices: 21,
            completed: 1,
            ..Default::default()
        };
        let old_state = GpuSurfaceState {
            generation_low: 41,
            active_bank: 0,
            valid: 1,
            regular_vertex_count: 7,
            regular_index_count: 11,
            transition_vertex_count: 5,
            transition_index_count: 7,
            ..Default::default()
        };
        let draw_page = GpuDrawPage {
            lod0_cell_size_m: 0.1,
            visible: 1,
            ..Default::default()
        };
        let old_regular_draw = DrawIndexedIndirectArgs {
            index_count: 11,
            instance_count: 1,
            first_index: 7,
            base_vertex: 3,
            first_instance: 0,
        };
        let old_transition_draw = DrawIndexedIndirectArgs {
            index_count: 7,
            instance_count: 1,
            first_index: 5,
            base_vertex: 2,
            first_instance: 0,
        };

        let job_buffer = initialized_buffer(
            &device,
            "Surface Publication Job",
            bytemuck::bytes_of(&job),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let metadata_buffer = initialized_buffer(
            &device,
            "Surface Publication Metadata",
            bytemuck::bytes_of(&metadata),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let regular_counters = initialized_buffer(
            &device,
            "Surface Publication Regular Counters",
            bytemuck::bytes_of(&regular_success),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let transition_counters = initialized_buffer(
            &device,
            "Surface Publication Transition Counters",
            bytemuck::bytes_of(&transition_success),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let state_buffer = initialized_buffer(
            &device,
            "Surface Publication State",
            bytemuck::bytes_of(&old_state),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let draw_page_buffer = initialized_buffer(
            &device,
            "Surface Publication Draw Page",
            bytemuck::bytes_of(&draw_page),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let regular_draw_buffer = initialized_buffer(
            &device,
            "Surface Publication Regular Draw",
            bytemuck::bytes_of(&old_regular_draw),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let transition_draw_buffer = initialized_buffer(
            &device,
            "Surface Publication Transition Draw",
            bytemuck::bytes_of(&old_transition_draw),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let feedback_buffer = initialized_buffer(
            &device,
            "Surface Publication Feedback",
            bytemuck::bytes_of(&GpuSurfaceFeedback::default()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let publish_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surface Publication Validation Bind Group"),
            layout: &publish_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &job_buffer),
                entry(1, &metadata_buffer),
                entry(2, &regular_counters),
                entry(5, &state_buffer),
                entry(8, &transition_counters),
                entry(14, &regular_draw_buffer),
                entry(15, &transition_draw_buffer),
                entry(16, &feedback_buffer),
            ],
        });
        let visibility_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surface Visibility Validation Bind Group"),
            layout: &visibility_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(5, &state_buffer),
                entry(13, &draw_page_buffer),
                entry(14, &regular_draw_buffer),
                entry(15, &transition_draw_buffer),
            ],
        });

        dispatch(&device, &queue, &publish_pipeline, &publish_bind_group);
        assert_eq!(
            read_one::<GpuSurfaceState>(&device, &queue, &state_buffer),
            old_state
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &regular_draw_buffer),
            old_regular_draw
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &transition_draw_buffer),
            old_transition_draw
        );
        assert_eq!(
            read_one::<GpuSurfaceFeedback>(&device, &queue, &feedback_buffer),
            GpuSurfaceFeedback {
                submitted_jobs: 1,
                stale_rejections: 1,
                ..Default::default()
            }
        );

        let current_metadata = GpuPageMeta::new(PageKey::new(0, [0, 0, 0]), [0, 0, 0], 0, 43, 0)
            .expect("validation metadata is valid");
        let overflow = GpuTransvoxelEmissionCounters {
            vertex_overflow: 1,
            ..regular_success
        };
        queue.write_buffer(&metadata_buffer, 0, bytemuck::bytes_of(&current_metadata));
        queue.write_buffer(&regular_counters, 0, bytemuck::bytes_of(&overflow));
        dispatch(&device, &queue, &publish_pipeline, &publish_bind_group);
        assert_eq!(
            read_one::<GpuSurfaceState>(&device, &queue, &state_buffer),
            old_state
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &regular_draw_buffer),
            old_regular_draw
        );
        assert_eq!(
            read_one::<GpuSurfaceFeedback>(&device, &queue, &feedback_buffer),
            GpuSurfaceFeedback {
                submitted_jobs: 2,
                stale_rejections: 1,
                overflow_rejections: 1,
                ..Default::default()
            }
        );

        let incomplete = GpuTransvoxelEmissionCounters {
            completed: 0,
            ..regular_success
        };
        queue.write_buffer(&regular_counters, 0, bytemuck::bytes_of(&incomplete));
        dispatch(&device, &queue, &publish_pipeline, &publish_bind_group);
        assert_eq!(
            read_one::<GpuSurfaceState>(&device, &queue, &state_buffer),
            old_state
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &regular_draw_buffer),
            old_regular_draw
        );
        assert_eq!(
            read_one::<GpuSurfaceFeedback>(&device, &queue, &feedback_buffer),
            GpuSurfaceFeedback {
                submitted_jobs: 3,
                stale_rejections: 1,
                overflow_rejections: 1,
                incomplete_rejections: 1,
                ..Default::default()
            }
        );

        queue.write_buffer(&regular_counters, 0, bytemuck::bytes_of(&regular_success));
        dispatch(&device, &queue, &publish_pipeline, &publish_bind_group);
        assert_eq!(
            read_one::<GpuSurfaceState>(&device, &queue, &state_buffer),
            GpuSurfaceState {
                generation_low: 43,
                active_bank: 1,
                valid: 1,
                regular_vertex_count: 19,
                regular_index_count: 27,
                transition_vertex_count: 13,
                transition_index_count: 21,
                ..Default::default()
            }
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &regular_draw_buffer),
            DrawIndexedIndirectArgs {
                index_count: 27,
                instance_count: 0,
                first_index: 64,
                base_vertex: 32,
                first_instance: 0,
            }
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &transition_draw_buffer),
            DrawIndexedIndirectArgs {
                index_count: 21,
                instance_count: 0,
                first_index: 48,
                base_vertex: 16,
                first_instance: 0,
            }
        );
        assert_eq!(
            read_one::<GpuSurfaceFeedback>(&device, &queue, &feedback_buffer),
            GpuSurfaceFeedback {
                submitted_jobs: 4,
                published_jobs: 1,
                stale_rejections: 1,
                overflow_rejections: 1,
                incomplete_rejections: 1,
                ..Default::default()
            }
        );

        dispatch(
            &device,
            &queue,
            &visibility_pipeline,
            &visibility_bind_group,
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &regular_draw_buffer)
                .instance_count,
            1
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &transition_draw_buffer)
                .instance_count,
            1
        );

        queue.write_buffer(
            &draw_page_buffer,
            0,
            bytemuck::bytes_of(&GpuDrawPage {
                visible: 0,
                ..draw_page
            }),
        );
        dispatch(
            &device,
            &queue,
            &visibility_pipeline,
            &visibility_bind_group,
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &regular_draw_buffer)
                .instance_count,
            0
        );
        assert_eq!(
            read_one::<DrawIndexedIndirectArgs>(&device, &queue, &transition_draw_buffer)
                .instance_count,
            0
        );
    });
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

fn compute_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: None,
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn initialized_buffer(
    device: &wgpu::Device,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Surface Publication Validation Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Planetary Surface Publication Validation Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
}

fn read_one<T: Pod + Copy>(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> T {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planetary Surface Publication Readback"),
        size: core::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planetary Surface Publication Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, core::mem::size_of::<T>() as u64);
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
    let value = *bytemuck::from_bytes::<T>(&mapped);
    drop(mapped);
    readback.unmap();
    value
}
