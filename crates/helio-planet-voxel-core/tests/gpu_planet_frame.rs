use bytemuck::{Pod, Zeroable};
use helio_planet_voxel_core::{
    GpuPageMeta, PageKey, PlanetFrameUniform, PlanetId, PlanetPosition, PlanetRenderFrame,
    LOD0_CELL_SIZE_METERS, MILLIMETER_INTERACTION_RADIUS_METERS, PLANET_VOXEL_LAYOUT_WGSL,
};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuParityInput {
    frame: PlanetFrameUniform,
    page: GpuPageMeta,
    local_lod0_cell: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuParityOutput {
    origin_x: [u32; 2],
    origin_y: [u32; 2],
    origin_z: [u32; 2],
    frame_index: [u32; 2],
    camera_local_m: [f32; 4],
}

#[derive(Clone, Copy)]
struct Case {
    camera: PlanetPosition,
    point: PlanetPosition,
}

#[test]
fn gpu_matches_canonical_frames_at_planet_scale_and_across_rebases() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!(
                "GPU_VALIDATION_SKIPPED_NO_ADAPTER: no primary or fallback adapter available"
            );
            return;
        };
        let info = adapter.get_info();
        eprintln!(
            "GPU_PLANET_FRAME_ADAPTER: name={:?} backend={:?} device_type={:?}",
            info.name, info.backend, info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Planet Frame Parity Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a validation device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("planet frame GPU validation error: {error:?}");
        }));

        let earth_cells = 63_710_000_i64;
        let orbit_cells = 67_710_000_i64;
        let cases = [
            // Ground, low orbit, antipode, and mixed-sign coordinates.
            case([earth_cells, 0, 0], [11, -7, 3]),
            case([orbit_cells, 120, -80], [-13, 9, 5]),
            case([-earth_cells, 0, 0], [17, 4, -6]),
            case([-earth_cells, earth_cells, -earth_cells], [-19, 21, 8]),
            // The same local relationship on either side of a page-origin rebase.
            case([earth_cells + 31, -1, 7], [5, -3, 2]),
            case([earth_cells + 32, 0, 8], [5, -3, 2]),
            // The documented millimeter-precision interaction bound.
            case(
                [earth_cells, 0, 0],
                [
                    (MILLIMETER_INTERACTION_RADIUS_METERS / LOD0_CELL_SIZE_METERS) as i64 - 1,
                    0,
                    0,
                ],
            ),
        ];

        let planet = PlanetId([0x5a; 16]);
        let mut inputs = Vec::with_capacity(cases.len());
        let mut expected = Vec::with_capacity(cases.len());
        for (index, case) in cases.into_iter().enumerate() {
            let frame = PlanetRenderFrame::new(planet, case.camera, index as u64 + 1);
            let uniform = PlanetFrameUniform::from_render_frame(frame);
            let (page, local) = PageKey::address_lod0_cell(0, case.point.lod0_cell()).unwrap();
            let page =
                GpuPageMeta::new(page, frame.origin_lod0_cell(), index as u32, 1, 0).unwrap();
            let subcell = case.point.subcell_m();
            let local_lod0_cell = [
                f32::from(local[0]) + (subcell[0] / LOD0_CELL_SIZE_METERS) as f32,
                f32::from(local[1]) + (subcell[1] / LOD0_CELL_SIZE_METERS) as f32,
                f32::from(local[2]) + (subcell[2] / LOD0_CELL_SIZE_METERS) as f32,
                0.0,
            ];
            inputs.push(GpuParityInput {
                frame: uniform,
                page,
                local_lod0_cell,
            });
            expected.push((uniform, frame.camera_local_meters(case.point).unwrap()));
        }

        let outputs = dispatch(&device, &queue, &inputs);
        assert_eq!(outputs.len(), expected.len());
        let mut max_error_m = 0.0_f64;
        for (index, (output, (uniform, expected_m))) in
            outputs.into_iter().zip(expected).enumerate()
        {
            assert_eq!(output.origin_x, uniform.origin_x, "case {index} origin x");
            assert_eq!(output.origin_y, uniform.origin_y, "case {index} origin y");
            assert_eq!(output.origin_z, uniform.origin_z, "case {index} origin z");
            assert_eq!(
                output.frame_index, uniform.frame_index,
                "case {index} frame index"
            );
            for (axis, expected_axis_m) in expected_m.into_iter().enumerate() {
                let error = (f64::from(output.camera_local_m[axis]) - expected_axis_m).abs();
                max_error_m = max_error_m.max(error);
                assert!(
                    error <= 0.001,
                    "case {index} axis {axis} exceeded 1 mm: gpu={} cpu={} error={error}",
                    output.camera_local_m[axis],
                    expected_axis_m
                );
            }
        }
        eprintln!("GPU_PLANET_FRAME_MAX_ERROR_METERS: {max_error_m:.9}");

        let camera =
            PlanetPosition::new([earth_cells + 17, -earth_cells - 5, 11], [0.03; 3]).unwrap();
        let frame = PlanetRenderFrame::new(planet, camera, 99);
        let uniform = PlanetFrameUniform::from_render_frame(frame);
        let left_page = PageKey::address_lod0_cell(0, frame.origin_lod0_cell())
            .unwrap()
            .0;
        let right_page = PageKey::new(
            0,
            [
                left_page.page_xyz[0] + 1,
                left_page.page_xyz[1],
                left_page.page_xyz[2],
            ],
        );
        let boundary_inputs = [
            GpuParityInput {
                frame: uniform,
                page: GpuPageMeta::new(left_page, frame.origin_lod0_cell(), 0, 1, 0).unwrap(),
                local_lod0_cell: [32.0, 13.25, 6.75, 0.0],
            },
            GpuParityInput {
                frame: uniform,
                page: GpuPageMeta::new(right_page, frame.origin_lod0_cell(), 1, 1, 0).unwrap(),
                local_lod0_cell: [0.0, 13.25, 6.75, 0.0],
            },
        ];
        let boundary_outputs = dispatch(&device, &queue, &boundary_inputs);
        assert_eq!(
            boundary_outputs[0].camera_local_m, boundary_outputs[1].camera_local_m,
            "adjacent pages must reconstruct their shared boundary bit-identically"
        );
    });
}

fn case(camera_cell: [i64; 3], point_delta_cell: [i64; 3]) -> Case {
    let camera = PlanetPosition::new(camera_cell, [0.037, 0.081, 0.019]).unwrap();
    let point_cell = [
        camera_cell[0] + point_delta_cell[0],
        camera_cell[1] + point_delta_cell[1],
        camera_cell[2] + point_delta_cell[2],
    ];
    let point = PlanetPosition::new(point_cell, [0.092, 0.006, 0.071]).unwrap();
    Case { camera, point }
}

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    inputs: &[GpuParityInput],
) -> Vec<GpuParityOutput> {
    let shader_source = format!(
        "{PLANET_VOXEL_LAYOUT_WGSL}\n{}",
        r#"
struct GpuParityInput {
    frame: PlanetFrameUniform,
    page: GpuPageMeta,
    local_lod0_cell: vec4<f32>,
}

struct GpuParityOutput {
    origin_x: vec2<u32>,
    origin_y: vec2<u32>,
    origin_z: vec2<u32>,
    frame_index: vec2<u32>,
    camera_local_m: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> inputs: array<GpuParityInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<GpuParityOutput>;

@compute @workgroup_size(64)
fn validate_planet_frame(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&inputs)) {
        return;
    }
    let input = inputs[index];
    let local_m = planet_camera_local_position_m(
        input.frame,
        input.page,
        input.local_lod0_cell.xyz,
    );
    outputs[index] = GpuParityOutput(
        input.frame.origin_x,
        input.frame.origin_y,
        input.frame.origin_z,
        input.frame.frame_index,
        vec4<f32>(local_m, 0.0),
    );
}
"#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Planet Frame Parity Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Planet Frame Parity Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("validate_planet_frame"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Planet Frame Parity Inputs"),
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_size = (inputs.len() * core::mem::size_of::<GpuParityOutput>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planet Frame Parity Outputs"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Planet Frame Parity Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planet Frame Parity Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Planet Frame Parity Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    read_buffer(device, queue, &output_buffer, output_size)
}

fn read_buffer<T: Pod + Copy>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> Vec<T> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Planet Frame Parity Readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Planet Frame Readback Encoder"),
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
