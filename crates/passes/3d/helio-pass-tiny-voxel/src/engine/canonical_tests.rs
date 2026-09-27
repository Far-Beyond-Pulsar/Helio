use wgpu::util::DeviceExt;

#[test]
fn compensated_gpu_position_retains_authored_cells_at_planetary_distances() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU required");
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let mut seed = 37u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 16_777_216.0
        };
        let mut inputs = Vec::<u32>::new();
        let mut expected = Vec::new();
        for i in 0..4096 {
            let origin = [63_717_583i32, -12_340_011, 999_991];
            let fraction = std::array::from_fn::<_, 3, _>(|_| random());
            let offset = std::array::from_fn::<_, 3, _>(|_| (random() - 0.5) * 256.0);
            let direction = glam::Vec3::from_array(std::array::from_fn(|_| random() - 0.5))
                .normalize()
                .to_array();
            let distance = [
                0.0f32,
                0.001,
                1.0,
                128.0,
                1000.0,
                300_000.0,
                1_000_000.0,
                8_400_000.0,
            ][i % 8];
            inputs.extend(origin.map(|x| x as u32));
            inputs.push(0);
            for value in [fraction, offset, direction] {
                inputs.extend(value.map(f32::to_bits));
                inputs.push(0);
            }
            inputs.extend([distance.to_bits(), 0, 0, 0]);
            expected.push(std::array::from_fn::<_, 3, _>(|a| {
                f64::from(origin[a])
                    + f64::from(fraction[a])
                    + (f64::from(offset[a]) + f64::from(direction[a]) * f64::from(distance)) * 10.0
            }));
        }
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planetary precision cases"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = expected.len() as u64 * 32;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let source = format!(
            "{}\n{}",
            include_str!("canonical_position.wgsl"),
            r#"
struct Input { origin:vec4<i32>, fraction:vec4<f32>, offset:vec4<f32>, direction:vec4<f32>, distance:vec4<f32> }
struct Output { cell:vec4<i32>, fraction:vec4<f32> }
@group(0) @binding(0) var<storage,read> inputs:array<Input>;
@group(0) @binding(1) var<storage,read_write> outputs:array<Output>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=inputs[id.x];
    let p=canonical_position(i.origin.xyz,i.fraction.xyz,i.offset.xyz,i.direction.xyz,i.distance.x);
    outputs[id.x]=Output(vec4<i32>(p.cell,0),vec4<f32>(p.fraction,0.0));
}"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production compensated ray position"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(expected.len() as u32 / 64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = readback.slice(..).get_mapped_range().unwrap();
        let words = bytemuck::cast_slice::<u8, u32>(&bytes);
        let mut maximum_error_m = 0.0f64;
        for (i, target) in expected.iter().enumerate() {
            for a in 0..3 {
                let cell = words[i * 8 + a] as i32;
                let fraction = f32::from_bits(words[i * 8 + 4 + a]);
                assert!((0.0..1.0).contains(&fraction));
                let error_m = (f64::from(cell) + f64::from(fraction) - target[a]).abs() * 0.1;
                maximum_error_m = maximum_error_m.max(error_m);
                assert!(error_m < 0.000002, "case {i} axis {a}: error {error_m} m");
                let margin = target[a].fract().abs().min(1.0 - target[a].fract().abs());
                if margin > 0.00002 {
                    assert_eq!(cell, target[a].floor() as i32, "case {i} axis {a}");
                }
            }
        }
        eprintln!(
            "CANONICAL_POSITION cases={} max_error_m={maximum_error_m:.12}",
            expected.len()
        );
    });
}
