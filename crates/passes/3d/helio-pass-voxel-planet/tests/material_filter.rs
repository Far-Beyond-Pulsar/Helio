//! Appearance-only procedural filtering; canonical material queries remain exact.
mod common;
use common::*;
use wgpu::util::DeviceExt;

#[test]
fn unresolved_outcrop_noise_stops_changing_snow_classification() {
    let Some(gpu) = gpu() else { return };
    let noise = include_str!("../shaders/noise.wgsl");
    let landform = include_str!("../shaders/landform.wgsl");
    let world = include_str!("../shaders/world.wgsl");
    let materials = world
        .split("const M_AIR")
        .nth(1)
        .unwrap()
        .split("// Face bases")
        .next()
        .unwrap();
    let source = format!(
        r#"
        {noise}
        const M_AIR{materials}
        {landform}
        @group(0) @binding(0) var<uniform> terrain:TerrainConstants;
        @group(0) @binding(1) var<storage,read> points:array<vec4<i32>>;
        @group(0) @binding(2) var<storage,read_write> answers:array<vec4<u32>>;
        var<private> material_footprint:f32=0.0;
        @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&points) {{return;}}
            let p=points[id.x].xyz;
            material_footprint=0.0;
            let canonical=ground_material(p,4000000,0,5,39999);
            material_footprint=0.1;
            let near=ground_material(p,4000000,0,5,39999);
            material_footprint=128.0;
            let far=ground_material(p,4000000,0,5,39999);
            // The6.4m octave is unresolved here; the51.2m octave remains.
            material_footprint=6.4;
            let middle=ground_material(p,4000000,0,5,39999);
            answers[id.x]=vec4<u32>(canonical,near,far,middle);
        }}
    "#
    );
    let mut constants = [0i32; 140];
    constants[1] = 100;
    constants[2] = 7;
    constants[3] = 123;
    constants[6] = 2_000_000;
    constants[9] = 16;
    constants[12 + 6 * 4] = 10;
    let points: Vec<[i32; 4]> = (0..4096i32)
        .map(|i| [i * 373 - 600000, i * 919 - 700000, i * 1571 - 1000000, 0])
        .collect();
    let terrain = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&constants),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let input = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&points),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (points.len() * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("material appearance filtering"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((points.len() as u32 + 63) / 64, 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let bytes = read_buffer(&gpu, &output, (points.len() * 16) as u64);
    let answers: &[[u32; 4]] = bytemuck::cast_slice(&bytes);
    let mut canonical_rock = 0usize;
    let mut middle_rock = 0usize;
    for (index, a) in answers.iter().enumerate() {
        assert_eq!(a[0], a[1], "near material changed at point{index}");
        assert_eq!(
            a[2], 5,
            "unresolved zero-mean outcrop still changed snow at point{index}"
        );
        canonical_rock += usize::from(a[0] != 5);
        middle_rock += usize::from(a[3] != 5);
    }
    assert!(
        canonical_rock > 0,
        "fixture must expose original outcrop class variation"
    );
    assert!(
        middle_rock > 0,
        "resolvable broad rock patches must remain visible"
    );
}
