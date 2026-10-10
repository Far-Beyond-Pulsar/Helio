//! The hardware 64-bit wide products (`wide64.wgsl`) give the same bits as
//! the 16-bit limb versions they replace (noise.wgsl's wide region).
mod common;
use common::{gpu, read_buffer};
use wgpu::util::DeviceExt;

#[test]
fn hardware_wide_products_match_the_limb_versions() {
    let Some(gpu) = gpu() else { return };
    if !gpu.device.features().contains(wgpu::Features::SHADER_INT64) {
        eprintln!("SKIP: no 64-bit shader integers");
        return;
    }
    let noise = include_str!("../shaders/noise.wgsl");
    let (_, rest) = noise.split_once("// wide:begin").unwrap();
    let (limbs, _) = rest.split_once("// wide:end").unwrap();
    let hardware = include_str!("../shaders/wide64.wgsl").replace("fn mul_wide(", "fn hw_wide(").replace("fn mul_shr(", "fn hw_shr(");
    let source = format!(
        "{limbs}\n{hardware}\n
        @group(0) @binding(0) var<storage, read> inputs: array<vec4<u32>>;
        @group(0) @binding(1) var<storage, read_write> mismatches: array<atomic<u32>>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
            if id.x >= arrayLength(&inputs) {{ return; }}
            let v = inputs[id.x];
            if any(mul_wide(v.x, v.y) != hw_wide(v.x, v.y)) {{ atomicAdd(&mismatches[0], 1u); }}
            if mul_shr(v.x, v.y, v.z) != hw_shr(v.x, v.y, v.z) {{ atomicAdd(&mismatches[1], 1u); }}
        }}"
    );
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wide products"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    // Random operands, the extremes, and every shift the callers use.
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng as u32
    };
    let mut inputs: Vec<[u32; 4]> = (0..200_000).map(|n| [next(), next(), n % 64, 0]).collect();
    for (a, b) in [(0, 0), (u32::MAX, u32::MAX), (u32::MAX, 1), (1 << 31, 1 << 31), (0xffff, 0x10000)] {
        for s in 0..64 {
            inputs.push([a, b, s, 0]);
        }
    }
    let input = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &[0u8; 8],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let bytes = read_buffer(&gpu, &output, 8);
    let wide = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let shifted = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!((wide, shifted), (0, 0), "mul_wide and mul_shr mismatches over {} inputs", inputs.len());
}
