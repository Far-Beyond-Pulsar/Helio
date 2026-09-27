//! Matched compressed-brick traversal control: same camera, finite canonical
//! source, target size and synthetic lighting as the exact-face raster probe.
use super::*;
use wgpu::util::DeviceExt;

const TRACE: &str = r#"
@group(0) @binding(2) var result:texture_storage_2d<rgba16float,write>;
@group(0) @binding(3) var<storage,read> surface_words:array<u32>;
@compute @workgroup_size(8,8)
fn trace_lit(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(result);if any(id.xy>=size) {return;}
    let ndc=(vec2<f32>(id.xy)+vec2<f32>(0.5))/vec2<f32>(size)*2.0-vec2<f32>(1.0);
    let rd=normalize(camera.forward.xyz+camera.right.xyz*(ndc.x*camera.eye.w*camera.right.w)-camera.up.xyz*(ndc.y*camera.eye.w));
    let hit=surface_trace(0u,camera.eye.xyz,rd,true);
    var color=vec4<f32>(0.0);
    if hit.material>3u {color=vec4<f32>(-1.0);}
    else if hit.material>0u {
        let face=hit.face-1u;
        var normal=vec3<f32>(0.0);normal[face/2u]=select(1.0,-1.0,(face&1u)!=0u);
        let palette=array<vec3<f32>,4>(vec3<f32>(0.0),vec3<f32>(0.18,0.42,0.055),vec3<f32>(0.35,0.12,0.035),vec3<f32>(0.46,0.49,0.56));
        let irradiance=0.12+0.88*max(0.0,dot(normal,camera.light.xyz));
        color=vec4<f32>(palette[hit.material]*irradiance,1.0);
    }
    textureStore(result,vec2<i32>(id.xy),color);
}
"#;

pub(super) fn run(gpu: &Gpu, fixture: usize, output: &std::path::Path) {
    let brick = Brick::from_samples(|q| material(fixture, q));
    let words = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(brick.words()),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let camera = gpu.buffer(
        std::mem::size_of::<Camera>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matched local compressed trace control"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    include_str!("../../raster.wgsl"),
                    crate::surface_cache::GPU_SHADER,
                    TRACE
                )
                .into(),
            ),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("trace_lit"),
            compilation_options: Default::default(),
            cache: None,
        });
    let make_group = |target: &Target| {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &target.color.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: words.as_entire_binding(),
                },
            ],
        })
    };
    let dispatch = |encoder: &mut wgpu::CommandEncoder,
                    group: &wgpu::BindGroup,
                    size: [u32; 2],
                    timing: Option<wgpu::ComputePassTimestampWrites<'_>>| {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: timing,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
    };
    for (lighting, light) in [Vec3::new(-0.4, 0.7, 0.5), Vec3::new(0.8, 0.15, -0.4)]
        .into_iter()
        .enumerate()
    {
        for movement in 0..2 {
            let size = [128, 72];
            let target = Target::new(gpu, size, Output::Lit, 1);
            let c = Camera::new(
                DVec3::new(16.137 + f64::from(movement) * 0.19, 25.17, 52.219),
                DVec3::new(0.0, -0.18, -1.0),
                size,
                light,
            );
            gpu.queue.write_buffer(&camera, 0, bytemuck::bytes_of(&c));
            let group = make_group(&target);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            dispatch(&mut encoder, &group, size, None);
            gpu.queue.submit([encoder.finish()]);
            std::fs::write(
                output.join(format!(
                    "f{fixture}-l{lighting}-m{movement}-trace-128x72.rgba16"
                )),
                target.read(gpu),
            )
            .unwrap();
        }
    }
    let size = [1280, 720];
    let target = Target::new(gpu, size, Output::Lit, 1);
    let group = make_group(&target);
    let c = Camera::new(
        DVec3::new(16.137, 25.17, 52.219),
        DVec3::new(0.0, -0.18, -1.0),
        size,
        Vec3::new(-0.4, 0.7, 0.5),
    );
    gpu.queue.write_buffer(&camera, 0, bytemuck::bytes_of(&c));
    let count = 144;
    let query = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: count * 2,
    });
    let resolve = gpu.buffer(
        u64::from(count) * 16,
        wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = gpu.buffer(
        u64::from(count) * 16,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    for repeat in 0..2 {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for i in 0..count {
            dispatch(
                &mut encoder,
                &group,
                size,
                Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query,
                    beginning_of_pass_write_index: Some(i * 2),
                    end_of_pass_write_index: Some(i * 2 + 1),
                }),
            );
        }
        encoder.resolve_query_set(&query, 0..count * 2, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, u64::from(count) * 16);
        gpu.queue.submit([encoder.finish()]);
        let times: Vec<_> = gpu
            .map(&readback)
            .chunks_exact(16)
            .skip(16)
            .map(|v| {
                let a = u64::from_le_bytes(v[..8].try_into().unwrap());
                let b = u64::from_le_bytes(v[8..].try_into().unwrap());
                assert!(b >= a);
                (b - a) as f64 * f64::from(gpu.queue.get_timestamp_period()) / 1e6
            })
            .collect();
        eprintln!("SURFACE_MESH_TRACE fixture={fixture} repeat={repeat} milliseconds={times:?}");
    }
}
