use helio_pass_tiny_voxel::landforms::{VoxelField, BOUNDS_SHADER, SAMPLING_SHADER, VOLUME_SHADER};
use wgpu::util::DeviceExt;

pub struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: [wgpu::ComputePipeline; 2],
}
impl Device {
    pub fn new() -> (Self, String) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let info = format!("{:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let source = format!(
            "{SAMPLING_SHADER}\n{BOUNDS_SHADER}\n{VOLUME_SHADER}\n{}",
            r#"
@group(0) @binding(2) var<storage,read> test_inputs:array<vec4<i32>>;
@group(0) @binding(3) var<storage,read_write> test_outputs:array<vec4<u32>>;
@compute @workgroup_size(64)
fn samples(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&test_inputs) {return;}
    let c=test_inputs[id.x].xyz;
    if !vf_valid(c) {test_outputs[id.x*2u]=vec4<u32>(0u);test_outputs[id.x*2u+1u]=vec4<u32>(0u);return;}
    let seed=lf_settings.pad0;
    let raw=vec3<i32>(vf_noise(c,14u,seed^73u),vf_noise(c,10u,seed^191u),vf_noise(c,7u,seed^311u));
    let sampled=vf_sample(c);
    test_outputs[id.x*2u]=vec4<u32>(vec3<u32>(raw),bitcast<u32>(vf_detail_values(raw)));
    test_outputs[id.x*2u+1u]=vec4<u32>(bitcast<u32>(vf_height(c)),sampled.material,sampled.source,sampled.valid);
}
@compute @workgroup_size(64)
fn regions(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x*2u+1u>=arrayLength(&test_inputs) {return;}
    let b=vf_classify_region(test_inputs[id.x*2u].xyz,test_inputs[id.x*2u+1u].xyz);
    test_outputs[id.x]=vec4<u32>(bitcast<vec2<u32>>(b.height),b.classification,b.valid);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("new voxel field qualification"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = ["samples", "regions"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        (
            Self {
                device,
                queue,
                pipelines,
            },
            info,
        )
    }
    pub fn run(&self, field: &VoxelField, regions: bool, input: &[u8], count: usize) -> Vec<u32> {
        let d = &self.device;
        let pipeline = &self.pipelines[usize::from(regions)];
        let storage = |bytes: &[u8]| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let settings = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&field.gpu_settings()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let heights = storage(bytemuck::cast_slice(
            field.landforms().snapshot().height_atlas(),
        ));
        let inputs = storage(input);
        let hierarchy = storage(bytemuck::cast_slice(field.landforms().hierarchy()));
        let edits = storage(if field.edits().is_empty() {
            &[0u8; 32]
        } else {
            bytemuck::cast_slice(field.edits())
        });
        let output = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (count * if regions { 16 } else { 32 }) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let buffers = [&settings, &heights, &inputs, &output, &hierarchy, &edits];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .filter(|(i, _)| regions || *i != 4)
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = d.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let values = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
        readback.unmap();
        values
    }
}
