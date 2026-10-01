//! Checks the vendored wgpu patch (Helio#311): raw backend handles are
//! reachable for compute/render pipelines, pipeline layouts and bind groups.
//!
//! Skips when no adapter exists, and on backends other than Vulkan and D3D12.

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { data[id.x] = id.x; }
"#;

#[test]
fn pipelines_layouts_and_bind_groups_expose_hal_handles() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: hal handle access");
            return;
        };
        let backend = adapter.get_info().backend;
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("available adapter must create a device");

        let layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hal access bgl"),
            entries: &[layout_entry],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hal access layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hal access shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hal access pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hal access buffer"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hal access bind group"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        match backend {
            wgpu::Backend::Vulkan => unsafe {
                use wgpu::hal::api::Vulkan;
                assert!(pipeline.as_hal::<Vulkan>().is_some(), "compute pipeline");
                assert!(layout.as_hal::<Vulkan>().is_some(), "pipeline layout");
                assert!(bind_group.as_hal::<Vulkan>().is_some(), "bind group");
                #[cfg(windows)]
                assert!(
                    pipeline.as_hal::<wgpu::hal::api::Dx12>().is_none(),
                    "a Vulkan pipeline must not downcast to another backend"
                );
            },
            #[cfg(windows)]
            wgpu::Backend::Dx12 => unsafe {
                use wgpu::hal::api::Dx12;
                assert!(pipeline.as_hal::<Dx12>().is_some(), "compute pipeline");
                assert!(layout.as_hal::<Dx12>().is_some(), "pipeline layout");
                assert!(bind_group.as_hal::<Dx12>().is_some(), "bind group");
            },
            other => eprintln!("hal handle access: nothing to check on {other:?}"),
        }
    });
}
