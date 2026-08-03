use std::sync::Arc;

use helio_pass_hiz::HiZBuildPass;

fn validate_depth_copy_round_trip(device: &wgpu::Device, queue: &wgpu::Queue, backend: &str) {
    const WIDTH: u32 = 13;
    const HEIGHT: u32 = 7;
    const CLEAR_DEPTH: f32 = 0.375;
    let bytes_per_row = (WIDTH * std::mem::size_of::<f32>() as u32)
        .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer_size = u64::from(bytes_per_row) * u64::from(HEIGHT);

    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("HiZ Portability Depth"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let r32 = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("HiZ Portability R32"),
        size: depth.size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let intermediate = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HiZ Portability Intermediate"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HiZ Portability Readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(bytes_per_row),
        rows_per_image: Some(HEIGHT),
    };
    let extent = depth.size();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("HiZ Portability Encoder"),
    });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("HiZ Portability Depth Clear"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_DEPTH),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &depth,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &intermediate,
            layout,
        },
        extent,
    );
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &intermediate,
            layout,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &r32,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        extent,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &r32,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout,
        },
        extent,
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("send map result");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll depth-copy round trip");
    receiver
        .recv()
        .expect("receive map result")
        .unwrap_or_else(|error| panic!("{backend} depth-copy readback must map: {error}"));

    let mapped = slice
        .get_mapped_range()
        .expect("read mapped depth-copy bytes");
    for row in 0..HEIGHT as usize {
        let start = row * bytes_per_row as usize;
        for column in 0..WIDTH as usize {
            let offset = start + column * std::mem::size_of::<f32>();
            let value = f32::from_ne_bytes(mapped[offset..offset + 4].try_into().unwrap());
            assert_eq!(
                value.to_bits(),
                CLEAR_DEPTH.to_bits(),
                "{backend} must preserve the exact depth bits at ({column}, {row})"
            );
        }
    }
    drop(mapped);
    readback.unmap();
}

async fn compile_on_available_backends() -> usize {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;

    for adapter in &adapters {
        let info = adapter.get_info();
        let backend = format!("{:?}", info.backend);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("HiZ Portability Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap_or_else(|error| panic!("{backend} adapter must create a device: {error}"));
        device.on_uncaptured_error(Arc::new(move |error| {
            panic!("HiZ {backend} validation error: {error:?}");
        }));

        let _pass = HiZBuildPass::new(&device, &queue, 1280, 720);
        if adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
        {
            validate_depth_copy_round_trip(&device, &queue, &format!("{:?}", info.backend));
        }
    }

    adapters.len()
}

#[test]
fn pipelines_compile_on_every_available_backend() {
    if pollster::block_on(compile_on_available_backends()) == 0 {
        eprintln!("skipping HiZ portability test: no GPU adapter available");
    }
}
