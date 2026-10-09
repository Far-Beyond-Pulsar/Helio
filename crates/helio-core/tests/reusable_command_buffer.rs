//! Checks the reusable-command-buffer patch in Helio's wgpu (the
//! `helio/v30.0.1-reusable-command-buffers` branch of Far-Beyond-Pulsar/wgpu,
//! Helio#330): a command buffer encoded once can be submitted again,
//! interleaved with `queue.write_buffer`, and each submission really executes.
//!
//! Skips when no adapter exists or the backend cannot reuse command buffers
//! (anything but Vulkan and D3D12).

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@compute @workgroup_size(1)
fn main() { atomicAdd(&counter, 1u); }
"#;

fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer) -> u32 {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reusable readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &readback, 0, 4);
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = readback.slice(..).get_mapped_range().expect("readback is mapped");
    let value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
    drop(mapped);
    readback.unmap();
    value
}

#[test]
fn reusable_command_buffers_resubmit_and_interleave_with_queue_writes() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: reusable command buffers");
            return;
        };
        let backend = adapter.get_info().backend;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("available adapter must create a device");

        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reusable counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reusable counter shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reusable counter pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reusable counter bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counter.as_entire_binding(),
            }],
        });

        // An unmarked command buffer is handed back untouched.
        let plain = device.create_command_encoder(&Default::default()).finish();
        let plain = plain
            .into_reusable()
            .expect_err("an unmarked command buffer must not become reusable");
        queue.submit([plain]);

        let mut encoder = device.create_command_encoder(&Default::default());
        if !encoder.mark_reusable() {
            assert!(
                !matches!(backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12),
                "{backend:?} must support reusable command buffers"
            );
            eprintln!("reusable command buffers: unsupported on {backend:?}, skipping");
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let increment = encoder
            .finish()
            .into_reusable()
            .expect("a marked command buffer must become reusable");

        queue.write_buffer(&counter, 0, &0u32.to_le_bytes());
        for _ in 0..3 {
            queue.submit_mixed([wgpu::SubmitItem::Reusable(&increment)]);
        }
        assert_eq!(read_u32(&device, &queue, &counter), 3, "three replays");

        // A queue write between replays must land before the next replay.
        queue.write_buffer(&counter, 0, &10u32.to_le_bytes());
        let tail = device.create_command_encoder(&Default::default()).finish();
        queue.submit_mixed([
            wgpu::SubmitItem::Reusable(&increment),
            wgpu::SubmitItem::Once(tail),
            wgpu::SubmitItem::Reusable(&increment),
        ]);
        assert_eq!(read_u32(&device, &queue, &counter), 12, "write, then two replays");

        // Dropping it while nothing is in flight, and again after reuse, is fine.
        drop(increment);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    });
}

// Keep many render submissions in flight and preserve a pixel from every
// frame. Reading only the final image would miss intermittent black frames.
#[test]
fn reusable_rendering_preserves_every_frame_under_queue_pressure() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let Ok(adapter) = instance.request_adapter(&Default::default()).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: render replay stress");
            return;
        };
        let backend = adapter.get_info().backend;
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        eprintln!("render replay stress backend: {backend:?}");
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("replay frame color"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("replay frame color"),
            source: wgpu::ShaderSource::Wgsl(r#"
@group(0) @binding(0) var<uniform> color: vec4<u32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
    return vec4(p[i], 0., 1.);
}
@fragment fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(color) / 255.;
}
"#.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("replay frame color"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs"),
                compilation_options: Default::default(), buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None, write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(), depth_stencil: None,
            multisample: Default::default(), multiview_mask: None, cache: None,
        });
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("replay stress target"),
            size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        if !encoder.mark_reusable() {
            eprintln!("render replay stress unsupported on {backend:?}");
            return;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("replay stress draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, depth_slice: None, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, timestamp_writes: None,
                occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &binding, &[]);
            for _ in 0..32 { pass.draw(0..3, 0..1); }
        }
        let draw = encoder.finish().into_reusable().unwrap();
        const FRAMES: u32 = 96;
        let pixels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("replay frame pixels"), size: u64::from(FRAMES) * 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for frame in 0..FRAMES {
            let color = [frame + 1, 255 - frame, 127, 255];
            queue.write_buffer(&params, 0, bytemuck::cast_slice(&color));
            let mut capture = device.create_command_encoder(&Default::default());
            capture.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &pixels,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: u64::from(frame) * 256,
                        bytes_per_row: Some(256), rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            queue.submit_mixed([
                wgpu::SubmitItem::Reusable(&draw),
                wgpu::SubmitItem::Once(capture.finish()),
            ]);
        }
        pixels.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = pixels.slice(..).get_mapped_range().unwrap();
        for frame in 0..FRAMES {
            let offset = frame as usize * 256;
            assert_eq!(&mapped[offset..offset + 4],
                &[frame as u8 + 1, 255 - frame as u8, 127, 255],
                "{backend:?}: frame {frame} became black or stale");
        }
        drop(mapped);
        pixels.unmap();
    });
}
