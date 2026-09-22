use super::*;
use crate::{TsrPass, TsrQuality, TsrUniform};
use wgpu::util::DeviceExt;

/// Exercise the independent diagnostic raster with known visibility states and
/// resized targets. Every pixel must overwrite the sentinel, including returns.
#[test]
fn diagnostics_measure_resets_visibility_and_resized_rasters() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut diagnostic =
        Diagnostics::new(&device, include_str!("../../shaders/tsr_main.wgsl"), 8, 6);
    let cameras = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 736,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    for (width, height) in [(8, 6), (11, 9), (5, 3)] {
        diagnostic.resize(&device, width, height);
        let tsr = TsrPass::new(
            &device,
            width,
            height,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            TsrQuality::Native,
        );
        let texture = |format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        let current = texture(wgpu::TextureFormat::Rgba16Float).create_view(&Default::default());
        let depth = texture(wgpu::TextureFormat::Depth32Float).create_view(&Default::default());
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(width * height) * 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for (reset, outside, invalid_depth, expected_flags) in [
            (true, false, false, 32.0),
            (false, true, false, 64.0),
            (false, false, true, 2.0),
            (false, false, false, 0.0),
        ] {
            let mut matrix = glam::Mat4::IDENTITY;
            if outside {
                matrix.w_axis.x = 4.0;
            }
            let uniform = TsrUniform {
                jitter_offset: [0.0; 2],
                previous_jitter: [0.0; 2],
                reactivity: 0.0,
                reset: u32::from(reset),
                time_delta: 1.0 / 60.0,
                tap_radius: 1,
                clip_to_previous: matrix.to_cols_array(),
            };
            queue.write_buffer(&tsr.uniform_buf, 0, bytemuck::bytes_of(&uniform));
            let mut entries = Vec::new();
            for (binding, view) in [
                (0, &current),
                (1, &tsr.history_view),
                (2, &depth),
                (7, &tsr.history_depth_view),
                (8, &tsr.moments[0].history_view),
                (9, &tsr.moments[1].history_view),
            ] {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
            for (binding, sampler) in [(3, &tsr.linear_sampler), (4, &tsr.point_sampler)] {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Sampler(sampler),
                });
            }
            for (binding, buffer) in [(5, &cameras), (6, &tsr.uniform_buf)] {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                });
            }
            diagnostic.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &vec![0xff; stage.size() as usize],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            diagnostic.bind(&device, &entries);
            let mut encoder = device.create_command_encoder(&Default::default());
            for (view, color) in [
                (
                    &current,
                    wgpu::Color {
                        r: 0.5,
                        g: 0.5,
                        b: 0.5,
                        a: 1.0,
                    },
                ),
                (
                    &tsr.history_view,
                    wgpu::Color {
                        r: 0.5,
                        g: 0.5,
                        b: 0.5,
                        a: 1.0,
                    },
                ),
                (
                    &tsr.history_depth_view,
                    wgpu::Color {
                        r: if invalid_depth { 1.0 } else { 0.5 },
                        g: if invalid_depth { 1.0 } else { 0.5 },
                        b: 0.0,
                        a: 0.0,
                    },
                ),
                (
                    &tsr.moments[0].history_view,
                    wgpu::Color {
                        r: 0.2,
                        g: 0.0,
                        b: 0.0,
                        a: 32.0,
                    },
                ),
                (
                    &tsr.moments[1].history_view,
                    wgpu::Color {
                        r: 0.0625,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    },
                ),
            ] {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            }
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.5),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
            }
            diagnostic.encode(&mut encoder);
            encoder.copy_buffer_to_buffer(&diagnostic.buffer, 0, &stage, 0, stage.size());
            queue.submit(Some(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            stage
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            {
                let bytes = stage.slice(..).get_mapped_range().unwrap();
                let records: &[[f32; 8]] = bytemuck::cast_slice(&bytes);
                for record in records {
                    assert!(record.iter().all(|v| v.is_finite()));
                    assert_eq!(record[3], expected_flags, "{width}x{height}: {record:?}");
                    if expected_flags == 0.0 {
                        assert!((record[1] - 32.0).abs() < 0.001);
                        assert!((record[2] - 0.0548334).abs() < 0.00001);
                        assert!(record[4] < 0.00001 && record[5] < 0.00001);
                    } else {
                        assert_eq!(record[2], 1.0);
                    }
                }
            }
            stage.unmap();
        }
    }
}
