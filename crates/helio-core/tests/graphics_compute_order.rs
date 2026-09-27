//! A post-lighting compute consumer must observe this frame, even at motion.
use helio_core::{PassContext, RenderGraph, RenderPass, ResourceKey, ResourceRegistry, Result};
use std::sync::Arc;
mod support;

struct Producer(wgpu::TextureView);
impl RenderPass for Producer {
    fn name(&self) -> &'static str {
        "changing graphics producer"
    }
    fn writes(&self) -> &'static [&'static str] {
        &["test_lighting"]
    }
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &self.0,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: (ctx.frame_num + 1) as f64,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })];
        let desc = wgpu::RenderPassDescriptor {
            color_attachments: &attachments,
            ..Default::default()
        };
        drop(ctx.begin_render_pass(&desc));
        Ok(())
    }
    fn publish<'a>(&self, frame: &mut ResourceRegistry<'a>) {
        frame.route_named_texture("test_lighting", &self.0, self.name());
    }
}

struct Consumer {
    pipeline: wgpu::ComputePipeline,
    output: wgpu::Buffer,
    ordered: bool,
}
impl RenderPass for Consumer {
    fn name(&self) -> &'static str {
        "post graphics capture"
    }
    fn reads(&self) -> &'static [&'static str] {
        &["test_lighting"]
    }
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let input = ctx
            .registry
            .texture_view(ResourceKey::new("test_lighting"))
            .unwrap();
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output.as_entire_binding(),
                },
            ],
        });
        let desc = wgpu::ComputePassDescriptor::default();
        let mut pass = if self.ordered {
            ctx.begin_graphics_compute_pass(&desc)
        } else {
            ctx.begin_compute_pass(&desc)
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }
}

#[test]
fn graphics_compute_reads_current_frame_with_previous_frame_negative_control() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        eprintln!("GRAPHICS_COMPUTE_ADAPTER {:?}", adapter.get_info());
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var input:texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(1) fn main() { output[0]=textureLoad(input,vec2<i32>(0),0).r; }
"#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        for ordered in [false, true] {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut graph = RenderGraph::new(&device, &queue);
            graph.add_pass(Box::new(Producer(view.clone())));
            graph.add_pass(Box::new(Consumer {
                pipeline: pipeline.clone(),
                output: output.clone(),
                ordered,
            }));
            graph.lock(1, 1);
            let mut scene = support::SceneInputAdapter::new(device.clone(), queue.clone());
            for frame in 0..4 {
                scene.frame_count = frame;
                graph.execute(&scene, &view, &view).unwrap();
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 4);
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let value = {
                    let bytes = readback.slice(..).get_mapped_range().unwrap();
                    f32::from_le_bytes(bytes[..4].try_into().unwrap())
                };
                readback.unmap();
                assert_eq!(
                    value,
                    (frame + u64::from(ordered)) as f32,
                    "ordered={ordered}, frame={frame}"
                );
            }
        }
    });
}
