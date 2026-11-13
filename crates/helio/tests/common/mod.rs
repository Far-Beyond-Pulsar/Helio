//! A GPU harness shared by the renderer's integration tests: one device at a
//! time, a renderer, and a frame read back as RGBA8.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use glam::Vec3;
use helio::{
    Camera, IntoGraph, NamedPipeline, PassBuildContext, PipelineFailure, Renderer, RendererBuilder,
    RendererConfig,
};
use helio_core::RenderGraph;
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// The renderer's own clear colour (what a rendering pipeline's frame starts as).
pub const PIPELINE_CLEAR: [f32; 4] = [0.0, 0.2, 0.0, 1.0];

/// One test at a time may hold a device (see Pulsar-Native#1647).
pub static GPU_DEVICE: Mutex<()> = Mutex::new(());

pub fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    GPU_DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct Gpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

pub fn gpu() -> Option<Gpu> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: helio::required_wgpu_features(adapter.features()),
                required_limits: helio::required_wgpu_limits(adapter.limits()),
                experimental_features: helio::required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Gpu {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    })
}

impl Gpu {
    pub fn mirror(&self) -> GpuMirrorHandle {
        let context = EngineGpuContext::new(Arc::clone(&self.device), Arc::clone(&self.queue));
        let store = SceneGpuStore::new(
            &context,
            SceneGpuConfig {
                classes: Vec::new(),
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        );
        GpuMirrorHandle::new(Arc::new(store), Arc::clone(&self.queue))
    }

    /// A renderer with the pipeline the builder is given (or none).
    pub fn renderer(&self, size: [u32; 2], pipeline: Option<NamedPipeline>) -> Renderer {
        self.renderer_in(size, pipeline, FORMAT)
    }

    /// [`renderer`](Self::renderer) drawing to `format`.
    pub fn renderer_in(
        &self,
        size: [u32; 2],
        pipeline: Option<NamedPipeline>,
        format: wgpu::TextureFormat,
    ) -> Renderer {
        let mut builder =
            RendererBuilder::new(RendererConfig::new(size[0], size[1], format), self.mirror())
                .with_external_device()
                .with_clear_color(PIPELINE_CLEAR);
        if let Some(NamedPipeline { id, pipeline }) = pipeline {
            builder = builder.with_pipeline(id, BoxedPipeline(pipeline));
        }
        builder.build(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            size[0],
            size[1],
            format,
        )
    }

    /// Renders one frame and reads it back as RGBA8.
    pub fn frame(&self, renderer: &mut Renderer) -> Frame {
        self.frame_in(renderer, FORMAT)
    }

    /// [`frame`](Self::frame) for a renderer that draws to `format` (the
    /// pixels come back as RGBA whatever the channel order).
    pub fn frame_in(&self, renderer: &mut Renderer, format: wgpu::TextureFormat) -> Frame {
        let (width, height) = (renderer.output_width(), renderer.output_height());
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let camera = Camera::perspective_look_at(
            Vec3::new(0.0, 2.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            1.0,
            width as f32 / height as f32,
            0.1,
            100.0,
        );
        renderer.render(&camera, &view).expect("the frame renders");

        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let start = (y * row) as usize;
            pixels.extend_from_slice(&mapped[start..start + (width * 4) as usize]);
        }
        drop(mapped);
        readback.unmap();
        if matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        Frame {
            width,
            height,
            pixels,
        }
    }
}

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Frame {
    pub fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        self.pixels[i..i + 4].try_into().unwrap()
    }

    /// Pixels the text drew: bright, in any channel.
    pub fn text_pixels(&self) -> usize {
        self.pixels
            .chunks_exact(4)
            .filter(|p| p[0] > 150 && p[1] > 150 && p[2] > 150)
            .count()
    }

    pub fn corner(&self) -> [u8; 4] {
        self.at(self.width - 1, self.height - 1)
    }
}

/// `Box<dyn IntoGraph>` as an `IntoGraph` for `with_pipeline`.
pub struct BoxedPipeline(pub Box<dyn IntoGraph>);

impl IntoGraph for BoxedPipeline {
    fn into_graph(
        self: Box<Self>,
        ctx: PassBuildContext<'_>,
    ) -> Result<RenderGraph, PipelineFailure> {
        self.0.into_graph(ctx)
    }
}

/// Whether `pixel` is the status screen's near-black field (its vignette and
/// grain included).
pub fn is_void(pixel: [u8; 4]) -> bool {
    pixel[..3].iter().all(|&c| c <= 12)
}

pub fn near(pixel: [u8; 4], color: [f32; 4]) -> bool {
    (0..3).all(|c| (i32::from(pixel[c]) - (color[c] * 255.0).round() as i32).abs() <= 2)
}

macro_rules! gpu_or_skip {
    ($name:literal) => {{
        match $crate::common::gpu() {
            Some(gpu) => gpu,
            None => {
                eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: {}", $name);
                return;
            }
        }
    }};
}
