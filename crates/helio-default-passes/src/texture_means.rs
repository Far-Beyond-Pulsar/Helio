//! The mean colour of textures, for shading ray hits without sampling them
//! (`helio_core::RayInstance::base_color` is a material's factor times its
//! base colour texture's mean). Measured on the GPU: a compute pass loads
//! the texture's coarsest mip (decoded to linear by the format) and
//! averages it, and the means are read back asynchronously. A texture's
//! mean is unknown (`None`) until its readback arrives, a frame or two
//! after it was first asked for.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Bytes between two results in the readback (storage offset alignment).
const STRIDE: u64 = 256;
/// Textures measured a dispatch at most.
const BATCH: usize = 64;

const SHADER: &str = r"
@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> mean: vec4<f32>;

// The view's one mip, averaged over at most 8 x 8 of its texels.
@compute @workgroup_size(1)
fn texture_mean() {
    let size = textureDimensions(texture);
    let n = min(size, vec2<u32>(8u));
    var sum = vec4<f32>(0.0);
    for (var y = 0u; y < n.y; y++) {
        for (var x = 0u; x < n.x; x++) {
            sum += textureLoad(texture, (vec2<u32>(x, y) * size) / n, 0);
        }
    }
    mean = sum / f32(n.x * n.y);
}
";

/// Measured texture means (module docs).
pub struct TextureMeans {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    known: HashMap<wgpu::Texture, [f32; 3]>,
    /// Asked for, not yet dispatched.
    queued: Vec<wgpu::Texture>,
    /// Dispatched, readback pending.
    in_flight: HashSet<wgpu::Texture>,
    arrived: Arc<Mutex<Vec<(wgpu::Texture, [f32; 3])>>>,
}

impl TextureMeans {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture mean"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let module = helio_core::shader::module_with(device, "texture mean", SHADER, &[]);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("texture mean"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = helio_core::shader::create_compute_pipeline(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some("texture_mean"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("texture_mean"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        Self {
            layout,
            pipeline,
            known: HashMap::new(),
            queued: Vec::new(),
            in_flight: HashSet::new(),
            arrived: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// `texture`'s mean linear RGB, if measured; else it is asked for.
    /// Textures that cannot be measured (not 2D, not float-sampled) have
    /// none.
    pub fn mean(&mut self, texture: &wgpu::Texture) -> Option<[f32; 3]> {
        if let Some(mean) = self.known.get(texture) {
            return Some(*mean);
        }
        let measurable = texture.dimension() == wgpu::TextureDimension::D2
            && texture
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            && matches!(
                texture.format().sample_type(None, None),
                Some(wgpu::TextureSampleType::Float { .. })
            );
        if measurable && !self.in_flight.contains(texture) && !self.queued.contains(texture) {
            self.queued.push(texture.clone());
        }
        None
    }

    /// Takes the readbacks that arrived; whether any did (means that were
    /// unknown are now known).
    pub fn collect(&mut self) -> bool {
        let arrived = std::mem::take(&mut *self.arrived.lock().unwrap());
        let any = !arrived.is_empty();
        for (texture, mean) in arrived {
            self.in_flight.remove(&texture);
            self.known.insert(texture, mean);
        }
        any
    }

    /// Measures the textures asked for since the last call.
    pub fn dispatch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.queued.is_empty() {
            return;
        }
        let batch: Vec<wgpu::Texture> = self.queued.drain(..self.queued.len().min(BATCH)).collect();
        let bytes = batch.len() as u64 * STRIDE;
        let results = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture means"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture means readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("texture means"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("texture means"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (k, texture) in batch.iter().enumerate() {
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: texture.mip_level_count() - 1,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("texture mean"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &results,
                                offset: k as u64 * STRIDE,
                                size: wgpu::BufferSize::new(16),
                            }),
                        },
                    ],
                });
                pass.set_bind_group(0, Some(&group), &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&results, 0, &readback, 0, bytes);
        queue.submit([encoder.finish()]);
        for texture in &batch {
            self.in_flight.insert(texture.clone());
        }
        let arrived = self.arrived.clone();
        let mapped = readback.clone();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let mut means = Vec::with_capacity(batch.len());
                if result.is_ok() {
                    let data = mapped.slice(..).get_mapped_range().unwrap();
                    for (k, texture) in batch.into_iter().enumerate() {
                        let at = k * STRIDE as usize;
                        let v: &[f32] = bytemuck::cast_slice(&data[at..at + 12]);
                        let mean = [v[0], v[1], v[2]];
                        means.push((
                            texture,
                            if mean.iter().all(|c| c.is_finite()) {
                                mean
                            } else {
                                [1.0; 3]
                            },
                        ));
                    }
                    drop(data);
                    mapped.unmap();
                } else {
                    // Failed: white, as before measuring.
                    means.extend(batch.into_iter().map(|t| (t, [1.0; 3])));
                }
                arrived.lock().unwrap().extend(means);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    /// A 2x2 sRGB texture of red, green, blue and white means linear grey
    /// 0.5, a frame after it is asked for.
    #[test]
    fn means_are_linear_averages_of_the_coarsest_mip() {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let texels: [u8; 16] = [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ];
        let texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &texels,
        );
        let mut means = TextureMeans::new(&device);
        assert_eq!(means.mean(&texture), None);
        means.dispatch(&device, &queue);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(means.collect());
        let mean = means.mean(&texture).unwrap();
        assert!(mean.iter().all(|c| (c - 0.5).abs() < 1e-3), "{mean:?}");
    }
}
