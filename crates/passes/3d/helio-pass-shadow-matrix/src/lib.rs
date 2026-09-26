//! GPU shadow matrix computation.
//!
//! Computes light-space view-projection matrices for all shadow-casting lights.
//! O(1) CPU — single compute dispatch regardless of light count.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub mod gpu_types;
pub use gpu_types::*;

const WORKGROUP_SIZE: u32 = 64;

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowMatrixUniforms {
    light_count: u32,
    shadow_atlas_size: u32,
    _pad: [u32; 2],
}

pub struct ShadowMatrixPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Buffers bound alongside the lights, kept to rebind when SceneDB
    /// reallocates the `"scene_lights"` buffer.
    shadow_matrix_buf: wgpu::Buffer,
    camera_buf: wgpu::Buffer,
    shadow_dirty_buf: wgpu::Buffer,
    shadow_hashes_buf: wgpu::Buffer,
    /// The lights buffer `bind_group` currently binds.
    bound_lights: wgpu::Buffer,
    shadow_atlas_size: u32,
}

impl ShadowMatrixPass {
    pub fn new(
        device: &wgpu::Device,
        lights_buf: &wgpu::Buffer,
        shadow_matrix_buf: &wgpu::Buffer,
        camera_buf: &wgpu::Buffer,
        shadow_dirty_buf: &wgpu::Buffer,
        shadow_hashes_buf: &wgpu::Buffer,
        shadow_atlas_size: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShadowMatrix Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/shadow_matrices.wgsl").into(),
            ),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowMatrix Uniforms"),
            size: std::mem::size_of::<ShadowMatrixUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowMatrix BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = Self::bind(
            device,
            &bind_group_layout,
            [lights_buf, shadow_matrix_buf, camera_buf, &uniform_buf, shadow_dirty_buf, shadow_hashes_buf],
        );

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ShadowMatrix PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ShadowMatrix Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("compute_shadow_matrices"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buf,
            bind_group,
            shadow_matrix_buf: shadow_matrix_buf.clone(),
            camera_buf: camera_buf.clone(),
            shadow_dirty_buf: shadow_dirty_buf.clone(),
            shadow_hashes_buf: shadow_hashes_buf.clone(),
            bound_lights: lights_buf.clone(),
            shadow_atlas_size: shadow_atlas_size.max(1),
        }
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffers: [&wgpu::Buffer; 6],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowMatrix BG"),
            layout,
            entries: &entries,
        })
    }
}

impl RenderPass for ShadowMatrixPass {
    fn name(&self) -> &'static str {
        "ShadowMatrix"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let lights = ctx.scene_buffers.get(helio_core::BufferKey::of("scene_lights"));
        // SceneDB grows the lights buffer with the scene: follow the
        // reallocation instead of reading the buffer bound at construction.
        if let Some(lights) = lights.filter(|lights| lights.buffer != self.bound_lights) {
            self.bound_lights = lights.buffer.clone();
            self.bind_group = Self::bind(
                ctx.device,
                &self.bind_group_layout,
                [
                    &self.bound_lights,
                    &self.shadow_matrix_buf,
                    &self.camera_buf,
                    &self.uniform_buf,
                    &self.shadow_dirty_buf,
                    &self.shadow_hashes_buf,
                ],
            );
        }
        let u = ShadowMatrixUniforms {
            light_count: lights.map_or(0, |lights| lights.row_capacity()),
            shadow_atlas_size: self.shadow_atlas_size,
            _pad: [0; 2],
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let count = ctx
            .scene_buffers
            .get(helio_core::BufferKey::of("scene_lights"))
            .map_or(0, |lights| lights.row_capacity());
        if count == 0 {
            return Ok(());
        }
        let wg = count.div_ceil(WORKGROUP_SIZE);
        let mut pass =
            unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ShadowMatrix"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }
}
