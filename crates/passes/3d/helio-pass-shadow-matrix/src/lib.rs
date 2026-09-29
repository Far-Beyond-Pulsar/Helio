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
    /// Faces the matrices buffer holds (its size / 64 bytes).
    face_capacity: u32,
    /// Bumped when light rows are uploaded or the camera moves: every caster's
    /// cached (static) faces are then re-rendered. Directional cascades follow
    /// the camera; SceneDB's content generation reports light edits.
    caster_generation: u64,
    last_lights_generation: Option<u64>,
    last_view_proj: [f32; 16],
    /// Advances every frame so ShadowPass runs its GPU-gated per-face path,
    /// which consumes the matrix pass's per-caster dirty flags and movement.
    frame_generation: u64,
    /// GPU shadow-caster allocation (`shadow_casters.wgsl`, Helio#246):
    /// turns `shadow_index` requests into atlas slots in the light rows.
    caster_pipeline: wgpu::ComputePipeline,
    caster_bind_group_layout: wgpu::BindGroupLayout,
    caster_bind_group: wgpu::BindGroup,
    caster_params_buf: wgpu::Buffer,
    /// `(epoch, content_generation, row_capacity, caster_capacity)` of the
    /// light rows the slots were last assigned for. Written slots stay valid
    /// until SceneDB re-uploads a row, which bumps the content generation.
    caster_key: Option<(u64, u64, u32, u32)>,
    /// Set by `prepare` when `caster_key` is stale; `execute` reallocates.
    caster_rebuild: Option<(u64, u64, u32, u32)>,
}

/// Uniforms of `shadow_casters.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CasterParams {
    row_count: u32,
    caster_capacity: u32,
    _pad: [u32; 2],
}

/// Most casters any consumer addresses (`MAX_SHADOW_LIGHTS` in the lighting
/// shaders, ShadowPass's per-caster arrays).
pub const MAX_SHADOW_CASTERS: u32 = 42;

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

        let caster_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow caster allocation"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_casters.wgsl").into()),
        });
        let caster_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow caster allocation BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let caster_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow caster allocation params"),
            size: std::mem::size_of::<CasterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let caster_bind_group =
            Self::bind_casters(device, &caster_bind_group_layout, lights_buf, &caster_params_buf);
        let caster_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow caster allocation PL"),
            bind_group_layouts: &[Some(&caster_bind_group_layout)],
            immediate_size: 0,
        });
        let caster_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow caster allocation"),
            layout: Some(&caster_layout),
            module: &caster_shader,
            entry_point: Some("assign_shadow_casters"),
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
            face_capacity: (shadow_matrix_buf.size() / std::mem::size_of::<GpuShadowMatrix>() as u64) as u32,
            caster_generation: 1,
            last_lights_generation: None,
            last_view_proj: [0.0; 16],
            frame_generation: 0,
            caster_pipeline,
            caster_bind_group_layout,
            caster_bind_group,
            caster_params_buf,
            caster_key: None,
            caster_rebuild: None,
        }
    }

    /// Atlas faces this pass computes matrices for (its matrix buffer's size).
    pub fn face_capacity(&self) -> u32 {
        self.face_capacity
    }

    /// Resolution of one atlas face.
    pub fn atlas_size(&self) -> u32 {
        self.shadow_atlas_size
    }

    /// Casters the atlas holds: six faces each, capped at what the lighting
    /// shaders address.
    pub fn caster_capacity(&self) -> u32 {
        (self.face_capacity / 6).min(MAX_SHADOW_CASTERS)
    }

    fn bind_casters(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        lights: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow caster allocation BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
            ],
        })
    }

    /// Record the caster allocation: one workgroup over every light row.
    fn record_caster_allocation(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Shadow caster allocation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.caster_pipeline);
        pass.set_bind_group(0, &self.caster_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    /// The matrices this pass computes (one per atlas face).
    pub fn matrices(&self) -> &wgpu::Buffer {
        &self.shadow_matrix_buf
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

    fn writes(&self) -> &'static [&'static str] {
        &["shadow_matrices"]
    }

    /// Publish this frame's matrices for the shadow, lighting, fog and lens
    /// passes. The Renderer published this before the SceneDB migration;
    /// without it every consumer skipped shadows entirely.
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let matrices: &'a wgpu::Buffer = unsafe { std::mem::transmute(&self.shadow_matrix_buf) };
        frame.write(
            helio_core::resource_keys::shadow_matrices(),
            ShadowMatricesFrameData {
                shadow_matrices: matrices,
                shadow_count: self.face_capacity,
                per_caster_dirty_gen: [self.caster_generation; 42],
                movable_objects_generation: self.frame_generation,
            },
            self.name(),
        );
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
            self.caster_bind_group = Self::bind_casters(
                ctx.device,
                &self.caster_bind_group_layout,
                &self.bound_lights,
                &self.caster_params_buf,
            );
            self.caster_key = None;
        }
        // Reallocate slots only when the light rows change (Helio#246): no
        // per-frame CPU scoring, and nothing at all while lights are idle.
        let caster_key = lights.map(|lights| {
            (lights.epoch, lights.content_generation, lights.row_capacity(), self.caster_capacity())
        });
        self.caster_rebuild = caster_key.filter(|key| self.caster_key != Some(*key));
        if let Some((_, _, row_count, caster_capacity)) = self.caster_rebuild {
            let params = CasterParams { row_count, caster_capacity, _pad: [0; 2] };
            ctx.queue.write_buffer(&self.caster_params_buf, 0, bytemuck::bytes_of(&params));
        }
        let u = ShadowMatrixUniforms {
            light_count: lights.map_or(0, |lights| lights.row_capacity()),
            shadow_atlas_size: self.shadow_atlas_size,
            _pad: [0; 2],
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        let lights_generation = lights.map(|lights| lights.content_generation);
        if lights_generation != self.last_lights_generation
            || ctx.camera_data.view_proj != self.last_view_proj
        {
            self.caster_generation += 1;
            self.last_lights_generation = lights_generation;
            self.last_view_proj = ctx.camera_data.view_proj;
        }
        self.frame_generation += 1;
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
        // Slots first: the matrices below, and every later pass, read the
        // `shadow_index` this writes into the same rows.
        if let Some(key) = self.caster_rebuild.take() {
            self.record_caster_allocation(unsafe { &mut *ctx.encoder_ptr });
            self.caster_key = Some(key);
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
