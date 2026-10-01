//! GPU shadow matrix computation.
//!
//! Computes light-space view-projection matrices for all shadow-casting lights.
//! O(1) CPU — single compute dispatch regardless of light count.

use bytemuck::{Pod, Zeroable};
use helio_core::{CommandRecorder, PassContext, PrepareContext, RenderPass, Result as HelioResult};

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
    /// What `per_caster_generation` was last bumped for: SceneDB's content
    /// generation reports light edits, and directional cascades follow the
    /// camera.
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
    /// `[caster count, light_type per slot]`, written by the allocation.
    caster_table: wgpu::Buffer,
    /// CPU-mappable copy of `caster_table`.
    caster_table_staging: wgpu::Buffer,
    caster_readback: CasterReadback,
    /// An allocation whose table still has to be copied out. The copy waits
    /// while the staging buffer is being mapped; the table persists on the
    /// GPU until the next allocation, so a later frame's copy is as good.
    caster_copy_wanted: Option<(CasterKey, u32)>,
    /// Bumped per allocation and echoed by the kernel into `caster_table`.
    caster_nonce: u32,
    /// View used for the most recent caster ranking. Kept separate from the
    /// matrix pass view so small camera jitter does not reshuffle slots.
    last_caster_view_proj: [f32; 16],
    pending_caster_view_proj: [f32; 16],
    /// The allocation's layout, with the `caster_key` it was read for.
    caster_layout: Option<(CasterKey, CasterLayout)>,
    /// Per-slot dirty generations (what `ShadowPass` compares). Light edits
    /// dirty every slot; camera movement only the camera-fitted ones.
    per_caster_generation: [u64; 42],
}

type CasterKey = (u64, u64, u32, u32);

/// Bytes of `caster_table`: the count plus one light type per slot.
const CASTER_TABLE_BYTES: u64 = 4 * (2 + MAX_SHADOW_CASTERS as u64);
const CAMERA_REBALANCE_EPSILON: f32 = 0.025;

/// Reading `caster_table` back after an allocation. Mapping has to wait for
/// the copy's submission, so the copy and the map request are a frame apart.
enum CasterReadback {
    Idle,
    /// The copy for this key is recorded; map it next frame.
    Copied((CasterKey, u32)),
    /// Map requested; the callback stores whether it succeeded.
    Mapping((CasterKey, u32), std::sync::Arc<std::sync::Mutex<Option<bool>>>),
}

/// Uniforms of `shadow_casters.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CasterParams {
    row_count: u32,
    caster_capacity: u32,
    /// Echoed into `caster_table`, so a readback proves it holds this
    /// allocation rather than an older one.
    nonce: u32,
    _pad: u32,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let caster_table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow caster table"),
            size: CASTER_TABLE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let caster_table_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow caster table readback"),
            size: CASTER_TABLE_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let caster_bind_group = Self::bind_casters(
            device,
            &caster_bind_group_layout,
            lights_buf,
            &caster_params_buf,
            &caster_table,
            camera_buf,
        );
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
            last_lights_generation: None,
            last_view_proj: [0.0; 16],
            frame_generation: 0,
            caster_pipeline,
            caster_bind_group_layout,
            caster_bind_group,
            caster_params_buf,
            caster_key: None,
            caster_rebuild: None,
            caster_table,
            caster_table_staging,
            caster_readback: CasterReadback::Idle,
            caster_copy_wanted: None,
            caster_nonce: 0,
            last_caster_view_proj: [0.0; 16],
            pending_caster_view_proj: [0.0; 16],
            caster_layout: None,
            per_caster_generation: [1; 42],
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
        caster_table: &wgpu::Buffer,
        camera: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow caster allocation BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: caster_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: camera.as_entire_binding() },
            ],
        })
    }

    /// The caster layout, only while it describes the current allocation.
    fn current_layout(&self) -> Option<CasterLayout> {
        match (&self.caster_layout, self.caster_key, &self.caster_rebuild) {
            (Some((key, layout)), Some(current), None) if *key == current => Some(*layout),
            _ => None,
        }
    }

    /// Advance the caster-table readback: request the map a frame after the
    /// copy was submitted, and take the result once the host's device poll
    /// has delivered it. Never blocks.
    fn poll_caster_readback(&mut self) {
        match &self.caster_readback {
            CasterReadback::Idle => {}
            CasterReadback::Copied(key) => {
                let key = *key;
                let done = std::sync::Arc::new(std::sync::Mutex::new(None));
                let callback_done = std::sync::Arc::clone(&done);
                self.caster_table_staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        if let Ok(mut done) = callback_done.lock() {
                            *done = Some(result.is_ok());
                        }
                    });
                self.caster_readback = CasterReadback::Mapping(key, done);
            }
            CasterReadback::Mapping(allocation, done) => {
                let (key, nonce) = *allocation;
                let Some(mapped) = done.lock().ok().and_then(|done| *done) else {
                    return;
                };
                let layout = if mapped {
                    let layout = self
                        .caster_table_staging
                        .slice(..)
                        .get_mapped_range()
                        .ok()
                        .and_then(|bytes| {
                            let words: &[u32] = bytemuck::cast_slice(&bytes);
                            // A copy lost with a failed frame leaves an older
                            // table (or zeros) behind; the nonce tells them apart.
                            (words[1 + MAX_SHADOW_CASTERS as usize] == nonce).then(|| {
                                let mut light_types = [0u32; 42];
                                light_types.copy_from_slice(&words[1..43]);
                                CasterLayout {
                                    caster_count: words[0].min(MAX_SHADOW_CASTERS),
                                    light_types,
                                }
                            })
                        });
                    self.caster_table_staging.unmap();
                    layout
                } else {
                    None
                };
                match layout {
                    Some(layout)
                        if self.caster_key == Some(key) && nonce == self.caster_nonce =>
                    {
                        self.caster_layout = Some((key, layout))
                    }
                    // Unreadable or stale: copy again while it is current.
                    None if self.caster_key == Some(key) && nonce == self.caster_nonce => {
                        self.caster_copy_wanted = Some((key, nonce));
                    }
                    None => {}
                }
                self.caster_readback = CasterReadback::Idle;
            }
        }
    }

    /// Record the caster allocation: one workgroup over every light row.
    fn record_caster_allocation(&self, encoder: &mut CommandRecorder<'_>) {
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
                per_caster_dirty_gen: self.per_caster_generation,
                movable_objects_generation: self.frame_generation,
                caster_layout: self.current_layout(),
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
        self.poll_caster_readback();
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
                &self.caster_table,
                &self.camera_buf,
            );
            self.caster_key = None;
        }
        // Light edits and meaningful camera changes trigger GPU allocation.
        // Scoring still runs over light rows only on the GPU; the CPU compares
        // camera matrices already available to the pass and never walks lights.
        let caster_key = lights.map(|lights| {
            (lights.epoch, lights.content_generation, lights.row_capacity(), self.caster_capacity())
        });
        let camera_rebalance = ctx.camera_data.view_proj.iter()
            .zip(self.last_caster_view_proj)
            .any(|(now, old)| (now - old).abs() > CAMERA_REBALANCE_EPSILON);
        if camera_rebalance {
            // The previous readback describes the old view's winners. Do not
            // publish it while a new allocation is pending.
            self.caster_layout = None;
        }
        self.caster_rebuild = caster_key.filter(|key| {
            self.caster_key != Some(*key) || camera_rebalance
        });
        if self.caster_rebuild.is_some() {
            self.pending_caster_view_proj = ctx.camera_data.view_proj;
        }
        if let Some((_, _, row_count, caster_capacity)) = self.caster_rebuild {
            self.caster_nonce = self.caster_nonce.wrapping_add(1);
            let params = CasterParams { row_count, caster_capacity, nonce: self.caster_nonce, _pad: 0 };
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
        let lights_changed = lights_generation != self.last_lights_generation;
        let camera_moved = ctx.camera_data.view_proj != self.last_view_proj;
        if lights_changed || camera_moved {
            // A light edit can change any caster. Camera movement only moves
            // the cascades fitted to the view, so once the layout is known,
            // point and spot casters keep their cached faces.
            let layout = self.current_layout();
            for (slot, generation) in self.per_caster_generation.iter_mut().enumerate() {
                if lights_changed || layout.map_or(true, |layout| layout.follows_camera(slot)) {
                    *generation += 1;
                }
            }
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
            self.record_caster_allocation(&mut ctx.graphics_cmds());
            self.caster_key = Some(key);
            self.last_caster_view_proj = self.pending_caster_view_proj;
            self.caster_copy_wanted = Some((key, self.caster_nonce));
        }
        // Copy the table out for the CPU once the staging buffer is free
        // (a buffer with a map pending cannot be written by a submission).
        if matches!(self.caster_readback, CasterReadback::Idle) {
            if let Some(key) = self.caster_copy_wanted.take() {
                ctx.graphics_cmds().copy_buffer_to_buffer(
                    &self.caster_table,
                    0,
                    &self.caster_table_staging,
                    0,
                    CASTER_TABLE_BYTES,
                );
                self.caster_readback = CasterReadback::Copied(key);
            }
        }
        let wg = count.div_ceil(WORKGROUP_SIZE);
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ShadowMatrix"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }
}
