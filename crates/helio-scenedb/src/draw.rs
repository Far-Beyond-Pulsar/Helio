//! The M3-b T7 minimal indirect-draw executor: the first GPU work that
//! turns the cull pass's output (`crate::cull::CullOutputBuffers`) into
//! actual pixels -- proving the C5 end-to-end path (SceneDB geometry +
//! SceneDB transforms + cull-produced commands -> pixels). This is
//! deliberately NOT the full Helio pass integration (that is M4): one
//! offscreen color target, one render pipeline, one `record`-shaped API
//! that runs ONE render pass issuing indirect draws from the cull pass's
//! command buffer.
//!
//! [`wgsl::DRAW_WGSL`] (in `crate::wgsl`) is the renderer-side source of
//! truth for the vertex/fragment shader; this module's doc comment there
//! covers the `@group(1)`-unbound choice and the read-only `DrawRecord`/
//! `DrawCullOutput` duplication rationale (`VERTEX_WRITABLE_STORAGE` is a
//! non-default feature this task's device does not request). The
//! **indirect-draw mechanism** (CPU-side loop of `draw_indexed_indirect`,
//! not `multi_draw_indexed_indirect`) is documented in full there too --
//! summary: both wgpu-30 methods gate on the same downlevel capability
//! (`DownlevelFlags::INDIRECT_EXECUTION`, universal on desktop backends,
//! not a `Features` flag), so availability was never the blocker;
//! `multi_draw_indexed_indirect` additionally requires its indirect
//! buffer's draw structs to be tightly packed (20-byte `DrawIndexedIndirect
//! Args` stride), which `CullOutputBuffers`' 32-byte `CullRecord` stride
//! does not satisfy without a repack pass this minimal executor chooses not
//! to add.

use crate::cull::CullOutputBuffers;
use crate::wgsl::{DRAW_WGSL, SCENE_BINDINGS_WGSL};

/// Rust twin of [`crate::wgsl::DRAW_WGSL`]'s `DrawUniforms` -- the draw
/// pass's per-dispatch constant input: just the view-proj matrix (no
/// frustum planes, no mesh/capacity bounds -- those are cull-only
/// concerns; this pass only ever transforms already-culled geometry).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawUniforms {
    pub view_proj: [f32; 16],
}
const _: () = assert!(std::mem::size_of::<DrawUniforms>() == 64);
// SAFETY: `#[repr(C)]`, `Copy`, one scalar-array field, no padding.
unsafe impl bytemuck::Zeroable for DrawUniforms {}
unsafe impl bytemuck::Pod for DrawUniforms {}

/// Deterministic `mesh_index -> RGBA8` mapping, the CPU-side twin of
/// [`crate::wgsl::DRAW_WGSL`]'s `mesh_color` WGSL function -- MUST stay
/// numerically identical (same multipliers, same `% 256`, same `/255.0`
/// rounding-to-u8 story) so a test can assert a painted pixel's color
/// against this function and know it proves the GPU-side `InstanceInfo.
/// mesh_index -> color` lookup ran, not just that some pixel got painted.
#[must_use]
pub fn mesh_color_rgba8(mesh_index: u32) -> [u8; 4] {
    let r = (mesh_index.wrapping_mul(37).wrapping_add(17)) % 256;
    let g = (mesh_index.wrapping_mul(91).wrapping_add(53)) % 256;
    let b = (mesh_index.wrapping_mul(131).wrapping_add(7)) % 256;
    [r as u8, g as u8, b as u8, 255]
}

/// The offscreen color target the draw executor renders into. `Rgba8Unorm`
/// at [`OffscreenTarget::WIDTH`]x[`OffscreenTarget::HEIGHT`] -- deliberately
/// chosen so `WIDTH * 4` (the tightly-packed RGBA8 row stride) is ALREADY a
/// multiple of 256 (wgpu's `copy_texture_to_buffer` `bytes_per_row`
/// alignment requirement), so [`Self::read_pixels`] needs no manual padding
/// or de-padding -- one `copy_texture_to_buffer` call, one contiguous
/// `Vec<u8>` out.
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl OffscreenTarget {
    pub const WIDTH: u32 = 64;
    pub const HEIGHT: u32 = 64;
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    /// `WIDTH * 4` bytes (RGBA8) -- exactly 256, the wgpu `bytes_per_row`
    /// alignment ceiling, chosen deliberately (see struct doc).
    pub const BYTES_PER_ROW: u32 = Self::WIDTH * 4;
    const _ALIGNED: () = assert!(Self::BYTES_PER_ROW % 256 == 0);

    #[must_use]
    pub fn new(device: &wgpu::Device, label: &str) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: Self::WIDTH, height: Self::HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Copies the whole target into a `MAP_READ` staging buffer and returns
    /// its bytes, tightly packed (`WIDTH * HEIGHT * 4`, row-major, no
    /// padding -- see struct doc for why `bytes_per_row` needs none here).
    pub fn read_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let byte_size = (Self::BYTES_PER_ROW * Self::HEIGHT) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("helio-scenedb-draw-target-readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(Self::BYTES_PER_ROW),
                    rows_per_image: Some(Self::HEIGHT),
                },
            },
            wgpu::Extent3d { width: Self::WIDTH, height: Self::HEIGHT, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map draw target readback"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range().expect("mapped range").to_vec();
        staging.unmap();
        data
    }
}

/// The draw pass itself: one render pipeline, one `@group(2)` bind-group
/// layout (`draw_cull_output`/`draw_uniforms`, `crate::wgsl::DRAW_WGSL`'s
/// module doc has the full rationale), one uniform buffer. Stateless across
/// dispatches beyond that -- same "rebuilt, never mutated in place" idiom
/// [`crate::cull::CullPass`] and [`crate::SceneDbBinding`] both follow.
pub struct DrawExecutor {
    pipeline: wgpu::RenderPipeline,
    output_layout: wgpu::BindGroupLayout,
    uniforms_buf: wgpu::Buffer,
}

impl DrawExecutor {
    /// `cull_layout` is [`crate::SceneDbBinding::cull_layout`] (`@group(0)`
    /// -- the caller already built a `SceneDbBinding`; this pass borrows
    /// only its cull-read layout, and only two of its five entries
    /// (`instances`/`instance_info`) are actually reachable from `vs_main`/
    /// `fs_main` -- never `draw_layout`, matching the plan's "prefer not
    /// binding it" instruction).
    #[must_use]
    pub fn new(device: &wgpu::Device, cull_layout: &wgpu::BindGroupLayout) -> Self {
        let shader_src = format!("{SCENE_BINDINGS_WGSL}\n{DRAW_WGSL}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helio-scenedb-draw-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let output_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("helio-scenedb-draw-output-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("helio-scenedb-draw-uniforms"),
            size: std::mem::size_of::<DrawUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // group(1) genuinely absent -- see DRAW_WGSL's module doc (mirrors
        // CULL_WGSL's own "never bound" trick for the exact same reason:
        // vs_main/fs_main never reference clusters/meshlets/cell_meta/
        // materials, so wgpu/naga's reachability-based validation permits
        // a pipeline layout with no entry at that index at all).
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("helio-scenedb-draw-pipeline-layout"),
            bind_group_layouts: &[Some(cull_layout), None, Some(&output_layout)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 12, // vec3<f32> local position, GeometryArena's stride for this trivial mesh
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("helio-scenedb-draw-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // No backface culling: this executor's job is to prove the
                // cull-to-draw handoff and the transform/color lookups, not
                // to exercise winding-order correctness -- irrelevant here
                // and would only add a way for an unrelated geometry choice
                // to accidentally fail Test 4.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OffscreenTarget::FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline, output_layout, uniforms_buf }
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: &DrawUniforms) {
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(uniforms));
    }

    /// Builds the `@group(2)` bind group over `output`'s combined
    /// counters+records buffer (read-only here -- see `DRAW_WGSL`'s module
    /// doc for why this is a distinct read-only WGSL view of the SAME
    /// buffer `crate::cull::CullPass` writes) and this pass's own uniform
    /// buffer. Rebuild whenever `output`'s underlying `wgpu::Buffer`
    /// changes, mirroring `CullPass::build_output_bind_group`'s idiom.
    #[must_use]
    pub fn build_output_bind_group(&self, device: &wgpu::Device, output: &CullOutputBuffers) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("helio-scenedb-draw-output-bind-group"),
            layout: &self.output_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.uniforms_buf.as_entire_binding() },
            ],
        })
    }

    /// Records ONE render pass into `target`: clears to transparent black,
    /// binds `vertex_buffer`/`index_buffer` (callers pass
    /// `GeometryArena::vertex_buffer`/`index_buffer` -- the real SceneDB-
    /// owned geometry residency, not a test-local buffer), then issues
    /// `command_count` `draw_indexed_indirect` calls, one per command slot,
    /// each reading its 20-byte `DrawIndexedIndirectArgs` prefix directly
    /// out of `output`'s buffer at `HEADER_BYTES + slot * RECORD_BYTES`
    /// (see this module's doc for why not `multi_draw_indexed_indirect`).
    ///
    /// `command_count` is the CPU-clamped visible-command count (design
    /// S14.2: `min(readback visible_count, output.capacity())`) -- the
    /// caller already has this from reading back `output`'s counters for
    /// its own assertions (the same readback pattern `tests/cull_pass.rs`
    /// established), so this function does not re-read it itself; it only
    /// consumes the number, never derives it, keeping this a pure
    /// `record`-shaped API with no hidden synchronous stalls of its own.
    ///
    /// `scene_cull_bind_group` is [`crate::SceneDbBinding::cull_bind_group`]
    /// (`@group(0)`); `output_bind_group` is
    /// [`Self::build_output_bind_group`]'s result (`@group(2)`). `@group(1)`
    /// is NEVER bound -- see this module's doc.
    ///
    /// A `command_count` of 0 still opens the render pass (so the clear
    /// happens -- callers that want a determinate empty frame get one)
    /// but issues no draw calls.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        scene_cull_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        output: &CullOutputBuffers,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        command_count: u32,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("helio-scenedb-draw-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if command_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene_cull_bind_group, &[]);
        pass.set_bind_group(2, output_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for slot in 0..command_count {
            let offset = CullOutputBuffers::HEADER_BYTES + slot as u64 * CullOutputBuffers::RECORD_BYTES;
            pass.draw_indexed_indirect(output.buffer(), offset);
        }
    }

    /// The strongest no-readback alternative to [`Self::record`]'s CPU-side
    /// per-slot loop (M3-b T9 review, defect 1): ONE
    /// `RenderPass::multi_draw_indexed_indirect` call issuing `count`
    /// GPU-side indirect draws, instead of `count` individual
    /// `draw_indexed_indirect` CPU calls. Gated only by
    /// `DownlevelFlags::INDIRECT_EXECUTION` (universal on desktop Vulkan/
    /// DX12/Metal, verified against `wgpu-30.0.0/src/api/render_pass.rs` --
    /// NOT behind `Features::MULTI_DRAW_INDIRECT_COUNT`, which gates only
    /// the separate `_count` variant), so this needs no extra
    /// `wgpu::Features` beyond what [`Self::record`] already requires.
    ///
    /// `multi_draw_indexed_indirect`'s own doc requires its indirect
    /// buffer's records to be TIGHTLY PACKED 20-byte
    /// `DrawIndexedIndirectArgs` -- [`crate::cull::CullOutputBuffers`]'s
    /// `CullRecord` stride is 32 bytes (this crate's own module docs have
    /// the group(2) storage-budget reason those extra 12 bytes exist), so
    /// this method does NOT read `output`'s indirect args directly (unlike
    /// [`Self::record`]) -- callers pass a SEPARATE, already tightly-packed
    /// `indirect_buffer` (a repack pass's output; see the M3-b T9 pass-
    /// timing bench for a worked compute-shader repack) as the args
    /// source. `output_bind_group` is STILL `output`'s own group(2) bind
    /// group, unchanged from [`Self::record`] -- the vertex shader's row
    /// lookup (`draw_cull_output.records[iid].row`, `wgsl.rs`'s
    /// `DRAW_WGSL` doc) reads the ORIGINAL 32-byte-strided `CullRecord`
    /// buffer, which the tightly-packed indirect-args buffer does not
    /// carry (no `row` field); only the indirect draw ARGS come from the
    /// packed buffer, not the row-lookup data.
    #[allow(clippy::too_many_arguments)]
    pub fn record_multi_indirect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        scene_cull_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: wgpu::BufferAddress,
        count: u32,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("helio-scenedb-draw-pass-multi-indirect"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene_cull_bind_group, &[]);
        pass.set_bind_group(2, output_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.multi_draw_indexed_indirect(indirect_buffer, indirect_offset, count);
    }
}
