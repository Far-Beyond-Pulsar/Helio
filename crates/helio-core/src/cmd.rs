//! Core-owned command recording interface (Helio#330, part 5).
//!
//! Passes record GPU work through [`CommandRecorder`], [`RenderCmds`] and
//! [`ComputeCmds`] instead of touching `wgpu::CommandEncoder`,
//! `wgpu::RenderPass` or `wgpu::ComputePass` directly. Method names and
//! signatures match wgpu's, so moving a pass over is a type swap. Every
//! command is encoded straight into the wgpu encoder or pass behind the handle.
//!
//! wgpu keeps owning resources, pipelines, bind groups and shaders: commands
//! take `&wgpu::Buffer`, `&wgpu::BindGroup` and friends as before.
//!
//! # Handles
//!
//! [`PassContext::render_cmds`](crate::PassContext::render_cmds),
//! [`PassContext::graphics_cmds`](crate::PassContext::graphics_cmds) and
//! [`PassContext::compute_cmds`](crate::PassContext::compute_cmds) return
//! handles that do not borrow the context, so a pass keeps using `ctx`
//! (its registry, its GPU scopes) while it records. A handle is valid until
//! `execute()` returns, and each stream takes one handle at a time.

use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;

/// Records into a render pass: the graph-opened pass of the current pass or
/// fused chain ([`PassContext::render_cmds`](crate::PassContext::render_cmds)),
/// or a self-managed one from [`CommandRecorder::begin_render_pass`].
pub struct RenderCmds<'a> {
    inner: RenderInner<'a>,
}

enum RenderInner<'a> {
    /// The pass the graph opened. It outlives `execute()`.
    Active(NonNull<wgpu::RenderPass<'static>>, PhantomData<&'a mut ()>),
    /// A pass this recorder opened itself; ends when dropped.
    Owned(wgpu::RenderPass<'a>),
}

/// Encodes `$wgpu` into the wgpu pass.
macro_rules! render {
    ($self:ident, |$p:ident| $wgpu:expr) => {
        match &mut $self.inner {
            // SAFETY: the graph keeps the pass alive and unaliased for the
            // duration of `execute()`; see the module docs.
            RenderInner::Active(ptr, _) => {
                let $p = unsafe { ptr.as_mut() };
                $wgpu
            }
            RenderInner::Owned($p) => $wgpu,
        }
    };
}

impl<'a> RenderCmds<'a> {
    pub(crate) fn from_active(ptr: NonNull<wgpu::RenderPass<'static>>) -> RenderCmds<'a> {
        RenderCmds {
            inner: RenderInner::Active(ptr, PhantomData),
        }
    }

    /// Wraps a render pass the caller opened on a plain wgpu encoder (tests,
    /// offline tools). Graph passes get theirs from the context.
    pub fn from_wgpu(pass: wgpu::RenderPass<'a>) -> RenderCmds<'a> {
        RenderCmds {
            inner: RenderInner::Owned(pass),
        }
    }

    pub fn set_pipeline(&mut self, pipeline: &wgpu::RenderPipeline) {
        render!(self, |p| p.set_pipeline(pipeline))
    }

    pub fn set_bind_group<'b, BG>(
        &mut self,
        index: u32,
        bind_group: BG,
        offsets: &[wgpu::DynamicOffset],
    ) where
        Option<&'b wgpu::BindGroup>: From<BG>,
    {
        render!(self, |p| p.set_bind_group(index, bind_group, offsets))
    }

    pub fn set_vertex_buffer<'b, B>(&mut self, slot: u32, buffer_slice: B)
    where
        Option<wgpu::BufferSlice<'b>>: From<B>,
    {
        render!(self, |p| p.set_vertex_buffer(slot, buffer_slice))
    }

    pub fn set_index_buffer(
        &mut self,
        buffer_slice: wgpu::BufferSlice<'_>,
        index_format: wgpu::IndexFormat,
    ) {
        render!(self, |p| p.set_index_buffer(buffer_slice, index_format))
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        render!(self, |p| p.set_viewport(x, y, w, h, min_depth, max_depth))
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        render!(self, |p| p.set_scissor_rect(x, y, width, height))
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        render!(self, |p| p.set_stencil_reference(reference))
    }

    pub fn set_blend_constant(&mut self, color: wgpu::Color) {
        render!(self, |p| p.set_blend_constant(color))
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        render!(self, |p| p.set_immediates(offset, data))
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        render!(self, |p| p.draw(vertices, instances))
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        render!(self, |p| p.draw_indexed(indices, base_vertex, instances))
    }

    pub fn draw_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        render!(self, |p| p.draw_indirect(indirect_buffer, indirect_offset))
    }

    pub fn draw_indexed_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        render!(self, |p| p
            .draw_indexed_indirect(indirect_buffer, indirect_offset))
    }

    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) {
        render!(self, |p| p.multi_draw_indirect(
            indirect_buffer,
            indirect_offset,
            count
        ))
    }

    pub fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) {
        render!(self, |p| p.multi_draw_indexed_indirect(
            indirect_buffer,
            indirect_offset,
            count
        ))
    }

    pub fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) {
        render!(self, |p| p.multi_draw_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count
        ))
    }

    pub fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) {
        render!(self, |p| p.multi_draw_indexed_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count
        ))
    }

    /// Replays pre-recorded render bundles. Only the graph's own bundle path
    /// should need this.
    pub fn execute_bundles<'b, I: IntoIterator<Item = &'b wgpu::RenderBundle>>(
        &mut self,
        bundles: I,
    ) {
        render!(self, |p| p.execute_bundles(bundles))
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        render!(self, |p| p.write_timestamp(query_set, query_index))
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        render!(self, |p| p.insert_debug_marker(label))
    }

    pub fn push_debug_group(&mut self, label: &str) {
        render!(self, |p| p.push_debug_group(label))
    }

    pub fn pop_debug_group(&mut self) {
        render!(self, |p| p.pop_debug_group())
    }
}

/// Records into a compute pass. Ends when dropped.
pub struct ComputeCmds<'a> {
    pass: wgpu::ComputePass<'a>,
}

/// Encodes `$wgpu` into the wgpu pass.
macro_rules! compute {
    ($self:ident, |$p:ident| $wgpu:expr) => {{
        let $p = &mut $self.pass;
        $wgpu
    }};
}

impl<'a> ComputeCmds<'a> {
    /// Wraps a compute pass the caller opened on a plain wgpu encoder (tests,
    /// offline tools). Graph passes open theirs through the context or a
    /// [`CommandRecorder`].
    pub fn from_wgpu(pass: wgpu::ComputePass<'a>) -> ComputeCmds<'a> {
        ComputeCmds { pass }
    }

    pub fn set_pipeline(&mut self, pipeline: &wgpu::ComputePipeline) {
        compute!(self, |p| p.set_pipeline(pipeline))
    }

    pub fn set_bind_group<'b, BG>(
        &mut self,
        index: u32,
        bind_group: BG,
        offsets: &[wgpu::DynamicOffset],
    ) where
        Option<&'b wgpu::BindGroup>: From<BG>,
    {
        compute!(self, |p| p.set_bind_group(index, bind_group, offsets))
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        compute!(self, |p| p.set_immediates(offset, data))
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        compute!(self, |p| p.dispatch_workgroups(x, y, z))
    }

    pub fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
    ) {
        compute!(self, |p| p
            .dispatch_workgroups_indirect(indirect_buffer, indirect_offset))
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        compute!(self, |p| p.write_timestamp(query_set, query_index))
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        compute!(self, |p| p.insert_debug_marker(label))
    }

    pub fn push_debug_group(&mut self, label: &str) {
        compute!(self, |p| p.push_debug_group(label))
    }

    pub fn pop_debug_group(&mut self) {
        compute!(self, |p| p.pop_debug_group())
    }
}

/// A command stream: opens render and compute passes and records transfers.
/// The replacement for `&mut wgpu::CommandEncoder` in pass code.
///
/// Get one from [`PassContext::graphics_cmds`](crate::PassContext::graphics_cmds)
/// (the graphics stream, in graph order) or
/// [`PassContext::compute_cmds`](crate::PassContext::compute_cmds) (the
/// pre-graphics compute stream). Helpers that used to take
/// `&mut wgpu::CommandEncoder` take `&mut CommandRecorder<'_>`.
pub struct CommandRecorder<'a> {
    encoder: NonNull<wgpu::CommandEncoder>,
    _marker: PhantomData<&'a mut wgpu::CommandEncoder>,
}

/// Encodes `$wgpu` into the wgpu encoder.
macro_rules! encoder {
    ($self:ident, |$e:ident| $wgpu:expr) => {{
        // SAFETY: the pointer comes from a live `&mut CommandEncoder` (the
        // graph's stream, or the caller of `from_encoder`) that this recorder
        // borrows exclusively.
        let $e = unsafe { $self.encoder.as_mut() };
        $wgpu
    }};
}

impl<'a> CommandRecorder<'a> {
    /// Records into a plain wgpu encoder: for tests, offline tools and
    /// anything outside a graph frame. Graph passes use the context.
    pub fn from_encoder(encoder: &'a mut wgpu::CommandEncoder) -> CommandRecorder<'a> {
        CommandRecorder {
            encoder: NonNull::from(encoder),
            _marker: PhantomData,
        }
    }

    /// For the graph: a recorder over one of its wgpu streams that does not
    /// borrow the context.
    pub(crate) fn from_ptr(ptr: *mut wgpu::CommandEncoder) -> CommandRecorder<'a> {
        CommandRecorder {
            encoder: NonNull::new(ptr).expect("graph command stream pointer is null"),
            _marker: PhantomData,
        }
    }

    /// Opens a self-managed render pass (for a pass whose
    /// `render_pass_descriptor` returns `None`). Ends when dropped.
    pub fn begin_render_pass<'b>(
        &'b mut self,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> RenderCmds<'b> {
        RenderCmds::from_wgpu(encoder!(self, |e| e.begin_render_pass(desc)))
    }

    /// Opens a compute pass on this stream. Ends when dropped.
    pub fn begin_compute_pass<'b>(
        &'b mut self,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> ComputeCmds<'b> {
        ComputeCmds::from_wgpu(encoder!(self, |e| e.begin_compute_pass(desc)))
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &wgpu::Buffer,
        source_offset: u64,
        destination: &wgpu::Buffer,
        destination_offset: u64,
        copy_size: impl Into<Option<u64>>,
    ) {
        let copy_size = copy_size.into();
        encoder!(self, |e| e.copy_buffer_to_buffer(
            source,
            source_offset,
            destination,
            destination_offset,
            copy_size
        ))
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        source: wgpu::TexelCopyBufferInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        encoder!(self, |e| e.copy_buffer_to_texture(
            source,
            destination,
            copy_size
        ))
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        encoder!(self, |e| e.copy_texture_to_buffer(
            source,
            destination,
            copy_size
        ))
    }

    pub fn copy_texture_to_texture(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        encoder!(self, |e| e.copy_texture_to_texture(
            source,
            destination,
            copy_size
        ))
    }

    pub fn clear_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        encoder!(self, |e| e.clear_buffer(buffer, offset, size))
    }

    pub fn clear_texture(
        &mut self,
        texture: &wgpu::Texture,
        subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        encoder!(self, |e| e.clear_texture(texture, subresource_range))
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        encoder!(self, |e| e.write_timestamp(query_set, query_index))
    }

    pub fn resolve_query_set(
        &mut self,
        query_set: &wgpu::QuerySet,
        query_range: Range<u32>,
        destination: &wgpu::Buffer,
        destination_offset: u64,
    ) {
        encoder!(self, |e| e.resolve_query_set(
            query_set,
            query_range,
            destination,
            destination_offset
        ))
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        encoder!(self, |e| e.insert_debug_marker(label))
    }

    pub fn push_debug_group(&mut self, label: &str) {
        encoder!(self, |e| e.push_debug_group(label))
    }

    pub fn pop_debug_group(&mut self) {
        encoder!(self, |e| e.pop_debug_group())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPUTE: &str = r#"
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@compute @workgroup_size(1)
fn main() { atomicAdd(&counter, 1u); }
"#;

    const RENDER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
"#;

    const SIZE: u32 = 4;
    const ROW: u64 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;

    struct Work {
        counter: wgpu::Buffer,
        group: wgpu::BindGroup,
        compute: wgpu::ComputePipeline,
        render: wgpu::RenderPipeline,
        target: wgpu::Texture,
        view: wgpu::TextureView,
        out: wgpu::Buffer,
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(crate::wgpu_instance_descriptor());
            let adapter = instance.request_adapter(&Default::default()).await.ok()?;
            adapter.request_device(&Default::default()).await.ok()
        })
    }

    fn work(device: &wgpu::Device) -> Work {
        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(COMPUTE.into()),
        });
        let compute = crate::shader::create_compute_pipeline(
            &device,
            wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counter.as_entire_binding(),
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(RENDER.into()),
        });
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let render = crate::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
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
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ROW + ROW * SIZE as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Work {
            counter,
            group,
            compute,
            render,
            target,
            view,
            out,
        }
    }

    /// A clear, a compute pass, a scissored render pass and two copies.
    fn record(cmds: &mut CommandRecorder<'_>, w: &Work) {
        cmds.clear_buffer(&w.counter, 0, None);
        {
            let mut pass = cmds.begin_compute_pass(&Default::default());
            pass.set_pipeline(&w.compute);
            pass.set_bind_group(0, &w.group, &[]);
            pass.dispatch_workgroups(5, 1, 1);
        }
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &w.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cmd test"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&w.render);
            pass.set_scissor_rect(0, 0, SIZE / 2, SIZE);
            pass.draw(0..3, 0..1);
        }
        cmds.copy_buffer_to_buffer(&w.counter, 0, &w.out, 0, 4);
        cmds.copy_texture_to_buffer(
            w.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &w.out,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: ROW,
                    bytes_per_row: Some(ROW as u32),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
    }

    fn run(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: &Work,
        encoder: wgpu::CommandEncoder,
    ) -> Vec<u8> {
        queue.submit([encoder.finish()]);
        let slice = w.out.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = slice.get_mapped_range().unwrap().to_vec();
        w.out.unmap();
        bytes
    }

    #[test]
    fn the_recorder_encodes_passes_and_copies() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let w = work(&device);

        let mut encoder = device.create_command_encoder(&Default::default());
        record(&mut CommandRecorder::from_encoder(&mut encoder), &w);
        let direct = run(&device, &queue, &w, encoder);

        assert_eq!(&direct[..4], &5u32.to_le_bytes(), "five dispatches counted");
        let pixel = |x: u64, y: u64| &direct[(ROW + y * ROW + x * 4) as usize..][..4];
        assert_eq!(
            pixel(0, 0),
            &[0, 255, 0, 255],
            "scissored draw covers the left half"
        );
        assert_eq!(
            pixel(3, 3),
            &[255, 0, 0, 255],
            "the clear shows on the right"
        );
    }
}
