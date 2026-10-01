//! Core-owned command recording interface (Helio#311).
//!
//! Passes record GPU work through [`CommandRecorder`], [`RenderCmds`] and
//! [`ComputeCmds`] instead of touching `wgpu::CommandEncoder`,
//! `wgpu::RenderPass` or `wgpu::ComputePass` directly. Method names and
//! signatures deliberately match wgpu's, so migrating a pass is a type swap,
//! but because every command now flows through a core-owned type, the core can
//! change what a command *does* (record it into a reusable native command
//! buffer, count it, validate it) without any pass noticing.
//!
//! wgpu keeps owning resources, pipelines, bind groups and shaders: commands
//! take `&wgpu::Buffer`, `&wgpu::BindGroup` and friends as before.
//!
//! # Backends
//!
//! Today every handle forwards to wgpu. The handle types wrap an enum so a
//! second variant (a native Vulkan or D3D12 recorder) can be added without
//! changing a single pass.
//!
//! # Lifetimes
//!
//! [`PassContext::render_cmds`](crate::PassContext::render_cmds),
//! [`PassContext::graphics_cmds`](crate::PassContext::graphics_cmds) and
//! [`PassContext::compute_cmds`](crate::PassContext::compute_cmds) return
//! handles that do not borrow the context (passes read `ctx.camera`, the
//! registry and so on while recording). A handle is valid until `execute()`
//! returns, and only one handle per stream may be used at a time, exactly the
//! contract the raw encoder pointers had.

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

macro_rules! with_render_pass {
    ($self:ident, |$p:ident| $body:expr) => {
        match &mut $self.inner {
            // SAFETY: the graph keeps the pass alive and unaliased for the
            // duration of `execute()`; see the module docs.
            RenderInner::Active(ptr, _) => {
                let $p = unsafe { ptr.as_mut() };
                $body
            }
            RenderInner::Owned($p) => $body,
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
        with_render_pass!(self, |p| p.set_pipeline(pipeline))
    }

    pub fn set_bind_group<'b, BG>(
        &mut self,
        index: u32,
        bind_group: BG,
        offsets: &[wgpu::DynamicOffset],
    ) where
        Option<&'b wgpu::BindGroup>: From<BG>,
    {
        with_render_pass!(self, |p| p.set_bind_group(index, bind_group, offsets))
    }

    pub fn set_vertex_buffer<'b, B>(&mut self, slot: u32, buffer_slice: B)
    where
        Option<wgpu::BufferSlice<'b>>: From<B>,
    {
        with_render_pass!(self, |p| p.set_vertex_buffer(slot, buffer_slice))
    }

    pub fn set_index_buffer(
        &mut self,
        buffer_slice: wgpu::BufferSlice<'_>,
        index_format: wgpu::IndexFormat,
    ) {
        with_render_pass!(self, |p| p.set_index_buffer(buffer_slice, index_format))
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        with_render_pass!(self, |p| p.set_viewport(x, y, w, h, min_depth, max_depth))
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        with_render_pass!(self, |p| p.set_scissor_rect(x, y, width, height))
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        with_render_pass!(self, |p| p.set_stencil_reference(reference))
    }

    pub fn set_blend_constant(&mut self, color: wgpu::Color) {
        with_render_pass!(self, |p| p.set_blend_constant(color))
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        with_render_pass!(self, |p| p.set_immediates(offset, data))
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        with_render_pass!(self, |p| p.draw(vertices, instances))
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        with_render_pass!(self, |p| p.draw_indexed(indices, base_vertex, instances))
    }

    pub fn draw_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        with_render_pass!(self, |p| p.draw_indirect(indirect_buffer, indirect_offset))
    }

    pub fn draw_indexed_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        with_render_pass!(self, |p| p
            .draw_indexed_indirect(indirect_buffer, indirect_offset))
    }

    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) {
        with_render_pass!(self, |p| p.multi_draw_indirect(
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
        with_render_pass!(self, |p| p.multi_draw_indexed_indirect(
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
        with_render_pass!(self, |p| p.multi_draw_indirect_count(
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
        with_render_pass!(self, |p| p.multi_draw_indexed_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count
        ))
    }

    /// Replays pre-recorded render bundles. Only the graph's own bundle path
    /// should need this.
    pub fn execute_bundles<'b, I: IntoIterator<Item = &'b wgpu::RenderBundle>>(&mut self, bundles: I) {
        with_render_pass!(self, |p| p.execute_bundles(bundles))
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        with_render_pass!(self, |p| p.write_timestamp(query_set, query_index))
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        with_render_pass!(self, |p| p.insert_debug_marker(label))
    }

    pub fn push_debug_group(&mut self, label: &str) {
        with_render_pass!(self, |p| p.push_debug_group(label))
    }

    pub fn pop_debug_group(&mut self) {
        with_render_pass!(self, |p| p.pop_debug_group())
    }
}

/// Records into a compute pass. Ends when dropped.
pub struct ComputeCmds<'a> {
    inner: ComputeInner<'a>,
}

enum ComputeInner<'a> {
    Owned(wgpu::ComputePass<'a>),
}

macro_rules! with_compute_pass {
    ($self:ident, |$p:ident| $body:expr) => {
        match &mut $self.inner {
            ComputeInner::Owned($p) => $body,
        }
    };
}

impl<'a> ComputeCmds<'a> {
    /// Wraps a compute pass the caller opened on a plain wgpu encoder (tests,
    /// offline tools). Graph passes open theirs through the context or a
    /// [`CommandRecorder`].
    pub fn from_wgpu(pass: wgpu::ComputePass<'a>) -> ComputeCmds<'a> {
        ComputeCmds {
            inner: ComputeInner::Owned(pass),
        }
    }

    pub fn set_pipeline(&mut self, pipeline: &wgpu::ComputePipeline) {
        with_compute_pass!(self, |p| p.set_pipeline(pipeline))
    }

    pub fn set_bind_group<'b, BG>(
        &mut self,
        index: u32,
        bind_group: BG,
        offsets: &[wgpu::DynamicOffset],
    ) where
        Option<&'b wgpu::BindGroup>: From<BG>,
    {
        with_compute_pass!(self, |p| p.set_bind_group(index, bind_group, offsets))
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        with_compute_pass!(self, |p| p.set_immediates(offset, data))
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        with_compute_pass!(self, |p| p.dispatch_workgroups(x, y, z))
    }

    pub fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
    ) {
        with_compute_pass!(self, |p| p
            .dispatch_workgroups_indirect(indirect_buffer, indirect_offset))
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        with_compute_pass!(self, |p| p.write_timestamp(query_set, query_index))
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        with_compute_pass!(self, |p| p.insert_debug_marker(label))
    }

    pub fn push_debug_group(&mut self, label: &str) {
        with_compute_pass!(self, |p| p.push_debug_group(label))
    }

    pub fn pop_debug_group(&mut self) {
        with_compute_pass!(self, |p| p.pop_debug_group())
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
    inner: RecorderInner<'a>,
}

enum RecorderInner<'a> {
    Wgpu(NonNull<wgpu::CommandEncoder>, PhantomData<&'a mut wgpu::CommandEncoder>),
}

macro_rules! with_encoder {
    ($self:ident, |$e:ident| $body:expr) => {
        match &mut $self.inner {
            // SAFETY: the pointer comes from a live `&mut CommandEncoder` (the
            // graph's stream, or the caller of `from_encoder`) that this
            // recorder borrows exclusively.
            RecorderInner::Wgpu(ptr, _) => {
                let $e = unsafe { ptr.as_mut() };
                $body
            }
        }
    };
}

impl<'a> CommandRecorder<'a> {
    /// Records into a plain wgpu encoder: for tests, offline tools and
    /// anything outside a graph frame. Graph passes use the context.
    pub fn from_encoder(encoder: &'a mut wgpu::CommandEncoder) -> CommandRecorder<'a> {
        CommandRecorder {
            inner: RecorderInner::Wgpu(NonNull::from(encoder), PhantomData),
        }
    }

    /// For the graph: a recorder over one of its streams that does not borrow
    /// the context.
    pub(crate) fn from_ptr(ptr: *mut wgpu::CommandEncoder) -> CommandRecorder<'a> {
        CommandRecorder {
            inner: RecorderInner::Wgpu(
                NonNull::new(ptr).expect("graph command stream pointer is null"),
                PhantomData,
            ),
        }
    }

    /// Opens a self-managed render pass (for a pass whose
    /// `render_pass_descriptor` returns `None`). Ends when dropped.
    pub fn begin_render_pass<'b>(
        &'b mut self,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> RenderCmds<'b> {
        with_encoder!(self, |e| RenderCmds::from_wgpu(e.begin_render_pass(desc)))
    }

    /// Opens a compute pass on this stream. Ends when dropped.
    pub fn begin_compute_pass<'b>(
        &'b mut self,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> ComputeCmds<'b> {
        with_encoder!(self, |e| ComputeCmds::from_wgpu(e.begin_compute_pass(desc)))
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &wgpu::Buffer,
        source_offset: u64,
        destination: &wgpu::Buffer,
        destination_offset: u64,
        copy_size: impl Into<Option<u64>>,
    ) {
        with_encoder!(self, |e| e.copy_buffer_to_buffer(
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
        with_encoder!(self, |e| e.copy_buffer_to_texture(source, destination, copy_size))
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        with_encoder!(self, |e| e.copy_texture_to_buffer(source, destination, copy_size))
    }

    pub fn copy_texture_to_texture(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        with_encoder!(self, |e| e.copy_texture_to_texture(source, destination, copy_size))
    }

    pub fn clear_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        with_encoder!(self, |e| e.clear_buffer(buffer, offset, size))
    }

    pub fn clear_texture(
        &mut self,
        texture: &wgpu::Texture,
        subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        with_encoder!(self, |e| e.clear_texture(texture, subresource_range))
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        with_encoder!(self, |e| e.write_timestamp(query_set, query_index))
    }

    pub fn resolve_query_set(
        &mut self,
        query_set: &wgpu::QuerySet,
        query_range: Range<u32>,
        destination: &wgpu::Buffer,
        destination_offset: u64,
    ) {
        with_encoder!(self, |e| e.resolve_query_set(
            query_set,
            query_range,
            destination,
            destination_offset
        ))
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        with_encoder!(self, |e| e.insert_debug_marker(label))
    }

    pub fn push_debug_group(&mut self, label: &str) {
        with_encoder!(self, |e| e.push_debug_group(label))
    }

    pub fn pop_debug_group(&mut self) {
        with_encoder!(self, |e| e.pop_debug_group())
    }
}
