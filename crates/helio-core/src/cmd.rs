//! Core-owned command recording interface (Helio#330, part 5).
//!
//! Passes record GPU work through [`CommandRecorder`], [`RenderCmds`] and
//! [`ComputeCmds`] instead of touching `wgpu::CommandEncoder`,
//! `wgpu::RenderPass` or `wgpu::ComputePass` directly. Method names and
//! signatures match wgpu's, so moving a pass over is a type swap.
//!
//! Today every command is encoded straight into wgpu. Owning the type the
//! commands go through is what later lets the graph record a frame once and
//! resubmit it while nothing it depends on changes: the core then decides what
//! a command does, and passes cannot tell.
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

/// Records into a render pass: the one the graph opened for the current pass
/// or fused chain ([`PassContext::render_cmds`](crate::PassContext::render_cmds)),
/// or one opened with [`CommandRecorder::begin_render_pass`].
pub struct RenderCmds<'a> {
    inner: RenderInner<'a>,
}

enum RenderInner<'a> {
    /// The pass the graph opened; it stays open after this handle is dropped.
    Active(NonNull<wgpu::RenderPass<'static>>, PhantomData<&'a mut ()>),
    /// A pass this handle owns; it ends when the handle is dropped.
    Owned(wgpu::RenderPass<'a>),
}

impl<'a> RenderCmds<'a> {
    /// # Safety
    ///
    /// `pass` must stay open and unaliased for `'a`.
    pub(crate) unsafe fn from_active(pass: NonNull<wgpu::RenderPass<'static>>) -> Self {
        Self {
            inner: RenderInner::Active(pass, PhantomData),
        }
    }

    /// Wraps a render pass opened on a plain wgpu encoder (tests, offline
    /// tools). Graph passes get theirs from the context.
    pub fn from_wgpu(pass: wgpu::RenderPass<'a>) -> Self {
        Self {
            inner: RenderInner::Owned(pass),
        }
    }

    fn pass(&mut self) -> &mut wgpu::RenderPass<'a> {
        match &mut self.inner {
            // SAFETY: `from_active`'s contract; the pointer is only narrowed
            // from `'static` to `'a`.
            RenderInner::Active(pass, _) => unsafe {
                &mut *(pass.as_ptr() as *mut wgpu::RenderPass<'a>)
            },
            RenderInner::Owned(pass) => pass,
        }
    }

    pub fn set_pipeline(&mut self, pipeline: &wgpu::RenderPipeline) {
        self.pass().set_pipeline(pipeline);
    }

    pub fn set_bind_group<'b, BG>(&mut self, index: u32, bind_group: BG, offsets: &[u32])
    where
        Option<&'b wgpu::BindGroup>: From<BG>,
    {
        self.pass().set_bind_group(index, bind_group, offsets);
    }

    pub fn set_vertex_buffer<'b, B>(&mut self, slot: u32, buffer_slice: B)
    where
        Option<wgpu::BufferSlice<'b>>: From<B>,
    {
        self.pass().set_vertex_buffer(slot, buffer_slice);
    }

    pub fn set_index_buffer(
        &mut self,
        buffer_slice: wgpu::BufferSlice<'_>,
        index_format: wgpu::IndexFormat,
    ) {
        self.pass().set_index_buffer(buffer_slice, index_format);
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.pass().set_viewport(x, y, w, h, min_depth, max_depth);
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.pass().set_scissor_rect(x, y, width, height);
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.pass().set_stencil_reference(reference);
    }

    pub fn set_blend_constant(&mut self, color: wgpu::Color) {
        self.pass().set_blend_constant(color);
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.pass().set_immediates(offset, data);
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.pass().draw(vertices, instances);
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.pass().draw_indexed(indices, base_vertex, instances);
    }

    pub fn draw_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        self.pass().draw_indirect(indirect_buffer, indirect_offset);
    }

    pub fn draw_indexed_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        self.pass().draw_indexed_indirect(indirect_buffer, indirect_offset);
    }

    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) {
        self.pass()
            .multi_draw_indirect(indirect_buffer, indirect_offset, count);
    }

    pub fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) {
        self.pass()
            .multi_draw_indexed_indirect(indirect_buffer, indirect_offset, count);
    }

    pub fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) {
        self.pass().multi_draw_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count,
        );
    }

    pub fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) {
        self.pass().multi_draw_indexed_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count,
        );
    }

    pub fn execute_bundles<'b, I: IntoIterator<Item = &'b wgpu::RenderBundle>>(
        &mut self,
        bundles: I,
    ) {
        self.pass().execute_bundles(bundles);
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        self.pass().write_timestamp(query_set, query_index);
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.pass().insert_debug_marker(label);
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.pass().push_debug_group(label);
    }

    pub fn pop_debug_group(&mut self) {
        self.pass().pop_debug_group();
    }
}

/// Records into a compute pass opened with
/// [`CommandRecorder::begin_compute_pass`] or the context's helpers.
pub struct ComputeCmds<'a> {
    pass: wgpu::ComputePass<'a>,
}

impl<'a> ComputeCmds<'a> {
    /// Wraps a compute pass opened on a plain wgpu encoder (tests, offline
    /// tools).
    pub fn from_wgpu(pass: wgpu::ComputePass<'a>) -> Self {
        Self { pass }
    }

    pub fn set_pipeline(&mut self, pipeline: &wgpu::ComputePipeline) {
        self.pass.set_pipeline(pipeline);
    }

    pub fn set_bind_group<'b, BG>(&mut self, index: u32, bind_group: BG, offsets: &[u32])
    where
        Option<&'b wgpu::BindGroup>: From<BG>,
    {
        self.pass.set_bind_group(index, bind_group, offsets);
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        self.pass.set_immediates(offset, data);
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.pass.dispatch_workgroups(x, y, z);
    }

    pub fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
    ) {
        self.pass
            .dispatch_workgroups_indirect(indirect_buffer, indirect_offset);
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        self.pass.write_timestamp(query_set, query_index);
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.pass.insert_debug_marker(label);
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.pass.push_debug_group(label);
    }

    pub fn pop_debug_group(&mut self) {
        self.pass.pop_debug_group();
    }
}

/// Records encoder-level commands (copies, clears, queries) and opens passes
/// on one of the graph's command streams.
pub struct CommandRecorder<'a> {
    encoder: NonNull<wgpu::CommandEncoder>,
    _marker: PhantomData<&'a mut wgpu::CommandEncoder>,
}

impl<'a> CommandRecorder<'a> {
    /// Records into a plain wgpu encoder (tests, offline tools).
    pub fn from_encoder(encoder: &'a mut wgpu::CommandEncoder) -> Self {
        Self {
            encoder: NonNull::from(encoder),
            _marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// `encoder` must be valid, and not otherwise used, for `'a`.
    pub(crate) unsafe fn from_ptr(encoder: *mut wgpu::CommandEncoder) -> Self {
        Self {
            encoder: NonNull::new(encoder).expect("graph command stream"),
            _marker: PhantomData,
        }
    }

    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        // SAFETY: the constructors' contracts.
        unsafe { self.encoder.as_mut() }
    }

    pub fn begin_render_pass(&mut self, desc: &wgpu::RenderPassDescriptor<'_>) -> RenderCmds<'_> {
        RenderCmds::from_wgpu(self.encoder().begin_render_pass(desc))
    }

    pub fn begin_compute_pass(&mut self, desc: &wgpu::ComputePassDescriptor<'_>) -> ComputeCmds<'_> {
        ComputeCmds::from_wgpu(self.encoder().begin_compute_pass(desc))
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &wgpu::Buffer,
        source_offset: u64,
        destination: &wgpu::Buffer,
        destination_offset: u64,
        copy_size: impl Into<Option<u64>>,
    ) {
        self.encoder().copy_buffer_to_buffer(
            source,
            source_offset,
            destination,
            destination_offset,
            copy_size,
        );
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        source: wgpu::TexelCopyBufferInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        self.encoder()
            .copy_buffer_to_texture(source, destination, copy_size);
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        self.encoder()
            .copy_texture_to_buffer(source, destination, copy_size);
    }

    pub fn copy_texture_to_texture(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        self.encoder()
            .copy_texture_to_texture(source, destination, copy_size);
    }

    pub fn clear_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        self.encoder().clear_buffer(buffer, offset, size);
    }

    pub fn clear_texture(
        &mut self,
        texture: &wgpu::Texture,
        subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        self.encoder().clear_texture(texture, subresource_range);
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        self.encoder().write_timestamp(query_set, query_index);
    }

    pub fn resolve_query_set(
        &mut self,
        query_set: &wgpu::QuerySet,
        query_range: Range<u32>,
        destination: &wgpu::Buffer,
        destination_offset: u64,
    ) {
        self.encoder()
            .resolve_query_set(query_set, query_range, destination, destination_offset);
    }

    pub fn build_acceleration_structures<'b>(
        &mut self,
        blas: impl IntoIterator<Item = &'b wgpu::BlasBuildEntry<'b>>,
        tlas: impl IntoIterator<Item = &'b wgpu::Tlas>,
    ) {
        self.encoder().build_acceleration_structures(blas, tlas);
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.encoder().insert_debug_marker(label);
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.encoder().push_debug_group(label);
    }

    pub fn pop_debug_group(&mut self) {
        self.encoder().pop_debug_group();
    }
}
