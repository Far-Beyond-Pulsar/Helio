//! Core-owned command recording interface (Helio#311).
//!
//! Passes record GPU work through [`CommandRecorder`], [`RenderCmds`] and
//! [`ComputeCmds`] instead of touching `wgpu::CommandEncoder`,
//! `wgpu::RenderPass` or `wgpu::ComputePass` directly. Method names and
//! signatures deliberately match wgpu's, so migrating a pass is a type swap,
//! but because every command flows through a core-owned type, the core decides
//! what a command *does*.
//!
//! wgpu keeps owning resources, pipelines, bind groups and shaders: commands
//! take `&wgpu::Buffer`, `&wgpu::BindGroup` and friends as before.
//!
//! # Backends
//!
//! * **wgpu**: the command is encoded straight into a wgpu encoder or pass.
//! * **recorded**: the command is appended to a [`crate::cmd_ir`] stream. The
//!   graph's recording cache compares a unit's streams with the ones it cached
//!   and resubmits the matching already-encoded, reusable command buffer, or
//!   encodes the stream once and caches it. See `docs/command_interface.md`.
//!
//! Passes cannot tell which backend they are recording into.
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

use crate::cmd_ir::{self, Cmd, Stream};

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
    /// Appends to a recorded stream. `owned` passes record their own end when
    /// dropped; the graph ends the passes it opened.
    Recorded {
        stream: NonNull<Stream>,
        owned: bool,
        _marker: PhantomData<&'a mut ()>,
    },
}

/// Encodes `$wgpu` into the wgpu pass, or appends `$cmd` to the recorded
/// stream. Exactly one of the two is evaluated.
macro_rules! render {
    ($self:ident, |$p:ident| $wgpu:expr, $cmd:expr) => {
        match &mut $self.inner {
            // SAFETY: the graph keeps the pass alive and unaliased for the
            // duration of `execute()`; see the module docs.
            RenderInner::Active(ptr, _) => {
                let $p = unsafe { ptr.as_mut() };
                $wgpu
            }
            RenderInner::Owned($p) => $wgpu,
            // SAFETY: the stream belongs to the unit being recorded and
            // outlives `execute()`; one handle records into it at a time.
            RenderInner::Recorded { stream, .. } => unsafe { stream.as_mut() }.push($cmd),
        }
    };
}

impl<'a> RenderCmds<'a> {
    pub(crate) fn from_active(ptr: NonNull<wgpu::RenderPass<'static>>) -> RenderCmds<'a> {
        RenderCmds {
            inner: RenderInner::Active(ptr, PhantomData),
        }
    }

    /// The graph-opened pass of a recorded unit.
    pub(crate) fn recorded_active(stream: NonNull<Stream>) -> RenderCmds<'a> {
        RenderCmds {
            inner: RenderInner::Recorded {
                stream,
                owned: false,
                _marker: PhantomData,
            },
        }
    }

    /// Opens a pass on a recorded stream; its end is recorded on drop.
    pub(crate) fn begin_recorded(
        mut stream: NonNull<Stream>,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> RenderCmds<'a> {
        // SAFETY: see `render!`.
        unsafe { stream.as_mut() }.push(Cmd::BeginRenderPass(Box::new(
            cmd_ir::RenderPassDesc::capture(desc),
        )));
        RenderCmds {
            inner: RenderInner::Recorded {
                stream,
                owned: true,
                _marker: PhantomData,
            },
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
        render!(
            self,
            |p| p.set_pipeline(pipeline),
            Cmd::SetRenderPipeline(pipeline.clone())
        )
    }

    pub fn set_bind_group<'b, BG>(
        &mut self,
        index: u32,
        bind_group: BG,
        offsets: &[wgpu::DynamicOffset],
    ) where
        Option<&'b wgpu::BindGroup>: From<BG>,
    {
        render!(self, |p| p.set_bind_group(index, bind_group, offsets), {
            let group: Option<&wgpu::BindGroup> = bind_group.into();
            Cmd::SetBindGroup {
                index,
                group: group.cloned(),
                offsets: offsets.to_vec(),
            }
        })
    }

    pub fn set_vertex_buffer<'b, B>(&mut self, slot: u32, buffer_slice: B)
    where
        Option<wgpu::BufferSlice<'b>>: From<B>,
    {
        render!(self, |p| p.set_vertex_buffer(slot, buffer_slice), {
            let slice: Option<wgpu::BufferSlice<'_>> = buffer_slice.into();
            Cmd::SetVertexBuffer {
                slot,
                buffer: slice.map(|s| (s.buffer().clone(), s.offset(), s.size())),
            }
        })
    }

    pub fn set_index_buffer(
        &mut self,
        buffer_slice: wgpu::BufferSlice<'_>,
        index_format: wgpu::IndexFormat,
    ) {
        render!(
            self,
            |p| p.set_index_buffer(buffer_slice, index_format),
            Cmd::SetIndexBuffer {
                buffer: buffer_slice.buffer().clone(),
                offset: buffer_slice.offset(),
                size: buffer_slice.size(),
                format: index_format,
            }
        )
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        render!(
            self,
            |p| p.set_viewport(x, y, w, h, min_depth, max_depth),
            Cmd::SetViewport([x, y, w, h, min_depth, max_depth])
        )
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        render!(
            self,
            |p| p.set_scissor_rect(x, y, width, height),
            Cmd::SetScissorRect([x, y, width, height])
        )
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        render!(
            self,
            |p| p.set_stencil_reference(reference),
            Cmd::SetStencilReference(reference)
        )
    }

    pub fn set_blend_constant(&mut self, color: wgpu::Color) {
        render!(
            self,
            |p| p.set_blend_constant(color),
            Cmd::SetBlendConstant(color)
        )
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        render!(
            self,
            |p| p.set_immediates(offset, data),
            Cmd::SetImmediates {
                offset,
                data: data.to_vec(),
            }
        )
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        render!(
            self,
            |p| p.draw(vertices, instances),
            Cmd::Draw {
                vertices,
                instances,
            }
        )
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        render!(
            self,
            |p| p.draw_indexed(indices, base_vertex, instances),
            Cmd::DrawIndexed {
                indices,
                base_vertex,
                instances,
            }
        )
    }

    pub fn draw_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        render!(
            self,
            |p| p.draw_indirect(indirect_buffer, indirect_offset),
            Cmd::DrawIndirect {
                buffer: indirect_buffer.clone(),
                offset: indirect_offset,
            }
        )
    }

    pub fn draw_indexed_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        render!(
            self,
            |p| p.draw_indexed_indirect(indirect_buffer, indirect_offset),
            Cmd::DrawIndexedIndirect {
                buffer: indirect_buffer.clone(),
                offset: indirect_offset,
            }
        )
    }

    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) {
        render!(
            self,
            |p| p.multi_draw_indirect(indirect_buffer, indirect_offset, count),
            Cmd::MultiDrawIndirect {
                buffer: indirect_buffer.clone(),
                offset: indirect_offset,
                count,
            }
        )
    }

    pub fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) {
        render!(
            self,
            |p| p.multi_draw_indexed_indirect(indirect_buffer, indirect_offset, count),
            Cmd::MultiDrawIndexedIndirect {
                buffer: indirect_buffer.clone(),
                offset: indirect_offset,
                count,
            }
        )
    }

    pub fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) {
        render!(
            self,
            |p| p.multi_draw_indirect_count(
                indirect_buffer,
                indirect_offset,
                count_buffer,
                count_offset,
                max_count
            ),
            Cmd::MultiDrawIndirectCount {
                buffer: indirect_buffer.clone(),
                offset: indirect_offset,
                count_buffer: count_buffer.clone(),
                count_offset,
                max_count,
            }
        )
    }

    pub fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) {
        render!(
            self,
            |p| p.multi_draw_indexed_indirect_count(
                indirect_buffer,
                indirect_offset,
                count_buffer,
                count_offset,
                max_count
            ),
            Cmd::MultiDrawIndexedIndirectCount {
                buffer: indirect_buffer.clone(),
                offset: indirect_offset,
                count_buffer: count_buffer.clone(),
                count_offset,
                max_count,
            }
        )
    }

    /// Replays pre-recorded render bundles. Only the graph's own bundle path
    /// should need this.
    pub fn execute_bundles<'b, I: IntoIterator<Item = &'b wgpu::RenderBundle>>(&mut self, bundles: I) {
        render!(
            self,
            |p| p.execute_bundles(bundles),
            Cmd::ExecuteBundles(bundles.into_iter().cloned().collect())
        )
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        render!(
            self,
            |p| p.write_timestamp(query_set, query_index),
            Cmd::WriteTimestamp {
                set: query_set.clone(),
                index: query_index,
            }
        )
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        render!(
            self,
            |p| p.insert_debug_marker(label),
            Cmd::InsertDebugMarker(cmd_ir::Label::new(Some(label)))
        )
    }

    pub fn push_debug_group(&mut self, label: &str) {
        render!(
            self,
            |p| p.push_debug_group(label),
            Cmd::PushDebugGroup(cmd_ir::Label::new(Some(label)))
        )
    }

    pub fn pop_debug_group(&mut self) {
        render!(self, |p| p.pop_debug_group(), Cmd::PopDebugGroup)
    }
}

impl Drop for RenderCmds<'_> {
    fn drop(&mut self) {
        if let RenderInner::Recorded {
            stream,
            owned: true,
            ..
        } = &mut self.inner
        {
            // SAFETY: see `render!`.
            unsafe { stream.as_mut() }.push(Cmd::EndRenderPass);
        }
    }
}

/// Records into a compute pass. Ends when dropped.
pub struct ComputeCmds<'a> {
    inner: ComputeInner<'a>,
}

enum ComputeInner<'a> {
    Owned(wgpu::ComputePass<'a>),
    Recorded {
        stream: NonNull<Stream>,
        _marker: PhantomData<&'a mut ()>,
    },
}

/// Encodes `$wgpu` into the wgpu pass, or appends `$cmd` to the recorded
/// stream. Exactly one of the two is evaluated.
macro_rules! compute {
    ($self:ident, |$p:ident| $wgpu:expr, $cmd:expr) => {
        match &mut $self.inner {
            ComputeInner::Owned($p) => $wgpu,
            // SAFETY: see `render!`.
            ComputeInner::Recorded { stream, .. } => unsafe { stream.as_mut() }.push($cmd),
        }
    };
}

impl<'a> ComputeCmds<'a> {
    /// Opens a pass on a recorded stream; its end is recorded on drop.
    pub(crate) fn begin_recorded(
        mut stream: NonNull<Stream>,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> ComputeCmds<'a> {
        // SAFETY: see `render!`.
        unsafe { stream.as_mut() }.push(Cmd::BeginComputePass(
            cmd_ir::ComputePassDesc::capture(desc),
        ));
        ComputeCmds {
            inner: ComputeInner::Recorded {
                stream,
                _marker: PhantomData,
            },
        }
    }

    /// Wraps a compute pass the caller opened on a plain wgpu encoder (tests,
    /// offline tools). Graph passes open theirs through the context or a
    /// [`CommandRecorder`].
    pub fn from_wgpu(pass: wgpu::ComputePass<'a>) -> ComputeCmds<'a> {
        ComputeCmds {
            inner: ComputeInner::Owned(pass),
        }
    }

    pub fn set_pipeline(&mut self, pipeline: &wgpu::ComputePipeline) {
        compute!(
            self,
            |p| p.set_pipeline(pipeline),
            Cmd::SetComputePipeline(pipeline.clone())
        )
    }

    pub fn set_bind_group<'b, BG>(
        &mut self,
        index: u32,
        bind_group: BG,
        offsets: &[wgpu::DynamicOffset],
    ) where
        Option<&'b wgpu::BindGroup>: From<BG>,
    {
        compute!(self, |p| p.set_bind_group(index, bind_group, offsets), {
            let group: Option<&wgpu::BindGroup> = bind_group.into();
            Cmd::SetBindGroup {
                index,
                group: group.cloned(),
                offsets: offsets.to_vec(),
            }
        })
    }

    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        compute!(
            self,
            |p| p.set_immediates(offset, data),
            Cmd::SetImmediates {
                offset,
                data: data.to_vec(),
            }
        )
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        compute!(
            self,
            |p| p.dispatch_workgroups(x, y, z),
            Cmd::Dispatch([x, y, z])
        )
    }

    pub fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
    ) {
        compute!(
            self,
            |p| p.dispatch_workgroups_indirect(indirect_buffer, indirect_offset),
            Cmd::DispatchIndirect {
                buffer: indirect_buffer.clone(),
                offset: indirect_offset,
            }
        )
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        compute!(
            self,
            |p| p.write_timestamp(query_set, query_index),
            Cmd::WriteTimestamp {
                set: query_set.clone(),
                index: query_index,
            }
        )
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        compute!(
            self,
            |p| p.insert_debug_marker(label),
            Cmd::InsertDebugMarker(cmd_ir::Label::new(Some(label)))
        )
    }

    pub fn push_debug_group(&mut self, label: &str) {
        compute!(
            self,
            |p| p.push_debug_group(label),
            Cmd::PushDebugGroup(cmd_ir::Label::new(Some(label)))
        )
    }

    pub fn pop_debug_group(&mut self) {
        compute!(self, |p| p.pop_debug_group(), Cmd::PopDebugGroup)
    }
}

impl Drop for ComputeCmds<'_> {
    fn drop(&mut self) {
        if let ComputeInner::Recorded { stream, .. } = &mut self.inner {
            // SAFETY: see `render!`.
            unsafe { stream.as_mut() }.push(Cmd::EndComputePass);
        }
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
    Recorded(NonNull<Stream>, PhantomData<&'a mut ()>),
}

/// Encodes `$wgpu` into the wgpu encoder, or appends `$cmd` to the recorded
/// stream. Exactly one of the two is evaluated.
macro_rules! encoder {
    ($self:ident, |$e:ident| $wgpu:expr, $cmd:expr) => {
        match &mut $self.inner {
            // SAFETY: the pointer comes from a live `&mut CommandEncoder` (the
            // graph's stream, or the caller of `from_encoder`) that this
            // recorder borrows exclusively.
            RecorderInner::Wgpu(ptr, _) => {
                let $e = unsafe { ptr.as_mut() };
                $wgpu
            }
            // SAFETY: see `render!`.
            RecorderInner::Recorded(stream, _) => unsafe { stream.as_mut() }.push($cmd),
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

    /// For the graph: a recorder over one of its wgpu streams that does not
    /// borrow the context.
    pub(crate) fn from_ptr(ptr: *mut wgpu::CommandEncoder) -> CommandRecorder<'a> {
        CommandRecorder {
            inner: RecorderInner::Wgpu(
                NonNull::new(ptr).expect("graph command stream pointer is null"),
                PhantomData,
            ),
        }
    }

    /// For the graph: a recorder over one of a unit's recorded streams.
    pub(crate) fn from_stream(stream: NonNull<Stream>) -> CommandRecorder<'a> {
        CommandRecorder {
            inner: RecorderInner::Recorded(stream, PhantomData),
        }
    }

    /// Opens a self-managed render pass (for a pass whose
    /// `render_pass_descriptor` returns `None`). Ends when dropped.
    pub fn begin_render_pass<'b>(
        &'b mut self,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> RenderCmds<'b> {
        match &mut self.inner {
            // SAFETY: see `encoder!`.
            RecorderInner::Wgpu(ptr, _) => {
                RenderCmds::from_wgpu(unsafe { ptr.as_mut() }.begin_render_pass(desc))
            }
            RecorderInner::Recorded(stream, _) => RenderCmds::begin_recorded(*stream, desc),
        }
    }

    /// Opens a compute pass on this stream. Ends when dropped.
    pub fn begin_compute_pass<'b>(
        &'b mut self,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> ComputeCmds<'b> {
        match &mut self.inner {
            // SAFETY: see `encoder!`.
            RecorderInner::Wgpu(ptr, _) => {
                ComputeCmds::from_wgpu(unsafe { ptr.as_mut() }.begin_compute_pass(desc))
            }
            RecorderInner::Recorded(stream, _) => ComputeCmds::begin_recorded(*stream, desc),
        }
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
        encoder!(
            self,
            |e| e.copy_buffer_to_buffer(
                source,
                source_offset,
                destination,
                destination_offset,
                copy_size
            ),
            Cmd::CopyBufferToBuffer {
                src: source.clone(),
                src_offset: source_offset,
                dst: destination.clone(),
                dst_offset: destination_offset,
                size: copy_size,
            }
        )
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        source: wgpu::TexelCopyBufferInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        encoder!(
            self,
            |e| e.copy_buffer_to_texture(source, destination, copy_size),
            Cmd::CopyBufferToTexture {
                src: cmd_ir::BufferCopy::capture(&source),
                dst: cmd_ir::TextureCopy::capture(&destination),
                size: copy_size,
            }
        )
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        encoder!(
            self,
            |e| e.copy_texture_to_buffer(source, destination, copy_size),
            Cmd::CopyTextureToBuffer {
                src: cmd_ir::TextureCopy::capture(&source),
                dst: cmd_ir::BufferCopy::capture(&destination),
                size: copy_size,
            }
        )
    }

    pub fn copy_texture_to_texture(
        &mut self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        encoder!(
            self,
            |e| e.copy_texture_to_texture(source, destination, copy_size),
            Cmd::CopyTextureToTexture {
                src: cmd_ir::TextureCopy::capture(&source),
                dst: cmd_ir::TextureCopy::capture(&destination),
                size: copy_size,
            }
        )
    }

    pub fn clear_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, size: Option<u64>) {
        encoder!(
            self,
            |e| e.clear_buffer(buffer, offset, size),
            Cmd::ClearBuffer {
                buffer: buffer.clone(),
                offset,
                size,
            }
        )
    }

    pub fn clear_texture(
        &mut self,
        texture: &wgpu::Texture,
        subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        encoder!(
            self,
            |e| e.clear_texture(texture, subresource_range),
            Cmd::ClearTexture {
                texture: texture.clone(),
                range: *subresource_range,
            }
        )
    }

    pub fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
        encoder!(
            self,
            |e| e.write_timestamp(query_set, query_index),
            Cmd::WriteTimestamp {
                set: query_set.clone(),
                index: query_index,
            }
        )
    }

    pub fn resolve_query_set(
        &mut self,
        query_set: &wgpu::QuerySet,
        query_range: Range<u32>,
        destination: &wgpu::Buffer,
        destination_offset: u64,
    ) {
        encoder!(
            self,
            |e| e.resolve_query_set(
                query_set,
                query_range,
                destination,
                destination_offset
            ),
            Cmd::ResolveQuerySet {
                set: query_set.clone(),
                range: query_range,
                dst: destination.clone(),
                offset: destination_offset,
            }
        )
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        encoder!(
            self,
            |e| e.insert_debug_marker(label),
            Cmd::InsertDebugMarker(cmd_ir::Label::new(Some(label)))
        )
    }

    pub fn push_debug_group(&mut self, label: &str) {
        encoder!(
            self,
            |e| e.push_debug_group(label),
            Cmd::PushDebugGroup(cmd_ir::Label::new(Some(label)))
        )
    }

    pub fn pop_debug_group(&mut self) {
        encoder!(self, |e| e.pop_debug_group(), Cmd::PopDebugGroup)
    }
}
