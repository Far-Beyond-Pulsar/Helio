//! The recorded form of a command stream (Helio#311).
//!
//! When the graph's recording cache is on, the [`crate::cmd`] handles append
//! [`Cmd`]s here instead of encoding into wgpu. A unit's recorded streams are
//! compared with the recordings cached for it: equal streams mean equal
//! commands against the same objects, so the cached, already-encoded
//! `wgpu::ReusableCommandBuffer` is submitted again; otherwise the streams are
//! encoded once with [`encode`] and cached.
//!
//! Every command owns clones of the wgpu objects it names (an `Arc` bump), so
//! a recording keeps them alive and can be compared and encoded after the pass
//! that recorded it dropped its own references. wgpu objects compare by
//! identity. Labels are carried for debuggers but never compared.

use std::num::NonZeroU32;
use std::ops::Range;

/// One recorded stream. Encoder-level, render-pass and compute-pass commands
/// share it; `BeginRenderPass`/`EndRenderPass` and
/// `BeginComputePass`/`EndComputePass` bracket the pass commands.
pub(crate) type Stream = Vec<Cmd>;

/// A debug label. Never part of a recording's identity.
#[derive(Clone, Debug, Default)]
pub(crate) struct Label(pub(crate) Option<Box<str>>);

impl Label {
    pub(crate) fn new(label: Option<&str>) -> Self {
        Self(label.map(Into::into))
    }

    fn get(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

impl PartialEq for Label {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PassTimestamps {
    pub(crate) set: wgpu::QuerySet,
    pub(crate) begin: Option<u32>,
    pub(crate) end: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ColorAttachment {
    pub(crate) view: wgpu::TextureView,
    pub(crate) depth_slice: Option<u32>,
    pub(crate) resolve_target: Option<wgpu::TextureView>,
    pub(crate) ops: wgpu::Operations<wgpu::Color>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DepthStencilAttachment {
    pub(crate) view: wgpu::TextureView,
    pub(crate) depth_ops: Option<wgpu::Operations<f32>>,
    pub(crate) stencil_ops: Option<wgpu::Operations<u32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderPassDesc {
    pub(crate) label: Label,
    pub(crate) color: Vec<Option<ColorAttachment>>,
    pub(crate) depth_stencil: Option<DepthStencilAttachment>,
    pub(crate) timestamp_writes: Option<PassTimestamps>,
    pub(crate) occlusion_query_set: Option<wgpu::QuerySet>,
    pub(crate) multiview_mask: Option<NonZeroU32>,
}

impl RenderPassDesc {
    pub(crate) fn capture(desc: &wgpu::RenderPassDescriptor<'_>) -> Self {
        Self {
            label: Label::new(desc.label),
            color: desc
                .color_attachments
                .iter()
                .map(|attachment| {
                    attachment.as_ref().map(|a| ColorAttachment {
                        view: a.view.clone(),
                        depth_slice: a.depth_slice,
                        resolve_target: a.resolve_target.cloned(),
                        ops: a.ops,
                    })
                })
                .collect(),
            depth_stencil: desc.depth_stencil_attachment.as_ref().map(|d| DepthStencilAttachment {
                view: d.view.clone(),
                depth_ops: d.depth_ops,
                stencil_ops: d.stencil_ops,
            }),
            timestamp_writes: desc.timestamp_writes.as_ref().map(|t| PassTimestamps {
                set: t.query_set.clone(),
                begin: t.beginning_of_pass_write_index,
                end: t.end_of_pass_write_index,
            }),
            occlusion_query_set: desc.occlusion_query_set.cloned(),
            multiview_mask: desc.multiview_mask,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ComputePassDesc {
    pub(crate) label: Label,
    pub(crate) timestamp_writes: Option<PassTimestamps>,
}

impl ComputePassDesc {
    pub(crate) fn capture(desc: &wgpu::ComputePassDescriptor<'_>) -> Self {
        Self {
            label: Label::new(desc.label),
            timestamp_writes: desc.timestamp_writes.as_ref().map(|t| PassTimestamps {
                set: t.query_set.clone(),
                begin: t.beginning_of_pass_write_index,
                end: t.end_of_pass_write_index,
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextureCopy {
    pub(crate) texture: wgpu::Texture,
    pub(crate) mip_level: u32,
    pub(crate) origin: wgpu::Origin3d,
    pub(crate) aspect: wgpu::TextureAspect,
}

impl TextureCopy {
    pub(crate) fn capture(info: &wgpu::TexelCopyTextureInfo<'_>) -> Self {
        Self {
            texture: info.texture.clone(),
            mip_level: info.mip_level,
            origin: info.origin,
            aspect: info.aspect,
        }
    }

    fn info(&self) -> wgpu::TexelCopyTextureInfo<'_> {
        wgpu::TexelCopyTextureInfo {
            texture: &self.texture,
            mip_level: self.mip_level,
            origin: self.origin,
            aspect: self.aspect,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BufferCopy {
    pub(crate) buffer: wgpu::Buffer,
    /// `TexelCopyBufferLayout` is not `PartialEq`: offset, bytes per row,
    /// rows per image.
    pub(crate) layout: (u64, Option<u32>, Option<u32>),
}

impl BufferCopy {
    pub(crate) fn capture(info: &wgpu::TexelCopyBufferInfo<'_>) -> Self {
        Self {
            buffer: info.buffer.clone(),
            layout: (
                info.layout.offset,
                info.layout.bytes_per_row,
                info.layout.rows_per_image,
            ),
        }
    }

    fn info(&self) -> wgpu::TexelCopyBufferInfo<'_> {
        wgpu::TexelCopyBufferInfo {
            buffer: &self.buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: self.layout.0,
                bytes_per_row: self.layout.1,
                rows_per_image: self.layout.2,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Cmd {
    // Encoder level.
    BeginRenderPass(Box<RenderPassDesc>),
    EndRenderPass,
    BeginComputePass(ComputePassDesc),
    EndComputePass,
    CopyBufferToBuffer {
        src: wgpu::Buffer,
        src_offset: u64,
        dst: wgpu::Buffer,
        dst_offset: u64,
        size: Option<u64>,
    },
    CopyBufferToTexture {
        src: BufferCopy,
        dst: TextureCopy,
        size: wgpu::Extent3d,
    },
    CopyTextureToBuffer {
        src: TextureCopy,
        dst: BufferCopy,
        size: wgpu::Extent3d,
    },
    CopyTextureToTexture {
        src: TextureCopy,
        dst: TextureCopy,
        size: wgpu::Extent3d,
    },
    ClearBuffer {
        buffer: wgpu::Buffer,
        offset: u64,
        size: Option<u64>,
    },
    ClearTexture {
        texture: wgpu::Texture,
        range: wgpu::ImageSubresourceRange,
    },
    ResolveQuerySet {
        set: wgpu::QuerySet,
        range: Range<u32>,
        dst: wgpu::Buffer,
        offset: u64,
    },

    // Encoder, render-pass or compute-pass level, depending on position.
    WriteTimestamp {
        set: wgpu::QuerySet,
        index: u32,
    },
    PushDebugGroup(Label),
    PopDebugGroup,
    InsertDebugMarker(Label),

    // Render or compute pass.
    SetBindGroup {
        index: u32,
        group: Option<wgpu::BindGroup>,
        offsets: Vec<u32>,
    },
    SetImmediates {
        offset: u32,
        data: Vec<u8>,
    },

    // Render pass.
    SetRenderPipeline(wgpu::RenderPipeline),
    SetVertexBuffer {
        slot: u32,
        /// Buffer, offset, size; `None` unbinds the slot.
        buffer: Option<(wgpu::Buffer, u64, u64)>,
    },
    SetIndexBuffer {
        buffer: wgpu::Buffer,
        offset: u64,
        size: u64,
        format: wgpu::IndexFormat,
    },
    SetViewport([f32; 6]),
    SetScissorRect([u32; 4]),
    SetStencilReference(u32),
    SetBlendConstant(wgpu::Color),
    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
    DrawIndexed {
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    },
    DrawIndirect {
        buffer: wgpu::Buffer,
        offset: u64,
    },
    DrawIndexedIndirect {
        buffer: wgpu::Buffer,
        offset: u64,
    },
    MultiDrawIndirect {
        buffer: wgpu::Buffer,
        offset: u64,
        count: u32,
    },
    MultiDrawIndexedIndirect {
        buffer: wgpu::Buffer,
        offset: u64,
        count: u32,
    },
    MultiDrawIndirectCount {
        buffer: wgpu::Buffer,
        offset: u64,
        count_buffer: wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    },
    MultiDrawIndexedIndirectCount {
        buffer: wgpu::Buffer,
        offset: u64,
        count_buffer: wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    },
    ExecuteBundles(Vec<wgpu::RenderBundle>),

    // Compute pass.
    SetComputePipeline(wgpu::ComputePipeline),
    Dispatch([u32; 3]),
    DispatchIndirect {
        buffer: wgpu::Buffer,
        offset: u64,
    },
}

impl Cmd {
    /// A short name for diagnostics.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Cmd::BeginRenderPass(_) => "BeginRenderPass",
            Cmd::EndRenderPass => "EndRenderPass",
            Cmd::BeginComputePass(_) => "BeginComputePass",
            Cmd::EndComputePass => "EndComputePass",
            Cmd::CopyBufferToBuffer { .. } => "CopyBufferToBuffer",
            Cmd::CopyBufferToTexture { .. } => "CopyBufferToTexture",
            Cmd::CopyTextureToBuffer { .. } => "CopyTextureToBuffer",
            Cmd::CopyTextureToTexture { .. } => "CopyTextureToTexture",
            Cmd::ClearBuffer { .. } => "ClearBuffer",
            Cmd::ClearTexture { .. } => "ClearTexture",
            Cmd::ResolveQuerySet { .. } => "ResolveQuerySet",
            Cmd::WriteTimestamp { .. } => "WriteTimestamp",
            Cmd::PushDebugGroup(_) => "PushDebugGroup",
            Cmd::PopDebugGroup => "PopDebugGroup",
            Cmd::InsertDebugMarker(_) => "InsertDebugMarker",
            Cmd::SetBindGroup { .. } => "SetBindGroup",
            Cmd::SetImmediates { .. } => "SetImmediates",
            Cmd::SetRenderPipeline(_) => "SetRenderPipeline",
            Cmd::SetVertexBuffer { .. } => "SetVertexBuffer",
            Cmd::SetIndexBuffer { .. } => "SetIndexBuffer",
            Cmd::SetViewport(_) => "SetViewport",
            Cmd::SetScissorRect(_) => "SetScissorRect",
            Cmd::SetStencilReference(_) => "SetStencilReference",
            Cmd::SetBlendConstant(_) => "SetBlendConstant",
            Cmd::Draw { .. } => "Draw",
            Cmd::DrawIndexed { .. } => "DrawIndexed",
            Cmd::DrawIndirect { .. } => "DrawIndirect",
            Cmd::DrawIndexedIndirect { .. } => "DrawIndexedIndirect",
            Cmd::MultiDrawIndirect { .. } => "MultiDrawIndirect",
            Cmd::MultiDrawIndexedIndirect { .. } => "MultiDrawIndexedIndirect",
            Cmd::MultiDrawIndirectCount { .. } => "MultiDrawIndirectCount",
            Cmd::MultiDrawIndexedIndirectCount { .. } => "MultiDrawIndexedIndirectCount",
            Cmd::ExecuteBundles(_) => "ExecuteBundles",
            Cmd::SetComputePipeline(_) => "SetComputePipeline",
            Cmd::Dispatch(_) => "Dispatch",
            Cmd::DispatchIndirect { .. } => "DispatchIndirect",
        }
    }
}

/// Describes where two streams first differ, for cache diagnostics.
pub(crate) fn first_difference(old: &[Cmd], new: &[Cmd]) -> String {
    match old.iter().zip(new).position(|(a, b)| a != b) {
        Some(i) if old[i].kind() == new[i].kind() => {
            format!("command {i} ({}) changed its arguments", new[i].kind())
        }
        Some(i) => format!(
            "command {i} changed from {} to {}",
            old[i].kind(),
            new[i].kind()
        ),
        None => format!("length changed from {} to {} commands", old.len(), new.len()),
    }
}

/// Encodes a recorded stream into a wgpu encoder.
pub(crate) fn encode(cmds: &[Cmd], encoder: &mut wgpu::CommandEncoder) {
    let mut i = 0;
    while i < cmds.len() {
        match &cmds[i] {
            Cmd::BeginRenderPass(desc) => {
                let color: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = desc
                    .color
                    .iter()
                    .map(|attachment| {
                        attachment.as_ref().map(|a| wgpu::RenderPassColorAttachment {
                            view: &a.view,
                            depth_slice: a.depth_slice,
                            resolve_target: a.resolve_target.as_ref(),
                            ops: a.ops,
                        })
                    })
                    .collect();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: desc.label.get(),
                    color_attachments: &color,
                    depth_stencil_attachment: desc.depth_stencil.as_ref().map(|d| {
                        wgpu::RenderPassDepthStencilAttachment {
                            view: &d.view,
                            depth_ops: d.depth_ops,
                            stencil_ops: d.stencil_ops,
                        }
                    }),
                    timestamp_writes: desc.timestamp_writes.as_ref().map(|t| {
                        wgpu::RenderPassTimestampWrites {
                            query_set: &t.set,
                            beginning_of_pass_write_index: t.begin,
                            end_of_pass_write_index: t.end,
                        }
                    }),
                    occlusion_query_set: desc.occlusion_query_set.as_ref(),
                    multiview_mask: desc.multiview_mask,
                });
                i += 1;
                while i < cmds.len() && !matches!(cmds[i], Cmd::EndRenderPass) {
                    encode_render(&cmds[i], &mut pass);
                    i += 1;
                }
            }
            Cmd::BeginComputePass(desc) => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: desc.label.get(),
                    timestamp_writes: desc.timestamp_writes.as_ref().map(|t| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: &t.set,
                            beginning_of_pass_write_index: t.begin,
                            end_of_pass_write_index: t.end,
                        }
                    }),
                });
                i += 1;
                while i < cmds.len() && !matches!(cmds[i], Cmd::EndComputePass) {
                    encode_compute(&cmds[i], &mut pass);
                    i += 1;
                }
            }
            // A stray end (its pass already closed) is ignored, as dropping a
            // pass twice would be.
            Cmd::EndRenderPass | Cmd::EndComputePass => {}
            cmd => encode_encoder(cmd, encoder),
        }
        i += 1;
    }
}

fn encode_encoder(cmd: &Cmd, encoder: &mut wgpu::CommandEncoder) {
    match cmd {
        Cmd::CopyBufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        } => encoder.copy_buffer_to_buffer(src, *src_offset, dst, *dst_offset, *size),
        Cmd::CopyBufferToTexture { src, dst, size } => {
            encoder.copy_buffer_to_texture(src.info(), dst.info(), *size)
        }
        Cmd::CopyTextureToBuffer { src, dst, size } => {
            encoder.copy_texture_to_buffer(src.info(), dst.info(), *size)
        }
        Cmd::CopyTextureToTexture { src, dst, size } => {
            encoder.copy_texture_to_texture(src.info(), dst.info(), *size)
        }
        Cmd::ClearBuffer {
            buffer,
            offset,
            size,
        } => encoder.clear_buffer(buffer, *offset, *size),
        Cmd::ClearTexture { texture, range } => encoder.clear_texture(texture, range),
        Cmd::ResolveQuerySet {
            set,
            range,
            dst,
            offset,
        } => encoder.resolve_query_set(set, range.clone(), dst, *offset),
        Cmd::WriteTimestamp { set, index } => encoder.write_timestamp(set, *index),
        Cmd::PushDebugGroup(label) => encoder.push_debug_group(label.get().unwrap_or_default()),
        Cmd::PopDebugGroup => encoder.pop_debug_group(),
        Cmd::InsertDebugMarker(label) => {
            encoder.insert_debug_marker(label.get().unwrap_or_default())
        }
        other => unreachable!(
            "Helio command stream: {} recorded outside a pass",
            other.kind()
        ),
    }
}

fn encode_render(cmd: &Cmd, pass: &mut wgpu::RenderPass<'_>) {
    match cmd {
        Cmd::SetRenderPipeline(pipeline) => pass.set_pipeline(pipeline),
        Cmd::SetBindGroup {
            index,
            group,
            offsets,
        } => pass.set_bind_group(*index, group.as_ref(), offsets),
        Cmd::SetImmediates { offset, data } => pass.set_immediates(*offset, data),
        Cmd::SetVertexBuffer { slot, buffer } => match buffer {
            Some((buffer, offset, size)) => {
                pass.set_vertex_buffer(*slot, buffer.slice(*offset..*offset + *size))
            }
            None => pass.set_vertex_buffer(*slot, None::<wgpu::BufferSlice<'_>>),
        },
        Cmd::SetIndexBuffer {
            buffer,
            offset,
            size,
            format,
        } => pass.set_index_buffer(buffer.slice(*offset..*offset + *size), *format),
        Cmd::SetViewport([x, y, w, h, min, max]) => pass.set_viewport(*x, *y, *w, *h, *min, *max),
        Cmd::SetScissorRect([x, y, w, h]) => pass.set_scissor_rect(*x, *y, *w, *h),
        Cmd::SetStencilReference(reference) => pass.set_stencil_reference(*reference),
        Cmd::SetBlendConstant(color) => pass.set_blend_constant(*color),
        Cmd::Draw {
            vertices,
            instances,
        } => pass.draw(vertices.clone(), instances.clone()),
        Cmd::DrawIndexed {
            indices,
            base_vertex,
            instances,
        } => pass.draw_indexed(indices.clone(), *base_vertex, instances.clone()),
        Cmd::DrawIndirect { buffer, offset } => pass.draw_indirect(buffer, *offset),
        Cmd::DrawIndexedIndirect { buffer, offset } => pass.draw_indexed_indirect(buffer, *offset),
        Cmd::MultiDrawIndirect {
            buffer,
            offset,
            count,
        } => pass.multi_draw_indirect(buffer, *offset, *count),
        Cmd::MultiDrawIndexedIndirect {
            buffer,
            offset,
            count,
        } => pass.multi_draw_indexed_indirect(buffer, *offset, *count),
        Cmd::MultiDrawIndirectCount {
            buffer,
            offset,
            count_buffer,
            count_offset,
            max_count,
        } => pass.multi_draw_indirect_count(buffer, *offset, count_buffer, *count_offset, *max_count),
        Cmd::MultiDrawIndexedIndirectCount {
            buffer,
            offset,
            count_buffer,
            count_offset,
            max_count,
        } => pass.multi_draw_indexed_indirect_count(
            buffer,
            *offset,
            count_buffer,
            *count_offset,
            *max_count,
        ),
        Cmd::ExecuteBundles(bundles) => pass.execute_bundles(bundles.iter()),
        Cmd::WriteTimestamp { set, index } => pass.write_timestamp(set, *index),
        Cmd::PushDebugGroup(label) => pass.push_debug_group(label.get().unwrap_or_default()),
        Cmd::PopDebugGroup => pass.pop_debug_group(),
        Cmd::InsertDebugMarker(label) => pass.insert_debug_marker(label.get().unwrap_or_default()),
        other => unreachable!(
            "Helio command stream: {} recorded inside a render pass",
            other.kind()
        ),
    }
}

fn encode_compute(cmd: &Cmd, pass: &mut wgpu::ComputePass<'_>) {
    match cmd {
        Cmd::SetComputePipeline(pipeline) => pass.set_pipeline(pipeline),
        Cmd::SetBindGroup {
            index,
            group,
            offsets,
        } => pass.set_bind_group(*index, group.as_ref(), offsets),
        Cmd::SetImmediates { offset, data } => pass.set_immediates(*offset, data),
        Cmd::Dispatch([x, y, z]) => pass.dispatch_workgroups(*x, *y, *z),
        Cmd::DispatchIndirect { buffer, offset } => {
            pass.dispatch_workgroups_indirect(buffer, *offset)
        }
        Cmd::WriteTimestamp { set, index } => pass.write_timestamp(set, *index),
        Cmd::PushDebugGroup(label) => pass.push_debug_group(label.get().unwrap_or_default()),
        Cmd::PopDebugGroup => pass.pop_debug_group(),
        Cmd::InsertDebugMarker(label) => pass.insert_debug_marker(label.get().unwrap_or_default()),
        other => unreachable!(
            "Helio command stream: {} recorded inside a compute pass",
            other.kind()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{first_difference, Cmd, Label};

    #[test]
    fn labels_never_affect_identity() {
        assert_eq!(
            Cmd::PushDebugGroup(Label::new(Some("a"))),
            Cmd::PushDebugGroup(Label::new(Some("b")))
        );
    }

    #[test]
    fn differences_name_the_first_changed_command() {
        let old = [Cmd::Dispatch([1, 1, 1]), Cmd::EndComputePass];
        let changed = [Cmd::Dispatch([2, 1, 1]), Cmd::EndComputePass];
        assert_eq!(
            first_difference(&old, &changed),
            "command 0 (Dispatch) changed its arguments"
        );
        assert_eq!(
            first_difference(&old, &old[..1]),
            "length changed from 2 to 1 commands"
        );
    }
}
