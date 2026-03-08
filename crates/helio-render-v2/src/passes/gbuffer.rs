//! G-buffer write pass.
//!
//! Renders scene geometry into four Rgba G-buffer textures + the shared depth
//! buffer.  No lighting is evaluated here — that happens in the subsequent
//! DeferredLightingPass.

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::Result;

/// The four G-buffer render targets.  Wrapped in Arc<Mutex<…>> so the Renderer
/// can swap in new views when the window is resized without rebuilding the graph.
pub struct GBufferTargets {
    pub albedo_view:   wgpu::TextureView,   // Rgba8Unorm
    pub normal_view:   wgpu::TextureView,   // Rgba16Float
    pub orm_view:      wgpu::TextureView,   // Rgba8Unorm
    pub emissive_view: wgpu::TextureView,   // Rgba16Float
}

pub struct GBufferPass {
    targets:   Arc<Mutex<GBufferTargets>>,
    pipeline:  Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    sorted_opaque_indices: Vec<usize>,
    /// Pre-computed material-order sort shared by DepthPrepassPass.
    shared_sorted: Arc<Mutex<Vec<usize>>>,
    /// Order-independent (XOR) hash of the opaque draw set — see DepthPrepassPass.
    sort_geom_hash: u64,
}

impl GBufferPass {
    pub fn new(
        targets:   Arc<Mutex<GBufferTargets>>,
        pipeline:  Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        shared_sorted: Arc<Mutex<Vec<usize>>>,
    ) -> Self {
        Self {
            targets,
            pipeline,
            draw_list,
            sorted_opaque_indices: Vec::new(),
            shared_sorted,
            sort_geom_hash: u64::MAX,
        }
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str { "gbuffer" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.read(ResourceHandle::named("shadow_atlas"));
        builder.read(ResourceHandle::named("rc_cascade0"));
        builder.read(ResourceHandle::named("sky_layer"));
        builder.write(ResourceHandle::named("gbuffer"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let draw_calls = self.draw_list.lock().unwrap();

        // Same commutative XOR hash as DepthPrepassPass for sort invalidation.
        let new_sort_hash = {
            let mut h: u64 = draw_calls.iter().filter(|dc| !dc.transparent_blend).count() as u64;
            for dc in draw_calls.iter() {
                if dc.transparent_blend || dc.instance_buffer.is_none() { continue; }
                let entry = (Arc::as_ptr(&dc.vertex_buffer) as u64)
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    ^ (Arc::as_ptr(dc.instance_buffer.as_ref().unwrap()) as u64)
                        .wrapping_mul(0x517cc1b727220a95)
                    ^ (dc.instance_count as u64).wrapping_mul(0x6c62272e07bb0142);
                h ^= entry.wrapping_mul(0xbf58476d1ce4e5b9);
            }
            h
        };

        // Sync sorted order from DepthPrepass (computed this frame); fall back to
        // self-sort only when the shared list is empty (graph ordering race, rare).
        if new_sort_hash != self.sort_geom_hash {
            let shared = self.shared_sorted.lock().unwrap();
            if !shared.is_empty() {
                self.sorted_opaque_indices.clear();
                self.sorted_opaque_indices.extend_from_slice(&shared);
            } else {
                self.sorted_opaque_indices.clear();
                self.sorted_opaque_indices.reserve(draw_calls.len());
                for (idx, dc) in draw_calls.iter().enumerate() {
                    if !dc.transparent_blend { self.sorted_opaque_indices.push(idx); }
                }
                self.sorted_opaque_indices.sort_unstable_by_key(|&i| {
                    Arc::as_ptr(&draw_calls[i].material_bind_group) as usize
                });
            }
            self.sort_geom_hash = new_sort_hash;
        }

        let targets = self.targets.lock().unwrap();

        let color_attachments = [
            // target 0: albedo
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.albedo_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            // target 1: normals
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.normal_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            // target 2: ORM
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.orm_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            // target 3: emissive
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.emissive_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];

        let depth_attachment = Some(wgpu::RenderPassDepthStencilAttachment {
            view: ctx.depth_view,
            // Load depth written by DepthPrepassPass so the GPU can reject occluded
            // GBuffer fragments before running the expensive material shader.
            // Must NOT clear here — that discards the prepass work entirely.
            depth_ops: Some(wgpu::Operations {
                load:  wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });

        // Need an explicit encoder::begin_render_pass here since we have 4 targets
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("G-Buffer Write Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Direct-encode all opaque draws — no RenderBundle compilation step.
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        let mut last_material: Option<usize> = None;
        for &idx in &self.sorted_opaque_indices {
            let dc = &draw_calls[idx];
            if dc.instance_buffer.is_none() { continue; }
            let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
            if last_material != Some(mat_ptr) {
                pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                last_material = Some(mat_ptr);
            }
            pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
            let inst_start = dc.instance_buffer_offset;
            let inst_end   = inst_start + dc.instance_count as u64 * INSTANCE_STRIDE;
            pass.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(inst_start..inst_end));
            pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
        }
        Ok(())
    }
}
