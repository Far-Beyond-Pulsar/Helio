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
    device:    Arc<wgpu::Device>,
    targets:   Arc<Mutex<GBufferTargets>>,
    pipeline:  Arc<wgpu::RenderPipeline>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    sorted_opaque_indices: Vec<usize>,
    /// Pre-computed front-to-back order shared by DepthPrepassPass.
    shared_sorted: Arc<Mutex<Vec<usize>>>,
    /// Cached RenderBundle — see DepthPrepassPass for rationale.
    bundle_cache: Option<wgpu::RenderBundle>,
    /// FNV hash of the opaque draw set — see DepthPrepassPass for full rationale.
    bundle_geom_hash: u64,
    frame_counter: u64,
    bundle_last_rebuild: u64,
}

impl GBufferPass {
    pub fn new(
        device:    Arc<wgpu::Device>,
        targets:   Arc<Mutex<GBufferTargets>>,
        pipeline:  Arc<wgpu::RenderPipeline>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        shared_sorted: Arc<Mutex<Vec<usize>>>,
    ) -> Self {
        Self {
            device,
            targets,
            pipeline,
            draw_list,
            sorted_opaque_indices: Vec::new(),
            shared_sorted,
            bundle_cache: None,
            bundle_geom_hash: u64::MAX,
            frame_counter: 0,
            bundle_last_rebuild: 0,
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
        self.frame_counter += 1;
        let draw_calls = self.draw_list.lock().unwrap();

        // Same geom hash as DepthPrepassPass — only structural identity, not transforms.
        let new_geom_hash = {
            let mut h: u64 = 0xcbf29ce484222325;
            for dc in draw_calls.iter() {
                if dc.transparent_blend { continue; }
                h ^= Arc::as_ptr(&dc.vertex_buffer) as u64;
                h = h.wrapping_mul(0x100000001b3);
                h ^= dc.instance_count as u64;
                h = h.wrapping_mul(0x100000001b3);
                if let Some(b) = &dc.instance_buffer {
                    h ^= Arc::as_ptr(b) as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            h
        };

        let set_changed = new_geom_hash != self.bundle_geom_hash;

        // Update sorted index list from DepthPrepass (already sorted this frame)
        // whenever the opaque set changes.  Fallback to self-sort if not available.
        if set_changed {
            let shared = self.shared_sorted.lock().unwrap();
            if !shared.is_empty() {
                self.sorted_opaque_indices.clear();
                self.sorted_opaque_indices.extend_from_slice(&shared);
            } else {
                let cam = ctx.camera_position;
                let fwd = ctx.camera_forward;
                self.sorted_opaque_indices.clear();
                self.sorted_opaque_indices.reserve(draw_calls.len());
                for (idx, dc) in draw_calls.iter().enumerate() {
                    if !dc.transparent_blend { self.sorted_opaque_indices.push(idx); }
                }
                self.sorted_opaque_indices.sort_unstable_by(|&ia, &ib| {
                    let a = &draw_calls[ia];
                    let b = &draw_calls[ib];
                    let za = (glam::Vec3::from(a.bounds_center) - cam).dot(fwd) - a.bounds_radius;
                    let zb = (glam::Vec3::from(b.bounds_center) - cam).dot(fwd) - b.bounds_radius;
                    za.partial_cmp(&zb).unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            Arc::as_ptr(&a.material_bind_group).cmp(
                                &Arc::as_ptr(&b.material_bind_group))
                        })
                });
            }
        }

        let can_rebuild = self.bundle_cache.is_none()
            || self.frame_counter.saturating_sub(self.bundle_last_rebuild) >= 4;

        if set_changed && can_rebuild {
            let _bundle_t = std::time::Instant::now();
            let mut benc = self.device.create_render_bundle_encoder(
                &wgpu::RenderBundleEncoderDescriptor {
                    label: Some("gbuffer_bundle"),
                    color_formats: &[
                        Some(wgpu::TextureFormat::Rgba8Unorm),
                        Some(wgpu::TextureFormat::Rgba16Float),
                        Some(wgpu::TextureFormat::Rgba8Unorm),
                        Some(wgpu::TextureFormat::Rgba16Float),
                    ],
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: 1,
                    multiview: None,
                },
            );
            benc.set_pipeline(&self.pipeline);
            benc.set_bind_group(0, ctx.global_bind_group, &[]);
            let mut last_material: Option<usize> = None;
            for &idx in &self.sorted_opaque_indices {
                let dc = &draw_calls[idx];
                let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
                if last_material != Some(mat_ptr) {
                    benc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                    last_material = Some(mat_ptr);
                }
                benc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                let inst_start = dc.instance_buffer_offset;
                let inst_end   = inst_start + dc.instance_count as u64 * INSTANCE_STRIDE;
                benc.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(inst_start..inst_end));
                benc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                benc.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
            }
            self.bundle_cache        = Some(benc.finish(&wgpu::RenderBundleDescriptor { label: None }));
            self.bundle_geom_hash    = new_geom_hash;
            self.bundle_last_rebuild = self.frame_counter;
            eprintln!(
                "⚠️ [GBuffer] Bundle rebuild: {} draw calls — {:.2}ms",
                self.sorted_opaque_indices.len(),
                _bundle_t.elapsed().as_secs_f32() * 1000.0,
            );
        }
        drop(draw_calls);

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

        if let Some(bundle) = &self.bundle_cache {
            pass.execute_bundles(std::iter::once(bundle));
        }

        Ok(())
    }
}
