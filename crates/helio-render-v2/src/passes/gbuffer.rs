//! G-buffer write pass.
//!
//! Renders scene geometry into four Rgba G-buffer textures + the shared depth
//! buffer.  No lighting is evaluated here — that happens in the subsequent
//! DeferredLightingPass.

use std::sync::{Arc, Mutex};
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
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
    cached_generation: u64,
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
            cached_generation: u64::MAX,
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

        // Rebuild RenderBundle when the draw list content changes.
        // The bundle pre-compiles all draw commands; the main encoder records a
        // single execute_bundles command so encoder.finish() cost is O(1).
        if ctx.draw_list_generation != self.cached_generation {
            // Read the sorted order published by DepthPrepassPass.
            let cam = ctx.camera_position;
            let fwd = ctx.camera_forward;
            let shared = self.shared_sorted.lock().unwrap();
            if !shared.is_empty() {
                self.sorted_opaque_indices.clear();
                self.sorted_opaque_indices.extend_from_slice(&shared);
            } else {
                self.sorted_opaque_indices.clear();
                self.sorted_opaque_indices.reserve(draw_calls.len());
                for (idx, dc) in draw_calls.iter().enumerate() {
                    if !dc.transparent_blend {
                        self.sorted_opaque_indices.push(idx);
                    }
                }
                self.sorted_opaque_indices.sort_unstable_by(|&ia, &ib| {
                    let a = &draw_calls[ia];
                    let b = &draw_calls[ib];
                    let za = (glam::Vec3::from(a.bounds_center) - cam).dot(fwd) - a.bounds_radius;
                    let zb = (glam::Vec3::from(b.bounds_center) - cam).dot(fwd) - b.bounds_radius;
                    za.partial_cmp(&zb)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            let ma = Arc::as_ptr(&a.material_bind_group) as usize;
                            let mb = Arc::as_ptr(&b.material_bind_group) as usize;
                            ma.cmp(&mb)
                        })
                });
            }
            drop(shared);

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
                benc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                benc.draw_indexed(0..dc.index_count, 0, 0..1);
            }
            self.bundle_cache = Some(benc.finish(&wgpu::RenderBundleDescriptor { label: None }));
            self.cached_generation = ctx.draw_list_generation;
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
            depth_ops: Some(wgpu::Operations {
                load:  wgpu::LoadOp::Clear(1.0),
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
