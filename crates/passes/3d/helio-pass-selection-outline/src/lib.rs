//! The editor's selection outline.
//!
//! Two passes around the selection mask phase
//! ([`helio_frame_api::Phase::SelectionMask`]):
//!
//! - [`SelectionMaskClearPass`] opens the phase: it clears the mask
//!   ([`helio_frame_api::selection::MASK_DEPTH`]). Every kind that can be
//!   selected then draws its selected geometry into it, depth only and
//!   unjittered (the meshes plugin's selection drawer for meshes).
//! - [`SelectionOutlinePass`] draws, over the final image, a line a constant
//!   number of pixels wide around everything in the mask: full strength where
//!   the selected surface is in front of the scene, dimmed where the scene
//!   hides it.
//!
//! How others do it: Unreal draws selected primitives into custom depth and
//! its editor composite finds the mask's edges, comparing against scene depth
//! to dim occluded parts; Unity's scene view draws selected renderers into a
//! mask and blurs its edge. This is Unreal's approach with the mask filled
//! through a draw phase, so the outline knows no kind.
//!
//! What it costs: the clear and the outline are one screen-sized pass each,
//! the outline 16 taps of the mask per pixel; what the drawers draw scales
//! with the selected geometry. Both passes record nothing outside the editor
//! (`RenderPass::set_editor_mode`).

pub mod feature;
pub use feature::SelectionOutlineFeature;

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::selection::MASK_DEPTH;

/// Clears the selection mask; records no draws. Opens
/// [`helio_frame_api::Phase::SelectionMask`].
#[derive(Default)]
pub struct SelectionMaskClearPass {
    editor: bool,
}

impl RenderPass for SelectionMaskClearPass {
    fn name(&self) -> &'static str {
        "SelectionMaskClear"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_depth(MASK_DEPTH, ResourceSize::MatchSurface);
    }

    fn set_editor_mode(&mut self, enabled: bool) {
        self.editor = enabled;
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
        _storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        if !self.editor {
            return None;
        }
        let mask =
            resources.read_texture_view(ResourceKey::new(MASK_DEPTH), "SelectionMaskClear")?;
        Some(wgpu::RenderPassDescriptor {
            label: Some("Selection Mask Clear"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: mask,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(helio_core::depth::CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}

/// Draws the outline of the selection mask over the final image.
pub struct SelectionOutlinePass {
    editor: bool,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(wgpu::TextureView, wgpu::TextureView)>,
}

impl SelectionOutlinePass {
    /// `target_format`: the format of `ctx.target`, the final image.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = helio_core::shader::module(
            device,
            "Selection Outline Shader",
            helio_core::include_wgsl!("../shaders/selection_outline.wgsl"),
        );
        let depth_texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Selection Outline BGL"),
            // 0: the selection mask, 1: the scene depth.
            entries: &[depth_texture(0), depth_texture(1)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Selection Outline PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = helio_core::shader::create_render_pipeline(
            device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Selection Outline Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        Self {
            editor: false,
            pipeline,
            bind_group_layout,
            bind_group: None,
            bind_group_key: None,
        }
    }
}

impl RenderPass for SelectionOutlinePass {
    fn name(&self) -> &'static str {
        "SelectionOutline"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(MASK_DEPTH);
    }

    fn set_editor_mode(&mut self, enabled: bool) {
        self.editor = enabled;
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        if !self.editor {
            return None;
        }
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Selection Outline"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.editor {
            return Ok(());
        }
        let Some(mask) = ctx
            .registry
            .read_texture_view(ResourceKey::new(MASK_DEPTH), "SelectionOutline")
        else {
            return Ok(());
        };
        let key = (mask.clone(), ctx.depth.clone());
        if self.bind_group_key.as_ref() != Some(&key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Selection Outline BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&key.0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&key.1),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }
        let Some(mut rp) = ctx.render_cmds() else {
            return Ok(());
        };
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        rp.draw(0..3, 0..1);
        Ok(())
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _size: helio_core::RenderSize) {
        self.bind_group = None;
        self.bind_group_key = None;
    }
}

#[cfg(test)]
mod tests {
    /// The outline's program is valid WGSL (validated without a device).
    #[test]
    fn the_outline_shader_validates() {
        let source = include_str!("../shaders/selection_outline.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
    }
}
