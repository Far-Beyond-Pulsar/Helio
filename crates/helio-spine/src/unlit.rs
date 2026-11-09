//! The lighting model's stand-in: the G-buffer drawn unlit (albedo plus
//! emission) into the scene colour, when no lighting model is installed.
//!
//! The lighting model is a plugin's: the lights register deferred lighting and
//! light-field shading (`helio_frame_api::lighting::LightingModel`). A preset
//! whose chosen model is not registered gets the spine's `unlit` model, which
//! draws this pass, so an
//! engine without the lights still shows its surfaces: the right failure,
//! as Bevy's `StandardMaterial::unlit` and Unreal's unlit view mode show
//! surfaces without any light. Covered pixels only; the background stays.

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::gbuffer;

/// Writes `albedo + emissive` of every covered G-buffer pixel over `pre_aa`.
pub struct UnlitPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: Option<(Vec<wgpu::TextureView>, wgpu::BindGroup)>,
}

impl UnlitPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = helio_core::shader::module(
            device,
            "Unlit",
            helio_core::include_wgsl!("../shaders/unlit.wgsl"),
        );
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let colour = wgpu::TextureSampleType::Float { filterable: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Unlit"),
            entries: &[
                texture(0, colour),
                texture(1, colour),
                texture(2, wgpu::TextureSampleType::Depth),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Unlit"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = helio_core::shader::create_render_pipeline(
            device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Unlit"),
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
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        Self {
            pipeline,
            layout,
            bind_group: None,
        }
    }
}

impl RenderPass for UnlitPass {
    fn name(&self) -> &'static str {
        "Unlit"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            gbuffer::GROUP,
            gbuffer::DEPTH,
            helio_frame_api::color::SCENE_COLOR,
        ]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(gbuffer::GROUP);
        builder.read(helio_frame_api::color::SCENE_COLOR);
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let target = resources.read(
            ResourceKey::new(helio_frame_api::color::SCENE_COLOR),
            "Unlit",
        )?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // Over the background the graph cleared it to.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Unlit"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let group = ctx
            .registry
            .read::<helio_core::ViewGroup<'_, 4>>(ResourceKey::new(gbuffer::GROUP), "Unlit")
            .ok_or_else(|| {
                helio_core::Error::InvalidPassConfig("Unlit needs the G-buffer".to_string())
            })?;
        let [albedo, _, _, emissive] = group.views;
        let views = vec![albedo.clone(), emissive.clone(), ctx.depth.clone()];
        if self.bind_group.as_ref().map(|(keys, _)| keys) != Some(&views) {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Unlit"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(emissive),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                ],
            });
            self.bind_group = Some((views, bind_group));
        }
        let Some((_, bind_group)) = &self.bind_group else {
            return Ok(());
        };
        let mut pass = ctx.render_cmds().unwrap();
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _size: helio_core::RenderSize) {
        self.bind_group = None;
    }
}
