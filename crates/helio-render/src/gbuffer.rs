use helio_core::gpu;

pub struct GBuffer {
    pub albedo: gpu::Texture,
    pub albedo_view: gpu::TextureView,
    pub normal: gpu::Texture,
    pub normal_view: gpu::TextureView,
    pub material: gpu::Texture,
    pub material_view: gpu::TextureView,
    pub depth: gpu::Texture,
    pub depth_view: gpu::TextureView,
    pub emissive: gpu::Texture,
    pub emissive_view: gpu::TextureView,
    pub velocity: gpu::Texture,
    pub velocity_view: gpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl GBuffer {
    pub fn new(context: &gpu::Context, width: u32, height: u32) -> Self {
        let albedo = context.create_texture(gpu::TextureDesc {
            name: "gbuffer_albedo",
            format: gpu::TextureFormat::Rgba8UnormSrgb,
            size: gpu::Extent { width, height, depth: 1 },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let albedo_view = context.create_texture_view(albedo, gpu::TextureViewDesc {
            name: "gbuffer_albedo_view",
            format: gpu::TextureFormat::Rgba8UnormSrgb,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });

        let normal = context.create_texture(gpu::TextureDesc {
            name: "gbuffer_normal",
            format: gpu::TextureFormat::Rgba16Float,
            size: gpu::Extent { width, height, depth: 1 },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let normal_view = context.create_texture_view(normal, gpu::TextureViewDesc {
            name: "gbuffer_normal_view",
            format: gpu::TextureFormat::Rgba16Float,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });

        let material = context.create_texture(gpu::TextureDesc {
            name: "gbuffer_material",
            format: gpu::TextureFormat::Rgba8Unorm,
            size: gpu::Extent { width, height, depth: 1 },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let material_view = context.create_texture_view(material, gpu::TextureViewDesc {
            name: "gbuffer_material_view",
            format: gpu::TextureFormat::Rgba8Unorm,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });

        let depth = context.create_texture(gpu::TextureDesc {
            name: "gbuffer_depth",
            format: gpu::TextureFormat::Depth32Float,
            size: gpu::Extent { width, height, depth: 1 },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let depth_view = context.create_texture_view(depth, gpu::TextureViewDesc {
            name: "gbuffer_depth_view",
            format: gpu::TextureFormat::Depth32Float,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });

        let emissive = context.create_texture(gpu::TextureDesc {
            name: "gbuffer_emissive",
            format: gpu::TextureFormat::Rgba16Float,
            size: gpu::Extent { width, height, depth: 1 },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let emissive_view = context.create_texture_view(emissive, gpu::TextureViewDesc {
            name: "gbuffer_emissive_view",
            format: gpu::TextureFormat::Rgba16Float,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });

        let velocity = context.create_texture(gpu::TextureDesc {
            name: "gbuffer_velocity",
            format: gpu::TextureFormat::Rg16Float,
            size: gpu::Extent { width, height, depth: 1 },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let velocity_view = context.create_texture_view(velocity, gpu::TextureViewDesc {
            name: "gbuffer_velocity_view",
            format: gpu::TextureFormat::Rg16Float,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });

        Self {
            albedo,
            albedo_view,
            normal,
            normal_view,
            material,
            material_view,
            depth,
            depth_view,
            emissive,
            emissive_view,
            velocity,
            velocity_view,
            width,
            height,
        }
    }

    pub fn clear(&self, encoder: &mut gpu::CommandEncoder) {
        let mut pass = encoder.render(
            "gbuffer_clear",
            gpu::RenderTargetSet {
                colors: &[
                    gpu::RenderTarget {
                        view: self.albedo_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: gpu::FinishOp::Store,
                    },
                    gpu::RenderTarget {
                        view: self.normal_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: gpu::FinishOp::Store,
                    },
                    gpu::RenderTarget {
                        view: self.material_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: gpu::FinishOp::Store,
                    },
                    gpu::RenderTarget {
                        view: self.emissive_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: gpu::FinishOp::Store,
                    },
                    gpu::RenderTarget {
                        view: self.velocity_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                        finish_op: gpu::FinishOp::Store,
                    },
                ],
                depth_stencil: None,
            },
        );
        drop(pass);
    }

    pub fn cleanup(&mut self, context: &gpu::Context) {
        context.destroy_texture_view(self.albedo_view);
        context.destroy_texture(self.albedo);
        context.destroy_texture_view(self.normal_view);
        context.destroy_texture(self.normal);
        context.destroy_texture_view(self.material_view);
        context.destroy_texture(self.material);
        context.destroy_texture_view(self.depth_view);
        context.destroy_texture(self.depth);
        context.destroy_texture_view(self.emissive_view);
        context.destroy_texture(self.emissive);
        context.destroy_texture_view(self.velocity_view);
        context.destroy_texture(self.velocity);
    }
}
