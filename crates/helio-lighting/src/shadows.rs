use helio_core::gpu;
use glam::Mat4;

pub const MAX_CASCADES: usize = 4;
pub const SHADOW_MAP_SIZE: u32 = 2048;

pub struct ShadowCascade {
    pub split_distance: f32,
    pub view_proj_matrix: Mat4,
    pub texture: Option<gpu::Texture>,
    pub texture_view: Option<gpu::TextureView>,
}

impl ShadowCascade {
    pub fn new() -> Self {
        Self {
            split_distance: 0.0,
            view_proj_matrix: Mat4::IDENTITY,
            texture: None,
            texture_view: None,
        }
    }
}

pub struct CascadedShadowMaps {
    pub cascades: [ShadowCascade; MAX_CASCADES],
    pub cascade_count: usize,
}

impl CascadedShadowMaps {
    pub fn new(context: &gpu::Context, cascade_count: usize) -> Self {
        let mut cascades = [(); MAX_CASCADES].map(|_| ShadowCascade::new());
        
        for i in 0..cascade_count {
            let texture = context.create_texture(gpu::TextureDesc {
                name: "shadow_cascade",
                format: gpu::TextureFormat::Depth32Float,
                size: gpu::Extent {
                    width: SHADOW_MAP_SIZE,
                    height: SHADOW_MAP_SIZE,
                    depth: 1,
                },
                array_layer_count: 1,
                mip_level_count: 1,
                dimension: gpu::TextureDimension::D2,
                usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
                sample_count: 1,
                external: None,
            });

            let texture_view = context.create_texture_view(
                texture,
                gpu::TextureViewDesc {
                    name: "shadow_cascade_view",
                    format: gpu::TextureFormat::Depth32Float,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &gpu::TextureSubresources::default(),
                },
            );

            cascades[i].texture = Some(texture);
            cascades[i].texture_view = Some(texture_view);
        }

        Self {
            cascades,
            cascade_count,
        }
    }

    pub fn update_cascade_splits(&mut self, near: f32, far: f32, lambda: f32) {
        for i in 0..self.cascade_count {
            let p = (i + 1) as f32 / self.cascade_count as f32;
            let log = near * (far / near).powf(p);
            let uniform = near + (far - near) * p;
            let d = lambda * log + (1.0 - lambda) * uniform;
            self.cascades[i].split_distance = d;
        }
    }

    pub fn cleanup(&mut self, context: &gpu::Context) {
        for cascade in &mut self.cascades {
            if let Some(view) = cascade.texture_view.take() {
                context.destroy_texture_view(view);
            }
            if let Some(texture) = cascade.texture.take() {
                context.destroy_texture(texture);
            }
        }
    }
}

pub struct ShadowAtlas {
    pub texture: Option<gpu::Texture>,
    pub texture_view: Option<gpu::TextureView>,
    pub size: u32,
    pub tile_size: u32,
}

impl ShadowAtlas {
    pub fn new(context: &gpu::Context, size: u32, tile_size: u32) -> Self {
        let texture = context.create_texture(gpu::TextureDesc {
            name: "shadow_atlas",
            format: gpu::TextureFormat::Depth32Float,
            size: gpu::Extent {
                width: size,
                height: size,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });

        let texture_view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "shadow_atlas_view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );

        Self {
            texture: Some(texture),
            texture_view: Some(texture_view),
            size,
            tile_size,
        }
    }

    pub fn cleanup(&mut self, context: &gpu::Context) {
        if let Some(view) = self.texture_view.take() {
            context.destroy_texture_view(view);
        }
        if let Some(texture) = self.texture.take() {
            context.destroy_texture(texture);
        }
    }
}

pub struct VirtualShadowMaps {
    pub page_table: Option<gpu::Texture>,
    pub page_table_view: Option<gpu::TextureView>,
    pub physical_pages: Option<gpu::Texture>,
    pub physical_pages_view: Option<gpu::TextureView>,
    pub page_size: u32,
    pub cache_size: u32,
}

impl VirtualShadowMaps {
    pub fn new(context: &gpu::Context, page_size: u32, cache_size: u32) -> Self {
        let page_table = context.create_texture(gpu::TextureDesc {
            name: "vsm_page_table",
            format: gpu::TextureFormat::R32Uint,
            size: gpu::Extent {
                width: 4096,
                height: 4096,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });

        let page_table_view = context.create_texture_view(
            page_table,
            gpu::TextureViewDesc {
                name: "vsm_page_table_view",
                format: gpu::TextureFormat::R32Uint,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );

        let physical_pages = context.create_texture(gpu::TextureDesc {
            name: "vsm_physical_pages",
            format: gpu::TextureFormat::Depth32Float,
            size: gpu::Extent {
                width: cache_size,
                height: cache_size,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });

        let physical_pages_view = context.create_texture_view(
            physical_pages,
            gpu::TextureViewDesc {
                name: "vsm_physical_pages_view",
                format: gpu::TextureFormat::Depth32Float,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );

        Self {
            page_table: Some(page_table),
            page_table_view: Some(page_table_view),
            physical_pages: Some(physical_pages),
            physical_pages_view: Some(physical_pages_view),
            page_size,
            cache_size,
        }
    }

    pub fn cleanup(&mut self, context: &gpu::Context) {
        if let Some(view) = self.page_table_view.take() {
            context.destroy_texture_view(view);
        }
        if let Some(texture) = self.page_table.take() {
            context.destroy_texture(texture);
        }
        if let Some(view) = self.physical_pages_view.take() {
            context.destroy_texture_view(view);
        }
        if let Some(texture) = self.physical_pages.take() {
            context.destroy_texture(texture);
        }
    }
}

pub struct ShadowSystem {
    pub cascaded_shadows: Option<CascadedShadowMaps>,
    pub shadow_atlas: Option<ShadowAtlas>,
    pub virtual_shadows: Option<VirtualShadowMaps>,
    pub pcss_enabled: bool,
    pub pcf_samples: u32,
}

impl ShadowSystem {
    pub fn new(context: &gpu::Context) -> Self {
        Self {
            cascaded_shadows: Some(CascadedShadowMaps::new(context, 4)),
            shadow_atlas: Some(ShadowAtlas::new(context, 4096, 512)),
            virtual_shadows: None,
            pcss_enabled: true,
            pcf_samples: 16,
        }
    }

    pub fn enable_virtual_shadows(&mut self, context: &gpu::Context) {
        if self.virtual_shadows.is_none() {
            self.virtual_shadows = Some(VirtualShadowMaps::new(context, 128, 8192));
        }
    }

    pub fn cleanup(&mut self, context: &gpu::Context) {
        if let Some(mut csm) = self.cascaded_shadows.take() {
            csm.cleanup(context);
        }
        if let Some(mut atlas) = self.shadow_atlas.take() {
            atlas.cleanup(context);
        }
        if let Some(mut vsm) = self.virtual_shadows.take() {
            vsm.cleanup(context);
        }
    }
}
