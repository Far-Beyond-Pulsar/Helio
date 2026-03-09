use std::sync::Arc;

/// All renderer-owned, resize-sensitive textures.
/// Shared via `Arc` with passes that need to sample them.
pub struct FrameTextures {
    pub depth:       Arc<wgpu::Texture>,
    pub depth_view:  wgpu::TextureView,

    // GBuffer targets
    pub gbuf_albedo:  Arc<wgpu::Texture>,
    pub gbuf_normal:  Arc<wgpu::Texture>,
    pub gbuf_orm:     Arc<wgpu::Texture>,
    pub gbuf_emissive: Arc<wgpu::Texture>,

    pub gbuf_albedo_view:  wgpu::TextureView,
    pub gbuf_normal_view:  wgpu::TextureView,
    pub gbuf_orm_view:     wgpu::TextureView,
    pub gbuf_emissive_view: wgpu::TextureView,

    // Pre-AA / post-deferred color
    pub pre_aa:      Arc<wgpu::Texture>,
    pub pre_aa_view: wgpu::TextureView,

    // TAA history (ping-pong)
    pub taa_history_a:      Option<Arc<wgpu::Texture>>,
    pub taa_history_b:      Option<Arc<wgpu::Texture>>,
    pub velocity:           Option<Arc<wgpu::Texture>>,

    pub width:  u32,
    pub height: u32,
    pub surface_format: wgpu::TextureFormat,
}

impl FrameTextures {
    pub fn new(
        device:         &wgpu::Device,
        width:          u32,
        height:         u32,
        surface_format: wgpu::TextureFormat,
        taa:            bool,
    ) -> Self {
        let depth        = mk_depth(device, width, height);
        let gbuf_albedo  = mk_color(device, width, height, wgpu::TextureFormat::Rgba8Unorm,   "gbuf_albedo");
        let gbuf_normal  = mk_color(device, width, height, wgpu::TextureFormat::Rgba16Float,  "gbuf_normal");
        let gbuf_orm     = mk_color(device, width, height, wgpu::TextureFormat::Rgba8Unorm,   "gbuf_orm");
        let gbuf_emissive= mk_color(device, width, height, wgpu::TextureFormat::Rgba16Float,  "gbuf_emissive");
        let pre_aa       = mk_color(device, width, height, surface_format,                    "pre_aa");

        let depth_view        = full_view(&depth);
        let gbuf_albedo_view  = full_view(&gbuf_albedo);
        let gbuf_normal_view  = full_view(&gbuf_normal);
        let gbuf_orm_view     = full_view(&gbuf_orm);
        let gbuf_emissive_view= full_view(&gbuf_emissive);
        let pre_aa_view       = full_view(&pre_aa);

        let (taa_history_a, taa_history_b, velocity) = if taa {
            let ha = mk_color(device, width, height, wgpu::TextureFormat::Rgba16Float, "taa_hist_a");
            let hb = mk_color(device, width, height, wgpu::TextureFormat::Rgba16Float, "taa_hist_b");
            let v  = mk_color(device, width, height, wgpu::TextureFormat::Rg16Float,   "velocity");
            (Some(Arc::new(ha)), Some(Arc::new(hb)), Some(Arc::new(v)))
        } else { (None, None, None) };

        FrameTextures {
            depth_view, gbuf_albedo_view, gbuf_normal_view, gbuf_orm_view, gbuf_emissive_view, pre_aa_view,
            depth:        Arc::new(depth),
            gbuf_albedo:  Arc::new(gbuf_albedo),
            gbuf_normal:  Arc::new(gbuf_normal),
            gbuf_orm:     Arc::new(gbuf_orm),
            gbuf_emissive: Arc::new(gbuf_emissive),
            pre_aa:       Arc::new(pre_aa),
            taa_history_a, taa_history_b, velocity,
            width, height, surface_format,
        }
    }
}

fn mk_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("depth"),
        size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Depth32Float,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[],
    })
}

fn mk_color(device: &wgpu::Device, w: u32, h: u32, fmt: wgpu::TextureFormat, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label:           Some(label),
        size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          fmt,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[],
    })
}

fn full_view(tex: &wgpu::Texture) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Minimal 1×1 stub textures for disabled features.
/// Using the same layout (same BGL) regardless of which features are enabled
/// is the key trick that lets all pipelines share a single bind group layout.
pub struct StubTextures {
    pub white_srgb:   wgpu::Texture,
    pub white_linear: wgpu::Texture,
    pub flat_normal:  wgpu::Texture,
    pub black_srgb:   wgpu::Texture,
    pub shadow_stub:  wgpu::Texture,   // 1×1×1 depth array
    pub cube_stub:    wgpu::Texture,   // 1×1 cube
}

impl StubTextures {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        use wgpu::util::DeviceExt;

        let white_srgb   = mk_stub_rgba8(device, queue, [255,255,255,255], wgpu::TextureFormat::Rgba8UnormSrgb,  "stub_white_srgb");
        let white_linear = mk_stub_rgba8(device, queue, [255,255,255,255], wgpu::TextureFormat::Rgba8Unorm,      "stub_white_linear");
        let flat_normal  = mk_stub_rgba8(device, queue, [127,127,255,255], wgpu::TextureFormat::Rgba8Unorm,      "stub_flat_normal");
        let black_srgb   = mk_stub_rgba8(device, queue, [0,0,0,255],       wgpu::TextureFormat::Rgba8UnormSrgb,  "stub_black_srgb");

        let shadow_stub = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("stub_shadow"),
            size:            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Depth32Float,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });

        let cube_stub = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("stub_cube"),
            size:            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8UnormSrgb,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });

        StubTextures { white_srgb, white_linear, flat_normal, black_srgb, shadow_stub, cube_stub }
    }
}

fn mk_stub_rgba8(device: &wgpu::Device, queue: &wgpu::Queue, pixel: [u8;4], fmt: wgpu::TextureFormat, label: &str) -> wgpu::Texture {
    use wgpu::util::DeviceExt;
    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label:           Some(label),
            size:            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          fmt,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &pixel,
    )
}

/// Create the standard linear+repeat sampler used everywhere.
pub fn linear_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label:            Some("linear_repeat"),
        address_mode_u:   wgpu::AddressMode::Repeat,
        address_mode_v:   wgpu::AddressMode::Repeat,
        address_mode_w:   wgpu::AddressMode::Repeat,
        mag_filter:       wgpu::FilterMode::Linear,
        min_filter:       wgpu::FilterMode::Linear,
        mipmap_filter:    wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

/// Comparison sampler for shadow map PCF sampling.
pub fn shadow_comparison_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label:            Some("shadow_comparison"),
        address_mode_u:   wgpu::AddressMode::ClampToEdge,
        address_mode_v:   wgpu::AddressMode::ClampToEdge,
        address_mode_w:   wgpu::AddressMode::ClampToEdge,
        mag_filter:       wgpu::FilterMode::Linear,
        min_filter:       wgpu::FilterMode::Linear,
        compare:          Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    })
}
