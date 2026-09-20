//! Shared, embedded CC0 architectural material for interactive and capture paths.
use helio_pass_gbuffer::MaterialComponent;
use pulsar_scenedb::{gpu::TextureStore, World};
use std::sync::{Arc, RwLock};

#[derive(Clone, Copy)]
pub struct StoneMaterial;

pub fn configure_sampler(renderer: &mut helio::Renderer) {
    renderer.set_material_sampler(&wgpu::SamplerDescriptor {
        label: Some("Architectural repeat trilinear sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    });
}

pub fn load(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    world: &mut World,
) -> Option<Arc<RwLock<TextureStore>>> {
    if std::env::var_os("HLFS_NO_STONE_TEXTURES").is_some()
        || std::env::var_os("HLFS_TEXTURE_CHECKER").is_some()
    {
        return None;
    }
    let materials: Vec<_> = world
        .query::<(&StoneMaterial,)>()
        .map(|(entity, _)| entity)
        .collect();
    if materials.is_empty() {
        return None;
    }
    let mut store = TextureStore::new(3);
    let base = upload(
        device,
        queue,
        &mut store,
        include_bytes!("assets/castle_wall_varriation/castle_wall_varriation_diff_1k.png"),
        true,
    );
    let normal = upload(
        device,
        queue,
        &mut store,
        include_bytes!("assets/castle_wall_varriation/castle_wall_varriation_nor_gl_1k.png"),
        false,
    );
    let arm = upload(
        device,
        queue,
        &mut store,
        include_bytes!("assets/castle_wall_varriation/castle_wall_varriation_arm_1k.png"),
        false,
    );
    for entity in materials {
        let mut material = world
            .get_mut::<MaterialComponent>(entity)
            .expect("stone material");
        // The photo provides the stone color without the old flat-color tint.
        material.base_color = [1.0; 4];
        material.roughness_metallic[0] = 1.0;
        material.tex_base_color = base;
        material.tex_roughness = arm;
        material.tex_occlusion = arm;
        if std::env::var_os("HLFS_NO_STONE_NORMALS").is_none() {
            material.tex_normal = normal;
            material.flags |= helio_mats::FLAG_HAS_NORMAL_MAP;
        }
    }
    Some(Arc::new(RwLock::new(store)))
}

fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    store: &mut TextureStore,
    bytes: &[u8],
    srgb: bool,
) -> u32 {
    let image = image::load_from_memory(bytes)
        .expect("embedded stone PNG")
        .to_rgba8();
    let (mut width, mut height) = image.dimensions();
    // These source maps are square powers of two. Explicitly reject an asset
    // replacement incompatible with this exact 2x2 box mip filter.
    assert!(width.is_power_of_two() && height == width);
    let count = width.ilog2() + 1;
    let slot = store
        .register(
            device,
            queue,
            &wgpu::TextureDescriptor {
                label: Some("CC0 architectural stone"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if srgb {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            image.as_raw(),
        )
        .expect("register stone texture");
    let texture = store.texture(slot).expect("registered texture");
    let mut pixels: Vec<[f32; 4]> = image
        .pixels()
        .map(|pixel| {
            std::array::from_fn(|c| {
                let x = pixel[c] as f32 / 255.0;
                if srgb && c < 3 {
                    srgb_to_linear(x)
                } else {
                    x
                }
            })
        })
        .collect();
    for level in 1..count {
        let w = width / 2;
        let h = height / 2;
        let mut next = vec![[0.0; 4]; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let source = pixels[((y * 2 + dy) * width + x * 2 + dx) as usize];
                        for c in 0..4 {
                            next[(y * w + x) as usize][c] += source[c] * 0.25;
                        }
                    }
                }
            }
        }
        let texels: Vec<u8> = next
            .iter()
            .flat_map(|pixel| {
                (0..4).map(move |c| {
                    let x = if srgb && c < 3 {
                        linear_to_srgb(pixel[c])
                    } else {
                        pixel[c]
                    };
                    (x.clamp(0.0, 1.0) * 255.0).round() as u8
                })
            })
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        pixels = next;
        width = w;
        height = h;
    }
    slot
}
