//! OpenXR swapchain creation and wgpu texture wrapping.
//!
//! OpenXR gives us raw `VkImage` handles (`u64`). To render into them with
//! wgpu we wrap each image with wgpu-hal's `texture_from_raw` and then hand the
//! resulting hal texture to `wgpu::Device::create_texture_from_hal`. OpenXR
//! owns both the image and its memory, so we pass an empty drop callback and
//! `TextureMemory::External` to make sure wgpu never destroys them.

use crate::graphics::WgpuGraphics;
use crate::{Result, XrError};

/// An OpenXR swapchain whose images are wrapped as wgpu textures.
///
/// All runtime images are enumerated and wrapped up-front (the `VkImage`
/// handles are stable for the swapchain's lifetime); each frame
/// [`XrSwapchain::acquire_image`] returns the index of the image to render to.
pub struct XrSwapchain {
    pub swapchain: openxr::Swapchain<WgpuGraphics>,
    /// One wgpu texture per swapchain image.
    pub textures: Vec<wgpu::Texture>,
    /// One `D2Array` view per swapchain image (the multiview render target).
    pub views: Vec<wgpu::TextureView>,
    /// One single-layer view per (image, array layer); the per-eye fallback
    /// render targets when the swapchain only has one array layer.
    pub layer_views: Vec<wgpu::TextureView>,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    /// Array layers the swapchain was created with (2 = multiview, 1 = fallback).
    pub array_size: u32,
    pub image_count: u32,
}

impl XrSwapchain {
    /// Create an OpenXR swapchain for `session` and wrap every image as a
    /// wgpu texture usable by `device`.
    ///
    /// A 2-layer array swapchain is requested so Helio can render both eyes in
    /// a single multiview pass; runtimes that only support 1 layer will reject
    /// that, in which case a 1-layer swapchain is created and the eyes are
    /// rendered per-eye.
    pub fn create(
        device: &wgpu::Device,
        session: &openxr::Session<WgpuGraphics>,
        width: u32,
        height: u32,
        requested_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let runtime_formats = session.enumerate_swapchain_formats()?;
        let vk_format = negotiate_vk_format(requested_format, &runtime_formats)?;
        let format =
            vk_format_to_wgpu(vk_format).ok_or(XrError::UnsupportedFormat(requested_format))?;

        let mut array_size = 2u32;
        let swapchain = match session.create_swapchain(&swapchain_info(vk_format, width, height, 2))
        {
            Ok(swapchain) => swapchain,
            Err(first) => {
                log::warn!("runtime rejected a 2-layer swapchain ({first}); retrying with 1 layer");
                array_size = 1;
                session
                    .create_swapchain(&swapchain_info(vk_format, width, height, 1))
                    .map_err(|second| {
                        XrError::Swapchain(format!(
                            "2-layer failed ({first}), 1-layer retry failed ({second})"
                        ))
                    })?
            }
        };

        let images = swapchain.enumerate_images()?;
        let image_count = images.len() as u32;

        let mut textures = Vec::with_capacity(images.len());
        let mut views = Vec::with_capacity(images.len());
        let mut layer_views = Vec::with_capacity(images.len() * array_size as usize);
        for (i, &raw_image) in images.iter().enumerate() {
            let texture = wrap_vk_image(device, raw_image, width, height, array_size, format)?;
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("OpenXR Swapchain Array View"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_array_layer: 0,
                array_layer_count: Some(array_size),
                ..Default::default()
            });
            for layer in 0..array_size {
                layer_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("OpenXR Swapchain Layer View"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                }));
            }
            log::debug!("wrapped OpenXR swapchain image {i} as a wgpu texture");
            textures.push(texture);
            views.push(view);
        }

        Ok(Self {
            swapchain,
            textures,
            views,
            layer_views,
            format,
            width: width.max(1),
            height: height.max(1),
            array_size,
            image_count,
        })
    }

    /// Acquire the next image and wait until the compositor has released it.
    ///
    /// Returns the index into [`XrSwapchain::textures`] / [`XrSwapchain::views`].
    pub fn acquire_image(&mut self) -> Result<u32> {
        let index = self.swapchain.acquire_image()?;
        self.swapchain.wait_image(openxr::Duration::INFINITE)?;
        Ok(index)
    }

    /// Present the last acquired image back to the compositor.
    pub fn present(&mut self) -> Result<()> {
        self.swapchain.release_image()?;
        Ok(())
    }

    /// The array (multiview) view for the given image index.
    pub fn view(&self, image_index: u32) -> Result<&wgpu::TextureView> {
        self.views
            .get(image_index as usize)
            .ok_or_else(|| XrError::TextureWrap(format!("image index {image_index} out of range")))
    }

    /// The single-layer view for a given image index and array layer.
    pub fn layer_view(&self, image_index: u32, layer: u32) -> Result<&wgpu::TextureView> {
        let slot = image_index as usize * self.array_size as usize + layer as usize;
        self.layer_views
            .get(slot)
            .ok_or_else(|| XrError::TextureWrap(format!("layer view {slot} out of range")))
    }
}

fn swapchain_info(
    vk_format: u32,
    width: u32,
    height: u32,
    array_size: u32,
) -> openxr::SwapchainCreateInfo<WgpuGraphics> {
    openxr::SwapchainCreateInfo {
        create_flags: openxr::SwapchainCreateFlags::EMPTY,
        usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
            | openxr::SwapchainUsageFlags::SAMPLED,
        format: vk_format,
        sample_count: 1,
        width: width.max(1),
        height: height.max(1),
        face_count: 1,
        array_size,
        mip_count: 1,
    }
}

/// Choose the `VkFormat` for the swapchain.
///
/// Prefers the caller's requested wgpu format if the runtime offers the
/// corresponding `VkFormat`; otherwise falls back to the first runtime format
/// this crate knows how to wrap.
fn negotiate_vk_format(requested: wgpu::TextureFormat, runtime_formats: &[u32]) -> Result<u32> {
    if let Some(vk) = wgpu_format_to_vk(requested) {
        if runtime_formats.contains(&vk) {
            return Ok(vk);
        }
    }
    runtime_formats
        .iter()
        .copied()
        .find_map(vk_format_to_wgpu)
        .and_then(wgpu_format_to_vk)
        .ok_or(XrError::UnsupportedFormat(requested))
}

/// Wrap a raw `VkImage` handle as a `wgpu::Texture`.
///
/// # Safety
///
/// - `device` must be the same Vulkan device that created `raw_image` (i.e.
///   the wgpu device the swapchain was created from).
/// - `raw_image` must be a valid `VkImage` whose dimensions / format match
///   `width` / `height` / `format`.
/// - OpenXR owns the image and its memory; wgpu is given a no-op drop callback
///   and `TextureMemory::External` so it never releases them.
pub fn wrap_vk_image(
    device: &wgpu::Device,
    raw_image: u64,
    width: u32,
    height: u32,
    array_size: u32,
    format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    use ash::vk::Handle as _;
    use wgpu::hal::vulkan::{Api, TextureMemory};

    let hal_device = unsafe { device.as_hal::<Api>() }.ok_or_else(|| {
        XrError::GraphicsUnavailable("wgpu device is not backed by the Vulkan backend".to_string())
    })?;

    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

    let descriptor = wgpu::TextureDescriptor {
        label: Some("OpenXR Swapchain Image"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: array_size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    };

    let hal_descriptor = wgpu::hal::TextureDescriptor {
        label: descriptor.label,
        size: descriptor.size,
        mip_level_count: descriptor.mip_level_count,
        sample_count: descriptor.sample_count,
        dimension: descriptor.dimension,
        format: descriptor.format,
        usage: map_texture_usage_to_hal(descriptor.usage),
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    };

    // OpenXR owns the VkImage and its backing memory; never let wgpu-hal
    // destroy them. The no-op drop callback still lets the hal texture be
    // dropped freely.
    let hal_texture = unsafe {
        hal_device.texture_from_raw(
            ash::vk::Image::from_raw(raw_image),
            &hal_descriptor,
            Some(Box::new(|| {})),
            TextureMemory::External,
        )
    };

    // UNINITIALIZED tells wgpu's tracker the image contents/layout are unknown
    // (OpenXR hands it to us that way on each acquire), so the first barrier it
    // emits may legally discard.
    let texture = unsafe {
        device.create_texture_from_hal::<Api>(
            hal_texture,
            &descriptor,
            wgpu::wgt::TextureUses::UNINITIALIZED,
        )
    };
    Ok(texture)
}

/// Map a user-facing `TextureUsages` bitmask onto the hal `TextureUses`
/// bitmask (the same mapping wgpu-core's `conv::map_texture_usage` performs).
pub fn map_texture_usage_to_hal(usage: wgpu::TextureUsages) -> wgpu::wgt::TextureUses {
    use wgpu::wgt::TextureUses as H;
    use wgpu::TextureUsages as U;

    let mut hal = H::empty();
    if usage.contains(U::COPY_SRC) {
        hal |= H::COPY_SRC;
    }
    if usage.contains(U::COPY_DST) {
        hal |= H::COPY_DST;
    }
    if usage.contains(U::TEXTURE_BINDING) {
        hal |= H::RESOURCE;
    }
    if usage.contains(U::STORAGE_BINDING) {
        hal |= H::STORAGE_READ_WRITE;
    }
    if usage.contains(U::RENDER_ATTACHMENT) {
        hal |= H::COLOR_TARGET;
    }
    if usage.contains(U::STORAGE_ATOMIC) {
        hal |= H::STORAGE_ATOMIC;
    }
    if usage.contains(U::TRANSIENT_ATTACHMENT) {
        hal |= H::TRANSIENT;
    }
    hal
}

/// Convert a wgpu texture format to its Vulkan `VkFormat` numeric value, for
/// the formats OpenXR swapchains are realistically created with.
fn wgpu_format_to_vk(format: wgpu::TextureFormat) -> Option<u32> {
    use wgpu::TextureFormat::*;
    Some(match format {
        Rgba8Unorm => 37,     // VK_FORMAT_R8G8B8A8_UNORM
        Rgba8UnormSrgb => 43, // VK_FORMAT_R8G8B8A8_SRGB
        Bgra8Unorm => 44,     // VK_FORMAT_B8G8R8A8_UNORM
        Bgra8UnormSrgb => 50, // VK_FORMAT_B8G8R8A8_SRGB
        Rgba16Float => 97,    // VK_FORMAT_R16G16B16A16_SFLOAT
        Rgba32Float => 109,   // VK_FORMAT_R32G32B32A32_SFLOAT
        Rgb10a2Unorm => 65,   // VK_FORMAT_A2R10G10B10_UNORM_PACK32
        _ => return None,
    })
}

/// Inverse of [`wgpu_format_to_vk`].
fn vk_format_to_wgpu(vk: u32) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat::*;
    Some(match vk {
        37 => Rgba8Unorm,
        43 => Rgba8UnormSrgb,
        44 => Bgra8Unorm,
        50 => Bgra8UnormSrgb,
        97 => Rgba16Float,
        109 => Rgba32Float,
        65 => Rgb10a2Unorm,
        _ => return None,
    })
}
