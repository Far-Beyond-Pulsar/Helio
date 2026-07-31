//! [`openxr::Graphics`] implementation for the wgpu/Vulkan backend.
//!
//! The `openxr` crate ships a `vulkan` module, but it has no wgpu module. The
//! [`Graphics`] trait only needs the *associated types* to line up with the
//! runtime's Vulkan binding; the actual Vulkan handles used to create the
//! session come from wgpu's `as_hal()` escape hatch (see
//! [`crate::session::vulkan_session_create_info`]).
//!
//! The associated types here are identical to the built-in
//! [`openxr::vulkan::Vulkan`] binding:
//!
//! - `Format` is `u32` (a raw `VkFormat`),
//! - `SwapchainImage` is `u64` (a raw `VkImage` handle).
//!
//! The `Requirements` / `SessionCreateInfo` / session-creation / format
//! conversion methods therefore forward straight to the built-in Vulkan
//! implementation. Only `enumerate_swapchain_images` is re-implemented, because
//! the built-in one is generic over `Swapchain<openxr::Vulkan>` while we deal in
//! `Swapchain<WgpuGraphics>` (the two only differ by a `PhantomData` marker).

use openxr::{Graphics, Instance, SystemId};

/// Marker type implementing [`openxr::Graphics`] for wgpu on Vulkan.
#[derive(Debug, Clone, Copy)]
pub struct WgpuGraphics;

impl Graphics for WgpuGraphics {
    type Requirements = <openxr::vulkan::Vulkan as Graphics>::Requirements;
    type SessionCreateInfo = <openxr::vulkan::Vulkan as Graphics>::SessionCreateInfo;
    type Format = <openxr::vulkan::Vulkan as Graphics>::Format;
    type SwapchainImage = <openxr::vulkan::Vulkan as Graphics>::SwapchainImage;

    fn raise_format(x: i64) -> Self::Format {
        <openxr::vulkan::Vulkan as Graphics>::raise_format(x)
    }

    fn lower_format(x: Self::Format) -> i64 {
        <openxr::vulkan::Vulkan as Graphics>::lower_format(x)
    }

    fn requirements(instance: &Instance, system: SystemId) -> openxr::Result<Self::Requirements> {
        <openxr::vulkan::Vulkan as Graphics>::requirements(instance, system)
    }

    unsafe fn create_session(
        instance: &Instance,
        system: SystemId,
        info: &Self::SessionCreateInfo,
    ) -> openxr::Result<openxr::sys::Session> {
        unsafe { <openxr::vulkan::Vulkan as Graphics>::create_session(instance, system, info) }
    }

    fn enumerate_swapchain_images(
        swapchain: &openxr::Swapchain<Self>,
    ) -> openxr::Result<Vec<Self::SwapchainImage>> {
        // Replicate the two-call enumeration from openxr's own vulkan binding
        // against the raw instance function pointer. `Swapchain<WgpuGraphics>`
        // is layout-identical to `Swapchain<openxr::vulkan::Vulkan>`; the
        // handle returned by `swapchain.as_raw()` is a plain `u64`.
        let fp = swapchain.instance().fp();
        let mut count = 0u32;
        let mut result = unsafe {
            (fp.enumerate_swapchain_images)(
                swapchain.as_raw(),
                count,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if result.into_raw() < 0 {
            return Err(result);
        }
        let mut buf = vec![
            openxr::sys::SwapchainImageVulkanKHR {
                ty: openxr::sys::SwapchainImageVulkanKHR::TYPE,
                next: std::ptr::null_mut(),
                image: 0,
            };
            count as usize
        ];
        result = unsafe {
            (fp.enumerate_swapchain_images)(
                swapchain.as_raw(),
                count,
                &mut count,
                buf.as_mut_ptr() as *mut _,
            )
        };
        if result.into_raw() < 0 {
            return Err(result);
        }
        buf.truncate(count as usize);
        Ok(buf.into_iter().map(|x| x.image).collect())
    }
}
