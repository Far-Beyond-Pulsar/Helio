use super::renderer_impl::Renderer;

impl Renderer {
    pub unsafe fn request_exclusive_fullscreen(
        &self,
        surface: &wgpu::Surface<'_>,
        raw_hwnd: *mut std::ffi::c_void,
    ) -> bool {
        #[cfg(not(target_os = "windows"))]
        {
            let _ = (surface, raw_hwnd);
            log::warn!("helio: request_exclusive_fullscreen is not supported on this platform");
            return false;
        }

        #[cfg(target_os = "windows")]
        exclusive_fullscreen_win(&self.device, surface, raw_hwnd)
    }

    pub unsafe fn exit_exclusive_fullscreen(&self, surface: &wgpu::Surface<'_>) {
        #[cfg(not(target_os = "windows"))]
        {
            let _ = surface;
        }

        #[cfg(target_os = "windows")]
        exit_exclusive_fullscreen_win(surface);
    }
}

#[cfg(target_os = "windows")]
fn exclusive_fullscreen_win(
    device: &wgpu::Device,
    surface: &wgpu::Surface<'_>,
    raw_hwnd: *mut std::ffi::c_void,
) -> bool {
    use windows::Win32::{
        Foundation::HWND,
        Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, DXGI_MWA_FLAGS},
    };

    let hwnd = HWND(raw_hwnd);

    let is_dx12 = unsafe { device.as_hal::<wgpu::hal::api::Dx12>() }.is_some();
    if is_dx12 {
        let result: windows::core::Result<()> = (|| unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
            factory.MakeWindowAssociation(hwnd, DXGI_MWA_FLAGS(0))?;

            let swap_chain = surface
                .as_hal::<wgpu::hal::api::Dx12>()
                .and_then(|s| s.swap_chain())
                .ok_or_else(|| windows::core::Error::from(
                    windows::Win32::Foundation::E_FAIL,
                ))?;
            swap_chain.SetFullscreenState(true, None)
        })();
        return match result {
            Ok(()) => true,
            Err(e) => {
                log::warn!("helio: DX12 exclusive fullscreen failed: {e}");
                false
            }
        };
    }

    let is_vulkan = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.is_some();
    if is_vulkan {
        log::warn!(
            "helio: exclusive fullscreen on Vulkan requires VK_EXT_full_screen_exclusive \
             to be enabled at instance-creation time; it cannot be activated post-hoc"
        );
        return false;
    }

    log::warn!("helio: request_exclusive_fullscreen: unrecognised or unsupported backend");
    false
}

#[cfg(target_os = "windows")]
fn exit_exclusive_fullscreen_win(surface: &wgpu::Surface<'_>) {
    unsafe {
        if let Some(swap_chain) = surface
            .as_hal::<wgpu::hal::api::Dx12>()
            .and_then(|s| s.swap_chain())
        {
            if let Err(e) = swap_chain.SetFullscreenState(false, None) {
                log::warn!("helio: exit_exclusive_fullscreen: SetFullscreenState(FALSE) failed: {e}");
            }
        }
    }
}
