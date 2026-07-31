//! Vulkan instance/device creation through OpenXR (`XR_KHR_vulkan_enable2`).
//!
//! A headset session requires the Vulkan instance and device to be created
//! *through* OpenXR — `xrCreateVulkanInstanceKHR` / `xrCreateVulkanDeviceKHR` —
//! so the runtime can inject its required extensions and pick the GPU the HMD
//! is actually driven by. wgpu then wraps the resulting handles:
//! [`wgpu::Instance::from_hal`], `expose_adapter` /
//! [`wgpu::Instance::create_adapter_from_hal`] and `device_from_raw` /
//! [`wgpu::Adapter::create_device_from_hal`].
//!
//! This is a port of the `indite` crate's `context.rs` (celphase, MIT /
//! Apache-2.0 dual licensed) adapted to the wgpu 30.0.0 hal API surface
//! (`ExposedAdapter.capabilities.limits`, `device_from_raw` taking
//! `limits: &Limits` and `memory_hints`).

use ash::vk::{self, Handle as _};
use wgpu::{
    Adapter, Device, DeviceDescriptor, ExperimentalFeatures, Features, Instance, InstanceFlags,
    MemoryBudgetThresholds, MemoryHints, Queue, Trace,
    hal::{api::Vulkan, Api},
};

use crate::{Result, XrError};

/// Bindless texture-table cap for the XR device (mirrors helio's native limit).
const MAX_TEXTURES: usize = 256;

/// Create a wgpu [`Instance`] whose underlying `VkInstance` was created through
/// OpenXR, so the runtime's required instance extensions are present and the
/// device it later exposes is known to be usable by the compositor.
pub fn create_wgpu_instance(
    xr_instance: &openxr::Instance,
    xr_system: openxr::SystemId,
) -> Result<Instance> {
    // Vulkan 1.1 guarantees multiview; request 1.2 to match wgpu-hal's
    // timeline-semaphore expectations (which it treats as core).
    let vk_target_version = vk::make_api_version(0, 1, 2, 0);
    let vk_target_version_xr = openxr::Version::new(1, 2, 0);

    // The `graphics_requirements` call is mandatory and must precede any other
    // Vulkan-via-OpenXR work; skipping it breaks the runtime in subtle ways.
    let reqs = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system)?;
    if vk_target_version_xr < reqs.min_api_version_supported
        || vk_target_version_xr.major() > reqs.max_api_version_supported.major()
    {
        return Err(XrError::Platform(format!(
            "OpenXR runtime requires Vulkan >= {} and <= {}",
            reqs.min_api_version_supported,
            reqs.max_api_version_supported.major()
        )));
    }

    let vk_entry = unsafe { ash::Entry::load().map_err(|e| XrError::Platform(format!("ash: {e}")))? };

    let flags = InstanceFlags::empty();
    let extensions =
        <Vulkan as Api>::Instance::desired_extensions(&vk_entry, vk_target_version, flags)
            .map_err(|e| XrError::Platform(format!("wgpu-hal desired_extensions: {e}")))?;
    let extensions_cchar: Vec<_> = extensions.iter().map(|s| s.as_ptr()).collect();

    let vk_app_info = vk::ApplicationInfo::default()
        .application_version(0)
        .engine_version(0)
        .api_version(vk_target_version);
    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&vk_app_info)
        .enabled_extension_names(&extensions_cchar);

    // Let OpenXR create the VkInstance (it merges in the extensions it needs).
    let get_instance_proc_addr = unsafe {
        std::mem::transmute::<
            ash::vk::PFN_vkGetInstanceProcAddr,
            openxr::sys::platform::VkGetInstanceProcAddr,
        >(vk_entry.static_fn().get_instance_proc_addr)
    };
    let vk_instance = unsafe {
        xr_instance.create_vulkan_instance(
            xr_system,
            get_instance_proc_addr,
            &instance_info as *const _ as *const _,
        )
    };
    let vk_instance = vk_instance
        .map_err(|e| XrError::Platform(format!("OpenXR create_vulkan_instance: {e}")))?
        .map_err(|e| XrError::Platform(format!("Vulkan create_vulkan_instance: {e}")))?;
    let vk_instance = unsafe {
        ash::Instance::load(vk_entry.static_fn(), vk::Instance::from_raw(vk_instance as _))
    };

    let hal_instance = unsafe {
        <Vulkan as Api>::Instance::from_raw(
            vk_entry,
            vk_instance,
            vk_target_version,
            0,
            None,
            extensions,
            flags,
            MemoryBudgetThresholds::default(),
            false,
            None,
        )
    }
    .map_err(|e| XrError::Platform(format!("wgpu-hal Instance::from_raw: {e}")))?;

    Ok(unsafe { Instance::from_hal::<Vulkan>(hal_instance) })
}

/// Create a wgpu [`Device`] + [`Queue`] on the physical device OpenXR assigned
/// to the HMD.
///
/// The physical device is selected by the runtime
/// (`xrGetVulkanGraphicsDevice2KHR`) and the logical device is created through
/// OpenXR (`xrCreateVulkanDeviceKHR`) so its required device extensions are
/// enabled. wgpu-hal's `device_from_raw` then wraps the resulting `VkDevice`.
///
/// `wanted_features` is masked down to what the adapter actually supports
/// (mirroring `helio::required_wgpu_features` semantics); the function fails if
/// `Features::MULTIVIEW` is not among the survivors. Limits are derived from
/// the adapter's capabilities with the bindless texture-table capped at 256 and
/// `max_multiview_view_count` raised to 2.
pub fn create_wgpu_device(
    xr_instance: &openxr::Instance,
    xr_system: openxr::SystemId,
    instance: &Instance,
    wanted_features: Features,
) -> Result<(Adapter, Device, Queue)> {
    let hal_instance = unsafe { instance.as_hal::<Vulkan>() }
        .ok_or_else(|| XrError::GraphicsUnavailable("wgpu instance is not Vulkan".to_string()))?;
    let shared = hal_instance.shared_instance();
    let raw_instance = shared.raw_instance();
    let vk_entry = shared.entry();

    let vk_physical_device = unsafe {
        xr_instance.vulkan_graphics_device(xr_system, raw_instance.handle().as_raw() as _)
    };
    let vk_physical_device = vk_physical_device.map_err(|e| {
        XrError::Platform(format!("OpenXR vulkan_graphics_device: {e} (is the runtime running?)"))
    })?;
    let vk_physical_device = vk::PhysicalDevice::from_raw(vk_physical_device as _);

    let vk_device_properties = unsafe { raw_instance.get_physical_device_properties(vk_physical_device) };
    if vk_device_properties.api_version < vk::API_VERSION_1_1 {
        return Err(XrError::Platform(format!(
            "Vulkan physical device does not support 1.1 (multiview) — got {}",
            vk_device_properties.api_version
        )));
    }

    let hal_adapter = hal_instance
        .expose_adapter(vk_physical_device)
        .ok_or_else(|| XrError::GraphicsUnavailable("wgpu-hal could not expose the HMD adapter".to_string()))?;

    // Mask the requested feature set down to what the adapter supports, the
    // same way helio's required_wgpu_features does (required | (wanted & available)).
    let required_features = wanted_features & hal_adapter.features;
    if !required_features.contains(Features::MULTIVIEW) {
        return Err(XrError::GraphicsUnavailable(
            "the HMD's Vulkan device does not support Features::MULTIVIEW".to_string(),
        ));
    }

    // Limits: cap the bindless texture table at MAX_TEXTURES and ask for two
    // multiview layers (min'd against what the adapter reports).
    let mut limits = hal_adapter.capabilities.limits.clone();
    limits.max_sampled_textures_per_shader_stage = limits
        .max_sampled_textures_per_shader_stage
        .min(MAX_TEXTURES as u32);
    limits.max_samplers_per_shader_stage = limits
        .max_samplers_per_shader_stage
        .min(MAX_TEXTURES as u32);
    limits.max_multiview_view_count = limits.max_multiview_view_count.max(2);

    // The device extensions wgpu needs for the requested features.
    let device_extensions = hal_adapter.adapter.required_device_extensions(required_features);
    let device_extensions_cchar: Vec<_> = device_extensions.iter().map(|s| s.as_ptr()).collect();
    let mut enabled_physical_device_features =
        hal_adapter
            .adapter
            .physical_device_features(&device_extensions, required_features);

    let queue_family_index = unsafe {
        raw_instance
            .get_physical_device_queue_family_properties(vk_physical_device)
            .into_iter()
            .enumerate()
            .find_map(|(i, info)| {
                if info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .ok_or_else(|| XrError::Platform("Vulkan device has no graphics queue".to_string()))?
    };

    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0]);
    let queue_infos = [queue_info];
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extensions_cchar);
    let device_info = enabled_physical_device_features.add_to_device_create(device_info);

    let get_instance_proc_addr = unsafe {
        std::mem::transmute::<
            ash::vk::PFN_vkGetInstanceProcAddr,
            openxr::sys::platform::VkGetInstanceProcAddr,
        >(vk_entry.static_fn().get_instance_proc_addr)
    };
    let vk_device = unsafe {
        xr_instance.create_vulkan_device(
            xr_system,
            get_instance_proc_addr,
            vk_physical_device.as_raw() as _,
            &device_info as *const _ as *const _,
        )
    };
    let vk_device = vk_device
        .map_err(|e| XrError::Platform(format!("OpenXR create_vulkan_device: {e}")))?
        .map_err(|e| XrError::Platform(format!("Vulkan create_vulkan_device: {e}")))?;
    let vk_device = unsafe {
        ash::Device::load(raw_instance.fp_v1_0(), vk::Device::from_raw(vk_device as _))
    };

    let memory_hints = MemoryHints::default();
    let hal_device = unsafe {
        hal_adapter.adapter.device_from_raw(
            vk_device,
            None,
            &device_extensions,
            required_features,
            &limits,
            &memory_hints,
            queue_family_index,
            0,
        )
    }
    .map_err(|e| XrError::GraphicsUnavailable(format!("wgpu-hal device_from_raw: {e:?}")))?;

    let wgpu_adapter = unsafe { instance.create_adapter_from_hal(hal_adapter) };
    let device_desc = DeviceDescriptor {
        label: Some("helio xr device"),
        required_features,
        required_limits: limits,
        experimental_features: ExperimentalFeatures::default(),
        memory_hints,
        trace: Trace::default(),
    };
    let (device, queue) =
        unsafe { wgpu_adapter.create_device_from_hal(hal_device, &device_desc) }.map_err(|e| {
            XrError::GraphicsUnavailable(format!("wgpu create_device_from_hal: {e:?}"))
        })?;

    Ok((wgpu_adapter, device, queue))
}
