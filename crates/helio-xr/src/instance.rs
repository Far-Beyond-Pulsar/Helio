//! OpenXR instance / system negotiation.

use openxr::sys::Handle as _;

use crate::{Result, XrError};

/// OpenXR debug-utils messenger callback.
///
/// Forwards runtime messages into the `log` crate. Severity is mapped onto the
/// log level; message *type* is reported as a raw mask (the openxr-sys bitmask
/// types do not implement `Debug`).
unsafe extern "system" fn debug_utils_callback(
    severity: openxr::sys::DebugUtilsMessageSeverityFlagsEXT,
    ty: openxr::sys::DebugUtilsMessageTypeFlagsEXT,
    data: *const openxr::sys::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> openxr::sys::Bool32 {
    if !data.is_null() {
        let message = unsafe { (*data).message };
        if !message.is_null() {
            let msg = unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy();
            let ty_mask = ty.into_raw();
            let msg = format!("[OpenXR 0x{ty_mask:x}] {msg}");
            if severity.contains(openxr::sys::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
                log::error!("{msg}");
            } else if severity.contains(openxr::sys::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
                log::warn!("{msg}");
            } else {
                log::debug!("{msg}");
            }
        }
    }
    openxr::sys::FALSE
}

/// A connected OpenXR instance and the system it drives.
///
/// Owns the raw `openxr::Instance`; when dropped, any debug-utils messenger is
/// destroyed *before* the instance goes away.
pub struct XrInstance {
    pub instance: openxr::Instance,
    pub system: openxr::SystemId,
    /// Vulkan version range required by the runtime for this system.
    pub requirements: openxr::vulkan::Requirements,
    debug_messenger: Option<openxr::sys::DebugUtilsMessengerEXT>,
}

impl XrInstance {
    /// Connect to the OpenXR runtime, create an instance with the Vulkan
    /// graphics binding enabled, and pick the HMD system.
    ///
    /// The loader is loaded dynamically (`openxr_loader` on the dynamic
    /// loader search path). With the crate's `linked` feature the entry points
    /// are resolved at link time instead.
    pub fn create(application_name: &str) -> Result<Self> {
        #[cfg(feature = "linked")]
        let entry = openxr::Entry::linked();
        #[cfg(not(feature = "linked"))]
        let entry = unsafe { openxr::Entry::load().map_err(XrError::Load)? };

        let supported = entry.enumerate_extensions()?;

        // Enable the Vulkan graphics binding (prefer XR_KHR_vulkan_enable2,
        // fall back to the legacy XR_KHR_vulkan_enable) plus debug utils if
        // the runtime offers them.
        // An empty ExtensionSet asks for nothing; never request an extension
        // the runtime does not advertise (create_instance would fail). The
        // struct is `#[non_exhaustive]`, so it is mutated rather than built
        // with a struct literal.
        let mut extensions = openxr::ExtensionSet::default();
        extensions.khr_vulkan_enable2 = supported.khr_vulkan_enable2;
        extensions.khr_vulkan_enable =
            !supported.khr_vulkan_enable2 && supported.khr_vulkan_enable;
        extensions.ext_debug_utils = supported.ext_debug_utils;

        let app_info = openxr::ApplicationInfo {
            application_name,
            application_version: 0,
            engine_name: "Helio",
            engine_version: 0,
            api_version: openxr::CURRENT_API_VERSION,
        };

        let instance = entry.create_instance(&app_info, &extensions, &[])?;

        let system = instance
            .system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY)
            .map_err(|e| XrError::Platform(format!("no HMD system available ({e})")))?;

        let requirements = instance.graphics_requirements::<openxr::vulkan::Vulkan>(system)?;

        let debug_messenger = Self::create_debug_messenger(&instance)?;

        if let Ok(props) = instance.properties() {
            log::info!(
                "OpenXR runtime '{}' v{} (system {}, Vulkan {}.{})",
                props.runtime_name,
                props.runtime_version,
                system.into_raw(),
                requirements.min_api_version_supported.major(),
                requirements.min_api_version_supported.minor(),
            );
        }

        Ok(Self {
            instance,
            system,
            requirements,
            debug_messenger,
        })
    }

    /// Whether the debug-utils messenger is active.
    pub fn debug_utils_enabled(&self) -> bool {
        self.debug_messenger.is_some()
    }

    fn create_debug_messenger(
        instance: &openxr::Instance,
    ) -> Result<Option<openxr::sys::DebugUtilsMessengerEXT>> {
        let Some(ext) = instance.exts().ext_debug_utils.as_ref() else {
            return Ok(None);
        };
        let create_info = openxr::sys::DebugUtilsMessengerCreateInfoEXT {
            ty: openxr::sys::DebugUtilsMessengerCreateInfoEXT::TYPE,
            next: std::ptr::null(),
            message_severities: openxr::sys::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | openxr::sys::DebugUtilsMessageSeverityFlagsEXT::INFO
                | openxr::sys::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | openxr::sys::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_types: openxr::sys::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | openxr::sys::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | openxr::sys::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | openxr::sys::DebugUtilsMessageTypeFlagsEXT::CONFORMANCE,
            user_callback: Some(debug_utils_callback),
            user_data: std::ptr::null_mut(),
        };
        let mut out = openxr::sys::DebugUtilsMessengerEXT::NULL;
        let result = unsafe {
            (ext.create_debug_utils_messenger)(instance.as_raw(), &create_info, &mut out)
        };
        if result.into_raw() < 0 {
            // Debug utils are best-effort; do not fail instance creation over
            // them.
            log::warn!("failed to create OpenXR debug utils messenger: {result}");
            return Ok(None);
        }
        log::debug!("OpenXR debug utils messenger active");
        Ok(Some(out))
    }
}

impl Drop for XrInstance {
    fn drop(&mut self) {
        // Destroy the messenger before `self.instance` (and with it the raw
        // XrInstance) is dropped.
        let messenger = self.debug_messenger.take();
        if let Some(messenger) = messenger {
            if let Some(ext) = self.instance.exts().ext_debug_utils.as_ref() {
                let destroy = ext.destroy_debug_utils_messenger;
                unsafe { (destroy)(messenger) };
            }
        }
    }
}
