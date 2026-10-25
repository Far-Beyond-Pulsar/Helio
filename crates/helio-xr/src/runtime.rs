//! One-call OpenXR + wgpu runtime bootstrap for Helio applications.
//!
//! Applications still own their window and event loop, but should not need to
//! repeat the fragile ordering of creating the OpenXR Vulkan device, binding a
//! session to it, attaching input actions, and constructing the eye swapchain.

use std::sync::Arc;

use crate::{
    Result, XrInput, XrInstance, XrSession, XrSwapchain, create_wgpu_device, create_wgpu_instance,
};

/// The OpenXR runtime, graphics device, session, input actions, and eye
/// swapchain assembled in the order required by OpenXR.
///
/// Keep this value alive for as long as the renderer is using its device. Pass
/// use `Renderer::install_xr_runtime`), and call `Renderer::render_xr` for
/// each XR frame.
pub struct XrRuntime {
    pub instance: XrInstance,
    pub session: XrSession,
    pub swapchain: XrSwapchain,
    pub input: XrInput,
    pub wgpu_instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl XrRuntime {
    /// Start the XR runtime with the feature set used by Helio's renderer.
    /// Optional GPU capabilities are requested opportunistically and filtered
    /// to the selected headset adapter by the Vulkan device factory.
    pub fn start_helio(
        application_name: &str,
        swapchain_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let required = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::MULTIVIEW
            | wgpu::Features::MULTISAMPLE_ARRAY;
        let optional = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::VERTEX_WRITABLE_STORAGE;
        Self::start(application_name, required | optional, swapchain_format)
    }

    /// Connect to the active OpenXR runtime and create its Vulkan-backed wgpu
    /// device, session, standard Helio controller actions, and swapchain.
    ///
    /// `requested_features` is filtered against the selected headset GPU by
    /// `create_wgpu_device`. Build the renderer with
    /// `RendererConfig::for_xr` and the returned device and queue.
    pub fn start(
        application_name: &str,
        requested_features: wgpu::Features,
        swapchain_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let instance = XrInstance::create(application_name)?;
        let wgpu_instance = create_wgpu_instance(&instance.instance, instance.system)?;
        let (adapter, device, queue) = create_wgpu_device(
            &instance.instance,
            instance.system,
            &wgpu_instance,
            requested_features,
        )?;

        // Declare and suggest actions before session creation. The runtime
        // resolves those bindings as part of creating the session.
        let input = XrInput::new(&instance.instance)?;
        let session = XrSession::create(
            &instance.instance,
            instance.system,
            &wgpu_instance,
            &device,
            &queue,
        )?;
        input.attach(&session.session)?;
        let swapchain = XrSwapchain::create(
            &device,
            &session.session,
            session.width,
            session.height,
            swapchain_format,
        )?;

        log::info!(
            "OpenXR ready — {}x{} eye buffer, {} array layer(s), format {:?}",
            session.width,
            session.height,
            swapchain.array_size,
            swapchain.format,
        );

        Ok(Self {
            instance,
            session,
            swapchain,
            input,
            wgpu_instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// The recommended per-eye target extent reported by the runtime.
    pub fn eye_extent(&self) -> (u32, u32) {
        (self.session.width, self.session.height)
    }

    /// Whether the runtime accepted a two-layer stereo swapchain.
    pub fn supports_array_stereo(&self) -> bool {
        self.swapchain.array_size >= 2
    }

    /// Create the PC mirror surface on the same Vulkan instance and adapter
    /// as the headset renderer. This accepts any wgpu surface target, so apps
    /// can pass their own `Arc<Window>` without making Helio own their event
    /// loop.
    pub fn create_mirror_surface<'window>(
        &self,
        target: impl Into<wgpu::SurfaceTarget<'window>>,
    ) -> Result<wgpu::Surface<'window>> {
        self.wgpu_instance
            .create_surface(target)
            .map_err(|error| crate::XrError::Platform(format!("mirror surface: {error}")))
    }

    /// Configure a mirror surface and choose an sRGB target format where the
    /// platform supports one. Returns the chosen format and alpha mode for
    /// renderer setup and subsequent surface reconfiguration.
    pub fn configure_mirror_surface(
        &self,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
    ) -> (wgpu::TextureFormat, wgpu::CompositeAlphaMode) {
        let (format, alpha_mode, _) = self.configure_mirror_surface_with_usage(
            surface,
            width,
            height,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        (format, alpha_mode)
    }

    /// Configure the mirror with app-requested usages, limited to those the
    /// platform surface supports. Render attachment remains required.
    pub fn configure_mirror_surface_with_usage(
        &self,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
        requested_usage: wgpu::TextureUsages,
    ) -> (
        wgpu::TextureFormat,
        wgpu::CompositeAlphaMode,
        wgpu::TextureUsages,
    ) {
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .find(|format| format.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let alpha_mode = caps.alpha_modes[0];
        let usage = requested_usage & caps.usages;
        assert!(
            usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
            "mirror surface does not support render attachment usage"
        );
        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage,
                format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: width.max(1),
                height: height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
        (format, alpha_mode, usage)
    }
}
