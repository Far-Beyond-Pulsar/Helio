//! Window and GPU startup shared by desktop and OpenXR Helio applications.
//!
//! `vr_support` is deliberately handled before the scene GPU mirror and
//! renderer are created: OpenXR must select the Vulkan device used by both.

use std::sync::Arc;

use crate::{
    Camera, Renderer, RendererBuilder, RendererConfig, required_experimental_features,
    required_wgpu_features, required_wgpu_limits, usable_adapter_features,
};

/// Controls Helio's optional automatic XR discovery during graphics startup.
#[derive(Debug, Clone)]
pub struct HelioGraphicsConfig {
    /// Try the active OpenXR runtime before creating a desktop WGPU device.
    /// If discovery or XR startup fails, startup continues in desktop mode.
    pub vr_support: bool,
    /// OpenXR application name shown by the runtime.
    pub application_name: String,
    /// Present-mode preference for the app's ordinary PC window.
    pub present_mode: wgpu::PresentMode,
    /// Requested usages for the ordinary PC window surface. Helio intersects
    /// these with the surface's supported usages before configuration.
    pub surface_usage: wgpu::TextureUsages,
    /// Adapter power preference for desktop fallback.
    pub power_preference: wgpu::PowerPreference,
}

impl Default for HelioGraphicsConfig {
    fn default() -> Self {
        Self {
            vr_support: true,
            application_name: "Helio Application".into(),
            present_mode: wgpu::PresentMode::Fifo,
            surface_usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            power_preference: wgpu::PowerPreference::HighPerformance,
        }
    }
}

/// Whether this app started with a live XR session or used desktop fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelioGraphicsMode {
    Desktop,
    Xr,
}

/// The app-facing XR input handle. Helio owns the OpenXR session lifetime, so
/// applications can read controller state without retaining or polling a raw
/// session handle.
pub struct HelioXrInput {
    input: helio_xr::XrInput,
    session: helio_xr::XrSessionHandle,
}

/// Controller actions and tracked poses sampled alongside the renderer's
/// most recently predicted OpenXR display frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct HelioXrFrameState {
    pub focused: bool,
    pub controllers: helio_xr::ControllerState,
    pub tracking: Option<helio_xr::TrackingState>,
    pub display_refresh_rate_hz: Option<f32>,
}

impl HelioXrInput {
    /// Sync controller actions and locate head/hand poses against the last
    /// display time prepared by Helio's renderer.
    pub fn frame_state(
        &mut self,
        renderer: &Renderer,
        world_from_stage: &glam::Mat4,
    ) -> helio_xr::Result<HelioXrFrameState> {
        let controllers = self.input.sync(&self.session)?;
        let tracking = if let Some(display_time) = renderer.xr_last_display_time() {
            Some(
                self.input
                    .tracking_state(&self.session, display_time, world_from_stage)?,
            )
        } else {
            None
        };
        Ok(HelioXrFrameState {
            focused: renderer.xr_session_focused(),
            controllers,
            tracking,
            display_refresh_rate_hz: renderer.xr_display_refresh_rate_hz(),
        })
    }

    /// Read the current standard Helio controller actions.
    pub fn controller_state(&self) -> helio_xr::Result<helio_xr::ControllerState> {
        self.input.sync(&self.session)
    }

    /// Apply app-owned locomotion to the XR stage. Tracking remains in the
    /// runtime's stage space; this rigid transform moves the player through
    /// the application world without moving scene objects.
    pub fn set_locomotion_transform(&self, renderer: &mut Renderer, world_from_stage: glam::Mat4) {
        renderer.set_xr_stage_transform(world_from_stage);
    }

    /// Locate tracked grip poses at the renderer's most recently predicted XR
    /// display time. Returns `None` before Helio has prepared an XR frame.
    pub fn grip_poses(
        &mut self,
        renderer: &Renderer,
        world_from_stage: &glam::Mat4,
    ) -> helio_xr::Result<Option<[Option<glam::Mat4>; 2]>> {
        let Some(display_time) = renderer.xr_last_display_time() else {
            return Ok(None);
        };
        self.input
            .grip_pose_matrices(&self.session, display_time, world_from_stage)
            .map(Some)
    }
}

/// The device, window surface, and optional XR runtime selected for a Helio app.
///
/// Construct this before creating SceneDB's GPU mirror. The returned device and
/// queue are the ones the mirror and renderer must share. Apps continue to own
/// the window and event loop; an `Arc<Window>` is a suitable surface target.
pub struct HelioGraphicsContext<'window> {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface<'window>,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface_config: wgpu::SurfaceConfiguration,
    mode: HelioGraphicsMode,
    xr_runtime: Option<helio_xr::XrRuntime>,
    xr_render_target: Option<(u32, u32, wgpu::TextureFormat)>,
}

impl<'window> HelioGraphicsContext<'window> {
    /// Create the window graphics context, trying OpenXR first when
    /// `config.vr_support` is enabled. The desktop fallback requests Helio's
    /// standard features and limits and selects an adapter compatible with the
    /// supplied window surface.
    pub async fn new(
        target: impl Into<wgpu::SurfaceTarget<'window>>,
        width: u32,
        height: u32,
        config: HelioGraphicsConfig,
    ) -> helio_core::Result<Self> {
        if config.vr_support {
            match helio_xr::XrRuntime::start_helio(
                &config.application_name,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            ) {
                Ok(runtime) => {
                    let surface = runtime.create_mirror_surface(target).map_err(|error| {
                        helio_core::Error::InvalidPassConfig(format!(
                            "OpenXR started but the PC mirror surface could not be created: {error}"
                        ))
                    })?;
                    let (format, alpha_mode, usage) = runtime.configure_mirror_surface_with_usage(
                        &surface,
                        width.max(1),
                        height.max(1),
                        config.surface_usage,
                    );
                    let caps = surface.get_capabilities(&runtime.adapter);
                    let present_mode = preferred_present_mode(&caps, config.present_mode);
                    let surface_config = wgpu::SurfaceConfiguration {
                        usage,
                        format,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                        width: width.max(1),
                        height: height.max(1),
                        present_mode,
                        alpha_mode,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    };
                    let (eye_width, eye_height) = runtime.eye_extent();
                    let eye_format = runtime.swapchain.format;
                    surface.configure(&runtime.device, &surface_config);
                    let instance = runtime.wgpu_instance.clone();
                    let adapter = runtime.adapter.clone();
                    let device = runtime.device.clone();
                    let queue = runtime.queue.clone();
                    return Ok(Self {
                        instance,
                        surface,
                        adapter,
                        device,
                        queue,
                        surface_config,
                        mode: HelioGraphicsMode::Xr,
                        xr_runtime: Some(runtime),
                        xr_render_target: Some((eye_width, eye_height, eye_format)),
                    });
                }
                Err(error) => log::info!(
                    "VR support requested, but OpenXR is unavailable; starting desktop mode: {error}"
                ),
            }
        }

        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let surface = instance.create_surface(target).map_err(|error| {
            helio_core::Error::InvalidPassConfig(format!("create window surface: {error}"))
        })?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|error| {
                helio_core::Error::InvalidPassConfig(format!("request WGPU adapter: {error}"))
            })?;
        let adapter_features = usable_adapter_features(&adapter);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Helio Device"),
                required_features: required_wgpu_features(adapter_features),
                required_limits: required_wgpu_limits(adapter.limits()),
                experimental_features: required_experimental_features(adapter_features),
                ..Default::default()
            })
            .await
            .map_err(|error| {
                helio_core::Error::InvalidPassConfig(format!("request WGPU device: {error}"))
            })?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|format| format.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let usage = config.surface_usage & caps.usages;
        if !usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
            return Err(helio_core::Error::InvalidPassConfig(
                "window surface does not support render attachment usage".into(),
            ));
        }
        let surface_config = wgpu::SurfaceConfiguration {
            usage,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: width.max(1),
            height: height.max(1),
            present_mode: preferred_present_mode(&caps, config.present_mode),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);
        Ok(Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_config,
            mode: HelioGraphicsMode::Desktop,
            xr_runtime: None,
            xr_render_target: None,
        })
    }

    pub fn mode(&self) -> HelioGraphicsMode {
        self.mode
    }

    pub fn is_xr(&self) -> bool {
        self.mode == HelioGraphicsMode::Xr
    }

    /// Renderer target extent and format selected for this context.
    pub fn render_target(&self) -> (u32, u32, wgpu::TextureFormat) {
        self.xr_render_target.unwrap_or((
            self.surface_config.width,
            self.surface_config.height,
            self.surface_config.format,
        ))
    }

    /// Build the renderer on the device selected by this context and attach
    /// XR automatically when available. The optional input/session pair is the
    /// only XR-specific value apps need when implementing controller behavior.
    /// The renderer config is adapted to the headset eye target or desktop
    /// surface before the graph is built.
    pub fn build_renderer(
        &mut self,
        builder: RendererBuilder,
        mut config: RendererConfig,
    ) -> helio_core::Result<(Renderer, Option<HelioXrInput>)> {
        config.enable_xr = self.is_xr();
        if let Some((width, height, format)) = self.xr_render_target {
            config.width = width;
            config.height = height;
            config.surface_format = format;
            config.render_scale = 1.0;
        } else {
            config.width = self.surface_config.width;
            config.height = self.surface_config.height;
            config.surface_format = self.surface_config.format;
        }
        let (width, height, surface_format) = (config.width, config.height, config.surface_format);
        let mut renderer = builder
            .with_renderer_config(config)
            .with_external_device()
            .build(
                self.device.clone(),
                self.queue.clone(),
                width,
                height,
                surface_format,
            );
        let input = if let Some(runtime) = self.xr_runtime.take() {
            renderer.set_xr_mirror_format(self.surface_config.format);
            let session_handle = runtime.session.session.clone();
            let input = renderer.install_xr_runtime(runtime)?;
            Some(HelioXrInput {
                input,
                session: session_handle,
            })
        } else {
            None
        };
        Ok((renderer, input))
    }

    /// Unified app frame entry point: desktop render/present or headset render
    /// plus Helio's PC-side stereo mirror.
    pub fn render_frame(&self, renderer: &mut Renderer, camera: &Camera) -> helio_core::Result<()> {
        if self.is_xr() {
            renderer.set_xr_camera(camera.clone());
        }
        renderer.render_window_frame(camera, &self.surface)
    }

    /// Resize the app window surface and update the renderer when in desktop
    /// mode. XR keeps its graph at the runtime's eye resolution while only the
    /// PC mirror surface changes size.
    pub fn resize(&mut self, renderer: &mut Renderer, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
        if !self.is_xr() {
            renderer.set_render_size(self.surface_config.width, self.surface_config.height);
        }
    }
}

fn preferred_present_mode(
    caps: &wgpu::SurfaceCapabilities,
    requested: wgpu::PresentMode,
) -> wgpu::PresentMode {
    if caps.present_modes.contains(&requested) {
        requested
    } else if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
        wgpu::PresentMode::Fifo
    } else {
        caps.present_modes[0]
    }
}
