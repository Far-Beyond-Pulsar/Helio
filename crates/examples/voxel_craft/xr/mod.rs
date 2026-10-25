//! OpenXR owns frame pacing and tracking; each eye uses the ordinary voxel graph.
//! Unlike a flat scene, the voxel source needs the *actual* absolute eye position
//! for each draw. Owning the frame here also lets gameplay use this frame's poses.

mod input;
pub mod rig;

use crate::gfx::Gpu;
use glam::{DMat4, DVec3};
use helio::Camera;
use helio_xr::{Result, SessionEvent, XrError, XrInstance, XrSession, XrSwapchain};
use std::sync::Arc;
use winit::window::Window;

pub enum Begin {
    Idle,
    Frame(rig::FrameInput),
    Exit,
}

struct Pending {
    time: openxr::Time,
    views: Vec<openxr::View>,
}

pub struct Runtime {
    // Drop references to external images and action spaces before their owners.
    mirror: helio::StereoMirror,
    input: input::Input,
    swapchain: XrSwapchain,
    session: XrSession,
    _instance: XrInstance,
    pending: Option<Pending>,
    refresh_rate_request_attempted: bool,
}

/// Same Vulkan-through-OpenXR bootstrap as foliage. Failure is a desktop fallback.
pub fn try_init(window: Arc<Window>) -> Option<(Gpu, Runtime)> {
    if std::env::var_os("VOXEL_CRAFT_SHOTS").is_some()
        || matches!(
            std::env::var("VOXEL_CRAFT_VR").as_deref(),
            Ok("off" | "0" | "false")
        )
    {
        return None;
    }
    match init(window) {
        Ok(value) => Some(value),
        Err(error) => {
            log::warn!("[XR] {error}; running Voxel Craft on the desktop");
            None
        }
    }
}

fn init(window: Arc<Window>) -> Result<(Gpu, Runtime)> {
    let instance = XrInstance::create("Voxel Craft")?;
    let wgpu_instance = helio_xr::create_wgpu_instance(&instance.instance, instance.system)?;
    // Include the voxel renderer's optional native int64 path, but no experimental
    // ray-query features: create_wgpu_device uses a non-experimental device.
    let features = (helio::required_wgpu_features(wgpu::Features::all())
        - wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        | wgpu::Features::MULTIVIEW
        | wgpu::Features::MULTISAMPLE_ARRAY;
    let (adapter, device, queue) = helio_xr::create_wgpu_device(
        &instance.instance,
        instance.system,
        &wgpu_instance,
        features,
    )?;
    let mut input = input::Input::new(&instance.instance)?;
    let session = XrSession::create(
        &instance.instance,
        instance.system,
        &wgpu_instance,
        &device,
        &queue,
    )?;
    input.attach(&session)?;
    let swapchain = XrSwapchain::create(
        &device,
        &session.session,
        session.width,
        session.height,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    )?;
    if session.view_config == openxr::ViewConfigurationType::PRIMARY_STEREO
        && swapchain.array_size < 2
    {
        return Err(XrError::Platform(
            "runtime cannot allocate a stereo array swapchain".into(),
        ));
    }
    let surface = wgpu_instance
        .create_surface(window)
        .map_err(|e| XrError::Platform(format!("mirror surface: {e}")))?;
    let caps = surface.get_capabilities(&adapter);
    let surface_format = caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .or_else(|| caps.formats.first().copied())
        .ok_or_else(|| XrError::Platform("headset GPU cannot present the mirror window".into()))?;
    let alpha_mode = *caps
        .alpha_modes
        .first()
        .ok_or_else(|| XrError::Platform("mirror has no alpha mode".into()))?;
    let mirror = helio::StereoMirror::new(&device, surface_format, &swapchain.views);
    log::info!(
        "[XR] Voxel Craft ready: {}x{} per eye on {}",
        session.width,
        session.height,
        adapter.get_info().name
    );
    let gpu = Gpu {
        surface,
        device: Arc::new(device),
        queue: Arc::new(queue),
        surface_format,
        alpha_mode,
    };
    Ok((
        gpu,
        Runtime {
            mirror,
            input,
            swapchain,
            session,
            _instance: instance,
            pending: None,
            refresh_rate_request_attempted: false,
        },
    ))
}

impl Runtime {
    pub fn size(&self) -> [u32; 2] {
        [self.session.width, self.session.height]
    }
    pub fn format(&self) -> wgpu::TextureFormat {
        self.swapchain.format
    }
    pub fn display_refresh_rate_hz(&self) -> Option<f32> {
        self.session.display_refresh_rate_hz()
    }

    pub fn begin(&mut self) -> Result<Begin> {
        debug_assert!(self.pending.is_none(), "previous XR frame must be ended");
        while let Some(event) = self.session.poll_events()? {
            if matches!(event, SessionEvent::Exit | SessionEvent::LossPending) {
                return Ok(Begin::Exit);
            }
        }
        if !self.session.session_begun {
            std::thread::sleep(std::time::Duration::from_millis(10));
            return Ok(Begin::Idle);
        }
        let frame = self.session.wait_frame()?;
        if !self.refresh_rate_request_attempted {
            if let Some(current_hz) = self.session.display_refresh_rate_hz() {
                self.refresh_rate_request_attempted = true;
                if (current_hz - 90.0).abs() >= 0.5 {
                    match self.session.request_display_refresh_rate_hz(90.0) {
                        Ok(true) => {}
                        Ok(false) => log::warn!(
                            "[XR] headset is at {current_hz:.1} Hz and its runtime does not offer a 90 Hz mode"
                        ),
                        Err(error) => log::warn!("[XR] could not request a 90 Hz headset mode: {error}"),
                    }
                } else {
                    self.refresh_rate_request_attempted = true;
                }
            }
        }
        self.session.begin_frame()?;
        self.pending = Some(Pending {
            time: frame.predicted_display_time,
            views: Vec::new(),
        });
        let result = (|| {
            if !frame.should_render {
                return Ok(Begin::Idle);
            }
            let (valid, views) = self
                .session
                .locate_raw_views(frame.predicted_display_time)?;
            if !valid.contains(
                openxr::ViewStateFlags::POSITION_VALID | openxr::ViewStateFlags::ORIENTATION_VALID,
            ) || views.is_empty()
            {
                return Ok(Begin::Idle);
            }
            if views.len() > self.swapchain.array_size as usize {
                return Err(XrError::Platform(
                    "view count exceeds the eye swapchain".into(),
                ));
            }
            let Some(input) = self
                .input
                .sample(&self.session, frame.predicted_display_time)?
            else {
                return Ok(Begin::Idle);
            };
            self.pending.as_mut().expect("begun frame").views = views;
            Ok(Begin::Frame(input))
        })();
        if !matches!(result, Ok(Begin::Frame(_))) {
            self.abort()?;
        }
        result
    }

    /// Render all eyes, release the acquired image, and end the frame even if a
    /// draw fails. The raw stage poses go back to OpenXR, never the world poses.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world_from_stage: DMat4,
        origin: DVec3,
        mirror: Option<(&wgpu::TextureView, [u32; 2])>,
        stats: helio::XrMirrorStats,
        mut draw_eye: impl FnMut(usize, &Camera, &wgpu::TextureView) -> helio_core::Result<()>,
    ) -> Result<()> {
        let Some(frame) = self.pending.take() else {
            return Ok(());
        };
        let image = match self.swapchain.acquire_image() {
            Ok(image) => image,
            Err(error) => {
                let _ = self.session.end_frame(frame.time, &self.swapchain, &[]);
                return Err(error);
            }
        };
        let stage = (DMat4::from_translation(-origin) * world_from_stage).as_mat4();
        let result = (|| -> Result<()> {
            for (eye, raw) in frame.views.iter().enumerate() {
                let pose = helio_xr::ViewPose::from_xr(raw, &stage);
                let camera = Camera::from_matrices(
                    pose.view_matrix(),
                    pose.projection(0.05),
                    pose.eye_position,
                    0.05,
                    40_000_000.0,
                );
                draw_eye(eye, &camera, self.swapchain.layer_view(image, eye as u32)?)
                    .map_err(|e| XrError::Platform(format!("eye {eye}: {e}")))?;
            }
            if let Some((view, size)) = mirror {
                self.mirror
                    .draw(device, queue, image, view, size, self.size(), stats);
            }
            Ok(())
        })();
        // Do not block the CPU for full GPU completion here. Helio's standard
        // `Renderer::render_xr` path submits the eye work and releases the same
        // OpenXR image without a device-wide wait; Vulkan queue ordering keeps
        // this frame's rendering ahead of later work on the bound XR queue.
        let released = self.swapchain.present();
        let views = if result.is_ok() && released.is_ok() {
            frame.views.as_slice()
        } else {
            &[]
        };
        let ended = self.session.end_frame(frame.time, &self.swapchain, views);
        result.and(released).and(ended)
    }

    fn abort(&mut self) -> Result<()> {
        if let Some(frame) = self.pending.take() {
            self.session.end_frame(frame.time, &self.swapchain, &[])?;
        }
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.abort();
    }
}
