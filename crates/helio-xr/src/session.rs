//! OpenXR session lifecycle, spaces and frame presentation.

use crate::camera::ViewPose;
use crate::graphics::WgpuGraphics;
use crate::{Result, XrError, XrSwapchain};

/// Events produced while pumping OpenXR events, distilled from the raw
/// `SessionStateChanged` / `InstanceLossPending` events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionEvent {
    /// The session reached `READY`; `Session::begin` has been called.
    Ready,
    /// The session is focused; rendering should happen.
    Focused,
    /// The session is idle/synchronized/visible; the runtime is not asking the
    /// app to render right now.
    Idle,
    /// The runtime requested the session to end.
    Exit,
    /// The runtime (or instance) is being lost.
    LossPending,
}

/// Views located for one frame: the raw OpenXR views (poses in the *stage*
/// space, needed for the composition layer) plus per-eye poses transformed
/// into the engine's world space (for building camera uniforms).
pub struct LocatedViews {
    /// Raw views in stage space.
    pub views: Vec<openxr::View>,
    /// Per-eye poses transformed into engine world space.
    pub view_poses: Vec<ViewPose>,
    /// Validity flags for the located views.
    pub view_state_flags: openxr::ViewStateFlags,
}

/// A live OpenXR session with the reference spaces Helio cares about.
pub struct XrSession {
    pub session: openxr::Session<WgpuGraphics>,
    pub frame_waiter: openxr::FrameWaiter,
    pub frame_stream: openxr::FrameStream<WgpuGraphics>,
    /// LOCAL reference space (seated origin).
    pub local: openxr::Space,
    /// STAGE reference space (room-scale, floor-level origin).
    pub stage: openxr::Space,
    pub view_config: openxr::ViewConfigurationType,
    pub environment_blend_mode: openxr::EnvironmentBlendMode,
    pub system: openxr::SystemId,
    /// Recommended swapchain resolution reported by the runtime.
    pub width: u32,
    pub height: u32,
    /// Latest session state (kept up to date by [`XrSession::poll_events`]).
    pub session_state: openxr::SessionState,
    /// Whether the last `wait_frame` said to render.
    pub should_render: bool,
    /// True once `xrBeginSession` has been accepted. The runtime transitions
    /// through SYNCHRONIZED/VISIBLE/FOCUSED in response to the app running its
    /// frame loop, so the renderer must start calling `wait_frame` as soon as
    /// this is set rather than waiting for FOCUSED.
    pub session_begun: bool,
}

impl XrSession {
    /// Create a session bound to the Vulkan handles owned by `device`.
    ///
    /// `device` must have been created through OpenXR (see [`crate::context`]);
    /// the session keeps a clone of the wgpu `Instance` + `Device` alive for as
    /// long as it lives so the underlying Vulkan objects outlive the session.
    pub fn create(
        instance: &openxr::Instance,
        system: openxr::SystemId,
        wgpu_instance: &wgpu::Instance,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Result<Self> {
        let create_info = vulkan_session_create_info(device)?;

        let view_config = pick_view_config(instance, system)?;

        let view_config_views = instance.enumerate_view_configuration_views(system, view_config)?;
        let (width, height) = view_config_views
            .first()
            .map(|v| (v.recommended_image_rect_width.max(1), v.recommended_image_rect_height.max(1)))
            .unwrap_or((1, 1));

        let blend_modes = instance.enumerate_environment_blend_modes(system, view_config)?;
        let environment_blend_mode = blend_modes
            .first()
            .copied()
            .unwrap_or(openxr::EnvironmentBlendMode::OPAQUE);

        let guard = Box::new((wgpu_instance.clone(), device.clone()));
        let (session, frame_waiter, frame_stream) = unsafe {
            instance.create_session_with_guard::<WgpuGraphics>(system, &create_info, guard)?
        };

        let identity = openxr::Posef {
            orientation: openxr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: openxr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let local = session.create_reference_space(openxr::ReferenceSpaceType::LOCAL, identity)?;
        let stage = session.create_reference_space(openxr::ReferenceSpaceType::STAGE, identity)?;

        Ok(Self {
            session,
            frame_waiter,
            frame_stream,
            local,
            stage,
            view_config,
            environment_blend_mode,
            system,
            width,
            height,
            session_state: openxr::SessionState::UNKNOWN,
            should_render: false,
            session_begun: false,
        })
    }

    /// Pump OpenXR events, driving the session state machine.
    ///
    /// Returns the most interesting transition (if any); the caller should
    /// poll this every frame and react to `SessionEvent::Focused` /
    /// `SessionEvent::Exit`.
    pub fn poll_events(&mut self) -> Result<Option<SessionEvent>> {
        let mut buffer = openxr::EventDataBuffer::new();
        loop {
            let event = self.session.instance().poll_event(&mut buffer)?;
            let Some(event) = event else {
                return Ok(None);
            };
            match event {
                openxr::Event::SessionStateChanged(changed) => {
                    let state = changed.state();
                    log::debug!("OpenXR session state changed to {state:?}");
                    self.session_state = state;
                    match state {
                        openxr::SessionState::READY => {
                            log::info!("[XR] session READY — beginning session");
                            match self.session.begin(self.view_config) {
                                Ok(_) => {
                                    self.session_begun = true;
                                    log::info!("[XR] session begun (waiting for SYNCHRONIZED/VISIBLE/FOCUSED)");
                                }
                                Err(e) => {
                                    log::error!("[XR] session begin failed: {e}");
                                }
                            }
                            return Ok(Some(SessionEvent::Ready));
                        }
                        openxr::SessionState::FOCUSED => {
                            return Ok(Some(SessionEvent::Focused));
                        }
                        openxr::SessionState::STOPPING => {
                            self.session.request_exit()?;
                            return Ok(Some(SessionEvent::Exit));
                        }
                        openxr::SessionState::EXITING => {
                            return Ok(Some(SessionEvent::Exit));
                        }
                        openxr::SessionState::LOSS_PENDING => {
                            return Ok(Some(SessionEvent::LossPending));
                        }
                        openxr::SessionState::UNKNOWN
                        | openxr::SessionState::IDLE
                        | openxr::SessionState::SYNCHRONIZED
                        | openxr::SessionState::VISIBLE => {
                            return Ok(Some(SessionEvent::Idle));
                        }
                        _ => {
                            // Future/unknown session states: keep pumping.
                        }
                    }
                }
                openxr::Event::InstanceLossPending(_) => {
                    return Ok(Some(SessionEvent::LossPending));
                }
                _ => {
                    // Interaction-profile, visibility-mask and other events are
                    // not needed by the render loop.
                }
            }
        }
    }

    /// Block until the compositor is ready for a new frame.
    pub fn wait_frame(&mut self) -> Result<openxr::FrameState> {
        let state = self.frame_waiter.wait()?;
        self.should_render = state.should_render;
        Ok(state)
    }

    /// Begin GPU work for the current frame.
    pub fn begin_frame(&mut self) -> Result<()> {
        self.frame_stream.begin()?;
        Ok(())
    }

    /// Locate the per-eye views in the *stage* space and transform them into
    /// engine world space via `world_from_stage`.
    pub fn locate_views(
        &self,
        display_time: openxr::Time,
        world_from_stage: &glam::Mat4,
    ) -> Result<LocatedViews> {
        let (view_state_flags, views) =
            self.session.locate_views(self.view_config, display_time, &self.stage)?;
        let view_poses = views
            .iter()
            .map(|view| ViewPose::from_xr(view, world_from_stage))
            .collect();
        Ok(LocatedViews {
            views,
            view_poses,
            view_state_flags,
        })
    }

    /// Locate raw views (poses in stage space) without the world transform.
    ///
    /// The layer submitted in [`XrSession::end_frame`] is anchored to the stage
    /// space, so these poses are the ones that must be fed back to
    /// `end_frame`.
    pub fn locate_raw_views(
        &self,
        display_time: openxr::Time,
    ) -> Result<(openxr::ViewStateFlags, Vec<openxr::View>)> {
        self.session
            .locate_views(self.view_config, display_time, &self.stage)
            .map_err(Into::into)
    }

    /// Submit the frame for presentation.
    ///
    /// `views` must be the raw views located in the *stage* space (see
    /// [`XrSession::locate_raw_views`]). With a 2-layer swapchain each eye is
    /// assigned its own array layer (Helio's multiview path); with a 1-layer
    /// swapchain the eyes are laid out side-by-side and rendered per-eye.
    pub fn end_frame(
        &mut self,
        display_time: openxr::Time,
        swapchain: &XrSwapchain,
        views: &[openxr::View],
    ) -> Result<()> {
        let blend = self.environment_blend_mode;
        if !self.should_render || views.is_empty() {
            self.frame_stream.end(display_time, blend, &[])?;
            return Ok(());
        }

        let multiview = swapchain.array_size >= 2;
        let sub_images: Vec<openxr::SwapchainSubImage<WgpuGraphics>> = views
            .iter()
            .enumerate()
            .map(|(i, _)| {
                openxr::SwapchainSubImage::new()
                    .swapchain(&swapchain.swapchain)
                    .image_array_index(if multiview { i as u32 } else { 0 })
                    .image_rect(sub_image_rect(
                        multiview,
                        i,
                        views.len(),
                        swapchain.width,
                        swapchain.height,
                    ))
            })
            .collect();

        let projection_views: Vec<openxr::CompositionLayerProjectionView<WgpuGraphics>> = views
            .iter()
            .zip(sub_images)
            .map(|(view, sub_image)| {
                openxr::CompositionLayerProjectionView::new()
                    .pose(view.pose)
                    .fov(view.fov)
                    .sub_image(sub_image)
            })
            .collect();

        let layer = openxr::CompositionLayerProjection::new()
            .space(&self.stage)
            .views(&projection_views);

        self.frame_stream.end(display_time, blend, &[&layer])?;
        Ok(())
    }

    /// Ask the runtime to end the session.
    pub fn request_exit(&self) -> Result<()> {
        self.session.request_exit()?;
        Ok(())
    }

    /// Access the underlying OpenXR instance.
    pub fn instance(&self) -> &openxr::Instance {
        self.session.instance()
    }
}

impl Drop for XrSession {
    fn drop(&mut self) {
        if self.session_state == openxr::SessionState::EXITING {
            return;
        }
        // Best-effort teardown: request exit then end, if the session began.
        if self.session_state == openxr::SessionState::READY
            || self.session_state == openxr::SessionState::FOCUSED
            || self.session_state == openxr::SessionState::VISIBLE
        {
            let _ = self.session.request_exit();
            let _ = self.session.end();
        }
    }
}

/// Extract the raw Vulkan handles wgpu owns and describe them in the form
/// OpenXR expects.
///
/// `device.as_hal::<vulkan::Api>()` returns wgpu-hal's `vulkan::Device`, which
/// exposes the underlying `VkInstance` / `VkPhysicalDevice` / `VkDevice` and
/// the queue family/index of wgpu's single queue.
pub fn vulkan_session_create_info(
    device: &wgpu::Device,
) -> Result<openxr::vulkan::SessionCreateInfo> {
    use ash::vk::Handle as _;

    let hal_device = unsafe { device.as_hal::<wgpu::hal::vulkan::Api>() }.ok_or_else(|| {
        XrError::GraphicsUnavailable("wgpu device is not backed by the Vulkan backend".to_string())
    })?;

    let instance =
        hal_device.shared_instance().raw_instance().handle().as_raw() as *const std::ffi::c_void;
    let physical_device =
        hal_device.raw_physical_device().as_raw() as *const std::ffi::c_void;
    let device = hal_device.raw_device().handle().as_raw() as *const std::ffi::c_void;

    Ok(openxr::vulkan::SessionCreateInfo {
        instance,
        physical_device,
        device,
        queue_family_index: hal_device.queue_family_index(),
        queue_index: hal_device.queue_index(),
    })
}

/// Pick the primary view configuration: stereo if the runtime offers it,
/// otherwise mono.
fn pick_view_config(
    instance: &openxr::Instance,
    system: openxr::SystemId,
) -> Result<openxr::ViewConfigurationType> {
    let configs = instance.enumerate_view_configurations(system)?;
    for preferred in [
        openxr::ViewConfigurationType::PRIMARY_STEREO,
        openxr::ViewConfigurationType::PRIMARY_MONO,
    ] {
        if configs.contains(&preferred) {
            return Ok(preferred);
        }
    }
    Err(XrError::Platform(
        "runtime offers neither PRIMARY_STEREO nor PRIMARY_MONO".to_string(),
    ))
}

/// Compute the image rectangle for eye `i` of `count`.
///
/// With a multi-layer (multiview) swapchain each eye owns its own array layer,
/// so every eye uses the *full* image rect. With a single-layer swapchain the
/// eyes share the image side-by-side, each getting `1/count` of the width.
fn sub_image_rect(
    multiview: bool,
    eye: usize,
    count: usize,
    width: u32,
    height: u32,
) -> openxr::Rect2Di {
    if multiview || count <= 1 {
        return openxr::Rect2Di {
            offset: openxr::Offset2Di { x: 0, y: 0 },
            extent: openxr::Extent2Di {
                width: width as i32,
                height: height as i32,
            },
        };
    }
    let w = (width / count as u32).max(1);
    openxr::Rect2Di {
        offset: openxr::Offset2Di {
            x: (eye as u32 * w) as i32,
            y: 0,
        },
        extent: openxr::Extent2Di {
            width: w as i32,
            height: height as i32,
        },
    }
}
