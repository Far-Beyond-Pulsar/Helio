//! The `wgpu::Instance` Helio's hosts, tests and tools make.

/// The instance descriptor for every Helio host, test and tool: the primary
/// backends (Vulkan, Metal, DX12, browser WebGPU) and never GL.
///
/// Helio can't run on GL (it needs indirect first-instance, storage buffers
/// in every stage, 64-bit integers), and on Windows a GL backend is harmful:
/// wgpu's WGL setup makes a GL context current on the creating thread and on
/// a helper thread, and NVIDIA's driver (616.64 and earlier) deadlocks when
/// such a thread exits while another thread is inside `vkCreateDevice`. The
/// exiting thread holds the Windows loader lock and waits on the driver;
/// device creation holds the driver's lock and calls `LoadLibrary`. Tests
/// that each make an instance and a device on parallel threads hit it.
///
/// `WGPU_BACKEND` still picks backends through `.with_env()`; add flags (for
/// hosts, `helio::recommended_instance_flags()`) or a display handle on top.
pub fn wgpu_instance_descriptor() -> wgpu::InstanceDescriptor {
    wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    }
}
