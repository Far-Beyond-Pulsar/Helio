#[cfg(webgpu)]
pub mod webgpu;
#[cfg(webgpu)]
pub(crate) use webgpu::{get_browser_gpu_property, ContextWebGpu};
