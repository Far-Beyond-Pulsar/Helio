use crate::{
    api::{impl_deferred_command_buffer_actions, SharedDeferredCommandBufferActions},
    *,
};

/// Handle to a command buffer on the GPU.
///
/// A `CommandBuffer` represents a complete sequence of commands that may be submitted to a command
/// queue with [`Queue::submit`]. A `CommandBuffer` is obtained by recording a series of commands to
/// a [`CommandEncoder`] and then calling [`CommandEncoder::finish`].
///
/// Corresponds to [WebGPU `GPUCommandBuffer`](https://gpuweb.github.io/gpuweb/#command-buffer).
#[derive(Debug)]
pub struct CommandBuffer {
    pub(crate) buffer: dispatch::DispatchCommandBuffer,
    /// Deferred actions recorded at encode time, to run at Queue::submit.
    pub(crate) actions: SharedDeferredCommandBufferActions,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(CommandBuffer: Send, Sync);

impl CommandBuffer {
    #[cfg(custom)]
    /// Returns custom implementation of CommandBuffer (if custom backend and is internally T)
    pub fn as_custom<T: custom::CommandBufferInterface>(&self) -> Option<&T> {
        self.buffer.as_custom()
    }

    // Expose map_buffer_on_submit/on_submitted_work_done on CommandBuffer as well,
    // so callers can schedule after finishing encoding.
    impl_deferred_command_buffer_actions!();
}

/// A command buffer that can be submitted any number of times, with
/// [`Queue::submit_mixed`]. Made with [`CommandBuffer::into_reusable`] from a
/// command buffer whose encoder was marked with
/// [`CommandEncoder::mark_reusable`].
///
/// Every submission is still preceded by wgpu's own transitions from the
/// resources' current states, memory-init clears, and resource-lifetime
/// tracking, exactly as for a normal command buffer; only the encoding (and its
/// validation) is not repeated. Resources it records keep living as long as it
/// does. Dropping it is always fine, including while a submission of it is in
/// flight.
///
/// HELIO PATCH: not part of upstream wgpu. Only the Vulkan and D3D12 backends
/// support it.
#[derive(Debug)]
pub struct ReusableCommandBuffer {
    #[cfg(wgpu_core)]
    pub(crate) inner: wgc::command::ReusableCommandBuffer,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ReusableCommandBuffer: Send, Sync);

impl CommandBuffer {
    /// Turns this command buffer into a [`ReusableCommandBuffer`]. Gives it
    /// back unchanged when that is not possible: its encoder was not marked
    /// reusable (or the backend cannot reuse command buffers), it builds or
    /// uses acceleration structures, it uses a surface texture, or it carries
    /// deferred actions (`map_buffer_on_submit`, `on_submitted_work_done`).
    ///
    /// HELIO PATCH: not part of upstream wgpu.
    pub fn into_reusable(self) -> Result<ReusableCommandBuffer, CommandBuffer> {
        #[cfg(wgpu_core)]
        {
            let has_actions = {
                let actions = self.actions.lock();
                !actions.buffer_mappings.is_empty()
                    || !actions.on_submitted_work_done_callbacks.is_empty()
            };
            if !has_actions {
                if let Some(core) = self.buffer.as_core_opt() {
                    if let Ok(inner) = core.context.command_buffer_make_reusable(core) {
                        return Ok(ReusableCommandBuffer { inner });
                    }
                }
            }
        }
        Err(self)
    }
}
