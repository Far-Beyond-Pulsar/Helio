//! HELIO PATCH: command buffers that can be submitted more than once.
//!
//! A normal [`CommandBuffer`] is consumed by its submission: its raw command
//! buffers and the encoder that owns them go to the queue's lifetime tracker
//! and are reset once the GPU is done. A [`ReusableCommandBuffer`] keeps them.
//! Every submission of it still gets its own small "transit" command buffer,
//! built exactly like the one `Queue::submit` builds for a normal command
//! buffer: memory-init clears, then barriers from the device tracker's current
//! states into the states the retained commands expect. The device tracker is
//! then advanced to the retained commands' end states, so wgpu's own later
//! submissions see the truth, and the submission keeps everything the
//! commands reference alive until the GPU has finished with it.
//!
//! What it does not do: re-validate the commands. They were validated once,
//! when they were encoded; resubmitting only re-checks what can change between
//! submissions (destroyed or mapped resources).

use alloc::{string::String, sync::Arc, vec::Vec};
use core::mem::{self, ManuallyDrop};

use thiserror::Error;

use super::{
    make_error_state, BakedCommands, CommandBuffer, CommandEncoderStatus, EncoderStateError,
    EncodingApi, InnerCommandEncoder,
};
use crate::{
    device::{queue::QueueSubmitError, Device, DeviceError},
    global::Global,
    id,
    init_tracker::BufferInitTrackerAction,
    resource::{BufferMapState, Labeled as _, RawResourceAccess as _, TextureInner},
    snatch::SnatchGuard,
    track::Tracker,
};

use super::memory_init::CommandBufferTextureMemoryActions;

/// The retained, already-encoded part of a [`ReusableCommandBuffer`]. Shared
/// with every in-flight submission of it, so it outlives the last one.
pub(crate) struct ReusableCommands {
    /// Owns the raw command buffers, in submission order.
    pub(crate) encoder: InnerCommandEncoder,
    /// Every resource the commands use, with the states they expect on entry
    /// and leave behind on exit.
    pub(crate) trackers: Tracker,
    buffer_memory_init_actions: Vec<BufferInitTrackerAction>,
    texture_memory_actions: CommandBufferTextureMemoryActions,
    _temp_resources: Vec<crate::device::queue::TempResource>,
    _indirect_draw_validation_resources: crate::indirect_validation::DrawResources,
}

impl core::fmt::Debug for ReusableCommands {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReusableCommands")
            .field("label", &self.encoder.label)
            .field("command_buffers", &self.encoder.list.len())
            .finish()
    }
}

/// A finished command buffer that can be submitted any number of times, made
/// with [`Global::command_buffer_make_reusable`].
#[derive(Debug)]
pub struct ReusableCommandBuffer {
    pub(crate) device: Arc<Device>,
    pub(crate) label: String,
    pub(crate) commands: Arc<ReusableCommands>,
}

crate::impl_resource_type!(ReusableCommandBuffer);
crate::impl_labeled!(ReusableCommandBuffer);
crate::impl_parent_device!(ReusableCommandBuffer);

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ReusableCommandBufferError {
    #[error("The command buffer is not finished, is invalid, or was already submitted")]
    NotFinished,
    #[error(
        "The command encoder was not marked reusable before encoding, or the backend cannot \
         resubmit command buffers"
    )]
    NotReusable,
    #[error("Command buffers that build or use acceleration structures cannot be reused")]
    AccelerationStructures,
    #[error("Command buffers that use surface textures cannot be reused")]
    SurfaceTexture,
    #[error(transparent)]
    Device(#[from] DeviceError),
}

impl CommandBuffer {
    /// Turns this finished command buffer into a [`ReusableCommandBuffer`].
    ///
    /// On an error other than [`ReusableCommandBufferError::Device`], the
    /// command buffer is left untouched and can still be submitted normally.
    pub fn make_reusable(&self) -> Result<ReusableCommandBuffer, ReusableCommandBufferError> {
        let snatch_guard = self.device.snatchable_lock.read();
        let mut status = self.data.lock();
        {
            let CommandEncoderStatus::Finished(data) = &*status else {
                return Err(ReusableCommandBufferError::NotFinished);
            };
            if !data.encoder.reusable {
                return Err(ReusableCommandBufferError::NotReusable);
            }
            if !data.as_actions.is_empty() {
                return Err(ReusableCommandBufferError::AccelerationStructures);
            }
            for texture in data.trackers.textures.used_resources() {
                if let Ok(TextureInner::Surface { .. }) = texture.try_inner(&snatch_guard) {
                    return Err(ReusableCommandBufferError::SurfaceTexture);
                }
            }
        }
        let CommandEncoderStatus::Finished(data) = mem::replace(
            &mut *status,
            make_error_state(EncoderStateError::Submitted),
        ) else {
            unreachable!("checked above");
        };
        drop(status);

        let mut baked = data.into_baked_commands();
        // Query resolves that had to wait for submission become part of the
        // retained commands, once.
        baked.process_deferred_query_set_resolves(&self.device, &snatch_guard)?;
        drop(snatch_guard);

        Ok(ReusableCommandBuffer {
            device: self.device.clone(),
            label: self.label().into(),
            commands: Arc::new(ReusableCommands {
                encoder: baked.encoder,
                trackers: baked.trackers,
                buffer_memory_init_actions: baked.buffer_memory_init_actions,
                texture_memory_actions: baked.texture_memory_actions,
                _temp_resources: baked.temp_resources,
                _indirect_draw_validation_resources: baked.indirect_draw_validation_resources,
            }),
        })
    }
}

impl ReusableCommands {
    /// Re-checks, for one submission, what can change after encoding.
    pub(crate) fn validate(&self, snatch_guard: &SnatchGuard) -> Result<(), QueueSubmitError> {
        for buffer in self.trackers.buffers.used_resources() {
            buffer.check_destroyed(snatch_guard)?;
            match *buffer.map_state.lock() {
                BufferMapState::Idle => (),
                _ => return Err(QueueSubmitError::BufferStillMapped(buffer.error_ident())),
            }
        }
        for texture in self.trackers.textures.used_resources() {
            texture.try_inner(snatch_guard)?;
        }
        for query_set in self.trackers.query_sets.used_resources() {
            query_set.try_raw(snatch_guard)?;
        }
        for bind_group in &self.trackers.bind_groups {
            bind_group.try_raw(snatch_guard)?;
        }
        Ok(())
    }
}

impl BakedCommands {
    /// An empty transit command buffer for one submission of `commands`,
    /// carrying their memory-init actions. The caller records the
    /// initialisation and barriers into it, exactly as for a normal command
    /// buffer's transit pass.
    pub(crate) fn reusable_transit(
        raw: alloc::boxed::Box<dyn hal::DynCommandEncoder>,
        device: &Arc<Device>,
        commands: &ReusableCommands,
    ) -> Self {
        BakedCommands {
            encoder: InnerCommandEncoder {
                raw: ManuallyDrop::new(raw),
                list: Vec::new(),
                device: device.clone(),
                is_open: false,
                api: EncodingApi::InternalUse,
                label: "(wgpu internal) Reusable transit".into(),
                reusable: false,
            },
            trackers: Tracker::new(device.ordered_buffer_usages, device.ordered_texture_usages),
            temp_resources: Vec::new(),
            indirect_draw_validation_resources: crate::indirect_validation::DrawResources::new(
                device.clone(),
            ),
            buffer_memory_init_actions: commands.buffer_memory_init_actions.clone(),
            texture_memory_actions: commands.texture_memory_actions.clone(),
            query_set_writes: Default::default(),
            deferred_query_set_resolves: Vec::new(),
        }
    }
}

/// One entry of [`Global::queue_submit_mixed`].
#[derive(Debug)]
pub enum MixedSubmission<'a> {
    /// A normal command buffer; consumed as by `queue_submit`.
    Once(id::CommandBufferId),
    /// A reusable command buffer; kept, and can be submitted again.
    Reusable(&'a ReusableCommandBuffer),
}

impl Global {
    /// Asks for this encoder's command buffers to be reusable. Must be called
    /// before the encoder is finished. Returns whether the backend supports
    /// it; when it returns `false`, the finished command buffer cannot be made
    /// reusable.
    pub fn command_encoder_mark_reusable(&self, encoder_id: id::CommandEncoderId) -> bool {
        let encoder = self.hub.command_encoders.get(encoder_id);
        let mut status = encoder.data.lock();
        let CommandEncoderStatus::Recording(data) = &mut *status else {
            return false;
        };
        if data.encoder.is_open {
            return false;
        }
        let supported = unsafe { data.encoder.raw.set_reusable(true) };
        data.encoder.reusable = supported;
        supported
    }

    /// See [`CommandBuffer::make_reusable`].
    pub fn command_buffer_make_reusable(
        &self,
        command_buffer_id: id::CommandBufferId,
    ) -> Result<ReusableCommandBuffer, ReusableCommandBufferError> {
        let command_buffer = self.hub.command_buffers.get(command_buffer_id);
        command_buffer.make_reusable()
    }

    /// Like `queue_submit`, but each entry is either a normal command buffer
    /// or a reusable one. Entries execute in order.
    pub fn queue_submit_mixed(
        &self,
        queue_id: id::QueueId,
        submissions: &[MixedSubmission<'_>],
    ) -> Result<crate::SubmissionIndex, (crate::SubmissionIndex, QueueSubmitError)> {
        let queue = self.hub.queues.get(queue_id);
        let command_buffer_guard = self.hub.command_buffers.read();
        let items = submissions
            .iter()
            .map(|submission| match submission {
                MixedSubmission::Once(id) => {
                    crate::device::queue::SubmitItem::Once(command_buffer_guard.get(*id))
                }
                MixedSubmission::Reusable(reusable) => {
                    crate::device::queue::SubmitItem::Reusable(reusable)
                }
            })
            .collect::<Vec<_>>();
        drop(command_buffer_guard);
        queue.submit_items(&items)
    }
}
