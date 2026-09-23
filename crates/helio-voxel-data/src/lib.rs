//! Renderer-independent voxel chunk data, revisioned publication, and CPU workers.
//!
//! The SceneDB component owns the canonical payload store. Renderers may read
//! snapshots of that store, but no GPU resource or render pass is owned here.

mod bounded_inbox;
mod chunk_codec;
mod data_api;
mod edit_worker;
mod edits;
mod generation;
mod generation_worker;
mod source_data;

pub use bounded_inbox::{
    BoundedVoxelInbox, VoxelInboxBatch, VoxelInboxClose, VoxelInboxDrain, VoxelInboxDrainBudget,
    VoxelInboxError, VoxelInboxInvalid, VoxelInboxLimits, VoxelPublicationFailure,
    VoxelPublicationFailureReason, VoxelPublicationOutcome, VoxelPublicationStartError,
    VoxelPublicationStatus, VoxelPublicationTicket, VoxelPublicationTicketState,
    VoxelPublicationWorker,
};
pub use chunk_codec::{VoxelChunkCodecError, VoxelMaterialChunk};
pub use data_api::{VoxelBatchReceipt, VoxelPayloadStore, VoxelSourceWriter, VoxelTerrainSnapshot};
pub use edit_worker::{
    VoxelEditAdmissionError, VoxelEditClose, VoxelEditJob, VoxelEditTicket, VoxelEditTicketState,
    VoxelEditWorker, VoxelEditWorkerStatus, VOXEL_EDIT_MAX_SAMPLES_PER_JOB,
    VOXEL_EDIT_PENDING_JOBS,
};
pub use edits::{VoxelEditError, VoxelSampleEdit};
pub use generation::{
    VoxelChunkGenerator, VoxelGeneratorDescriptor, VoxelGeneratorRegistry,
    VOXEL_BUILTIN_GENERATOR_VERSION, VOXEL_FLAT_GENERATOR, VOXEL_PLANET_GENERATOR,
};
pub use generation_worker::{
    VoxelGenerationAdmissionError, VoxelGenerationClose, VoxelGenerationJob, VoxelGenerationStatus,
    VoxelGenerationTicket, VoxelGenerationTicketState, VoxelGenerationWorker,
    VOXEL_GENERATION_MAX_CHUNKS_PER_JOB, VOXEL_GENERATION_PENDING_JOBS,
};
pub use source_data::{
    VoxelBatchRevision, VoxelChunkBatch, VoxelChunkKey, VoxelChunkOp, VoxelChunkPayload,
    VoxelChunkUpdate, VoxelDomain, VoxelSourceId, VoxelTerrainId, VoxelUpdateError,
    MAX_VOXEL_BATCH_PAYLOAD_BYTES, MAX_VOXEL_BATCH_UPDATES, MAX_VOXEL_CHUNK_PAYLOAD_BYTES,
    VOXEL_CHUNK_EDGE, VOXEL_CHUNK_ENCODING_RAW, VOXEL_CHUNK_SAMPLES, VOXEL_CHUNK_SCHEMA_VERSION,
};
