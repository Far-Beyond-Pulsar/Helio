//! Renderer-independent voxel data: a voxel object's chunks (keys, 8^3
//! material samples, revisioned publication, snapshots, the sample edit
//! worker) and a voxel world's brush journal.
//!
//! The SceneDB component owns the canonical payload store. Renderers may read
//! snapshots of that store, but no GPU resource or render pass is owned here.

mod brush;
mod chunk_codec;
mod data_api;
mod edit_worker;
mod edits;
mod ids;
mod source_data;

pub use brush::{
    VoxelBrushEdit, VoxelBrushOp, VoxelBrushShape, VoxelEditBase, VoxelEditJournal, VoxelKeptChunks,
};
pub use chunk_codec::{VoxelChunkCodecError, VoxelMaterialChunk};
pub use data_api::{
    VoxelBatchReceipt, VoxelPayloadKey, VoxelPayloadStore, VoxelSourceWriter, VoxelStoredPayload,
    VoxelTerrainSnapshot,
};
pub use edit_worker::{
    VoxelEditAdmissionError, VoxelEditClose, VoxelEditJob, VoxelEditTicket, VoxelEditTicketState,
    VoxelEditWorker, VoxelEditWorkerStatus, VOXEL_EDIT_MAX_SAMPLES_PER_JOB,
    VOXEL_EDIT_PENDING_JOBS,
};
pub use edits::{VoxelEditError, VoxelSampleEdit};
pub use ids::{
    VOXEL_MESH_RENDERER, VOXEL_TERRAIN_GENERATOR, VOXEL_TERRAIN_GENERATOR_VERSION,
    VOXEL_TERRAIN_RENDERER,
};
pub use source_data::{
    VoxelBatchRevision, VoxelChunkBatch, VoxelChunkKey, VoxelChunkOp, VoxelChunkPayload,
    VoxelChunkUpdate, VoxelDomain, VoxelFormatDescriptor, VoxelFormatId, VoxelFormatRegistry,
    VoxelSourceId, VoxelTerrainId, VoxelUpdateError, MAX_VOXEL_BATCH_PAYLOAD_BYTES,
    MAX_VOXEL_BATCH_UPDATES, MAX_VOXEL_CHUNK_PAYLOAD_BYTES, MAX_VOXEL_FORMAT_PAYLOAD_BYTES,
    VOXEL_CHUNK_EDGE, VOXEL_CHUNK_ENCODING_RAW, VOXEL_CHUNK_SAMPLES, VOXEL_CHUNK_SCHEMA_VERSION,
};
