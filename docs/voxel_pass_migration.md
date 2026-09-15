# Voxel pass-boundary migration

The retained voxel migration is complete for the voxel mesh, voxel raymarch,
and planetary voxel paths:

- `VoxelTerrain` is pass-owned authored input. Mesh uploads and raymarch brick
  uploads are explicit bounded commands; no voxel volume is stored in Helio's
  `Scene` or `GpuScene`.
- `VoxelMeshPass` owns surface extraction metadata, voxel data, dirty-brick
  work, and generated meshlet storage.
- `VoxelRayMarchPass` owns volume descriptors, brick/data pools, the bounded
  edit ring, and their generation ordering.
- Planetary addressing, page contracts, residency, GPU POD layouts, and WGSL
  layout text live under `helio-pass-planetary-voxel`.

The frame infrastructure in `libhelio` was intentionally left untouched. Its
`generic transient resource registry::voxels` slot and `VoxelsFrameData` type are now unused by the
renderer and remain only as a required follow-up interface deletion. Removing
them must be coordinated with any external graph/pass consumers of the frozen
frame ABI; this migration deliberately does not edit `crates/libhelio`.
