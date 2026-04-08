//! GPU instance data for GPU-driven indirect rendering.
//!
//! All geometry in the scene is submitted as a flat array of `GpuInstanceData`.
//! The GPU culling compute shaders read this array and emit `DrawIndexedIndirect`
//! commands — the CPU never iterates the draw list.

use bytemuck::{Pod, Zeroable};

/// Per-instance data for GPU-driven rendering. 144 bytes.
///
/// Uploaded once when instances change (dirty tracking), then read-only on GPU.
/// The vertex shader uses `instance_index` to look up this data from a storage buffer.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuInstanceData {
///     transform:    mat4x4<f32>,  // 64 bytes — model matrix
///     normal_mat_0: vec4<f32>,    // 16 bytes — row 0 of normal matrix
///     normal_mat_1: vec4<f32>,    // 16 bytes — row 1
///     normal_mat_2: vec4<f32>,    // 16 bytes — row 2
///     bounds:       vec4<f32>,    // 16 bytes — bounding sphere
///     mesh_id:      u32,          //  4 bytes
///     material_id:  u32,          //  4 bytes
///     flags:        u32,          //  4 bytes
///     lightmap_index: u32,        //  4 bytes — index into lightmap atlas regions buffer
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuInstanceData {
    /// Model matrix columns 0–3 (column-major, 64 bytes)
    pub model: [f32; 16],
    /// Normal matrix (inverse-transpose of upper-left 3x3, padded to 3×vec4, 48 bytes)
    pub normal_mat: [f32; 12],
    /// Bounding sphere center in world space (xyz) + radius (w)
    pub bounds: [f32; 4],
    /// Mesh index into the global mesh table
    pub mesh_id: u32,
    /// Material index into the global material table
    pub material_id: u32,
    /// Flags (bit 0 = casts_shadow, bit 1 = receives_shadow)
    pub flags: u32,
    /// Index into the lightmap atlas regions buffer (0xFFFFFFFF = no lightmap)
    pub lightmap_index: u32,
}

/// Per-instance AABB in world space for GPU culling. 32 bytes.
///
/// # WGSL equivalent
/// ```wgsl
/// struct GpuAabb {
///     min: vec3<f32>,
///     _pad0: f32,
///     max: vec3<f32>,
///     _pad1: f32,
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuInstanceAabb {
    pub min: [f32; 3],
    pub _pad0: f32,
    pub max: [f32; 3],
    pub _pad1: f32,
}

