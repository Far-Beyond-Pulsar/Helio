//! The sun's occlusion by terrain that does not draw into the shadow maps
//! (voxel terrain, which is traced, not rasterised): a camera-centred
//! clipmap of orthographic depth maps along the sun. A terrain pass renders
//! it and publishes [`SUN_OCCLUSION_MAPS`] (`R32Float` texture array, a layer
//! a level: each texel the distance along the sun from its level's near
//! plane to the first occluder, 0 where there is none) and
//! [`SUN_OCCLUSION_FRAME`] (a [`SunOcclusionFrame`] uniform). Passes lighting
//! anything the terrain should shade from the sun (volumetric fog, ...)
//! opt into [`SUN_OCCLUSION_SNIPPET`] (`//!use sun_occlusion`) and multiply
//! the sun by `sun_occlusion(p)`.

use bytemuck::{Pod, Zeroable};

/// Graph key of the depth maps (texture array, a layer a level).
pub const SUN_OCCLUSION_MAPS: &str = "sun_occlusion_maps";
/// Graph key of the frame (uniform buffer, [`SunOcclusionFrame`]).
pub const SUN_OCCLUSION_FRAME: &str = "sun_occlusion_frame";
/// Levels of the clipmap.
pub const SUN_OCCLUSION_LEVELS: usize = 4;
/// Texel format of the maps.
pub const SUN_OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Marker opting a shader into [`SUN_OCCLUSION_SNIPPET`].
pub const SUN_OCCLUSION_MARKER: &str = "//!use sun_occlusion";
/// `SunOcclusionFrame` and `sun_occlusion(p)` (`sun_occlusion.wgsl`).
pub const SUN_OCCLUSION_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(SUN_OCCLUSION_MARKER, "sun_occlusion.wgsl");

/// `SunOcclusionFrame` (`sun_occlusion.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SunOcclusionFrame {
    /// The levels' axes: right (w: the levels published, 0: none), up, and
    /// forward from the sun (w: the maps' side in texels).
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub forward: [f32; 4],
    /// Per level: its near plane's centre relative to the camera, and its
    /// half extent (m).
    pub levels: [[f32; 4]; SUN_OCCLUSION_LEVELS],
}

pub const SUN_OCCLUSION_FRAME_BYTES: u64 = std::mem::size_of::<SunOcclusionFrame>() as u64;
const _: () = assert!(SUN_OCCLUSION_FRAME_BYTES == 112);
