//! The sky feature's public contract.
//!
//! What other features read of the sky: the graph keys and sizes of the
//! resources it publishes every frame, the WGSL that reads them, and
//! stand-ins for when nothing publishes them. Holds no state; a feature that
//! reads the sky depends on this crate, never on `helio-pass-sky`.

// The sky's lighting inputs (the atmosphere frame and sky table) are the
// frame's `sky` contract; the sky publishes them.
pub use helio_frame_api::lighting::sky::*;
pub use helio_frame_api::lighting::{DIRECTIONAL_VISIBILITY, SUN_VISIBILITY};

/// Graph key of the clouds' shadow map (texture, [`CLOUD_SHADOW_FORMAT`]).
pub const CLOUD_SHADOW_MAP: &str = "cloud_shadow_map";
/// Graph key of the map's frame (uniform buffer, `CloudShadowFrame` in
/// `shaders/cloud_shadow.wgsl`).
pub const CLOUD_SHADOW_FRAME: &str = "cloud_shadow_frame";
pub const CLOUD_SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const CLOUD_SHADOW_FRAME_BYTES: u64 = 80;
/// Texels along each side of each of the shadow map's cascades (the map is
/// two of them wide).
pub const CLOUD_SHADOW_SIDE: u32 = 256;
/// Marker opting a shader into [`CLOUD_SHADOW_SNIPPET`].
pub const CLOUD_SHADOW_MARKER: &str = "//!use cloud_shadow";
/// `CloudShadowFrame` and `cloud_shadow(p)` (`shaders/cloud_shadow.wgsl`).
pub const CLOUD_SHADOW_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(CLOUD_SHADOW_MARKER, "shaders/cloud_shadow.wgsl");

/// Placeholder resources for a consumer of the clouds' shadow while nothing
/// publishes it: a 1x1 map and a zeroed frame (no clouds).
pub fn cloud_shadow_fallback(device: &wgpu::Device) -> (wgpu::TextureView, wgpu::Buffer) {
    let map = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("No cloud shadow"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CLOUD_SHADOW_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let frame = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("No cloud shadow frame"),
        size: CLOUD_SHADOW_FRAME_BYTES,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    (map, frame)
}
