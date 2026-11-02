//! The sky feature's public contract.
//!
//! What other features read of the sky: the graph keys and sizes of the
//! resources it publishes every frame, the WGSL that reads them, and
//! stand-ins for when nothing publishes them. Holds no state; a feature that
//! reads the sky depends on this crate, never on `helio-pass-sky`.

/// Graph key of the per-frame atmosphere buffer (`AtmosphereFrame`).
pub const ATMOSPHERE_FRAME: &str = "atmosphere_frame";
/// Graph key of the sky-view LUT.
pub const ATMOSPHERE_SKY_VIEW: &str = "atmosphere_sky_view";
/// Bytes of `AtmosphereFrame`. The buffer is usable as a uniform as well as
/// storage: consumers already at the per-stage storage-buffer limit (the
/// deferred lighting pass) bind it as a uniform.
pub const ATMOSPHERE_FRAME_BYTES: u64 = 528;
/// Graph key of the sky's irradiance table (`SKY_TABLE_*` in
/// `atmosphere_common.wgsl`): the sky over a surface anywhere in the air,
/// by its altitude and sun angle. Usable as a uniform.
pub const ATMOSPHERE_SKY_TABLE: &str = "atmosphere_sky_table";
/// Entries of the sky table: 16 altitudes by 24 sun cosines, six vec4s
/// each.
pub const SKY_TABLE_ENTRIES: u64 = 16 * 24;
pub const ATMOSPHERE_SKY_TABLE_BYTES: u64 = SKY_TABLE_ENTRIES * 6 * 16;

/// Marker opting a shader into the shared atmosphere WGSL.
pub const ATMOSPHERE_MARKER: &str = "//!use atmosphere";

/// The atmosphere's shared WGSL (`AtmosphereFrame`, the medium, the LUT
/// parameterisations) as a snippet: any pass reading the published
/// `atmosphere_frame` opts in with [`ATMOSPHERE_MARKER`].
pub const ATMOSPHERE_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(ATMOSPHERE_MARKER, "shaders/atmosphere_common.wgsl");

/// Marker opting a shader into [`ATMOSPHERE_SKY_TABLE_SNIPPET`].
pub const ATMOSPHERE_SKY_TABLE_MARKER: &str = "//!use atmosphere_sky_table";

/// The sky over a surface anywhere in the atmosphere from the published
/// sky table (`atmosphere_sky_table_eval`); with [`ATMOSPHERE_SNIPPET`],
/// for a shader binding the table as `atmosphere_sky_table`.
pub const ATMOSPHERE_SKY_TABLE_SNIPPET: helio_core::shader::ShaderSnippet = helio_core::wgsl_snippet!(
    ATMOSPHERE_SKY_TABLE_MARKER,
    "shaders/atmosphere_sky_table.wgsl"
);

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

/// Graph key of the sun's visibility at each pixel (full resolution,
/// `Rgba16Float`: x the visibility, yzw the sun's direction it holds for),
/// as the voxel terrain's `directional_visibility`, which it includes.
pub const SUN_VISIBILITY: &str = "sun_visibility";
/// Graph key of the terrain's sun visibility a voxel pass publishes.
pub const DIRECTIONAL_VISIBILITY: &str = "directional_visibility";

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
