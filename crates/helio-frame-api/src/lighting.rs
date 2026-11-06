//! What a lighting model reads that is not the G-buffer: generic contracts,
//! never named after the kind that publishes them, each with a stand-in for
//! when nothing does.
//!
//! | Contract | Form | Published today by | Stand-in |
//! |---|---|---|---|
//! | [`INDIRECT_DIFFUSE`] | buffer, 2 words a pixel | GI probes' resolve | sky/ambient only |
//! | [`INDIRECT_SPECULAR_LAYERS`] | rgba16f textures, straight alpha | SSR, planar reflections | no layer |
//! | [`ENVIRONMENT_PROBES`] + [`ENVIRONMENT_CUBES`] | probe rows + cube array | reflection captures, bake | sky cube only |
//! | [`SUN_VISIBILITY`] / [`DIRECTIONAL_VISIBILITY`] | rgba16f, full res | clouds / terrain | fully visible |
//! | [`TRANSMISSION`] | rgba16f, internal res | water caustics | nothing added |
//! | [`sky`] | atmosphere frame + sky table | the atmosphere | no atmosphere |
//! | [`LIGHT_LIST`] | SceneDB rows | lights | no lights |
//! | [`DIRECTIONAL_LIGHT`] | SceneDB row, direction toward it | lights | none |
//! | [`VIEW_REMAPS`] | SceneDB `mat4` per chain + G-buffer tag | portals | no remap |
//!
//! A lighting model (deferred, forward, HLFS) binds every contract it uses,
//! the stand-in where it is absent; a publisher writes it at
//! [`crate::Slot::LightingInputs`] (or before the G-buffer when it needs no
//! G-buffer). Neither side knows the other.

use helio_feature_api::BuildEnv;

/// Indirect diffuse irradiance at each pixel: a storage buffer of
/// `vec2<u32>` per internal-resolution pixel, row-major,
/// `x = pack2x16float(r, g)`, `y = pack2x16float(b, weight)`. The lighting
/// mixes its own ambient towards `rgb * albedo` by `weight`. Absent (or one
/// element long): the lighting's own ambient alone.
pub const INDIRECT_DIFFUSE: &str = "indirect_diffuse";

/// Screen-space indirect specular layers, composited over the environment
/// probes in index order: each is an `Rgba16Float` texture loaded at the
/// lighting pixel, `rgb` the reflected radiance before Fresnel and `a` its
/// coverage (`spec = mix(spec, rgb * F, a)`). A publisher takes the next free
/// layer with [`claim_specular_layer`] when it builds, so publishers built
/// later win where they cover.
pub const INDIRECT_SPECULAR_LAYERS: [&str; 2] = ["indirect_specular_0", "indirect_specular_1"];

/// Indirect-specular layers claimed by the nodes built so far.
struct ClaimedLayers(usize);

/// The next free indirect-specular layer for a publisher building now, or
/// `None` when every layer is taken.
pub fn claim_specular_layer(env: &mut BuildEnv<'_>) -> Option<&'static str> {
    let claimed = env.shared(|| ClaimedLayers(0));
    let layer = INDIRECT_SPECULAR_LAYERS.get(claimed.0).copied();
    claimed.0 += 1;
    layer
}

/// The latest indirect-specular layer claimed so far (what a composite built
/// after its publisher reads), if any.
pub fn last_specular_layer(env: &mut BuildEnv<'_>) -> Option<&'static str> {
    let claimed = env.shared(|| ClaimedLayers(0)).0;
    claimed
        .checked_sub(1)
        .and_then(|index| INDIRECT_SPECULAR_LAYERS.get(index).copied())
}

/// The environment's parallax probes: SceneDB rows (`ReflectionCaptureRow`
/// layout) each naming a layer of [`ENVIRONMENT_CUBES`]. The base the
/// specular layers composite over.
pub const ENVIRONMENT_PROBES: &str = "reflection_captures";
/// The probes' prefiltered cube array (graph resource) and its sampler.
pub const ENVIRONMENT_CUBES: &str = "baked_reflection";
pub const ENVIRONMENT_CUBES_SAMPLER: &str = "baked_reflection_sampler";

/// The sun's visibility at each pixel (full resolution, `Rgba16Float`: `x`
/// the visibility, `yzw` the sun direction it holds for): occluders' and
/// media's together. Absent: the occluders' [`DIRECTIONAL_VISIBILITY`],
/// else the sun is unoccluded.
pub const SUN_VISIBILITY: &str = "sun_visibility";
/// The occluders' part of the sun's visibility (same form): what media
/// combine into [`SUN_VISIBILITY`].
pub const DIRECTIONAL_VISIBILITY: &str = "directional_visibility";

/// Radiance media add at a surface (light focused through water: caustics),
/// at each internal-resolution pixel: `Rgba16Float`, `rgb` added to the
/// lighting's indirect term. Absent: nothing added.
pub const TRANSMISSION: &str = "transmission";
pub const TRANSMISSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The scene's lights: SceneDB rows (`GpuLight`), whatever kinds of light
/// the lights publish.
pub const LIGHT_LIST: &str = "scene_lights";

/// The lights' directional light (the sun of what reads one: water, media):
/// one SceneDB row, a `vec4<f32>`, the unit direction toward the light and
/// 1, published with the light list. Zero, or absent: no directional light.
pub const DIRECTIONAL_LIGHT: &str = "directional_light";

/// View remaps: per chain, the matrix taking a point of the host world into
/// the world the pixels of that chain show (portals). SceneDB buffer of 256
/// `mat4x4<f32>`; a pixel belongs to chain `c` when its G-buffer lightmap UV
/// is `(-1, -3 - c)` ([`VIEW_REMAP_SNIPPET`]'s `view_remap_of`). Absent: no
/// pixel is remapped. Bound as a uniform (fragment stages are at their
/// storage-buffer limit).
pub const VIEW_REMAPS: &str = "view_remaps";
/// Chains [`VIEW_REMAPS`] holds.
pub const VIEW_REMAP_CHAINS: u64 = 256;
pub const VIEW_REMAPS_BYTES: u64 = VIEW_REMAP_CHAINS * 64;

/// Marker opting a shader into [`VIEW_REMAP_SNIPPET`].
pub const VIEW_REMAP_MARKER: &str = "//!use view_remap";
/// `view_remap_of(lightmap_uv) -> i32`: the remap chain a pixel belongs to,
/// or -1.
pub const VIEW_REMAP_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(VIEW_REMAP_MARKER, "shaders/view_remap.wgsl");

/// The sky: what lights a surface from the air above it and what the air
/// takes from the sun on its way. Published by an atmosphere; read by
/// lighting, GI, fog, water, terrain.
pub mod sky {
    /// Graph key of the per-frame atmosphere buffer (`AtmosphereFrame`).
    pub const ATMOSPHERE_FRAME: &str = "atmosphere_frame";
    /// Graph key of the sky-view LUT.
    pub const ATMOSPHERE_SKY_VIEW: &str = "atmosphere_sky_view";
    /// Bytes of `AtmosphereFrame`. The buffer is usable as a uniform as well
    /// as storage: consumers already at the per-stage storage-buffer limit
    /// (the deferred lighting pass) bind it as a uniform.
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
}

/// How the graph's lighting model takes the indirect-specular layers: a
/// setting its preset supplies (absent: [`SpecularLayers::Read`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpecularLayers {
    /// The lighting composites them itself (deferred): publishers run at
    /// [`crate::Slot::LightingInputs`].
    #[default]
    Read,
    /// The lighting does not read them (HLFS): publishers run after it, at
    /// [`crate::Slot::Lighting`], and add their layer to the lit colour.
    CompositedAfter,
    /// No screen-space layers (forward): publishers build nothing.
    None,
}

/// The graph's [`SpecularLayers`].
pub fn specular_layers(env: &BuildEnv<'_>) -> SpecularLayers {
    env.setting::<SpecularLayers>().copied().unwrap_or_default()
}

/// The lighting's own ambient light where nothing better (the sky, indirect
/// diffuse) covers a surface: a colour and an intensity (a renderer
/// config's `features`; the renderer publishes it in the frame's
/// `RenderEnvironment`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: [0.05, 0.05, 0.08],
            intensity: 1.0,
        }
    }
}
