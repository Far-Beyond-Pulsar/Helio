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
//! | [`sun_transmittance`] | world-space map along the sun + frame | clouds | fully lit |
//! | [`TRANSMISSION`] | rgba16f, internal res | water caustics | nothing added |
//! | [`sky`] | atmosphere frame + sky table | the atmosphere | no atmosphere |
//! | [`VIEW_REMAPS`] | SceneDB `mat4` per chain + G-buffer tag | portals | no remap |
//!
//! A lighting model (deferred, forward, HLFS) binds every contract it uses,
//! the stand-in where it is absent; a publisher writes it at
//! [`crate::Slot::LightingInputs`] (or before the G-buffer when it needs no
//! G-buffer). Neither side knows the other. The lights themselves (their
//! list, shadows, the directional light) are not frame contracts: they are
//! the lights' own, and what reads them reads the lights' api.
//!
//! The model is a kind's too: a preset asks for one ([`LightingModel`]) and
//! places what is installed at its point, or the spine's unlit stand-in
//! when nothing is.

use helio_feature_api::{BuildEnv, ExtensionPoint};

/// The lighting model a graph resolves its G-buffer with: a setting its
/// preset supplies. Deferred and light-field models are nodes placed at
/// their [`LightingModel::point`] by whatever owns lighting (the lights
/// plugin); a preset whose point holds nothing draws the G-buffer unlit
/// (`helio_spine::UnlitPass`). How others do it: Bevy's deferred lighting
/// node is `bevy_pbr`'s, not the core pipeline's, and a core pipeline
/// without it shows no lit surfaces; Unity SRP's renderer asset picks
/// forward or deferred and its passes come with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightingModel {
    /// A full-screen pass over the G-buffer.
    Deferred,
    /// Hybrid light-field shading of the G-buffer.
    LightField,
    /// The forward phase's drawers light what they draw (through the
    /// lights' forward function); nothing resolves a G-buffer.
    Forward,
}

impl LightingModel {
    /// Where the model's nodes are placed (`None`: forward, which draws
    /// lit).
    pub fn point(self) -> Option<ExtensionPoint> {
        match self {
            Self::Deferred => Some(ExtensionPoint("lighting.model.deferred")),
            Self::LightField => Some(ExtensionPoint("lighting.model.light_field")),
            Self::Forward => None,
        }
    }
}

/// The graph's [`LightingModel`], when its preset said.
pub fn lighting_model(env: &BuildEnv<'_>) -> Option<LightingModel> {
    env.setting::<LightingModel>().copied()
}

/// What a host asks of whichever lighting model is installed (a graph's
/// builder sets it; Helio's presets from their renderer config).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightingSettings {
    /// The lighting's debug view (0: none).
    pub debug_mode: u32,
    /// Environment-cubemap indirect specular.
    pub environment_reflections: bool,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            debug_mode: 0,
            environment_reflections: true,
        }
    }
}

/// The graph's [`LightingSettings`].
pub fn lighting_settings(env: &BuildEnv<'_>) -> LightingSettings {
    env.setting::<LightingSettings>().copied().unwrap_or_default()
}

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

/// The sun's transmittance through the media above the scene at any point,
/// for what lights points no pixel holds (probes' rays, a medium's froxels);
/// surfaces read the per-pixel [`SUN_VISIBILITY`], which the publisher folds
/// it into. A world-space Beer shadow map along the sun round the camera
/// (`shaders/sun_transmittance.wgsl`): two cascades side by side,
/// [`FORMAT`] texels, and a frame (`SunTransmittanceFrame` in WGSL,
/// [`FRAME_BYTES`]).
///
/// A medium (clouds) publishes its map and frame by its own resource names
/// with [`publish`] when its node builds, before the readers' (at
/// [`crate::Slot::Prepare`]); a reader asks [`current`] when it builds,
/// declares the read and binds them, or [`stand_in`] when nothing
/// published: fully lit. Neither side knows the other.
///
/// [`FORMAT`]: sun_transmittance::FORMAT
/// [`FRAME_BYTES`]: sun_transmittance::FRAME_BYTES
/// [`publish`]: sun_transmittance::publish
/// [`current`]: sun_transmittance::current
/// [`stand_in`]: sun_transmittance::stand_in
pub mod sun_transmittance {
    use helio_feature_api::BuildEnv;

    /// The resources a medium published: the map (a texture view) and its
    /// frame (a uniform buffer).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SunTransmittance {
        pub map: &'static str,
        pub frame: &'static str,
    }

    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub const FRAME_BYTES: u64 = 80;

    /// Marker opting a shader into [`SNIPPET`].
    pub const MARKER: &str = "//!use sun_transmittance";
    /// `SunTransmittanceFrame` and `sun_transmittance_at(p)`, for a shader
    /// binding `sun_transmittance_map` and `sun_transmittance_frame`.
    pub const SNIPPET: helio_core::shader::ShaderSnippet =
        helio_core::wgsl_snippet!(MARKER, "shaders/sun_transmittance.wgsl");

    struct Published(Option<SunTransmittance>);

    /// Publishes the frame's sun transmittance for the nodes built after
    /// this one.
    pub fn publish(env: &mut BuildEnv<'_>, resources: SunTransmittance) {
        env.shared(|| Published(None)).0 = Some(resources);
    }

    /// The sun transmittance nodes built so far published, if any.
    pub fn current(env: &mut BuildEnv<'_>) -> Option<SunTransmittance> {
        env.shared(|| Published(None)).0
    }

    /// What a reader binds while nothing publishes: a 1x1 map and a zeroed
    /// frame (no medium: the sun reaches everywhere).
    pub fn stand_in(device: &wgpu::Device) -> (wgpu::TextureView, wgpu::Buffer) {
        let map = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("No sun transmittance"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let frame = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("No sun transmittance frame"),
            size: FRAME_BYTES,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        (map, frame)
    }
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

/// Where the graph's [`crate::Slot::LightingInputs`] runs relative to its
/// lighting model: a setting its preset supplies (absent:
/// [`InputsOrder::BeforeLighting`]). Either way the slot runs once the depth
/// is complete.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputsOrder {
    /// Before the lighting, which reads what the slot publishes (deferred,
    /// HLFS).
    #[default]
    BeforeLighting,
    /// After a draw phase that lit the scene (forward): only what draws
    /// later reads the slot's results, so a publisher whose one reader is
    /// the lighting (indirect diffuse, transmission) builds nothing, while
    /// one with other readers (clouds seen from the camera, for the sky's
    /// composite) still builds.
    AfterLighting,
}

/// The graph's [`InputsOrder`].
pub fn inputs_order(env: &BuildEnv<'_>) -> InputsOrder {
    env.setting::<InputsOrder>().copied().unwrap_or_default()
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
