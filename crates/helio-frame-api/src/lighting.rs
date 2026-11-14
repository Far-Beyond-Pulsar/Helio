//! What a lighting model reads that is not the G-buffer: generic contracts,
//! never named after the kind that publishes them, each with a stand-in for
//! when nothing does.
//!
//! | Contract | Form | Published today by | Stand-in |
//! |---|---|---|---|
//! | [`INDIRECT_DIFFUSE`] | buffer, 2 words a pixel | GI probes' resolve, then a bake's lightmaps | sky/ambient only |
//! | [`INDIRECT_SPECULAR_LAYERS`] | rgba16f textures, straight alpha, any resolution | SSR (half res), planar reflections | no layer |
//! | [`ENVIRONMENT_PROBES`] + [`ENVIRONMENT_CUBES`] | probe rows + cube array | reflection captures, bake | sky cube only |
//! | [`SUN_VISIBILITY`] / [`DIRECTIONAL_VISIBILITY`] | rgba16f, full res | clouds / terrain | fully visible |
//! | [`sun_transmittance`] | world-space map along the sun + frame | clouds | fully lit |
//! | [`TRANSMISSION`] | rgba16f, internal res | water caustics | nothing added |
//! | [`sky`] | sky frame (medium, suns) + sky table | the sky | no sky |
//! | [`VIEW_REMAPS`] | SceneDB `mat4` per chain + G-buffer tag | portals | no remap |
//!
//! A lighting model binds every contract it uses, the stand-in where it is
//! absent; a publisher writes it at [`crate::Slot::LightingInputs`] (or
//! before the G-buffer when it needs no G-buffer). Neither side knows the
//! other. The lights themselves (their list, shadows, the directional
//! light) are not frame contracts: they are the lights' own, and what reads
//! them reads the lights' api.
//!
//! The model is a plugin's too: it registers a [`LightingModel`] (an id, its
//! passes at its point, what it reads), a preset chooses one by id, and the
//! spine's unlit stand-in ([`model::UNLIT`]) is the model used when the
//! chosen id is not registered.

use helio_feature_api::{BuildEnv, ExtensionPoint, FeatureRegistrar, NodeRegistry};

/// The ids of the lighting models Helio's own presets choose, and of the
/// stand-in that replaces a chosen model nobody registered.
pub mod model {
    /// The spine's stand-in: the G-buffer drawn unlit (registered by the
    /// spine; used when the chosen model is absent).
    pub const UNLIT: &str = "unlit";
    /// A full-screen pass over the G-buffer.
    pub const DEFERRED: &str = "deferred";
    /// The forward phase's drawers light what they draw; nothing resolves a
    /// G-buffer.
    pub const FORWARD: &str = "forward";
}

/// How a lighting model turns geometry into lit colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shading {
    /// The G-buffer phase draws surfaces and the model's nodes at its point
    /// resolve it.
    GBuffer,
    /// The forward phase's drawers light what they draw; nothing resolves a
    /// G-buffer.
    Forward,
}

/// The anti-aliasing a pipeline ends its scene-linear chain with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Antialiasing {
    /// TSR when the renderer's config asks for it, FXAA otherwise.
    TemporalOrFxaa,
    /// TSR when the renderer's config asks for it, none otherwise.
    TemporalOrNone,
    /// FXAA, never TSR.
    Fxaa,
}

/// A pipeline a lighting model brings next to Helio's shipped ones: the
/// model it chooses, an anti-aliasing and the entry a host lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelPipeline {
    /// The pipeline's id in a host's registry.
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    /// The lighting model it chooses.
    pub model: &'static str,
    pub aa: Antialiasing,
}

/// A lighting model, as the plugin that brings it registers it
/// ([`RegisterLightingModel::lighting_model`]): its id, where its passes join
/// the graph, and what it reads of the frame. A preset names the id.
#[derive(Clone, Debug)]
pub struct LightingModel {
    /// The id a preset chooses it by.
    pub id: &'static str,
    /// How it shades.
    pub shading: Shading,
    /// Where its passes are placed (the nodes the registrar placed there with
    /// [`helio_feature_api::NodeDescriptor::at`]). Forward models have none:
    /// their drawers light as they draw.
    pub point: Option<ExtensionPoint>,
    /// How it takes the indirect-specular layers.
    pub specular_layers: SpecularLayers,
    /// Where the frame's lighting inputs run relative to it.
    pub inputs_order: InputsOrder,
    /// Whether it leaves transparency to the reactive mask a temporal
    /// resolve reads (it does not light what draws after it).
    pub reactive_transparency: bool,
    /// Pipelines it brings.
    pub pipelines: Vec<ModelPipeline>,
}

impl LightingModel {
    /// A model of `id` that shades `shading`, with no point of its own, the
    /// layers read before it and no pipelines of its own.
    pub fn new(id: &'static str, shading: Shading) -> Self {
        Self {
            id,
            shading,
            point: None,
            specular_layers: SpecularLayers::Read,
            inputs_order: InputsOrder::BeforeLighting,
            reactive_transparency: false,
            pipelines: Vec::new(),
        }
    }

    /// The model's passes are placed at `point`.
    pub fn at(mut self, point: ExtensionPoint) -> Self {
        self.point = Some(point);
        self
    }

    /// How the model takes the indirect-specular layers.
    pub fn with_specular_layers(mut self, layers: SpecularLayers) -> Self {
        self.specular_layers = layers;
        self
    }

    /// Where the frame's lighting inputs run relative to the model.
    pub fn with_inputs_order(mut self, order: InputsOrder) -> Self {
        self.inputs_order = order;
        self
    }

    /// The model leaves transparency to the reactive mask TSR reads.
    pub fn with_reactive_transparency(mut self) -> Self {
        self.reactive_transparency = true;
        self
    }

    /// The model brings `pipeline`.
    ///
    /// # Panics
    ///
    /// When `pipeline` chooses another model.
    pub fn with_pipeline(mut self, pipeline: ModelPipeline) -> Self {
        assert_eq!(
            pipeline.model, self.id,
            "pipeline `{}` chooses another model",
            pipeline.id
        );
        self.pipelines.push(pipeline);
        self
    }
}

/// What a feature registers a lighting model through.
pub trait RegisterLightingModel {
    /// Registers `model`; a later one with the same id replaces the earlier.
    fn lighting_model(&mut self, model: LightingModel);
}

impl RegisterLightingModel for FeatureRegistrar<'_> {
    fn lighting_model(&mut self, model: LightingModel) {
        self.provide(model);
    }
}

/// Every registered lighting model, one per id (the latest registration), in
/// the order their ids were first registered.
pub fn registered_models(registry: &NodeRegistry) -> Vec<&LightingModel> {
    let mut models: Vec<&LightingModel> = Vec::new();
    for model in registry.provided::<LightingModel>() {
        match models.iter_mut().find(|known| known.id == model.id) {
            Some(known) => *known = model,
            None => models.push(model),
        }
    }
    models
}

/// The registered lighting model `id`, if any.
pub fn registered_model<'r>(registry: &'r NodeRegistry, id: &str) -> Option<&'r LightingModel> {
    registry
        .provided::<LightingModel>()
        .filter(|model| model.id == id)
        .last()
}

/// The lighting model a preset that chose `id` gets: the registered one, or
/// the spine's unlit stand-in when nothing registered `id` (the plugin that
/// brings it is absent), so a host without it still shows its surfaces.
///
/// # Panics
///
/// When the spine's stand-in is not registered either (a registry without
/// the spine builds no graph).
pub fn resolve_model<'r>(registry: &'r NodeRegistry, id: &str) -> &'r LightingModel {
    registered_model(registry, id)
        .or_else(|| registered_model(registry, model::UNLIT))
        .expect("the spine registers the unlit lighting model")
}

/// The id of the lighting model a graph resolves its G-buffer with: a
/// setting its preset supplies (the resolved model, see [`resolve_model`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightingModelId(pub &'static str);

/// The graph's [`LightingModelId`], when its preset said.
pub fn lighting_model(env: &BuildEnv<'_>) -> Option<&'static str> {
    env.setting::<LightingModelId>().map(|id| id.0)
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
    env.setting::<LightingSettings>()
        .copied()
        .unwrap_or_default()
}

/// Indirect diffuse irradiance at each pixel: a storage buffer of
/// `vec2<u32>` per internal-resolution pixel, row-major,
/// `x = pack2x16float(r, g)`, `y = pack2x16float(b, weight)`, `rgb` the
/// irradiance over pi (the radiance a white Lambertian surface reflects).
/// The lighting mixes its own ambient towards `rgb * albedo` by `weight`.
/// Absent (or one element long): the lighting's own ambient alone.
///
/// Publishers run at [`crate::Slot::LightingInputs`] in order; a later one
/// refines the buffer an earlier one published in place (a bake's lightmaps
/// over the GI probes, on the surfaces it baked) and publishes its own only
/// where none was.
pub const INDIRECT_DIFFUSE: &str = "indirect_diffuse";

/// Screen-space indirect specular layers, composited over the environment
/// probes in index order: each is an `Rgba16Float` texture at whatever
/// resolution its publisher traces (SSR: half the internal resolution), `rgb`
/// the reflected radiance before Fresnel and `a` its coverage. A reader takes
/// a layer at a lighting pixel with [`INDIRECT_SPECULAR_SNIPPET`]'s
/// `indirect_specular_at` (never a `textureLoad` at the pixel, which reads a
/// smaller layer at the wrong place) and composites it as
/// `spec * (1 - a) + rgb * F`. A publisher takes the next free layer with
/// [`claim_specular_layer`] when it builds, so publishers built later win
/// where they cover.
pub const INDIRECT_SPECULAR_LAYERS: [&str; 2] = ["indirect_specular_0", "indirect_specular_1"];

/// Marker opting a shader into [`INDIRECT_SPECULAR_SNIPPET`].
pub const INDIRECT_SPECULAR_MARKER: &str = "//!use frame_indirect_specular";
/// `indirect_specular_at(layer, uv)`: an [`INDIRECT_SPECULAR_LAYERS`] layer
/// at a lighting pixel whatever its resolution (premultiplied bilinear).
pub const INDIRECT_SPECULAR_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(INDIRECT_SPECULAR_MARKER, "shaders/indirect_specular.wgsl");

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
pub const ENVIRONMENT_PROBES: &str = "environment_probes";
/// The probes' prefiltered cube array (graph resource) and its sampler.
pub const ENVIRONMENT_CUBES: &str = "environment_cubes";
pub const ENVIRONMENT_CUBES_SAMPLER: &str = "environment_cubes_sampler";

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
/// `mat4x4<f32>`; a pixel belongs to chain `c` when its G-buffer surface tag
/// is `(-1, -3 - c)` (the prelude's `gbuffer_tag_view_remap` and `view_remap_of`). Absent: no
/// pixel is remapped. Bound as a uniform (fragment stages are at their
/// storage-buffer limit).
pub const VIEW_REMAPS: &str = "view_remaps";
/// Chains [`VIEW_REMAPS`] holds.
pub const VIEW_REMAP_CHAINS: u64 = 256;
pub const VIEW_REMAPS_BYTES: u64 = VIEW_REMAP_CHAINS * 64;

/// The sky: what lights a surface from the air above it and what the air
/// takes from the sun on its way, as other kinds read it: the participating
/// medium (Rayleigh, Mie, ozone; aerial perspective, terrain and water read
/// it), the sky's radiance (the sky-view LUT, the sky table, its irradiance)
/// and the suns (the lights' directional lights, packed). Published by the
/// sky plugin, which keeps what only it draws (clouds, stars, a panorama) in
/// a block of its own; read by lighting, GI, fog, water, terrain.
pub mod sky {
    /// Graph key of the per-frame sky buffer (`SkyFrame` in `frame_sky.wgsl`).
    pub const SKY_FRAME: &str = "sky_frame";
    /// Graph key of the sky-view LUT.
    pub const SKY_VIEW: &str = "sky_view";
    /// Bytes of `SkyFrame`. The buffer is usable as a uniform as well
    /// as storage: consumers already at the per-stage storage-buffer limit
    /// (the deferred lighting pass) bind it as a uniform.
    pub const SKY_FRAME_BYTES: u64 = 1872;
    /// Graph key of the sky's irradiance table (`SKY_TABLE_*` in
    /// `frame_sky.wgsl`): the sky over a surface anywhere in the air, by its
    /// altitude and sun angle. Usable as a uniform.
    pub const SKY_TABLE: &str = "sky_table";
    /// Entries of the sky table: 16 altitudes by 24 sun cosines, six vec4s
    /// each.
    pub const SKY_TABLE_ENTRIES: u64 = 16 * 24;
    pub const SKY_TABLE_BYTES: u64 = SKY_TABLE_ENTRIES * 6 * 16;

    /// Marker opting a shader into the frame's sky WGSL.
    pub const SKY_MARKER: &str = "//!use frame_sky";

    /// The frame's sky WGSL (`SkyFrame`, `SkyMedium`, the LUT
    /// parameterisations, transmittance and integration) as a snippet: any
    /// pass reading the published [`SKY_FRAME`] opts in with [`SKY_MARKER`].
    pub const SKY_SNIPPET: helio_core::shader::ShaderSnippet =
        helio_core::wgsl_snippet!(SKY_MARKER, "shaders/frame_sky.wgsl");

    /// Marker opting a shader into [`SKY_TABLE_SNIPPET`].
    pub const SKY_TABLE_MARKER: &str = "//!use sky_table";

    /// The sky over a surface anywhere in the air from the published sky
    /// table (`sky_table_eval`); with [`SKY_SNIPPET`], for a shader binding
    /// the frame as `sky_frame` and the table as `sky_table`. It reads both
    /// through those bindings: a frame
    /// passed by value would be copied whole into per-invocation memory on
    /// every call, since its directional lights are indexed at run time.
    pub const SKY_TABLE_SNIPPET: helio_core::shader::ShaderSnippet =
        helio_core::wgsl_snippet!(SKY_TABLE_MARKER, "shaders/sky_table.wgsl");
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
    /// The lighting does not read them (a light-field model): publishers run after it, at
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
    /// a light-field model).
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
