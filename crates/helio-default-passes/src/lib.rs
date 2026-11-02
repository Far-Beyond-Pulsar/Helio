mod background;
pub mod environment_join;
pub mod graph_nodes;
pub mod graphs;
pub mod hierarchy_resolve;
pub mod pipelines;
pub mod portal_projection;
pub mod portals;
pub mod presets;
pub mod scene_join;
pub mod texture_means;
pub mod virtual_geometry;
pub mod water_shore;
pub use helio_vg_format::debug_view as vg_debug_view;
pub use virtual_geometry::{
    virtual_geometry_supported, with_virtual_geometry, VgInputsPass, VgStreamBeginPass,
    VgStreamEndPass,
};

use std::sync::Arc;

use helio::RendererConfig;
use helio_pass_billboard::BillboardPass;
use helio_pass_corona::CoronaPass;
use helio_pass_debug_overlay::DebugOverlayPass;
use helio_pass_decal::DecalPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_flare::LensFlarePass;
use helio_pass_foliage_gbuffer::FoliageGBufferPass;
use helio_pass_foliage_place::FoliagePlacePass;
use helio_pass_forward_lit::ForwardLitPass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_gi::{GiMeshTracePass, GiPass, GiResolvePass};
use helio_pass_hiz::HiZBuildPass;
use helio_pass_hlfs::HlfsPass;
use helio_pass_indirect_dispatch::IndirectDispatchPass;
use helio_pass_lens::{LensPass, LENS_COLOR};
use helio_pass_light_cull::LightCullPass;
use helio_pass_object_batch::ObjectBatchPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_perf_overlay::{
    PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayPass,
};
use helio_pass_planar_reflection::PlanarReflectionPass;
use helio_pass_portal_cull::PortalCullPass;
use helio_pass_portal_instances::{PortalEditorOverlayPass, PortalInstancePass, PortalMaskPass};
use helio_pass_postprocess::{
    FogCompositePass, PostProcessPass, PostProcessVolumeBlendPass, FOGGED_HDR, FOGGED_HDR_FORMAT,
};
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_dirty::ShadowDirtyPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_simple_cube::SimpleCubePass;
use helio_pass_sky::{AtmosphereCompositePass, AtmospherePass, CloudTracePass};

use background::BackgroundPass;
use helio_pass_sprite_batch::SpriteBatchPass;
use helio_pass_sprite_cull::SpriteCullPass;
use helio_pass_ssr::SsrPass;
use helio_pass_tsr::TsrPass;
use helio_pass_volumetric_fog::VolumetricFogPass;
use helio_pass_water_sim::WaterSimPass;

/// Spotlight icon embedded at compile time — used as the editor billboard sprite.
static SPOTLIGHT_PNG: &[u8] = include_bytes!("../../../spotlight.png");

fn scene_buffer_or_dummy(
    scene_db: &helio::SceneDbHandle,
    device: &wgpu::Device,
    key: pulsar_scenedb::gpu::BufferKey,
    label: &str,
    fallback_size: u64,
) -> pulsar_scenedb::gpu::BufferHandle {
    scene_db
        .store()
        .resolve_buffer_handle(key)
        .unwrap_or_else(|| pulsar_scenedb::gpu::BufferHandle {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: fallback_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            epoch: 0,
            row_bytes: 0,
            content_generation: 0,
        })
}

/// Which of this crate's passes shader hot reload may swap on their own
/// (see [`helio_core::SwapPolicy`] for what a swap does and why a policy is
/// needed).
///
/// Derived by auditing every node constructor ([`graph_nodes`]) for handles
/// the nodes create and share between passes. Those passes form a group and
/// are swapped together:
///
/// * shadows: `ShadowMatrixPass` + `ShadowDirtyPass` (dirty-flag buffer),
///   `ShadowDirtyPass` + `ShadowPass` (face dirty and geometry count buffers)
/// * `HiZBuildPass` + `OcclusionCullPass` (Hi-Z sampler)
/// * `PortalCullPass` + `PortalInstancePass` (portal output buffers)
/// * `FoliagePlacePass` + `FoliageGBufferPass` (blade arena, tile table,
///   visible blades, indirect buffer)
/// * every perf-overlay pass (one shared `PerfOverlayShared`)
/// * `SpriteCullPass` + `SpriteBatchPass` (draw order and indirect buffers)
///
/// Everything else listed as independent takes only renderer-owned handles
/// (camera, debug camera and cull-stats buffers, debug-draw state, SceneDB,
/// the debug-overlay state) and plain configuration; those are the same objects
/// in the live and the replacement graph. `GBufferPass`, `ForwardLitPass`
/// and `TransparentPass` are left out on purpose: they build part of their
/// WGSL at runtime and keep material/template registrations, so edits to them
/// rebuild the whole graph. So are passes other features add (the voxel
/// terrain's), which this crate cannot vouch for.
///
/// Dropping the replacement's unswapped passes has no side effects beyond the
/// ones a resize already has: none of these constructors spawns a thread,
/// registers into a global or shared registry, or writes a file, and the
/// replacement graph's pipeline cache has no persistence path.
pub(crate) fn default_swap_policy() -> helio_core::SwapPolicy {
    use helio_core::graph::type_name_of as name;
    helio_core::SwapPolicy::new()
        .independent::<ObjectBatchPass>()
        .independent::<IndirectDispatchPass>()
        .independent::<BackgroundPass>()
        .independent::<AtmospherePass>()
        .independent::<GiPass>()
        .independent::<GiMeshTracePass>()
        .independent::<GiResolvePass>()
        .independent::<CloudTracePass>()
        .independent::<AtmosphereCompositePass>()
        .independent::<LightCullPass>()
        .independent::<DecalPass>()
        .independent::<SsrPass>()
        .independent::<helio_pass_ssr::SsrCompositePass>()
        .independent::<PlanarReflectionPass>()
        .independent::<DeferredLightPass>()
        .independent::<HlfsPass>()
        .independent::<PortalMaskPass>()
        .independent::<PortalEditorOverlayPass>()
        .independent::<BillboardPass>()
        .independent::<CoronaPass>()
        .independent::<WaterSimPass>()
        .independent::<helio::DebugDrawPass>()
        .independent::<DebugOverlayPass>()
        .independent::<PostProcessVolumeBlendPass>()
        .independent::<VolumetricFogPass>()
        .independent::<FogCompositePass>()
        .independent::<TsrPass>()
        .independent::<FxaaPass>()
        .independent::<LensPass>()
        .independent::<LensFlarePass>()
        .independent::<PostProcessPass>()
        .group_types(&[
            name::<ShadowMatrixPass>(),
            name::<ShadowDirtyPass>(),
            name::<ShadowPass>(),
        ])
        .group_types(&[name::<HiZBuildPass>(), name::<OcclusionCullPass>()])
        .group_types(&[name::<PortalCullPass>(), name::<PortalInstancePass>()])
        .group_types(&[name::<FoliagePlacePass>(), name::<FoliageGBufferPass>()])
        .group_types(&[
            name::<PerfOverlayAnalyzerPass>(),
            name::<PerfOverlayCostAnalyzerPass>(),
            name::<PerfOverlayPass>(),
        ])
        .group_types(&[name::<SpriteCullPass>(), name::<SpriteBatchPass>()])
}

/// Most sprites the 2D overlay draws at once.
const MAX_VISIBLE_SPRITES: u32 = 16_384;

fn convert_perf_mode(mode: helio::PerfOverlayMode) -> helio_pass_perf_overlay::PerfOverlayMode {
    use helio::PerfOverlayMode as H;
    use helio_pass_perf_overlay::PerfOverlayMode as P;
    match mode {
        H::Disabled => P::Disabled,
        H::PassOverdraw => P::PassOverdraw,
        H::ShaderComplexity => P::ShaderComplexity,
        H::TileLightCount => P::TileLightCount,
        H::PassOutput => P::PassOutput,
    }
}

/// The foliage passes `config` asks for: placement, the boxes expansion (between
/// placement and the G-buffer) and the G-buffer draw (after it), which share the
/// placement's buffers.
pub(crate) fn foliage_passes(
    device: &wgpu::Device,
    config: &RendererConfig,
) -> (
    FoliagePlacePass,
    helio_pass_foliage_gbuffer::FoliageBoxesPass,
    FoliageGBufferPass,
) {
    let foliage = config
        .features
        .get_or_default::<helio_pass_foliage_place::FoliageSettings>();
    let defaults = helio_pass_foliage_place::FoliageRange::default();
    let place = FoliagePlacePass::new_with(
        device,
        foliage.quality,
        helio_pass_foliage_place::FoliageRange {
            range: foliage.reach(),
            tufts_per_m2: foliage.tufts_per_m2.unwrap_or(defaults.tufts_per_m2),
            blades_per_tuft: foliage.blades_per_tuft.unwrap_or(defaults.blades_per_tuft),
            ..defaults
        },
    );
    let mut draw = FoliageGBufferPass::new(
        device,
        Arc::clone(&place.blade_arena),
        Arc::clone(&place.tile_table),
        Arc::clone(&place.tile_frames),
        Arc::clone(&place.visible_blades),
        Arc::clone(&place.foliage_indirect),
        Arc::clone(&place.bands_buffer),
        Arc::clone(&place.interactors),
    );
    draw.quality = foliage.quality;
    let boxes = draw.boxes_pass(device);
    (place, boxes, draw)
}

/// Draws `terrain` in the graphs `config` builds: the voxel node adds its
/// chunk passes, and GI traces its probes against it (without a TLAS). A
/// terrain without passes leaves GI tracing the scene's meshes.
pub fn with_voxel_terrain(
    mut config: RendererConfig,
    terrain: helio_pass_voxel_planet::VoxelTerrain,
) -> RendererConfig {
    let traced = !terrain.passes.is_empty();
    config.features.insert(terrain);
    config
        .features
        .edit::<helio_pass_gi::GiSettings>()
        .terrain_traced = traced;
    config
}

/// A renderer config with a host's [`helio::helio_settings::RENDERER`]
/// values: the renderer's own, and the default features' settings
/// (foliage). Engines build their config with this rather than
/// `RendererConfig::with_settings`, which knows no feature.
pub fn config_with_host_settings(
    config: RendererConfig,
    settings: &helio::helio_settings::Resolved,
) -> RendererConfig {
    let mut config = config.with_settings(settings);
    let foliage = config
        .features
        .get_or_default::<helio_pass_foliage_place::FoliageSettings>()
        .with_host_settings(settings);
    config.features.insert(foliage);
    config
        .features
        .edit::<helio_pass_gi::GiSettings>()
        .probe_spacing = settings.optional("gi_probe_spacing").map(|m| m as f32);
    config
}
