use std::sync::Arc;

use helio::DebugDrawState;
use helio::GraphRebuilder;
use helio::RendererConfig;
use helio_pass_billboard::BillboardPass;
use helio_pass_corona::CoronaPass;
use helio_pass_debug_overlay::{DebugOverlayPass, DebugOverlayState};
use helio_foliage_core::FoliageQuality;
use helio_pass_flare::LensFlarePass;
use helio_pass_forward_lit::ForwardLitPass;
use helio_pass_foliage_gbuffer::FoliageGBufferPass;
use helio_pass_foliage_place::FoliagePlacePass;
use helio_pass_decal::DecalPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_hiz::HiZBuildPass;
use helio_pass_hlfs::HlfsPass;
use helio_pass_indirect_dispatch::IndirectDispatchPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_perf_overlay::{
    PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass, PerfOverlayPass, PerfOverlayShared,
};
use helio_pass_planar_reflection::PlanarReflectionPass;
use helio_pass_postprocess::{PostProcessPass, PostProcessVolumeBlendPass};
use helio_pass_radiance_cascades::RadianceCascadesPass;
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_cull::ShadowCullPass;
use helio_pass_shadow_dirty::ShadowDirtyPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_simple_cube::SimpleCubePass;
use helio_pass_sky::SkyPass;
use helio_pass_sky_lut::SkyLutPass;
use helio_pass_ssr::SsrPass;
use helio_pass_taa::TaaPass;
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_pass_volumetric_fog::VolumetricFogPass;
use helio_pass_voxel_mesh::VoxelMeshPass;
use helio_pass_water_sim::WaterSimPass;

use helio_core::RenderGraph;

use helio::Scene;

/// Spotlight icon embedded at compile time — used as the editor billboard sprite.
static SPOTLIGHT_PNG: &[u8] = include_bytes!("../../../spotlight.png");

/// Create a new graph, honouring the caller's device ownership.
///
/// When `config.enable_xr` the graph is put into OpenXR multiview mode:
/// every pool texture is allocated as a 2-layer array and the executor forces
/// `multiview_mask = 0b11` on all render passes. Note that the graph's internal
/// resolution stays `config.internal_width()/internal_height()` — in XR mode
/// the application is expected to size `RendererConfig` to the eye resolution
/// reported by the OpenXR runtime (via `XrSession::width`/`height`), since the
/// graph does not talk to the runtime itself.
fn new_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    owns_device: bool,
    config: &RendererConfig,
) -> RenderGraph {
    let mut graph = if owns_device {
        RenderGraph::new(device, queue)
    } else {
        RenderGraph::new_with_external_device(device, queue)
    };
    graph.with_xr_mode(config.enable_xr);
    graph
}

fn add_common_early_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: &RendererConfig,
    cull_stats_buf: &wgpu::Buffer,
    w: u32,
    h: u32,
) -> Arc<std::sync::Mutex<PerfOverlayShared>> {
    let gpu_scene = scene.gpu_scene();
    let camera_buf = gpu_scene.camera.buffer();

    let hiz_pass = HiZBuildPass::new(device, queue, w, h);
    let hiz_sampler = Arc::clone(&hiz_pass.hiz_sampler);

    let shadow_dirty_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Flags"),
        size: 42 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let shadow_hashes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Hashes"),
        size: 42 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    graph.add_pass(Box::new(ShadowMatrixPass::new(
        device,
        gpu_scene.lights.buffer(),
        gpu_scene.shadow_matrices.buffer(),
        camera_buf,
        &shadow_dirty_buf,
        &shadow_hashes_buf,
        config.shadow_atlas_size,
    )));

    let shadow_dirty_pass = ShadowDirtyPass::new(device, Arc::clone(&shadow_dirty_buf));
    let face_dirty_buf = Arc::clone(&shadow_dirty_pass.face_dirty_buf);
    let face_geom_count_buf = Arc::clone(&shadow_dirty_pass.face_geom_count_buf);
    graph.add_pass(Box::new(shadow_dirty_pass));

    let shadow_cull_pass = ShadowCullPass::new(device, Arc::clone(&face_dirty_buf));
    let face_cull_indirect = Arc::clone(&shadow_cull_pass.face_indirect_buf);
    let face_cull_counts = Arc::clone(&shadow_cull_pass.face_counts_buf);
    graph.add_pass(Box::new(shadow_cull_pass));

    graph.add_pass(Box::new(ShadowPass::new(
        device,
        queue,
        face_dirty_buf,
        face_geom_count_buf,
        face_cull_indirect,
        face_cull_counts,
        config.shadow_atlas_size,
        config.shadow_face_capacity,
    )));

    if scene.sky_context().has_sky {
        graph.add_pass(Box::new(SkyLutPass::new(device, camera_buf)));

        graph.add_pass(Box::new(SkyPass::new(
            device,
            camera_buf,
            config.surface_format,
        )));
    }

    graph.add_pass(Box::new(IndirectDispatchPass::new(
        device,
        cull_stats_buf.clone(),
    )));
    graph.add_pass(Box::new(hiz_pass));
    let mut occlusion_cull =
        OcclusionCullPass::new(device, hiz_sampler, w, h, cull_stats_buf.clone());
    if let Some(meta) = graph
        .find_pass::<HiZBuildPass>()
        .and_then(|p| p.static_hiz_metadata())
    {
        occlusion_cull.set_static_hiz_metadata(
            meta.world_bounds_min,
            meta.world_bounds_max,
            meta.grid_resolution,
        );
    }
    graph.add_pass(Box::new(occlusion_cull));

    let perf_overlay_shared = PerfOverlayShared::new(device, w, h);
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(
        &perf_overlay_shared,
    ))));

    perf_overlay_shared
}

fn add_geometry_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    scene: &Scene,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
) {
    let camera_buf = scene.gpu_scene().camera.buffer();

    // Foliage placement is a compute pass and must be added *before* GBufferPass, not
    // between it and FoliageGBufferPass. It is deliberately not `chain_transparent` (it
    // records on the main encoder so it reads this frame's Hi-Z rather than last
    // frame's — plan §6.2), which means the chain scan cannot skip over it: sitting
    // between the two raster passes it would break the very subpass fusion
    // FoliageGBufferPass exists to join. Nothing renders wrong either way, which is
    // exactly why this is worth a comment — the cost is a silent tile store/reload.
    let foliage_buffers = config.enable_foliage.then(|| {
        let place_pass = FoliagePlacePass::new_with_density(
            device,
            FoliageQuality::default(),
            config.foliage_blades_per_m2,
        );
        let handles = (
            Arc::clone(&place_pass.blade_arena),
            Arc::clone(&place_pass.tile_table),
            Arc::clone(&place_pass.visible_blades),
            Arc::clone(&place_pass.foliage_indirect),
            place_pass.blades_per_tile(),
        );
        graph.add_pass(Box::new(place_pass));
        handles
    });

    graph.add_pass(Box::new(GBufferPass::new(device)));

    // Foliage rasterisation goes immediately after GBufferPass and before
    // VirtualGeometry. After GBuffer because it composites into the same eight targets
    // with LoadOp::Load; before VirtualGeometry because chain formation requires an exact
    // attachment-view match and VG binds only seven attachments — it omits
    // gbuffer_velocity — so anything downstream of VG can never fuse with the G-buffer.
    if let Some((blade_arena, tile_table, visible_blades, foliage_indirect, blades_per_tile)) =
        foliage_buffers
    {
        graph.add_pass(Box::new(FoliageGBufferPass::new(
            device,
            blade_arena,
            tile_table,
            visible_blades,
            foliage_indirect,
            blades_per_tile,
        )));
    }

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));
}

fn add_forward_geometry_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    scene: &Scene,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    render_all_opaque: bool,
) {
    let camera_buf = scene.gpu_scene().camera.buffer();

    let mut fl_pass = ForwardLitPass::new(device, config.surface_format);
    fl_pass.render_all_opaque = render_all_opaque;
    graph.add_pass(Box::new(fl_pass));

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));
}

fn add_late_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    w: u32,
    h: u32,
) {
    let camera_buf = scene.gpu_scene().camera.buffer();

    let spotlight = image::load_from_memory(SPOTLIGHT_PNG)
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (sw, sh) = spotlight.dimensions();
    let mut billboard_pass = BillboardPass::new_with_sprite_rgba(
        device,
        queue,
        camera_buf,
        config.surface_format,
        spotlight.as_raw(),
        sw,
        sh,
    );
    billboard_pass.set_occluded_by_geometry(true);
    graph.add_pass(Box::new(billboard_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    graph.add_pass(Box::new(CoronaPass::new(
        device,
        queue,
        camera_buf,
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    graph.add_pass(Box::new(WaterSimPass::new(
        device,
        camera_buf,
        w,
        h,
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    // Editor overlay: grid, and the wireframe bounds of every scene volume.
    //
    // Position is load-bearing in both directions. It must come after
    // DeferredLightPass, which writes the same "pre_aa" target and would
    // otherwise paint over it — that is why the grid never appeared while this
    // sat with the early passes. It must also come before FXAA/post-process
    // consume pre_aa, or it would be drawn into an image nothing reads again.
    //
    // Depth-tested against the internal-res scene depth so geometry genuinely
    // in front occludes the bounds, exactly as BillboardPass does.
    graph.add_pass(Box::new(helio::DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state,
        true,
        true,
    )));
}

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

fn add_final_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) {
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    let mut perf_overlay_pass =
        PerfOverlayPass::new(device, Arc::clone(perf), config.surface_format);
    perf_overlay_pass.set_mode(convert_perf_mode(config.perf_overlay_mode));
    graph.add_pass(Box::new(perf_overlay_pass));

    // User debug lines/tris, drawn at output resolution over the final image.
    // The editor overlay is a separate instance added in add_late_passes,
    // because it needs the internal-res scene depth to occlude correctly.
    graph.add_pass(Box::new(helio::DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state,
        false,
        false,
    )));

    if let Some(shared) = debug_overlay {
        graph.add_pass(Box::new(DebugOverlayPass::new(
            device,
            queue,
            Arc::clone(shared),
            config.surface_format,
            config.width,
            config.height,
        )));
    }
}

pub fn build_default_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_default_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        None,
    )
}

pub fn build_default_graph_with_user_effects(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    user_effects: &'static str,
) -> RenderGraph {
    build_default_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
        Some(user_effects),
    )
}

pub fn build_default_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_default_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
        None,
    )
}

fn build_default_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
    user_effects: Option<&'static str>,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);

    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        cull_stats_buf,
        iw,
        ih,
    );

    graph.add_pass(Box::new(LightCullPass::new(device, iw, ih)));

    graph.add_pass(Box::new(RadianceCascadesPass::new(
        device,
        scene.gpu_scene().lights.buffer(),
    )));

    add_geometry_passes(&mut graph, device, scene, &config, &perf);

    let camera_buf = scene.gpu_scene().camera.buffer();

    // Decal pass — projects decals into the G-buffer after it's been written.
    // Runs as a compute pass between GBuffer and deferred lighting.
    let decal_buf = scene.gpu_scene().decals.buffer();
    graph.add_pass(Box::new(DecalPass::new(
        device, queue, decal_buf, camera_buf, iw, ih,
    )));

    // SSR pass — screen-space reflections for glossy/metallic surfaces.
    // Runs after GBuffer (needs normals + depth + Hi-Z), before deferred lighting.
    //
    // Off by default: this is one of the graph's most expensive passes. When it
    // is absent DeferredLightPass binds its 1×1 black fallback for `ssr_trace`,
    // so the only loss is the reflection contribution. Also compiled out on
    // Apple targets entirely (REFLECTIONS_SUPPORTED), which render it incorrectly.
    if config.enable_ssr && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(SsrPass::new(device, queue, camera_buf, iw, ih)));
    }

    // Planar reflection pass — reflects the scene across world-space planes.
    // Runs before deferred lighting so DeferredLightPass can composite its
    // output alongside SSR (planar_reflection texture) in a single draw call.
    //
    // Off by default: cost scales with scene complexity times reflection-plane
    // count. DeferredLightPass falls back to a 1×1 black `planar_reflection`.
    // Also compiled out on Apple targets (REFLECTIONS_SUPPORTED) — see SSR above.
    if config.enable_planar_reflections && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(PlanarReflectionPass::new(
            device,
            camera_buf,
            config.surface_format,
        )));
    }

    let mut deferred_light_pass =
        DeferredLightPass::new(device, queue, camera_buf, config.surface_format);
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    deferred_light_pass.set_env_reflections(config.enable_environment_reflections);
    graph.add_pass(Box::new(deferred_light_pass));
    graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(perf.clone())));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(perf.clone())));

    // Voxel mesh pass — real triangles with depth testing, composited over
    // deferred lighting. When no voxel volumes are present the pass is a no-op
    // (extract pass has zero dirty bricks → no geometry emitted).
    graph.add_pass(Box::new(VoxelMeshPass::new_composited(
        device,
        queue,
        config.surface_format,
    )));

    add_late_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        iw,
        ih,
    );

    // Before AA, at internal resolution: fog accumulates against internal-res
    // depth, and the AA pass then resolves it with the rest of the frame.
    graph.add_pass(Box::new(PostProcessVolumeBlendPass::new(device)));
    graph.add_pass(Box::new(VolumetricFogPass::new(device)));

    // Transparent pass — alpha-blended geometry (simple fixed shader).
    let camera_buf = scene.gpu_scene().camera.buffer();
    let instances_buf = scene.gpu_scene().instances.buffer();
    graph.add_pass(Box::new(helio_pass_transparent::TransparentPass::new(device, camera_buf, instances_buf, config.surface_format)));

    graph.add_pass(Box::new(LensFlarePass::new(
        device,
        queue,
        scene.gpu_scene().lights.buffer(),
        iw,
        ih,
        config.surface_format,
    )));

    graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));

    graph.add_pass(Box::new(PostProcessPass::new_with_user_effects(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
        user_effects,
    )));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(iw, ih);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let effect_snippet = user_effects;
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, scene, config, debug_state, debug_camera_buf, cull_stats_buf| {
            build_default_graph_internal(
                device,
                queue,
                scene,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
                effect_snippet,
            )
        },
    );
    graph.set_graph_data(rebuilder);

    graph
}

pub fn build_fxaa_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_fxaa_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
    )
}

pub fn build_fxaa_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_fxaa_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
    )
}

fn build_fxaa_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);

    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        cull_stats_buf,
        iw,
        ih,
    );

    graph.add_pass(Box::new(LightCullPass::new(device, iw, ih)));

    graph.add_pass(Box::new(RadianceCascadesPass::new(
        device,
        scene.gpu_scene().lights.buffer(),
    )));

    add_geometry_passes(&mut graph, device, scene, &config, &perf);

    let camera_buf = scene.gpu_scene().camera.buffer();

    // Decal pass
    let decal_buf = scene.gpu_scene().decals.buffer();
    graph.add_pass(Box::new(DecalPass::new(
        device, queue, decal_buf, camera_buf, iw, ih,
    )));

    // Both off by default; see the notes in the primary graph builder above.
    // DeferredLightPass binds 1×1 black fallbacks when either pass is absent.
    // Also compiled out on Apple targets (REFLECTIONS_SUPPORTED).
    if config.enable_ssr && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(SsrPass::new(device, queue, camera_buf, iw, ih)));
    }

    if config.enable_planar_reflections && helio_core::REFLECTIONS_SUPPORTED {
        graph.add_pass(Box::new(PlanarReflectionPass::new(
            device,
            camera_buf,
            config.surface_format,
        )));
    }

    let mut deferred_light_pass =
        DeferredLightPass::new(device, queue, camera_buf, config.surface_format);
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    deferred_light_pass.set_env_reflections(config.enable_environment_reflections);
    graph.add_pass(Box::new(deferred_light_pass));
    graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(Arc::clone(
        &perf,
    ))));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf))));

    add_late_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        iw,
        ih,
    );

    // Before TAA, at internal resolution. Fog accumulates in the same space as the
    // depth it reads, and TAA then resolves it along with everything else — which
    // is why the pass needs no jitter handling of its own.
    graph.add_pass(Box::new(PostProcessVolumeBlendPass::new(device)));
    graph.add_pass(Box::new(VolumetricFogPass::new(device)));

    graph.add_pass(Box::new(TaaPass::new(
        device,
        iw,
        ih,
        config.width,
        config.height,
        config.surface_format,
    )));

    graph.add_pass(Box::new(PostProcessPass::new_with_user_effects(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
        None,
    )));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(iw, ih);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, scene, config, debug_state, debug_camera_buf, cull_stats_buf| {
            build_fxaa_graph_internal(
                device,
                queue,
                scene,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
            )
        },
    );
    graph.set_graph_data(rebuilder);

    graph
}

fn build_hlfs_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);

    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        cull_stats_buf,
        iw,
        ih,
    );

    add_geometry_passes(&mut graph, device, scene, &config, &perf);

    let camera_buf = scene.gpu_scene().camera.buffer();

    // Decal pass
    let decal_buf = scene.gpu_scene().decals.buffer();
    graph.add_pass(Box::new(DecalPass::new(
        device, queue, decal_buf, camera_buf, iw, ih,
    )));

    let mut hlfs_pass = HlfsPass::new(device, queue, iw, ih, config.surface_format);
    hlfs_pass.set_shadow_quality(config.shadow_quality, queue);
    graph.add_pass(Box::new(hlfs_pass));

    add_late_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        iw,
        ih,
    );

    // Before TAA, at internal resolution. Fog accumulates in the same space as the
    // depth it reads, and TAA then resolves it along with everything else — which
    // is why the pass needs no jitter handling of its own.
    graph.add_pass(Box::new(PostProcessVolumeBlendPass::new(device)));
    graph.add_pass(Box::new(VolumetricFogPass::new(device)));

    graph.add_pass(Box::new(TaaPass::new(
        device,
        iw,
        ih,
        config.width,
        config.height,
        config.surface_format,
    )));

    graph.add_pass(Box::new(PostProcessPass::new_with_user_effects(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
        None,
    )));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(iw, ih);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, scene, config, debug_state, debug_camera_buf, cull_stats_buf| {
            build_hlfs_graph_internal(
                device,
                queue,
                scene,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
            )
        },
    );
    graph.set_graph_data(rebuilder);

    graph
}

pub fn build_hlfs_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_hlfs_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
    )
}

pub fn build_fxaa_hlfs_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_fxaa_hlfs_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
    )
}

pub fn build_fxaa_hlfs_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_fxaa_hlfs_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
    )
}

fn build_fxaa_hlfs_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    let w = config.width;
    let h = config.height;

    let mut graph = new_graph(device, queue, owns_device, &config);

    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        cull_stats_buf,
        w,
        h,
    );

    add_geometry_passes(&mut graph, device, scene, &config, &perf);

    let camera_buf = scene.gpu_scene().camera.buffer();

    // Decal pass
    let decal_buf = scene.gpu_scene().decals.buffer();
    graph.add_pass(Box::new(DecalPass::new(
        device, queue, decal_buf, camera_buf, w, h,
    )));

    let mut hlfs_pass = HlfsPass::new(device, queue, w, h, config.surface_format);
    hlfs_pass.set_shadow_quality(config.shadow_quality, queue);
    graph.add_pass(Box::new(hlfs_pass));

    add_late_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        w,
        h,
    );

    // Before AA, at internal resolution: fog accumulates against internal-res
    // depth, and the AA pass then resolves it with the rest of the frame.
    graph.add_pass(Box::new(PostProcessVolumeBlendPass::new(device)));
    graph.add_pass(Box::new(VolumetricFogPass::new(device)));

    graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));

    graph.add_pass(Box::new(PostProcessPass::new_with_user_effects(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
        None,
    )));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(w, h);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, scene, config, debug_state, debug_camera_buf, cull_stats_buf| {
            build_fxaa_hlfs_graph_internal(
                device,
                queue,
                scene,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
            )
        },
    );
    graph.set_graph_data(rebuilder);

    graph
}

pub fn build_simple_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
) -> RenderGraph {
    let mut graph = RenderGraph::new(device, queue);
    graph.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));

    let rebuilder: GraphRebuilder = Arc::new(
        move |device, _queue, _scene, _config, _debug_state, _debug_camera_buf, _cull_stats_buf| {
            let mut g = RenderGraph::new(device, _queue);
            g.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));
            g
        },
    );
    graph.set_graph_data(rebuilder);

    graph
}

// ── Forward-mode graph builders ─────────────────────────────────────────────

pub fn build_forward_opaque_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_forward_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
    )
}

pub fn build_forward_opaque_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_forward_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
    )
}

pub fn build_forward_only_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_forward_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
    )
}

pub fn build_forward_only_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_forward_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
    )
}

fn build_forward_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device, &config);

    let perf = add_common_early_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        cull_stats_buf,
        iw,
        ih,
    );

    graph.add_pass(Box::new(LightCullPass::new(device, iw, ih)));

    graph.add_pass(Box::new(RadianceCascadesPass::new(
        device,
        scene.gpu_scene().lights.buffer(),
    )));

    // Forward geometry pass replaces G-buffer + decal + deferred light + SSR + planar reflections
    add_forward_geometry_passes(&mut graph, device, scene, &config, &perf, true);

    // Voxel mesh pass — real triangles with depth testing, composited over
    // the forward-lit output.
    graph.add_pass(Box::new(VoxelMeshPass::new_composited(
        device,
        queue,
        config.surface_format,
    )));

    add_late_passes(
        &mut graph,
        device,
        queue,
        scene,
        &config,
        &perf,
        debug_state.clone(),
        debug_camera_buf,
        iw,
        ih,
    );

    // Before AA, at internal resolution: fog accumulates against internal-res
    // depth, and the AA pass then resolves it with the rest of the frame.
    graph.add_pass(Box::new(PostProcessVolumeBlendPass::new(device)));
    graph.add_pass(Box::new(VolumetricFogPass::new(device)));

    // Transparent pass — alpha-blended geometry (simple fixed shader).
    let camera_buf = scene.gpu_scene().camera.buffer();
    let instances_buf = scene.gpu_scene().instances.buffer();
    graph.add_pass(Box::new(helio_pass_transparent::TransparentPass::new(device, camera_buf, instances_buf, config.surface_format)));

    graph.add_pass(Box::new(LensFlarePass::new(
        device,
        queue,
        scene.gpu_scene().lights.buffer(),
        iw,
        ih,
        config.surface_format,
    )));

    graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));

    graph.add_pass(Box::new(PostProcessPass::new_with_user_effects(
        device,
        queue,
        config.width,
        config.height,
        config.surface_format,
        None,
    )));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.lock(iw, ih);

    let overlay_owned = debug_overlay.map(Arc::clone);
    let rebuilder: GraphRebuilder = Arc::new(
        move |device, queue, scene, config, debug_state, debug_camera_buf, cull_stats_buf| {
            build_forward_graph_internal(
                device,
                queue,
                scene,
                config,
                debug_state,
                debug_camera_buf,
                cull_stats_buf,
                owns_device,
                overlay_owned.as_ref(),
            )
        },
    );
    graph.set_graph_data(rebuilder);

    graph
}
