//! Helio's spine as a feature: the core pipeline every preset is built around
//! and its extension points' neighbours.
//!
//! [`SpineFeature`] registers one node per spine pass: scene prep (the
//! lights' shadow views, the background, the depth pyramid, light binning),
//! deferred lighting, TSR and the tone map, plus the host inputs they read.
//! Everything else joins at the frame's slots and draw phases
//! (`helio_frame_api`): an object kind's batches, culling and drawers
//! register from its own crates (a meshes plugin's object batch in
//! `Slot::Prepare`, its culling around the pyramid in `Slot::Visibility`,
//! its drawers in the phases). TSR and the tone map work on the scene colour
//! the colour chain hands them.
//!
//! The spine knows no renderer: its nodes read the build environment,
//! [`SpineSettings`] and the lights' `ShadowSettings`, which a graph's builder
//! sets (Helio's presets map their renderer config into them).

use helio_feature_api::{pass_node, BuildEnv, Feature, FeatureRegistrar, ParamDesc};
use helio_frame_api::{color, host_input, FrameNode, Slot};
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_hiz::HiZBuildPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_postprocess::PostProcessPass;
use helio_pass_shadow_matrix::{LightDirtyFlags, ShadowMatrixPass, ShadowSettings};
use helio_pass_tsr::{TsrPass, TsrQuality};

mod background;
pub use background::BackgroundPass;

/// What the spine's nodes read that is not the frame: a graph's builder sets
/// it (Helio's presets from their renderer config).
#[derive(Clone, Debug)]
pub struct SpineSettings {
    /// The colour `pre_aa` is cleared to before anything draws.
    pub background_color: [f32; 4],
    /// The deferred lighting's debug view (0: none).
    pub debug_mode: u32,
    /// Environment-cubemap indirect specular.
    pub environment_reflections: bool,
}

impl Default for SpineSettings {
    fn default() -> Self {
        Self {
            background_color: [0.0, 0.0, 0.0, 1.0],
            debug_mode: 0,
            environment_reflections: true,
        }
    }
}

fn settings(env: &BuildEnv<'_>) -> SpineSettings {
    env.setting::<SpineSettings>().cloned().unwrap_or_default()
}

/// The scene handle the spine reads SceneDB's buffers through.
fn scene_db(env: &BuildEnv<'_>) -> Result<pulsar_scenedb::gpu::GpuMirrorHandle, String> {
    env.require::<pulsar_scenedb::gpu::GpuMirrorHandle>()
        .cloned()
}

/// SceneDB's buffer under `key`, or an empty stand-in of `fallback_size`
/// bytes while the scene has none.
pub fn scene_buffer_or_dummy(
    scene_db: &pulsar_scenedb::gpu::GpuMirrorHandle,
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

/// Resources the renderer supplies every frame that the spine's passes read
/// (features declare their own, `helio_frame_api::host_input`).
const HOST_INPUTS: [&str; 2] = ["material_textures", "render_environment"];

/// The lights' shadow settings this graph is built with.
fn shadow_settings(env: &BuildEnv<'_>) -> ShadowSettings {
    env.setting::<ShadowSettings>().copied().unwrap_or_default()
}

fn tsr_quality(name: &str) -> TsrQuality {
    match name {
        "performance" => TsrQuality::Performance,
        "quality" => TsrQuality::Quality,
        "native" => TsrQuality::Native,
        _ => TsrQuality::Balanced,
    }
}

/// The spine's nodes and host inputs.
pub struct SpineFeature;

impl Feature for SpineFeature {
    fn id(&self) -> &'static str {
        "helio.spine"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        for name in HOST_INPUTS {
            registrar.node(host_input(name));
        }
        register_scene_prep(registrar);
        register_lighting(registrar);
        register_post(registrar);
    }
}

fn register_scene_prep(registry: &mut FeatureRegistrar<'_>) {
    const CATEGORY: &str = "Passes: Scene Prep";

    registry.node(pass_node(
        "shadow_matrix",
        "Shadows: Light Matrices",
        CATEGORY,
        "Computes the view-projection of every shadow-casting light face.",
        |build| {
            let device = build.env.device;
            let (budget, atlas_size) =
                shadow_settings(build.env).atlas(device.limits().max_texture_dimension_2d)?;
            let lights = scene_buffer_or_dummy(
                &scene_db(build.env)?,
                device,
                pulsar_scenedb::gpu::BufferKey::of(helio_frame_api::lighting::LIGHT_LIST),
                "SceneDB Lights",
                96,
            );
            let matrices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Matrices"),
                size: helio_pass_shadow_matrix::MAX_SHADOW_FACES as u64
                    * std::mem::size_of::<helio_pass_shadow_matrix::GpuShadowMatrix>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let dirty = LightDirtyFlags::new(device);
            let hashes = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shadow Hashes"),
                size: helio_pass_shadow_matrix::MAX_SHADOW_CASTERS as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let (_, ih) = build.env.internal_size();
            build.graph.add_pass(Box::new(
                ShadowMatrixPass::new(
                    device,
                    &lights.buffer,
                    &matrices,
                    build.env.camera_buffer,
                    &dirty.0,
                    &hashes,
                    atlas_size,
                )
                .with_budget(budget, ih),
            ));
            build.env.shared(|| dirty);
            Ok(())
        },
    ));

    registry.node(pass_node(
        "background",
        "Scene: Clear Background",
        CATEGORY,
        "Clears the scene colour to the background the sky and lighting draw over.",
        |build| {
            let pass = BackgroundPass::new(
                build.env.scene_format(),
                settings(build.env).background_color,
            );
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));

    registry.node(
        pass_node(
            "hiz",
            "Culling: Depth Pyramid (Hi-Z)",
            CATEGORY,
            "Builds the hierarchical depth pyramid occlusion culling tests against.",
            |build| {
                let (iw, ih) = build.env.internal_size();
                let pass = HiZBuildPass::new(build.env.device, build.env.queue, iw, ih);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        )
        .in_slot(Slot::Visibility, 0),
    );

    registry.node(pass_node(
        "light_cull",
        "Lighting: Tile Light Binning",
        CATEGORY,
        "Bins the scene's lights into screen tiles.",
        |build| {
            let (iw, ih) = build.env.internal_size();
            let pass = LightCullPass::new(build.env.device, iw, ih);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));
}

fn register_lighting(registry: &mut FeatureRegistrar<'_>) {
    const CATEGORY: &str = "Passes: Lighting";

    registry.node(
        pass_node(
            "deferred_light",
            "Lighting: Deferred",
            CATEGORY,
            "Lights the GBuffer with every light, shadow, reflection and the atmosphere.",
            |build| {
                let spine = settings(build.env);
                let shadows = shadow_settings(build.env);
                let queue = build.env.queue;
                let mut pass = DeferredLightPass::new(
                    build.env.device,
                    queue,
                    build.env.camera_buffer,
                    build.env.scene_format(),
                );
                pass.set_shadow_quality(shadows.quality, queue);
                pass.set_ies_profiles(
                    build
                        .env
                        .setting::<helio_pass_deferred_light::IesProfiles>()
                        .map(|profiles| profiles.0.clone()),
                );
                pass.debug_mode = spine.debug_mode;
                pass.set_env_reflections(spine.environment_reflections);
                build.graph.add_pass(Box::new(pass));
                Ok(())
            },
        )
        .with_marker_after(),
    );
}

fn register_post(registry: &mut FeatureRegistrar<'_>) {
    const CATEGORY: &str = "Passes: Post";

    registry.node(
        pass_node(
            "tsr",
            "Anti-Aliasing: TSR",
            CATEGORY,
            "Temporal anti-aliasing and upscaling. The renderer's internal resolution follows the project's TSR setting.",
            |build| {
                let (iw, ih) = build.env.internal_size();
                let frame = build.env.frame;
                let quality = tsr_quality(&build.text("quality"));
                let tsr = TsrPass::new(
                    build.env.device,
                    iw,
                    ih,
                    frame.width,
                    frame.height,
                    frame.surface_format,
                    quality,
                )
                .with_color_input(color::current(build.env))
                .with_color_alias(color::alias(build.env))
                .with_intermediate_output()
                .with_motion_vectors();
                let stochastic = build.boolean("stochastic_mask")
                    || build
                        .env
                        .setting::<helio_pass_tsr::StochasticSamples>()
                        .is_some();
                let tsr = if stochastic {
                    tsr.with_stochastic_mask()
                } else {
                    tsr
                };
                let tsr = if build.boolean("transparency_reactivity") {
                    tsr.with_transparency_reactivity()
                } else {
                    tsr
                };
                build.graph.add_pass(Box::new(tsr));
                color::advance(build.env, helio_pass_tsr::TSR_COLOR);
                Ok(())
            },
        )
        .with_param(ParamDesc::choice(
            "quality",
            "Quality",
            &["performance", "balanced", "quality", "native"],
            "balanced",
        ))
        .with_param(ParamDesc::boolean(
            "stochastic_mask",
            "Stochastic mask",
            false,
        ))
        .with_param(ParamDesc::boolean(
            "transparency_reactivity",
            "Transparency reactivity",
            false,
        )),
    );

    registry.node(
        pass_node(
            "post_process",
            "Post: Tone Map + Grade",
            CATEGORY,
            "Bloom, exposure, tone mapping and colour grading into the output.",
            |build| {
                let input = color::current(build.env);
                let alias = color::alias(build.env);
                let frame = build.env.frame;
                let effects = build
                    .env
                    .setting::<helio_pass_postprocess::UserEffects>()
                    .map(|effects| effects.0);
                build.graph.add_pass(Box::new(
                    PostProcessPass::new_with_user_effects(
                        build.env.device,
                        build.env.queue,
                        frame.width,
                        frame.height,
                        frame.surface_format,
                        effects,
                    )
                    .with_color_input(input)
                    .with_color_alias(alias)
                    .with_color_grading_lut(
                        build
                            .env
                            .setting::<helio_pass_postprocess::ColorGradingLut>()
                            .map(|lut| lut.0.clone()),
                    ),
                ));
                Ok(())
            },
        )
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_colour_chain_starts_at_the_lit_scene_colour() {
        assert_eq!(helio_frame_api::color::SCENE_COLOR, "pre_aa");
    }
}
