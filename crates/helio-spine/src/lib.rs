//! Helio's spine as a feature: the core pipeline every preset is built around
//! and its extension points' neighbours.
//!
//! [`SpineFeature`] registers one node per spine pass: scene prep (the
//! background, the G-buffer's clear of last resort, the depth pyramid), the
//! lighting model's unlit stand-in, TSR and the tone map, plus the host
//! inputs they read. Everything else joins at the frame's slots and draw
//! phases (`helio_frame_api`): an object kind's batches, culling and drawers
//! register from its own crates (a meshes plugin's object batch in
//! `Slot::Prepare`, its culling around the pyramid in `Slot::Visibility`,
//! its drawers in the phases), and so do the lights (their shadow views
//! before the shadow casters, their binning, the lighting model at its
//! point). TSR and the tone map work on the scene colour the colour chain
//! hands them.
//!
//! The spine knows no renderer: its nodes read the build environment and
//! [`SpineSettings`], which a graph's builder sets (Helio's presets map
//! their renderer config into it).

use helio_feature_api::{pass_node, BuildEnv, Feature, FeatureRegistrar, ParamDesc};
use helio_frame_api::lighting::{model, LightingModel, RegisterLightingModel, Shading};
use helio_frame_api::{color, host_input, FrameNode, Slot};
use helio_pass_hiz::HiZBuildPass;
use helio_pass_postprocess::PostProcessPass;
use helio_pass_tsr::{TsrPass, TsrQuality};

mod background;
mod gbuffer_clear;
mod unlit;
pub use background::BackgroundPass;
pub use gbuffer_clear::GBufferClearPass;
pub use unlit::UnlitPass;

/// What the spine's nodes read that is not the frame: a graph's builder sets
/// it (Helio's presets from their renderer config).
#[derive(Clone, Debug)]
pub struct SpineSettings {
    /// The colour `pre_aa` is cleared to before anything draws.
    pub background_color: [f32; 4],
}

impl Default for SpineSettings {
    fn default() -> Self {
        Self {
            background_color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

fn settings(env: &BuildEnv<'_>) -> SpineSettings {
    env.setting::<SpineSettings>().cloned().unwrap_or_default()
}

/// Resources the renderer supplies every frame that the spine's passes read
/// (features declare their own, `helio_frame_api::host_input`, and may
/// supply them: the materials' texture table).
const HOST_INPUTS: [&str; 1] = [helio_core::resource_keys::RENDER_ENVIRONMENT];

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
        // Shader hot reload may swap these alone: they take only
        // renderer-owned handles (camera, debug camera and cull-stats buffers,
        // debug-draw state, SceneDB) and plain configuration, the same objects
        // in the live and the replacement graph (`helio_core::SwapPolicy`).
        registrar.swappable::<BackgroundPass>();
        registrar.swappable::<UnlitPass>();
        registrar.swappable::<TsrPass>();
        registrar.swappable::<PostProcessPass>();
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
        "gbuffer_clear",
        "Scene: Clear G-Buffer",
        CATEGORY,
        "Clears the frame's G-buffer and depth when no object kind installed opens the G-buffer phase.",
        |build| {
            build.graph.add_pass(Box::new(GBufferClearPass));
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
}

fn register_lighting(registry: &mut FeatureRegistrar<'_>) {
    // The stand-in model: what a preset gets when the model it chose is not
    // registered. Its pass is placed by the preset (it has no point).
    registry.lighting_model(LightingModel::new(model::UNLIT, Shading::GBuffer));
    registry.node(pass_node(
        "unlit",
        "Lighting: Unlit (No Lighting Model)",
        "Passes: Lighting",
        "Draws the G-buffer unlit (albedo + emissive) when no lighting model is installed.",
        |build| {
            let pass = UnlitPass::new(build.env.device, build.env.scene_format());
            build.graph.add_pass(Box::new(pass));
            Ok(())
        },
    ));
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
        )),
    );

    registry.node(pass_node(
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
                .with_glare(color::glare(build.env))
                .with_color_grading_lut(
                    build
                        .env
                        .setting::<helio_pass_postprocess::ColorGradingLut>()
                        .map(|lut| lut.0.clone()),
                ),
            ));
            Ok(())
        },
    ));
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_colour_chain_starts_at_the_lit_scene_colour() {
        assert_eq!(helio_frame_api::color::SCENE_COLOR, "pre_aa");
    }
}
