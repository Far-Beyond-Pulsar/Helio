mod harness;
mod sdf_common;
mod examples {
    #[cfg(feature = "render_v2_basic")] pub mod render_v2_basic;
    #[cfg(feature = "render_v2_sky")]   pub mod render_v2_sky;
    #[cfg(feature = "sdf_blend")]       pub mod sdf_blend;
    #[cfg(feature = "sdf_boolean")]     pub mod sdf_boolean;
    #[cfg(feature = "sdf_demo")]        pub mod sdf_demo;
    #[cfg(feature = "sdf_grid")]        pub mod sdf_grid;
    #[cfg(feature = "sdf_morph")]       pub mod sdf_morph;
    #[cfg(feature = "sdf_move")]        pub mod sdf_move;
    #[cfg(feature = "sdf_multi")]       pub mod sdf_multi;
    #[cfg(feature = "sdf_pulse")]       pub mod sdf_pulse;
    #[cfg(feature = "sdf_terrain")]     pub mod sdf_terrain;
}

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "render_v2_basic")]
    { harness::run_scene(examples::render_v2_basic::RenderV2Basic::new()); return; }

    #[cfg(feature = "render_v2_sky")]
    { harness::run_scene(examples::render_v2_sky::RenderV2Sky::new()); return; }

    #[cfg(feature = "sdf_blend")]
    { harness::run_scene(sdf_common::SdfScene::new(examples::sdf_blend::BlendUpdater)); return; }

    #[cfg(feature = "sdf_boolean")]
    { harness::run_scene(sdf_common::SdfScene::new(examples::sdf_boolean::BooleanUpdater)); return; }

    #[cfg(feature = "sdf_demo")]
    { harness::run_scene(examples::sdf_demo::SdfDemo); return; }

    #[cfg(feature = "sdf_grid")]
    { harness::run_scene(sdf_common::SdfScene::new(examples::sdf_grid::GridUpdater)); return; }

    #[cfg(feature = "sdf_morph")]
    { harness::run_scene(sdf_common::SdfScene::new(examples::sdf_morph::MorphUpdater)); return; }

    #[cfg(feature = "sdf_move")]
    { harness::run_scene(sdf_common::SdfScene::new(examples::sdf_move::MoveUpdater)); return; }

    #[cfg(feature = "sdf_multi")]
    { harness::run_scene(sdf_common::SdfScene::new(examples::sdf_multi::MultiUpdater)); return; }

    #[cfg(feature = "sdf_pulse")]
    { harness::run_scene(sdf_common::SdfScene::new(examples::sdf_pulse::PulseUpdater)); return; }

    #[cfg(feature = "sdf_terrain")]
    { harness::run_scene(sdf_common::SdfScene::new(examples::sdf_terrain::TerrainUpdater)); return; }

    // If no feature is enabled, log a helpful error.
    #[allow(unreachable_code)]
    {
        console_error_panic_hook::set_once();
        wasm_logger::init(wasm_logger::Config::default());
        log::error!("helio-wasm-examples built without a demo feature flag. \
                     Rebuild with --features <example_name>.");
    }
}
