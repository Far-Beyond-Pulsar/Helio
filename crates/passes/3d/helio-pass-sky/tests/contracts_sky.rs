use helio_pass_sky::{
    RenderParams, SimParams, RENDER_PARAMS_SIZE, SIM_PARAMS_SIZE, VOLUME_SIZE,
};

#[test]
fn authored_contract_sizes_and_volume_are_exact() {
    assert_eq!(std::mem::size_of::<SimParams>(), SIM_PARAMS_SIZE as usize);
    assert_eq!(
        std::mem::size_of::<RenderParams>(),
        RENDER_PARAMS_SIZE as usize
    );
    assert_eq!(VOLUME_SIZE.width, 96);
    assert_eq!(VOLUME_SIZE.height, 48);
    assert_eq!(VOLUME_SIZE.depth_or_array_layers, 96);
}

#[test]
fn authored_shaders_are_kept_verbatim_in_source() {
    assert!(helio_pass_sky::SIM_SHADER.contains("@workgroup_size(4, 4, 4)"));
    assert!(helio_pass_sky::SIM_SHADER.contains("fn main("));
    assert!(!helio_pass_sky::SIM_SHADER.contains("fn cs_main("));
    assert!(helio_pass_sky::RENDER_SHADER.contains("array<vec2<f32>, 3>"));
}
