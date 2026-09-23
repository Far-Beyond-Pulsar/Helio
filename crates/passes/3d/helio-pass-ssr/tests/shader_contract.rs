#[test]
fn raster_and_ray_query_sources_validate_with_shared_gbuffer_contract() {
    for source in [
        include_str!("../shaders/ssr_trace.wgsl"),
        include_str!("../shaders/ssr_trace_rt.wgsl"),
        include_str!("../shaders/ssr_compose.wgsl"),
    ] {
        let resolved = helio_core::shader::resolve_with(source, &[helio_pass_hiz::HIZ_SNIPPET]);
        let module = naga::front::wgsl::parse_str(&resolved)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&resolved)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("SSR shader validation");
        // Both paths must inherit normal decoding instead of defining a second
        // contract that can drift when G-buffer attachments change.
        assert!(helio_core::shader::uses_prelude(source));
        assert!(!source.contains("fn helio_gbuffer_normal"));
    }
}

#[test]
fn half_resolution_contract_reconstructs_full_resolution_inputs() {
    let raster = include_str!("../shaders/ssr_trace.wgsl");
    let rt = include_str!("../shaders/ssr_trace_rt.wgsl");
    let compose = include_str!("../shaders/ssr_compose.wgsl");

    for source in [raster, rt] {
        assert!(source.contains("let source_dims = textureDimensions(gbuf_depth)"));
        assert!(source.contains("let source_px = clamp"));
    }
    assert!(compose.contains("let reflection_dims=textureDimensions(reflection)"));
}

#[test]
#[ignore = "requires hardware ray queries"]
fn hybrid_pipeline_accepts_shared_camera_and_normal_contract() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("adapter");
        let features = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        assert!(
            adapter.features().contains(features),
            "hardware RT is required"
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                required_limits: adapter.limits(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .expect("device");
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSR contract camera"),
            size: (2 * std::mem::size_of::<helio_core::GpuCameraUniforms>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let _pass = helio_pass_ssr::SsrPass::new(&device, &queue, &camera, 64, 64);
        let _composite = helio_pass_ssr::SsrCompositePass::new(
            &device,
            &camera,
            wgpu::TextureFormat::Rgba16Float,
        );
        assert!(
            errors.pop().await.is_none(),
            "SSR pipeline validation failed"
        );
    });
}
