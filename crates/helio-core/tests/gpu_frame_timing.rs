use helio_core::Profiler;

/// An outer timing scope must include work submitted on the graphics encoder
/// after the nested compute scope. This is a scope/ordering regression, not a
/// benchmark or a hardware performance threshold.
#[test]
fn whole_frame_timestamp_spans_both_encoders_and_nested_scopes() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        assert!(
            adapter.features().contains(features),
            "timestamp-capable GPU required"
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                ..Default::default()
            })
            .await
            .unwrap();
        let mut profiler = Profiler::new(&device, &queue);
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timed graphics-encoder work"),
            size: 8 * 1024 * 1024,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut compute = device.create_command_encoder(&Default::default());
        let mut graphics = device.create_command_encoder(&Default::default());
        profiler.begin_gpu_pass(&mut compute, "__graph_frame");
        profiler.begin_gpu_pass(&mut compute, "inner");
        profiler.end_gpu_pass(&mut compute, "inner");
        graphics.clear_buffer(&scratch, 0, None);
        profiler.end_gpu_pass(&mut graphics, "__graph_frame");
        profiler.resolve_gpu_queries(&mut graphics, 17);
        queue.submit([compute.finish(), graphics.finish()]);
        let timings = profiler.read_gpu_timestamps_blocking(&device);
        assert_eq!(timings.len(), 2, "both nested scopes must complete");
        let outer = timings
            .iter()
            .find(|s| s.name == "__graph_frame")
            .unwrap()
            .duration_ns;
        let inner = timings
            .iter()
            .find(|s| s.name == "inner")
            .unwrap()
            .duration_ns;
        assert!(
            outer > inner,
            "outer scope must include graphics-encoder work: {outer} <= {inner}"
        );
        profiler.update_snapshot(17, ["inner"].into_iter());
        let snapshot = profiler.timing_snapshot();
        assert_eq!(snapshot.gpu_frame_index, Some(17));
        assert_eq!(snapshot.query_overflows, 0);
        assert_eq!(snapshot.gpu_frame_ms, Some(outer as f32 / 1_000_000.0));
        assert_eq!(snapshot.total_gpu_ms, snapshot.gpu_frame_ms);
        assert_eq!(
            snapshot.passes.len(),
            1,
            "the outer scope is not a render pass"
        );
        assert_eq!(snapshot.passes[0].gpu_ms, Some(inner as f32 / 1_000_000.0));
    });
}
