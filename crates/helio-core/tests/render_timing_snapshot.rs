use helio_core::{profiling::GpuProfiler, GpuTimingAvailability, Profiler, RenderTimingSnapshot};

#[test]
fn empty_snapshot_is_explicitly_pending_and_allocation_free_to_read() {
    let snapshot = RenderTimingSnapshot::default();
    assert_eq!(snapshot.generation, 0);
    assert_eq!(snapshot.gpu_availability, GpuTimingAvailability::Pending);
    assert!(snapshot.passes.is_empty());
    assert_eq!(snapshot.total_cpu_ms, None);
    assert_eq!(snapshot.total_gpu_ms, None);
    assert_eq!(snapshot.gpu_frame_index, None);
    assert_eq!(snapshot.gpu_lag_frames, None);
}

#[test]
fn unsupported_gpu_state_and_pass_order_are_stable() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: timing snapshot");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Timing Snapshot Unsupported Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("available adapter must create a timing snapshot device");
        let mut profiler = Profiler::new(&device, &queue);
        {
            let _scope = profiler.scope("SecondPass");
        }
        {
            let _scope = profiler.scope("FirstPass");
        }
        profiler.update_snapshot(9, ["FirstPass", "SecondPass"].into_iter());

        let snapshot = profiler.timing_snapshot();
        assert_eq!(snapshot.generation, 1);
        assert_eq!(snapshot.cpu_frame_index, 9);
        assert_eq!(
            snapshot.gpu_availability,
            GpuTimingAvailability::Unsupported
        );
        assert_eq!(
            snapshot
                .passes
                .iter()
                .map(|pass| pass.name)
                .collect::<Vec<_>>(),
            ["FirstPass", "SecondPass"]
        );
        assert!(snapshot.passes.iter().all(|pass| pass.cpu_ms.is_some()));
        assert!(snapshot.passes.iter().all(|pass| pass.gpu_ms.is_none()));
    });
}

#[test]
fn deferred_readback_attributes_work_after_an_external_poll() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: deferred timing readback");
            return;
        };
        let required =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !adapter.features().contains(required) {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_TIMESTAMPS");
            return;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Deferred Timing Readback Device"),
                required_features: required,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("timestamp-capable adapter must create a timing device");
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            panic!("deferred timing GPU validation error: {error:?}");
        }));

        let mut profiler = GpuProfiler::new(&device, &queue);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Forced Slow Pass Timing Encoder"),
        });
        profiler.begin_pass(&mut encoder, "ForcedSlowPass");
        {
            let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Forced Slow Pass"),
                timestamp_writes: None,
            });
        }
        profiler.end_pass(&mut encoder, "ForcedSlowPass");
        profiler.resolve_queries(&mut encoder, 17);
        queue.submit([encoder.finish()]);

        // The Helio-owned method only queues map_async. It returns before any
        // host poll and therefore cannot introduce a wait into the frame.
        assert!(profiler.read_timestamps_deferred().is_empty());
        assert_eq!(profiler.pending_readbacks(), 1);

        // This poll represents GPUI/winit's owner-driven event-loop cadence,
        // not a call made by Helio's external-device render path.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let timings = profiler.read_timestamps_deferred();
        let timing_count = timings.len();
        let timing_name = timings.first().map(|timing| timing.name);
        assert_eq!(profiler.last_completed_frame(), Some(17));
        assert_eq!(timing_count, 1);
        assert_eq!(timing_name, Some("ForcedSlowPass"));
        assert_eq!(profiler.pending_readbacks(), 0);
    });
}

async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force_fallback_adapter in [false, true] {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            return Some(adapter);
        }
    }
    None
}
