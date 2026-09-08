use helio_core::{GpuTimingAvailability, RenderPassTiming, RenderTimingSnapshot};

/// Pulsar-Native's timing bridge constructs this public struct exhaustively.
/// Keep that caller compiling without requiring a synchronized host update.
#[test]
fn downstream_timing_snapshot_literal_remains_source_compatible() {
    let snapshot = RenderTimingSnapshot {
        generation: 4,
        cpu_frame_index: 18,
        gpu_frame_index: None,
        gpu_lag_frames: None,
        gpu_availability: GpuTimingAvailability::Unsupported,
        total_cpu_ms: Some(7.0),
        total_gpu_ms: None,
        readback_drops: 0,
        query_overflows: 1,
        passes: vec![RenderPassTiming {
            name: "Draw",
            cpu_ms: Some(7.0),
            gpu_ms: None,
        }],
    };
    assert_eq!(snapshot.total_gpu_ms, None);
    assert_eq!(snapshot.passes[0].name, "Draw");
}
