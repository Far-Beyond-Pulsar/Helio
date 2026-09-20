use helio_core::{
    FocusedTimingGroup, FocusedTimingReport, GpuTimingAvailability, RenderPassTiming,
    RenderTimingSnapshot,
};

#[test]
fn cathedral_stages_are_classified_without_render_changes() {
    assert_eq!(
        FocusedTimingGroup::classify("HLFS"),
        Some(FocusedTimingGroup::Hlfs)
    );
    assert_eq!(
        FocusedTimingGroup::classify("SsrPass"),
        Some(FocusedTimingGroup::RtReflections)
    );
    assert_eq!(
        FocusedTimingGroup::classify("ShadowCull"),
        Some(FocusedTimingGroup::Shadows)
    );
    assert_eq!(
        FocusedTimingGroup::classify("VolumetricFogPass"),
        Some(FocusedTimingGroup::VolumetricFog)
    );
    assert_eq!(
        FocusedTimingGroup::classify("HLFS temporal filter"),
        Some(FocusedTimingGroup::Denoising)
    );
    assert_eq!(
        FocusedTimingGroup::classify("TSR"),
        Some(FocusedTimingGroup::Denoising)
    );
}

#[test]
fn report_sums_gpu_samples_by_focus_group() {
    let snapshot = RenderTimingSnapshot {
        generation: 7,
        gpu_frame_index: Some(6),
        gpu_availability: GpuTimingAvailability::Available,
        passes: vec![
            RenderPassTiming {
                name: "HLFS",
                cpu_ms: Some(1.0),
                gpu_ms: Some(4.0),
            },
            RenderPassTiming {
                name: "SsrPass",
                cpu_ms: Some(0.5),
                gpu_ms: Some(2.0),
            },
            RenderPassTiming {
                name: "SsrCompositePass",
                cpu_ms: None,
                gpu_ms: Some(0.5),
            },
            RenderPassTiming {
                name: "Shadow",
                cpu_ms: None,
                gpu_ms: Some(1.25),
            },
            RenderPassTiming {
                name: "VolumetricFogPass",
                cpu_ms: None,
                gpu_ms: Some(0.75),
            },
            RenderPassTiming {
                name: "TSR",
                cpu_ms: None,
                gpu_ms: Some(0.4),
            },
            RenderPassTiming {
                name: "GBuffer",
                cpu_ms: Some(2.0),
                gpu_ms: Some(3.0),
            },
        ],
        ..Default::default()
    };
    let report = FocusedTimingReport::from_snapshot(&snapshot);
    assert_eq!(report.gpu_ms(FocusedTimingGroup::Hlfs), Some(4.0));
    assert_eq!(report.gpu_ms(FocusedTimingGroup::RtReflections), Some(2.5));
    assert_eq!(report.gpu_ms(FocusedTimingGroup::Shadows), Some(1.25));
    assert_eq!(report.gpu_ms(FocusedTimingGroup::VolumetricFog), Some(0.75));
    assert_eq!(report.gpu_ms(FocusedTimingGroup::Denoising), Some(0.4));
    assert!(!report.format_gpu_line().contains("GBuffer"));
}
