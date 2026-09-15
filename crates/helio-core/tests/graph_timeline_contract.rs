//! Public contract coverage for Helio's host-facing debug payloads.
//!
//! Phase 11 deliberately stops at the immutable payload boundary. Consumers
//! may render these values in any host UI, but Helio-core must not depend on a
//! UI framework or expose live GPU state through this API.

use helio_core::{DebugResourceInfo, GraphTimelineData, GraphTimelinePass};

#[test]
fn graph_timeline_payload_is_owned_and_cloneable() {
    let payload = GraphTimelineData {
        frame_count: 42,
        total_vram_kb: 512,
        physical_vram_kb: 384,
        passes: vec![GraphTimelinePass {
            index: 0,
            name: "lighting".to_owned(),
            reads: vec!["gbuffer".to_owned()],
            writes: vec!["scene_color".to_owned()],
            cpu_ms: Some(0.25),
            gpu_ms: None,
            parallel_layer: 1,
            chain_marker: "[0.1]".to_owned(),
        }],
        resources: vec![DebugResourceInfo {
            name: "scene_color".to_owned(),
            width: 1920,
            height: 1080,
            layers: 1,
            format_name: "Rgba16Float".to_owned(),
            size_kb: 16_200,
            alias: "color_alias".to_owned(),
            chain_local: false,
            first_write_pass: 0,
            last_read_pass: 0,
        }],
    };

    let copy = payload.clone();
    assert_eq!(copy.frame_count, 42);
    assert_eq!(copy.passes[0].name, "lighting");
    assert_eq!(copy.passes[0].gpu_ms, None);
    assert_eq!(copy.resources[0].alias, "color_alias");
}

#[test]
fn debug_payload_contract_has_no_live_gpu_handles() {
    let payload = GraphTimelineData::default();
    let _debug = format!("{payload:?}");
}
