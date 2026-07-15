use nebula_core::{
    error::NebulaError,
    progress::{
        ChannelReporter, NullReporter, ProgressEvent, ProgressEventKind,
        ProgressReporter,
    },
};

// ── NullReporter ──────────────────────────────────────────────────────────────

#[test]
fn null_reporter_begin_is_no_op() {
    let r = NullReporter;
    r.begin("test_pass", 3); // Must not panic
}

#[test]
fn null_reporter_step_is_no_op() {
    let r = NullReporter;
    r.step("test_pass", 0, "doing something");
}

#[test]
fn null_reporter_finish_is_no_op() {
    let r = NullReporter;
    r.finish("test_pass", true, "done");
    r.finish("test_pass", false, "failed");
}

#[test]
fn null_reporter_is_object_safe() {
    let r: Box<dyn ProgressReporter> = Box::new(NullReporter);
    r.begin("p", 1);
    r.step("p", 0, "hi");
    r.finish("p", true, "");
}

// ── ChannelReporter ───────────────────────────────────────────────────────────

#[test]
fn channel_reporter_begin_sends_event() {
    let (r, rx) = ChannelReporter::new();
    r.begin("ao", 4);
    let ev = rx.recv().expect("expected event");
    assert_eq!(ev.pass, "ao");
    match ev.kind {
        ProgressEventKind::Begin { total_steps } => assert_eq!(total_steps, 4),
        _ => panic!("unexpected event kind"),
    }
}

#[test]
fn channel_reporter_step_sends_event() {
    let (r, rx) = ChannelReporter::new();
    r.step("lightmap", 2, "baking texels");
    let ev = rx.recv().expect("expected event");
    assert_eq!(ev.pass, "lightmap");
    match ev.kind {
        ProgressEventKind::Step { step, message } => {
            assert_eq!(step, 2);
            assert_eq!(message, "baking texels");
        }
        _ => panic!("unexpected event kind"),
    }
}

#[test]
fn channel_reporter_finish_success_sends_event() {
    let (r, rx) = ChannelReporter::new();
    r.finish("nav", true, "4096 polygons");
    let ev = rx.recv().expect("expected event");
    match ev.kind {
        ProgressEventKind::Finish { success, message } => {
            assert!(success);
            assert_eq!(message, "4096 polygons");
        }
        _ => panic!("unexpected event kind"),
    }
}

#[test]
fn channel_reporter_finish_failure_sends_event() {
    let (r, rx) = ChannelReporter::new();
    r.finish("pvs", false, "timeout");
    let ev = rx.recv().expect("expected event");
    match ev.kind {
        ProgressEventKind::Finish { success, .. } => assert!(!success),
        _ => panic!(),
    }
}

#[test]
fn channel_reporter_sequence_ordering() {
    let (r, rx) = ChannelReporter::new();
    r.begin("probe", 3);
    r.step("probe", 0, "face 0");
    r.step("probe", 1, "face 1");
    r.finish("probe", true, "done");

    let events: Vec<ProgressEvent> = rx.try_iter().collect();
    assert_eq!(events.len(), 4);
    assert!(matches!(events[0].kind, ProgressEventKind::Begin { .. }));
    assert!(matches!(events[1].kind, ProgressEventKind::Step { .. }));
    assert!(matches!(events[2].kind, ProgressEventKind::Step { .. }));
    assert!(matches!(events[3].kind, ProgressEventKind::Finish { .. }));
}

// ── NebulaError ──────────────────────────────────────────────────────────────

#[test]
fn nebula_error_gpu_display() {
    let e = NebulaError::Gpu("device lost".to_owned());
    assert!(e.to_string().contains("device lost"));
}

#[test]
fn nebula_error_serialize_display() {
    let e = NebulaError::Serialize("encode failed".to_owned());
    assert!(e.to_string().contains("encode failed"));
}

#[test]
fn nebula_error_deserialize_display() {
    let e = NebulaError::Deserialize("invalid magic".to_owned());
    assert!(e.to_string().contains("invalid magic"));
}

#[test]
fn nebula_error_bake_failed_display() {
    let e = NebulaError::BakeFailed { pass: "ao".to_owned(), reason: "no geometry".to_owned() };
    let s = e.to_string();
    assert!(s.contains("ao"));
    assert!(s.contains("no geometry"));
}

#[test]
fn nebula_error_readback_timeout_display() {
    let e = NebulaError::ReadbackTimeout { ms: 5000 };
    assert!(e.to_string().contains("5000"));
}

#[test]
fn nebula_error_unsupported_display() {
    let e = NebulaError::Unsupported("hardware RT".to_owned());
    assert!(e.to_string().contains("hardware RT"));
}

#[test]
fn nebula_error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
    let e: NebulaError = io_err.into();
    assert!(matches!(e, NebulaError::Io(_)));
}

// ── BakeContext (GPU-dependent — skipped without adapter) ────────────────────

#[test]
#[ignore = "requires GPU adapter"]
fn bake_context_new_succeeds() {
    pollster::block_on(async {
        let ctx = nebula_core::BakeContext::new().await;
        assert!(ctx.is_ok(), "BakeContext::new failed: {:?}", ctx.err());
    });
}

#[test]
#[ignore = "requires GPU adapter"]
fn bake_context_adapter_info_non_empty_name() {
    pollster::block_on(async {
        let ctx = nebula_core::BakeContext::new().await.unwrap();
        assert!(!ctx.adapter_info.name.is_empty());
    });
}
