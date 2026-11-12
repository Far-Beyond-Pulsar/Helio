//! The status screen: what a renderer draws when it has no pipeline or its
//! pipeline is invalid.
//!
//! Helio builds this graph itself (one pass, `helio_pass_status_screen`, a
//! fullscreen shader over a rounded-rectangle signed distance field) so a
//! failure is readable on the output frame, in the editor viewport and in a
//! game, with no host drawing anything. This module only says what the screen
//! says for a [`PipelineStatus`]; the pass lays the text out for the frame
//! size and draws it.

use std::sync::Arc;

use helio_core::{RenderGraph, RenderSize};
use helio_pass_status_screen::{Detail, StatusContent, StatusScreenPass, Tone};

use super::pipeline_status::PipelineStatus;

/// What to do to assign a pipeline, when the host has not said. A name in
/// backticks is drawn as a code chip.
pub(crate) const DEFAULT_HINT: &str =
    "Call `Renderer::set_pipeline` with a `NamedPipeline`, or build the renderer with \
     `RendererBuilder::with_graph`.";

/// The reasons the renderer gives when the pipeline's graph was valid but
/// could not be built: the screen says the build failed, not that the graph
/// is invalid.
pub(crate) const REASON_PANIC: &str = "the pipeline panicked while building";
pub(crate) const REASON_GPU: &str =
    "the GPU rejected what the pipeline built (an invalid shader or pipeline state)";

/// `text` with its first letter capitalised: a reason is a sentence here.
fn sentence(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// The text `status` shows, or `None` for `Ready`, which has none.
/// `rebuilds` says the pipeline is built again when a shader, plugin or its
/// asset changes.
pub(crate) fn content_for(
    status: &PipelineStatus,
    hint: &str,
    rebuilds: bool,
) -> Option<StatusContent> {
    match status {
        PipelineStatus::Ready => None,
        PipelineStatus::None => Some(StatusContent {
            tone: Tone::Idle,
            kicker: "No pipeline".into(),
            title: "No render pipeline assigned".into(),
            pipeline: None,
            reason: "This renderer has no render pipeline, so there is nothing to draw.".into(),
            details: Vec::new(),
            hint: hint.to_owned(),
            rebuilds: false,
        }),
        PipelineStatus::Invalid {
            id,
            reason,
            details,
        } => {
            let build_failed = reason == REASON_PANIC || reason == REASON_GPU;
            let (kicker, title) = if build_failed {
                ("Pipeline build failed", "Render pipeline failed to build")
            } else {
                ("Pipeline invalid", "Render pipeline invalid")
            };
            Some(StatusContent {
                tone: Tone::Alert,
                kicker: kicker.into(),
                title: title.into(),
                pipeline: Some(id.clone()),
                reason: sentence(reason),
                details: details.iter().map(|line| Detail::parse(line)).collect(),
                hint: format!("Fix the pipeline, or assign another one. {hint}"),
                rebuilds,
            })
        }
    }
}

/// The status screen's graph for `size`.
pub(crate) fn graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    owns_device: bool,
    surface_format: wgpu::TextureFormat,
    size: RenderSize,
    content: StatusContent,
) -> RenderGraph {
    let mut graph = if owns_device {
        RenderGraph::new(device, queue)
    } else {
        RenderGraph::new_with_external_device(device, queue)
    };
    graph.with_xr_mode(false);
    graph.add_pass(Box::new(StatusScreenPass::new(
        device,
        queue,
        surface_format,
        size.output,
        content,
    )));
    graph.lock_sized(size);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_pass_status_screen::Severity;

    fn invalid(reason: &str, details: Vec<String>) -> PipelineStatus {
        PipelineStatus::Invalid {
            id: "projects/game/post.rendergraph".into(),
            reason: reason.into(),
            details,
        }
    }

    #[test]
    fn ready_has_no_screen() {
        assert!(content_for(&PipelineStatus::Ready, "", false).is_none());
    }

    #[test]
    fn none_is_the_quiet_screen_with_the_hosts_hint() {
        let content = content_for(&PipelineStatus::None, "Use World Settings.", true).unwrap();
        assert_eq!(content.tone, Tone::Idle);
        assert_eq!(content.kicker, "No pipeline");
        assert_eq!(content.title, "No render pipeline assigned");
        assert_eq!(content.pipeline, None);
        assert_eq!(content.hint, "Use World Settings.");
        assert!(!content.rebuilds, "nothing to rebuild without a pipeline");
        assert!(content.details.is_empty());
    }

    #[test]
    fn invalid_names_the_pipeline_the_reason_and_the_details() {
        let status = invalid(
            "the graph does not validate",
            vec![
                "error: missing node `voxel.terrain` (plugin voxel disabled)".into(),
                "warning: routing: Output is never written".into(),
                "adapter has 8 GiB".into(),
            ],
        );
        let content = content_for(&status, "Pick one in World Settings.", true).unwrap();
        assert_eq!(content.tone, Tone::Alert);
        assert_eq!(content.kicker, "Pipeline invalid");
        assert_eq!(content.title, "Render pipeline invalid");
        assert_eq!(
            content.pipeline.as_deref(),
            Some("projects/game/post.rendergraph")
        );
        assert_eq!(content.reason, "The graph does not validate");
        assert!(content.rebuilds);
        assert!(content.hint.ends_with("Pick one in World Settings."));
        let severities: Vec<_> = content.details.iter().map(|d| d.severity).collect();
        assert_eq!(
            severities,
            [Severity::Error, Severity::Warning, Severity::Note],
            "the marker sets the severity; a line with none is a note"
        );
        assert_eq!(
            content.details[0].text,
            "missing node `voxel.terrain` (plugin voxel disabled)"
        );
    }

    #[test]
    fn a_build_failure_is_not_called_invalid() {
        for reason in [REASON_PANIC, REASON_GPU] {
            let content = content_for(&invalid(reason, Vec::new()), "", false).unwrap();
            assert_eq!(content.kicker, "Pipeline build failed");
            assert_eq!(content.title, "Render pipeline failed to build");
        }
    }
}
