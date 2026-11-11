//! The status screen: what a renderer draws when it has no pipeline or its
//! pipeline is invalid.
//!
//! Helio builds this graph itself (one text pass, the debug overlay's: its
//! character grids and font are the engine's one on-screen text renderer) so a
//! failure is readable on the output frame, in the editor viewport and in a
//! game, with no host drawing anything. The text is laid out once per grid size
//! and written into the overlay's grids each frame.

use std::sync::{Arc, Mutex};

use helio_core::{RenderGraph, RenderSize};
use helio_pass_debug_overlay::{DebugOverlayPass, DebugOverlayState, ROW_H};

use super::pipeline_status::PipelineStatus;

/// What to do to assign a pipeline, when the host has not said.
pub(crate) const DEFAULT_HINT: &str =
    "Call Renderer::set_pipeline(Some(NamedPipeline::new(id, pipeline))), or build the renderer \
     with RendererBuilder::with_graph.";

/// Wrapped lines one reason, and one detail, may take before they end in `...`.
const REASON_LINES: usize = 6;
const DETAIL_LINES: usize = 3;

/// Body text starts at this small-grid row (below the title) and column.
const BODY_ROW: u32 = 3;
const BODY_COL: u32 = 2;

/// The text of a status screen, before it is wrapped to a grid.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StatusText {
    pub title: String,
    pub header: Vec<String>,
    pub details: Vec<String>,
    pub footer: Vec<String>,
    accent: [f32; 3],
}

/// The text `status` shows. `Ready` has none.
pub(crate) fn text_for(status: &PipelineStatus, hint: &str) -> Option<StatusText> {
    match status {
        PipelineStatus::Ready => None,
        PipelineStatus::None => Some(StatusText {
            title: "No render pipeline assigned".into(),
            header: vec![
                "This renderer has no render pipeline, so there is nothing to draw.".into(),
            ],
            details: Vec::new(),
            footer: vec![format!("To assign one: {hint}")],
            accent: [0.25, 0.45, 0.95],
        }),
        PipelineStatus::Invalid {
            id,
            reason,
            details,
        } => Some(StatusText {
            title: "Render pipeline invalid".into(),
            header: vec![format!("Pipeline: {id}"), format!("Reason: {reason}")],
            details: details.clone(),
            footer: vec![format!("Fix the pipeline, or assign another one. {hint}")],
            accent: [0.95, 0.2, 0.2],
        }),
    }
}

/// Printable ASCII only (the font has nothing else), one line per `\n`.
fn paragraphs(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| {
            line.chars()
                .map(|c| match c {
                    ' '..='~' => c,
                    '\t' => ' ',
                    _ => '?',
                })
                .collect()
        })
        .collect()
}

/// Breaks `text` into lines of at most `width` characters: at spaces where it
/// can, inside a word that is longer than a line where it cannot. Lines after
/// the first of a paragraph start with `indent`.
fn wrap(text: &str, width: usize, indent: &str) -> Vec<String> {
    let width = width.max(indent.len() + 1);
    let mut lines = Vec::new();
    for paragraph in paragraphs(text) {
        let mut cur = String::new();
        let mut emitted = 0usize;
        for mut word in paragraph.split(' ').filter(|w| !w.is_empty()) {
            loop {
                let prefix = if emitted == 0 { "" } else { indent };
                if cur.is_empty() {
                    cur.push_str(prefix);
                }
                let has_word = cur.len() > prefix.len();
                let need = word.len() + usize::from(has_word);
                if cur.len() + need <= width {
                    if has_word {
                        cur.push(' ');
                    }
                    cur.push_str(word);
                    break;
                }
                if has_word {
                    lines.push(std::mem::take(&mut cur));
                    emitted += 1;
                    continue;
                }
                // Longer than a whole line: split it (the text is ASCII).
                let (head, tail) = word.split_at(width - cur.len());
                cur.push_str(head);
                lines.push(std::mem::take(&mut cur));
                emitted += 1;
                word = tail;
                if word.is_empty() {
                    break;
                }
            }
        }
        if !cur.is_empty() || emitted == 0 {
            lines.push(cur);
        }
    }
    lines
}

/// `lines` cut to `max`, the last one ending in `...` when something was cut.
fn clip(mut lines: Vec<String>, max: usize) -> Vec<String> {
    if lines.len() > max {
        lines.truncate(max);
        if let Some(last) = lines.last_mut() {
            last.push_str(" ...");
        }
    }
    lines
}

/// The body of `text` wrapped to `cols` columns and cut to `rows` rows. What
/// does not fit is replaced by a count, so a long list never hides that it is
/// long.
pub(crate) fn layout_body(text: &StatusText, cols: usize, rows: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (index, line) in text.header.iter().enumerate() {
        let wrapped = wrap(line, cols, "  ");
        // The reason is the one header line worth more room than the rest.
        let limit = if index == 0 { 2 } else { REASON_LINES };
        out.extend(clip(wrapped, limit));
    }
    let mut hidden = 0usize;
    let footer: Vec<String> = text
        .footer
        .iter()
        .flat_map(|line| clip(wrap(line, cols, ""), REASON_LINES))
        .collect();
    // Rows the details may use: all but the footer, its gap and the count line.
    let reserved = footer.len() + 2;
    if !text.details.is_empty() {
        out.push(String::new());
        out.push(format!("Details ({}):", text.details.len()));
        let budget = rows.saturating_sub(out.len() + reserved + 1);
        let mut used = 0usize;
        for (index, detail) in text.details.iter().enumerate() {
            let lines = clip(wrap(&format!("- {detail}"), cols, "  "), DETAIL_LINES);
            if used + lines.len() > budget {
                hidden = text.details.len() - index;
                break;
            }
            used += lines.len();
            out.extend(lines);
        }
        if hidden > 0 {
            out.push(format!("... and {hidden} more (see the log)"));
        }
    }
    out.push(String::new());
    out.extend(footer);
    out.truncate(rows);
    out
}

/// What the overlay's populate hook has laid out, and for which grid.
struct Layout {
    grid: (u32, u32),
    title: String,
    body: Vec<String>,
}

fn populate(text: Arc<StatusText>) -> Box<dyn Fn(&mut DebugOverlayState) + Send + Sync> {
    let layout: Mutex<Option<Layout>> = Mutex::new(None);
    Box::new(move |state| {
        let grid = (state.small_cols(), state.small_rows());
        let mut cached = layout.lock().unwrap_or_else(|e| e.into_inner());
        if cached.as_ref().is_none_or(|l| l.grid != grid) {
            let cols = grid.0.saturating_sub(BODY_COL + 1) as usize;
            let rows = grid.1.saturating_sub(BODY_ROW + 1) as usize;
            let big_cols = state.grid_cols().saturating_sub(2) as usize;
            let title = paragraphs(&text.title)
                .into_iter()
                .next()
                .unwrap_or_default()
                .chars()
                .take(big_cols)
                .collect();
            *cached = Some(Layout {
                grid,
                title,
                body: layout_body(&text, cols, rows),
            });
        }
        let layout = cached.as_ref().expect("laid out above");
        // The overlay writes its own stats first; the status screen is all
        // there is on this frame.
        state.clear();
        state.add_bar(
            0.0,
            0.0,
            6.0,
            (state.grid_rows() * ROW_H) as f32,
            text.accent[0],
            text.accent[1],
            text.accent[2],
            1.0,
        );
        state.write_text(1, 0, &layout.title);
        for (row, line) in layout.body.iter().enumerate() {
            state.write_small(BODY_COL, BODY_ROW + row as u32, line);
        }
    })
}

/// The status screen's graph for `size`. `None` for a status that has no
/// screen (`Ready`).
pub(crate) fn graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    owns_device: bool,
    surface_format: wgpu::TextureFormat,
    size: RenderSize,
    text: StatusText,
) -> RenderGraph {
    let mut graph = if owns_device {
        RenderGraph::new(device, queue)
    } else {
        RenderGraph::new_with_external_device(device, queue)
    };
    graph.with_xr_mode(false);
    let state = DebugOverlayState::new();
    {
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        state.enabled = true;
        state.populate = Some(populate(Arc::new(text)));
    }
    let [width, height] = size.output;
    graph.add_pass(Box::new(DebugOverlayPass::new(
        device,
        queue,
        state,
        surface_format,
        width,
        height,
    )));
    graph.lock_sized(size);
    graph
}

/// The colour the frame is cleared to under the status screen.
pub(crate) fn clear_color(status: &PipelineStatus) -> Option<[f32; 4]> {
    match status {
        PipelineStatus::Ready => None,
        PipelineStatus::None => Some([0.03, 0.04, 0.08, 1.0]),
        PipelineStatus::Invalid { .. } => Some([0.10, 0.03, 0.03, 1.0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid(details: Vec<String>) -> PipelineStatus {
        PipelineStatus::Invalid {
            id: "projects/game/post.rendergraph".into(),
            reason: "the graph does not validate".into(),
            details,
        }
    }

    #[test]
    fn wrap_breaks_at_spaces_and_inside_long_words() {
        let lines = wrap("alpha beta gamma delta", 11, "");
        assert_eq!(lines, ["alpha beta", "gamma delta"]);
        let lines = wrap("abcdefghijklmnopqrstuvwxyz", 10, "");
        assert_eq!(lines, ["abcdefghij", "klmnopqrst", "uvwxyz"]);
        assert!(lines.iter().all(|l| l.len() <= 10));
    }

    #[test]
    fn wrap_indents_continuations_and_keeps_to_the_width() {
        let lines = wrap(
            "- missing node `a.b` (plugin voxel disabled) in graph",
            24,
            "  ",
        );
        assert!(lines.len() > 1);
        assert!(lines[0].starts_with("- "));
        assert!(lines[1..].iter().all(|l| l.starts_with("  ")));
        assert!(lines.iter().all(|l| l.len() <= 24), "{lines:?}");
    }

    #[test]
    fn non_ascii_is_replaced_not_dropped() {
        assert_eq!(paragraphs("a\u{e9}\tb"), ["a? b"]);
    }

    #[test]
    fn the_body_names_the_pipeline_reason_and_details() {
        let text = text_for(
            &invalid(vec![
                "missing node `voxel.terrain` (plugin voxel disabled)".into(),
                "routing: Output is never written".into(),
            ]),
            "Pick one in World Settings.",
        )
        .unwrap();
        let body = layout_body(&text, 70, 40).join("\n");
        assert!(body.contains("Pipeline: projects/game/post.rendergraph"));
        assert!(body.contains("Reason: the graph does not validate"));
        assert!(body.contains("Details (2):"));
        assert!(body.contains("plugin voxel disabled"));
        assert!(body.contains("Output is never written"));
        assert!(body.contains("Pick one in World Settings."));
    }

    #[test]
    fn a_long_list_is_truncated_with_a_count() {
        let details = (0..200).map(|i| format!("diagnostic number {i}")).collect();
        let text = text_for(&invalid(details), "hint").unwrap();
        let body = layout_body(&text, 60, 20);
        assert!(body.len() <= 20);
        let joined = body.join("\n");
        assert!(joined.contains("more (see the log)"), "{joined}");
        assert!(joined.contains("hint"), "the footer survives truncation");
    }

    #[test]
    fn none_says_how_to_assign_one() {
        let text = text_for(&PipelineStatus::None, "Use World Settings.").unwrap();
        assert_eq!(text.title, "No render pipeline assigned");
        let body = layout_body(&text, 60, 20).join("\n");
        assert!(body.contains("To assign one: Use World Settings."));
        assert!(text_for(&PipelineStatus::Ready, "").is_none());
    }
}
