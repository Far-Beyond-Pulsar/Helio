//! What a status screen says, before it is laid out for a size.

/// How serious one detail line is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    /// What a line with no marker is.
    Note,
}

impl Severity {
    /// The marker a detail line carries to say its severity. This is the
    /// convention `PipelineFailure::details` follows: a graph validation
    /// diagnostic already prints as `error: ...` / `warning: ...`.
    pub const fn marker(self) -> &'static str {
        match self {
            Self::Error => "error:",
            Self::Warning => "warning:",
            Self::Note => "note:",
        }
    }

    /// `line` without its leading marker, and the severity the marker names
    /// (case-insensitive). A line with no marker is a [`Severity::Note`].
    pub fn split(line: &str) -> (Self, &str) {
        let line = line.trim_start();
        for severity in [Self::Error, Self::Warning, Self::Note] {
            let marker = severity.marker();
            if line.len() >= marker.len()
                && line.is_char_boundary(marker.len())
                && line[..marker.len()].eq_ignore_ascii_case(marker)
            {
                return (severity, line[marker.len()..].trim_start());
            }
        }
        (Self::Note, line)
    }
}

/// One line behind the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detail {
    pub severity: Severity,
    /// The text; an identifier between backticks is drawn as a code chip.
    pub text: String,
}

impl Detail {
    /// A detail from a line that may carry a severity marker.
    pub fn parse(line: &str) -> Self {
        let (severity, text) = Severity::split(line);
        Self {
            severity,
            text: text.to_owned(),
        }
    }
}

/// How alarmed the screen looks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tone {
    /// Something failed: the edge and the kicker are red.
    #[default]
    Alert,
    /// Nothing is wrong, there is just nothing to draw (no pipeline): the
    /// same panel with a dimmer edge, a dashed icon and a quieter kicker.
    Idle,
}

/// The text of a status screen. [`crate::layout::compute`] lays it out for a
/// frame size.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StatusContent {
    pub tone: Tone,
    /// The small uppercase line over the title (`Pipeline invalid`).
    pub kicker: String,
    pub title: String,
    /// The id of the pipeline the screen is about.
    pub pipeline: Option<String>,
    /// What went wrong, in a sentence.
    pub reason: String,
    pub details: Vec<Detail>,
    /// What to do about it.
    pub hint: String,
    /// The pipeline is built again when a shader, plugin or its asset changes.
    pub rebuilds: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marker_sets_the_severity_and_is_removed() {
        assert_eq!(
            Severity::split("error: node `a` is missing"),
            (Severity::Error, "node `a` is missing")
        );
        assert_eq!(
            Severity::split("Warning:   routing: x"),
            (Severity::Warning, "routing: x")
        );
        assert_eq!(Severity::split("NOTE: ok"), (Severity::Note, "ok"));
    }

    #[test]
    fn a_line_with_no_marker_is_a_note() {
        assert_eq!(
            Severity::split("adapter has 8 GiB"),
            (Severity::Note, "adapter has 8 GiB")
        );
        // A marker must be the whole first word's prefix, not a substring.
        assert_eq!(Severity::split("errors: 3").0, Severity::Note);
        assert_eq!(Severity::split("").0, Severity::Note);
        assert_eq!(Severity::split("\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}").0, Severity::Note);
    }

    #[test]
    fn every_marker_round_trips() {
        for severity in [Severity::Error, Severity::Warning, Severity::Note] {
            let line = format!("{} text", severity.marker());
            assert_eq!(Detail::parse(&line).severity, severity);
            assert_eq!(Detail::parse(&line).text, "text");
        }
    }
}
