//! Lays a [`StatusContent`] out for a frame size: the dialog's rectangle, and
//! the text runs, chips, rules and icon inside it, all in physical pixels.
//!
//! Everything is measured in one unit, `u = clamp(10px, 1.9% of the shorter
//! side, 17px)`, so type and spacing scale with the view. The dialog is as
//! wide as its longest line up to `64u` (at least `30u`, never more than 90%
//! of the width), wraps what does not fit, and is never taller than 90% of the
//! height: when the details do not fit, they are dropped from the end and
//! counted (`... and N more (see the log)`), then the other blocks are
//! shortened.
//!
//! The text is the engine's one 8x8 bitmap font (`helio_pass_debug_overlay`),
//! so every character has the same advance; each style has its own size and
//! spacing. Wrapping is in pixels, over those advances.

use crate::abi::{
    Line, Params, Rect, FLAG_SRGB_TARGET, ICON_ALERT, ICON_IDLE, LINE_BOLD, MAX_CHARS, MAX_LINES,
    MAX_RECTS, RECT_BORDERED, RECT_FADE, RECT_SOLID,
};
use crate::content::{Severity, StatusContent, Tone};
use crate::theme;

/// Advance over font size, for the fixed-pitch and the display styles.
const MONO_ADVANCE: f32 = 0.68;
const SANS_ADVANCE: f32 = 0.62;
/// Glyph cell height over font size.
const CELL: f32 = 0.95;
/// Glyph width over cell height, in the shader.
pub const GLYPH_ASPECT: f32 = 0.60;
/// Coverage a glyph pixel needs to count: regular and bold text.
pub const THRESHOLD_REGULAR: f32 = 0.5;
pub const THRESHOLD_BOLD: f32 = 0.34;

/// What fraction of the frame the dialog may fill, per side.
pub const MAX_FRACTION: f32 = 0.9;

/// The layout unit for a frame, in pixels.
pub fn unit_for(width: u32, height: u32) -> f32 {
    (1.9 * width.min(height) as f32 / 100.0).clamp(10.0, 17.0)
}

/// A laid-out screen: what the shader draws.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub screen: [u32; 2],
    pub unit: f32,
    /// The dialog: x, y, width, height.
    pub dialog: [f32; 4],
    pub radius: f32,
    pub lines: Vec<Line>,
    pub rects: Vec<Rect>,
    pub chars: Vec<u32>,
    /// Centre x, centre y, size, kind.
    pub icon: [f32; 4],
    pub tone: Tone,
    /// Details drawn, and details left out.
    pub details_shown: usize,
    pub details_hidden: usize,
    /// The text runs as strings, in drawing order (for tests and logs).
    pub text: Vec<String>,
}

impl Layout {
    /// The uniform for this layout.
    pub fn params(&self, srgb_target: bool) -> Params {
        let u = self.unit;
        let edge = match self.tone {
            Tone::Alert => theme::EDGE_ALERT,
            Tone::Idle => theme::EDGE_IDLE,
        };
        let (icon_ring, icon_ink) = match self.tone {
            Tone::Alert => (theme::ICON_RING_ALERT, theme::ICON_INK_ALERT),
            Tone::Idle => (theme::ICON_RING_IDLE, theme::ICON_INK_IDLE),
        };
        Params {
            screen: [self.screen[0] as f32, self.screen[1] as f32],
            unit: u,
            flags: if srgb_target { FLAG_SRGB_TARGET } else { 0 },
            dialog: self.dialog,
            shape: [self.radius, 1.0, 1.6 * u, 1.2 * u],
            glyphs: [
                helio_pass_debug_overlay::FONT_FIRST_GLYPH,
                helio_pass_debug_overlay::FONT_ATLAS_COLUMNS,
                helio_pass_debug_overlay::FONT_GLYPH_TEXELS,
                0,
            ],
            counts: [self.lines.len() as u32, self.rects.len() as u32, 0, 0],
            text: [GLYPH_ASPECT, THRESHOLD_REGULAR, THRESHOLD_BOLD, 0.0],
            icon: self.icon,
            void_color: theme::VOID,
            shadow: theme::SHADOW,
            panel_a: theme::PANEL_A,
            panel_b: theme::PANEL_B,
            edge,
            rim: theme::RIM,
            chip_border: theme::CHIP_BORDER,
            icon_ring,
            icon_ink,
        }
    }
}

/// Lays `content` out for a `width` x `height` frame.
pub fn compute(content: &StatusContent, width: u32, height: u32) -> Layout {
    let (width, height) = (width.max(1), height.max(1));
    let unit = unit_for(width, height);
    let screen = [width as f32, height as f32];
    let limit = MAX_FRACTION * screen[1];
    let fits = |layout: &Option<Composed>| {
        layout
            .as_ref()
            .is_some_and(|composed| composed.height <= limit)
    };

    let total = content.details.len();
    let mut budget = Budget::full(total);
    let mut composed = compose(content, screen, unit, &budget);
    if !fits(&composed) && total > 0 {
        // The most details that fit: the layout only grows with them.
        let mut best: Option<(usize, Composed)> = None;
        let (mut low, mut high) = (0usize, (total - 1).min(MAX_LINES));
        while low <= high {
            let mid = (low + high) / 2;
            let trial = compose(content, screen, unit, &Budget { details: mid, ..budget.clone() });
            if fits(&trial) {
                best = trial.map(|t| (mid, t));
                low = mid + 1;
            } else if mid == 0 {
                break;
            } else {
                high = mid - 1;
            }
        }
        // Not even one detail fits beside everything else: shorten the other
        // blocks, the least important first, before dropping them all.
        if best.is_none() {
            let mut shrunk = Budget {
                details: 1,
                ..budget.clone()
            };
            for step in Budget::steps().into_iter().take(Budget::KEEP_DETAIL_STEPS) {
                step(&mut shrunk);
                let trial = compose(content, screen, unit, &shrunk);
                if fits(&trial) {
                    best = trial.map(|t| (1, t));
                    budget = shrunk;
                    break;
                }
            }
        }
        budget.details = best.as_ref().map_or(0, |(count, _)| *count);
        composed = match best {
            Some((_, best)) => Some(best),
            None => compose(content, screen, unit, &budget),
        };
    }
    // Still too tall (a long reason or hint on a small frame): shorten the
    // other blocks, the least important first.
    for step in Budget::steps() {
        if fits(&composed) {
            break;
        }
        step(&mut budget);
        composed = compose(content, screen, unit, &budget);
    }
    // Not even the header fits (a frame a few dozen pixels tall): draw what
    // there is; nothing else is possible.
    let composed = composed.unwrap_or_else(|| {
        compose(content, screen, unit, &Budget::minimal()).expect("the header always composes")
    });

    let size = [composed.width.ceil(), composed.height.ceil()];
    let origin = [
        ((screen[0] - size[0]) / 2.0).round().max(0.0),
        ((screen[1] - size[1]) / 2.0).round().max(0.0),
    ];
    let mut lines = composed.lines;
    let mut rects = composed.rects;
    for line in &mut lines {
        line.place[0] += origin[0];
        line.place[1] += origin[1];
    }
    for rect in &mut rects {
        rect.bounds[0] += origin[0];
        rect.bounds[1] += origin[1];
    }
    Layout {
        screen: [width, height],
        unit,
        dialog: [origin[0], origin[1], size[0], size[1]],
        radius: 1.9 * unit,
        lines,
        rects,
        chars: composed.chars,
        icon: [
            composed.icon[0] + origin[0],
            composed.icon[1] + origin[1],
            composed.icon[2],
            match content.tone {
                Tone::Alert => ICON_ALERT,
                Tone::Idle => ICON_IDLE,
            },
        ],
        tone: content.tone,
        details_shown: composed.details_shown,
        details_hidden: total - composed.details_shown,
        text: composed.text,
    }
}

// ---------------------------------------------------------------------------
// Budget: what the dialog may spend, shrunk until it fits.

#[derive(Clone, Debug)]
struct Budget {
    details: usize,
    /// Whether the details section is drawn at all.
    section: bool,
    reason_lines: usize,
    pipeline_lines: usize,
    hint_lines: usize,
    hint: bool,
    rebuilds: bool,
}

impl Budget {
    fn full(details: usize) -> Self {
        Self {
            details,
            section: true,
            reason_lines: 6,
            pipeline_lines: 3,
            hint_lines: 6,
            hint: true,
            rebuilds: true,
        }
    }

    fn minimal() -> Self {
        Self {
            details: 0,
            section: false,
            reason_lines: 0,
            pipeline_lines: 0,
            hint_lines: 0,
            hint: false,
            rebuilds: false,
        }
    }

    /// The first steps that keep the details (the rest drop blocks).
    const KEEP_DETAIL_STEPS: usize = 6;

    #[allow(clippy::type_complexity)]
    fn steps() -> [fn(&mut Budget); 10] {
        [
            |b| b.hint_lines = 3,
            |b| b.reason_lines = 3,
            |b| b.pipeline_lines = 1,
            |b| b.rebuilds = false,
            |b| b.hint_lines = 1,
            |b| b.reason_lines = 1,
            |b| b.hint = false,
            |b| b.pipeline_lines = 0,
            |b| b.section = false,
            |b| b.reason_lines = 0,
        ]
    }
}

// ---------------------------------------------------------------------------
// Text cells and wrapping.

/// One character of a run: printable ASCII (the font has nothing else), and
/// whether it is inside a code chip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    ch: u8,
    code: bool,
}

fn ascii(c: char) -> u8 {
    match c {
        ' '..='~' => c as u8,
        '\t' => b' ',
        _ => b'?',
    }
}

/// `text` as one run of cells per paragraph. With `chips`, an identifier in
/// backticks is a code chip (an unpaired backtick is dropped).
fn paragraphs(text: &str, chips: bool, upper: bool) -> Vec<Vec<Cell>> {
    text.lines()
        .map(|line| {
            let mut cells = Vec::with_capacity(line.len());
            let parts: Vec<&str> = if chips {
                line.split('`').collect()
            } else {
                vec![line]
            };
            // Parts alternate plain, code, plain...; the last code part has
            // no closing backtick when the count is even.
            let closed = parts.len() % 2 == 1;
            for (index, part) in parts.iter().enumerate() {
                let code = chips && index % 2 == 1 && (closed || index + 1 < parts.len());
                for c in part.chars() {
                    let ch = ascii(c);
                    cells.push(Cell {
                        ch: if upper { ch.to_ascii_uppercase() } else { ch },
                        code,
                    });
                }
            }
            cells
        })
        .collect()
}

/// How wide cells are: plain text, chip text, and the chip's padding.
#[derive(Clone, Copy, Debug)]
struct Metrics {
    advance: f32,
    code_advance: f32,
    pad: f32,
}

fn measure(cells: &[Cell], m: &Metrics) -> f32 {
    let mut width = 0.0;
    let mut previous = false;
    for cell in cells {
        if cell.code != previous {
            width += m.pad;
        }
        width += if cell.code { m.code_advance } else { m.advance };
        previous = cell.code;
    }
    if previous {
        width += m.pad;
    }
    width
}

/// Breaks each paragraph into lines no wider than `max` pixels: at spaces
/// where it can, inside a word that is longer than a line where it cannot.
fn wrap(paragraphs: &[Vec<Cell>], max: f32, m: &Metrics) -> Vec<Vec<Cell>> {
    let mut lines = Vec::new();
    for paragraph in paragraphs {
        let mut current: Vec<Cell> = Vec::new();
        let mut produced = false;
        for word in paragraph.split(|c| c.ch == b' ').filter(|w| !w.is_empty()) {
            let mut rest = word;
            loop {
                let mut trial = current.clone();
                if let Some(last) = trial.last() {
                    let code = last.code && rest[0].code;
                    trial.push(Cell { ch: b' ', code });
                }
                trial.extend_from_slice(rest);
                if measure(&trial, m) <= max {
                    current = trial;
                    break;
                }
                if !current.is_empty() {
                    lines.push(std::mem::take(&mut current));
                    produced = true;
                    continue;
                }
                // Longer than a whole line: split it (at least one character
                // per line, so this ends).
                let mut take = 1;
                while take < rest.len() && measure(&rest[..=take], m) <= max {
                    take += 1;
                }
                lines.push(rest[..take].to_vec());
                produced = true;
                rest = &rest[take..];
                if rest.is_empty() {
                    break;
                }
            }
        }
        if !current.is_empty() || !produced {
            lines.push(current);
        }
    }
    lines
}

/// `lines` cut to `max`, the last one ending in `...` when something was cut.
fn clip(mut lines: Vec<Vec<Cell>>, max: usize) -> Vec<Vec<Cell>> {
    if lines.len() > max {
        lines.truncate(max.max(1));
        if let Some(last) = lines.last_mut() {
            let code = last.last().is_some_and(|c| c.code);
            let keep = last.len().saturating_sub(3);
            last.truncate(keep);
            last.extend((0..3).map(|_| Cell { ch: b'.', code }));
        }
    }
    lines
}

// ---------------------------------------------------------------------------
// Styles.

#[derive(Clone, Copy, Debug)]
struct Face {
    /// Font size in pixels.
    size: f32,
    advance: f32,
    color: [f32; 4],
    bold: bool,
}

impl Face {
    fn cell(&self) -> f32 {
        CELL * self.size
    }
}

#[derive(Clone, Copy, Debug)]
struct Chip {
    face: Face,
    pad: f32,
    height: f32,
    radius: f32,
    fill: [f32; 4],
    bordered: bool,
}

#[derive(Clone, Copy, Debug)]
struct Style {
    text: Face,
    chip: Chip,
}

impl Style {
    fn metrics(&self) -> Metrics {
        Metrics {
            advance: self.text.advance,
            code_advance: self.chip.face.advance,
            pad: self.chip.pad,
        }
    }
}

fn face(size: f32, family: f32, tracking: f32, color: [f32; 4], bold: bool) -> Face {
    Face {
        size,
        advance: size * (family + tracking),
        color,
        bold,
    }
}

/// A style whose code chips are `chip_size` fixed-pitch text on a fill.
fn style(text: Face, u: f32, chip_size: f32, pad_em: f32, fill: [f32; 4], bordered: bool) -> Style {
    let chip_face = face(chip_size, MONO_ADVANCE, 0.0, theme::INK, false);
    Style {
        text,
        chip: Chip {
            face: chip_face,
            pad: pad_em * chip_size,
            height: 1.45 * chip_size,
            radius: if bordered { 0.35 * u } else { 0.3 * u },
            fill,
            bordered,
        },
    }
}

// ---------------------------------------------------------------------------
// Composing: everything relative to the dialog's top-left corner.

struct Composed {
    width: f32,
    height: f32,
    lines: Vec<Line>,
    rects: Vec<Rect>,
    chars: Vec<u32>,
    icon: [f32; 3],
    details_shown: usize,
    text: Vec<String>,
}

#[derive(Default)]
struct Builder {
    lines: Vec<Line>,
    rects: Vec<Rect>,
    chars: Vec<u32>,
    text: Vec<String>,
    overflow: bool,
}

impl Builder {
    fn rect(&mut self, bounds: [f32; 4], color: [f32; 4], radius: f32, kind: f32) {
        if self.rects.len() >= MAX_RECTS {
            self.overflow = true;
            return;
        }
        self.rects.push(Rect {
            bounds,
            color,
            shape: [radius, kind, 0.0, 0.0],
        });
    }

    /// One run of characters whose cell starts at (`x`, `y`).
    fn run(&mut self, x: f32, y: f32, face: &Face, cells: &[Cell], color: [f32; 4]) {
        if cells.is_empty() {
            return;
        }
        if self.lines.len() >= MAX_LINES || self.chars.len() + cells.len() > MAX_CHARS {
            self.overflow = true;
            return;
        }
        self.lines.push(Line {
            place: [x.round(), y.round(), face.advance, face.cell()],
            color,
            text: [
                self.chars.len() as u32,
                cells.len() as u32,
                if face.bold { LINE_BOLD } else { 0 },
                0,
            ],
        });
        self.chars.extend(cells.iter().map(|c| u32::from(c.ch)));
        self.text
            .push(cells.iter().map(|c| c.ch as char).collect());
    }

    /// A wrapped line in a box `lh` tall at `top`, with its chips. Returns
    /// nothing: the caller knows the box.
    fn line(&mut self, x: f32, top: f32, lh: f32, cells: &[Cell], style: &Style) {
        let mut x = x;
        let mut index = 0;
        while index < cells.len() {
            let code = cells[index].code;
            let end = cells[index..]
                .iter()
                .position(|c| c.code != code)
                .map_or(cells.len(), |n| index + n);
            let face = if code { &style.chip.face } else { &style.text };
            let width = (end - index) as f32 * face.advance;
            let y = top + (lh - face.cell()) / 2.0;
            if code {
                let chip = &style.chip;
                x += chip.pad;
                self.rect(
                    [
                        (x - chip.pad).round(),
                        (top + (lh - chip.height) / 2.0).round(),
                        (width + 2.0 * chip.pad).round(),
                        chip.height.round(),
                    ],
                    chip.fill,
                    chip.radius,
                    if chip.bordered {
                        RECT_BORDERED
                    } else {
                        RECT_SOLID
                    },
                );
                self.run(x, y, &style.chip.face, &cells[index..end], theme::INK);
                x += width + chip.pad;
            } else {
                self.run(x, y, face, &cells[index..end], face.color);
                x += width;
            }
            index = end;
        }
    }
}

fn natural_width(lines: &[Vec<Cell>], m: &Metrics) -> f32 {
    lines.iter().map(|l| measure(l, m)).fold(0.0, f32::max)
}

fn compose(content: &StatusContent, screen: [f32; 2], u: f32, budget: &Budget) -> Option<Composed> {
    let shown = budget.details.min(content.details.len());
    let hidden = if budget.section { content.details.len() - shown } else { 0 };

    // The styles.
    let kicker_color = match content.tone {
        Tone::Alert => theme::ERR,
        Tone::Idle => theme::INK_SOFT,
    };
    let kicker = face(0.72 * u, MONO_ADVANCE, 0.14, kicker_color, false);
    let title = face(1.55 * u, SANS_ADVANCE, 0.0, theme::TITLE, true);
    let label = face(0.68 * u, MONO_ADVANCE, 0.10, theme::INK_DIM, false);
    let pipe_id = face(0.86 * u, MONO_ADVANCE, 0.0, theme::INK, false);
    let section = face(0.70 * u, MONO_ADVANCE, 0.14, theme::INK_DIM, false);
    let reason_style = style(
        face(1.08 * u, SANS_ADVANCE, 0.0, theme::INK, false),
        u,
        0.86 * u,
        0.35,
        theme::CHIP_FILL,
        false,
    );
    let row_style = style(
        face(0.86 * u, MONO_ADVANCE, 0.0, theme::INK_SOFT, false),
        u,
        0.86 * u,
        0.35,
        theme::CHIP_FILL,
        false,
    );
    let more_face = face(0.86 * u, MONO_ADVANCE, 0.0, theme::INK_DIM, false);
    let hint_style = style(
        face(0.98 * u, SANS_ADVANCE, 0.0, theme::INK_SOFT, false),
        u,
        0.86 * u,
        0.45,
        theme::CHIP_FILL_HINT,
        true,
    );
    let live_face = face(0.78 * u, MONO_ADVANCE, 0.0, theme::INK_DIM, false);

    let plain = |face: &Face| Metrics {
        advance: face.advance,
        code_advance: face.advance,
        pad: 0.0,
    };
    let reason_cells = paragraphs(&content.reason, true, false);
    let hint_cells = paragraphs(&content.hint, true, false);
    let detail_cells: Vec<Vec<Vec<Cell>>> = content.details[..shown]
        .iter()
        .map(|d| paragraphs(&d.text, true, false))
        .collect();
    let more_text = format!("... and {hidden} more (see the log)");
    let more_cells = paragraphs(&more_text, false, false);
    let live_text = "Rebuilds when a shader, plugin or the asset changes";
    let live_cells = paragraphs(live_text, false, false);
    let id_cells = content
        .pipeline
        .as_ref()
        .map(|id| paragraphs(id, false, false));
    let label_cells = paragraphs("PIPELINE", false, false);
    let kicker_cells = paragraphs(&content.kicker, false, true);
    let title_cells = paragraphs(&content.title, false, false);
    let section_text = format!("DETAILS - {}", content.details.len());
    let section_cells = paragraphs(&section_text, false, false);

    let icon = 3.2 * u;
    let icon_gap = 1.2 * u;
    let mark_col = 1.1 * u + 0.7 * u;
    let rm = reason_style.metrics();
    let dm = row_style.metrics();
    let hm = hint_style.metrics();

    // The width: the longest line, within the limits.
    let has_details = budget.section && !content.details.is_empty();
    let mut natural = icon
        + icon_gap
        + natural_width(&kicker_cells, &plain(&kicker)).max(natural_width(&title_cells, &plain(&title)));
    if let Some(id) = &id_cells {
        natural = natural.max(
            natural_width(&label_cells, &plain(&label))
                + 0.8 * u
                + natural_width(id, &plain(&pipe_id)),
        );
    }
    natural = natural.max(natural_width(&reason_cells, &rm));
    for cells in &detail_cells {
        natural = natural.max(mark_col + natural_width(cells, &dm));
    }
    if hidden > 0 {
        natural = natural.max(mark_col + natural_width(&more_cells, &plain(&more_face)));
    }
    if has_details {
        natural = natural.max(natural_width(&section_cells, &plain(&section)));
    }
    if budget.hint {
        natural = natural.max(natural_width(&hint_cells, &hm));
    }
    if budget.rebuilds && content.rebuilds {
        natural = natural.max(0.55 * u + 0.6 * u + natural_width(&live_cells, &plain(&live_face)));
    }
    let (pad_x, pad_top, pad_bottom) = (2.2 * u, 1.9 * u, 1.7 * u);
    let max_width = (MAX_FRACTION * screen[0]).min(64.0 * u);
    let min_width = (30.0 * u).min(MAX_FRACTION * screen[0]);
    let width = (natural + 2.0 * pad_x).clamp(min_width, max_width).ceil();
    let inner = width - 2.0 * pad_x;
    let x0 = pad_x;

    let mut b = Builder::default();
    let mut y = pad_top;

    // Header: the icon, the kicker and the title.
    let title_lines = clip(wrap(&title_cells, inner - icon - icon_gap, &plain(&title)), 2);
    let title_lh = 1.2 * title.size;
    let block = kicker.size + 0.4 * u + title_lines.len() as f32 * title_lh;
    let header = icon.max(block);
    let text_x = x0 + icon + icon_gap;
    let mut ty = y + (header - block) / 2.0;
    for line in &kicker_cells {
        b.run(text_x, ty + (kicker.size - kicker.cell()) / 2.0, &kicker, line, kicker.color);
    }
    ty += kicker.size + 0.4 * u;
    for line in &title_lines {
        b.run(text_x, ty + (title_lh - title.cell()) / 2.0, &title, line, title.color);
        ty += title_lh;
    }
    let icon_center = [x0 + icon / 2.0, y + header / 2.0, icon];
    y += header;

    // The pipeline id.
    if let (Some(id), true) = (&id_cells, budget.pipeline_lines > 0) {
        y += 1.3 * u;
        let label_width = natural_width(&label_cells, &plain(&label));
        let id_x = x0 + label_width + 0.8 * u;
        let lh = 1.35 * pipe_id.size;
        let lines = clip(
            wrap(id, inner - label_width - 0.8 * u, &plain(&pipe_id)),
            budget.pipeline_lines,
        );
        b.run(
            x0,
            y + (lh - label.cell()) / 2.0,
            &label,
            &label_cells[0],
            label.color,
        );
        for line in &lines {
            b.run(id_x, y + (lh - pipe_id.cell()) / 2.0, &pipe_id, line, pipe_id.color);
            y += lh;
        }
    }

    // The reason.
    if !content.reason.is_empty() && budget.reason_lines > 0 {
        y += 1.2 * u;
        let lh = 1.5 * reason_style.text.size;
        for line in clip(wrap(&reason_cells, inner, &rm), budget.reason_lines) {
            b.line(x0, y, lh, &line, &reason_style);
            y += lh;
        }
    }

    // The details.
    if has_details {
        y += 1.5 * u;
        let label_width = natural_width(&section_cells, &plain(&section));
        b.run(x0, y + (section.size - section.cell()) / 2.0, &section, &section_cells[0], section.color);
        let rule_x = x0 + label_width + 0.8 * section.size;
        b.rect(
            [
                rule_x.round(),
                (y + section.size / 2.0).round(),
                (x0 + inner - rule_x).round().max(0.0),
                1.0,
            ],
            theme::RULE_FADE,
            0.0,
            RECT_FADE,
        );
        y += section.size + 0.6 * u;
        let lh = 1.45 * row_style.text.size;
        let text_x = x0 + mark_col;
        let row_width = inner - mark_col;
        let mut first = true;
        let mark = |b: &mut Builder, y: f32, severity: Severity| {
            let col = 1.1 * u;
            let cell = row_style.text.cell();
            let top = y + (lh - cell) / 2.0;
            match severity {
                Severity::Note => {
                    let d = (0.34 * u).round().max(2.0);
                    b.rect(
                        [
                            (x0 + (col - d) / 2.0).round(),
                            (y + (lh - d) / 2.0).round(),
                            d,
                            d,
                        ],
                        theme::NOTE,
                        d / 2.0,
                        RECT_SOLID,
                    );
                }
                _ => {
                    let (ch, color) = match severity {
                        Severity::Error => (b'x', theme::ERR),
                        _ => (b'!', theme::WARN),
                    };
                    let mut face = row_style.text;
                    face.bold = true;
                    b.run(
                        x0 + (col - face.advance) / 2.0,
                        top,
                        &face,
                        &[Cell { ch, code: false }],
                        color,
                    );
                }
            }
        };
        for (detail, cells) in content.details[..shown].iter().zip(&detail_cells) {
            if !first {
                y += 0.45 * u;
            }
            first = false;
            let lines = clip(wrap(cells, row_width, &dm), 4);
            mark(&mut b, y, detail.severity);
            for line in &lines {
                b.line(text_x, y, lh, line, &row_style);
                y += lh;
            }
            if b.overflow {
                return None;
            }
        }
        if hidden > 0 {
            if !first {
                y += 0.45 * u;
            }
            b.run(
                text_x,
                y + (lh - more_face.cell()) / 2.0,
                &more_face,
                &more_cells[0],
                more_face.color,
            );
            y += lh;
        }
    }

    // The hint, over a divider, and the rebuild line.
    let rebuilds = budget.rebuilds && content.rebuilds;
    if budget.hint || rebuilds {
        y += 1.6 * u;
        b.rect(
            [x0, y.round(), inner.round(), 1.0],
            theme::RULE,
            0.0,
            RECT_SOLID,
        );
        y += 1.0 + 1.2 * u;
        if budget.hint && !content.hint.is_empty() {
            let lh = 1.5 * hint_style.text.size;
            for line in clip(wrap(&hint_cells, inner, &hm), budget.hint_lines) {
                b.line(x0, y, lh, &line, &hint_style);
                y += lh;
            }
        }
        if rebuilds {
            if budget.hint && !content.hint.is_empty() {
                y += 0.7 * u;
            }
            let lh = 1.3 * live_face.size;
            let dot = (0.55 * u).round().max(3.0);
            b.rect(
                [x0.round(), (y + (lh - dot) / 2.0).round(), dot, dot],
                theme::ERR,
                dot / 2.0,
                RECT_SOLID,
            );
            b.run(
                x0 + 0.55 * u + 0.6 * u,
                y + (lh - live_face.cell()) / 2.0,
                &live_face,
                &live_cells[0],
                live_face.color,
            );
            y += lh;
        }
    }
    y += pad_bottom;

    if b.overflow {
        return None;
    }
    Some(Composed {
        width,
        height: y.ceil(),
        lines: b.lines,
        rects: b.rects,
        chars: b.chars,
        icon: icon_center,
        details_shown: shown,
        text: b.text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::Detail;

    fn cells_of(text: &str) -> Vec<Vec<Cell>> {
        paragraphs(text, false, false)
    }

    fn strings(lines: &[Vec<Cell>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.iter().map(|c| c.ch as char).collect())
            .collect()
    }

    /// One pixel per character, no chip padding: widths are character counts.
    const CHARS: Metrics = Metrics {
        advance: 1.0,
        code_advance: 1.0,
        pad: 0.0,
    };

    fn sample(details: usize) -> StatusContent {
        StatusContent {
            tone: Tone::Alert,
            kicker: "Pipeline invalid".into(),
            title: "Render pipeline invalid".into(),
            pipeline: Some("assets/pipelines/forest.rendergraph".into()),
            reason: "The graph does not validate: 2 nodes are not available.".into(),
            details: (0..details)
                .map(|i| Detail {
                    severity: match i % 7 {
                        0 => Severity::Warning,
                        3 => Severity::Note,
                        _ => Severity::Error,
                    },
                    text: format!(
                        "missing node `helio.pass.layer_{}` (plugin `pulsar.legacy` disabled)",
                        200 + i
                    ),
                })
                .collect(),
            hint: "Enable the plugin in `Plugins`, delete the node, or assign another pipeline."
                .into(),
            rebuilds: true,
        }
    }

    fn idle() -> StatusContent {
        StatusContent {
            tone: Tone::Idle,
            kicker: "No pipeline".into(),
            title: "No render pipeline assigned".into(),
            pipeline: None,
            reason: "This renderer has no render pipeline, so there is nothing to draw.".into(),
            details: Vec::new(),
            hint: "Call `Renderer::set_pipeline`, or build the renderer with `RendererBuilder::with_graph`."
                .into(),
            rebuilds: false,
        }
    }

    #[test]
    fn wrap_breaks_at_spaces_and_inside_long_words() {
        let lines = wrap(&cells_of("alpha beta gamma delta"), 11.0, &CHARS);
        assert_eq!(strings(&lines), ["alpha beta", "gamma delta"]);
        let lines = strings(&wrap(&cells_of("abcdefghijklmnopqrstuvwxyz"), 10.0, &CHARS));
        assert_eq!(lines, ["abcdefghij", "klmnopqrst", "uvwxyz"]);
        assert!(lines.iter().all(|l| l.len() <= 10));
    }

    #[test]
    fn wrap_keeps_to_the_width_and_never_drops_text() {
        let text = "missing node `a.b` (plugin voxel disabled) in graph, and a_very_long_identifier_that_needs_splitting";
        let wrapped = wrap(&paragraphs(text, true, false), 24.0, &CHARS);
        assert!(wrapped.len() > 1);
        assert!(wrapped.iter().all(|l| l.len() <= 24), "{wrapped:?}");
        let joined: String = strings(&wrapped).join("");
        for word in ["missing", "voxel", "splitting", "a.b"] {
            assert!(joined.contains(word), "{joined}");
        }
    }

    #[test]
    fn wrap_counts_chip_padding_in_the_width() {
        let m = Metrics {
            advance: 1.0,
            code_advance: 1.0,
            pad: 2.0,
        };
        let cells = paragraphs("ab `cd` ef", true, false);
        // "ab " + 2 + "cd" + 2 = 7, then " ef".
        assert_eq!(measure(&cells[0], &m), 3.0 + 2.0 + 2.0 + 2.0 + 3.0);
        let lines = wrap(&cells, 9.0, &m);
        assert_eq!(strings(&lines), ["ab cd", "ef"]);
        let lines = wrap(&cells, 8.0, &m);
        assert_eq!(strings(&lines), ["ab", "cd", "ef"]);
    }

    #[test]
    fn backticks_make_chips_and_an_unpaired_one_is_dropped() {
        let cells = &paragraphs("a `b` c", true, false)[0];
        assert_eq!(
            cells.iter().filter(|c| c.code).count(),
            1,
            "{cells:?}"
        );
        let cells = &paragraphs("a `b c", true, false)[0];
        assert!(cells.iter().all(|c| !c.code));
        assert_eq!(cells.len(), 5);
    }

    #[test]
    fn non_ascii_is_replaced_not_dropped() {
        assert_eq!(strings(&cells_of("a\u{e9}\tb")), ["a? b"]);
    }

    #[test]
    fn the_unit_follows_the_shorter_side_within_limits() {
        assert!((unit_for(1280, 720) - 13.68).abs() < 1e-3);
        assert_eq!(unit_for(420, 320), 10.0, "never below 10px");
        assert_eq!(unit_for(7680, 4320), 17.0, "never above 17px");
        assert_eq!(unit_for(720, 405), 10.0);
    }

    fn assert_fits(layout: &Layout, label: &str) {
        let [x, y, w, h] = layout.dialog;
        let [sw, sh] = layout.screen.map(|v| v as f32);
        assert!(x >= 0.0 && y >= 0.0 && x + w <= sw && y + h <= sh, "{label}: inside");
        assert!(w <= 0.9 * sw + 1.0, "{label}: width {w} of {sw}");
        assert!(h <= 0.9 * sh + 1.0, "{label}: height {h} of {sh}");
        assert!(
            ((x + w / 2.0) - sw / 2.0).abs() <= 1.0 && ((y + h / 2.0) - sh / 2.0).abs() <= 1.0,
            "{label}: centred"
        );
        assert!(layout.lines.len() <= MAX_LINES && layout.chars.len() <= MAX_CHARS);
        // Every run lies in the dialog.
        for line in &layout.lines {
            let right = line.place[0] + line.text[1] as f32 * line.place[2];
            assert!(
                line.place[0] >= x - 1.0 && right <= x + w + 1.0,
                "{label}: run {:?} sticks out of [{x}, {}]",
                line.place,
                x + w
            );
            assert!(line.place[1] >= y && line.place[1] + line.place[3] <= y + h + 1.0, "{label}: run in height");
        }
    }

    #[test]
    fn every_state_fits_at_the_review_sizes() {
        for (w, h) in [(1280, 720), (720, 405), (420, 320), (1920, 1080), (3840, 2160)] {
            for (name, content) in [
                ("none", idle()),
                ("invalid", sample(3)),
                ("stress", sample(46)),
            ] {
                let layout = compute(&content, w, h);
                assert_fits(&layout, &format!("{name} {w}x{h}"));
            }
        }
    }

    #[test]
    fn a_short_message_makes_a_small_dialog() {
        let layout = compute(&idle(), 1280, 720);
        assert!(layout.dialog[2] < 0.7 * 1280.0, "{:?}", layout.dialog);
        assert!(layout.dialog[3] < 0.4 * 720.0, "{:?}", layout.dialog);
        assert_eq!(layout.details_hidden, 0);
    }

    #[test]
    fn the_dialog_is_at_least_30_units_and_at_most_64() {
        let u = unit_for(1280, 720);
        let mut short = idle();
        short.title = "Hi".into();
        short.kicker = "A".into();
        short.reason = "B".into();
        short.hint = "C".into();
        let layout = compute(&short, 1280, 720);
        assert!(layout.dialog[2] >= 30.0 * u - 1.0, "{:?}", layout.dialog);
        let layout = compute(&sample(5), 1280, 720);
        assert!(layout.dialog[2] <= 64.0 * u + 1.0, "{:?}", layout.dialog);
    }

    #[test]
    fn long_details_wrap_inside_the_dialog() {
        let mut content = sample(1);
        content.details[0].text = "word ".repeat(80);
        let layout = compute(&content, 1280, 720);
        assert_fits(&layout, "wrapped");
        let rows = layout
            .text
            .iter()
            .filter(|t| t.starts_with("word"))
            .count();
        assert!(rows > 1, "the detail wraps onto several lines: {:?}", layout.text);
        assert!(rows <= 4, "and stops at four, ending in `...`");
        assert!(layout.text.iter().any(|t| t.ends_with("...")));
    }

    #[test]
    fn a_long_list_is_truncated_with_a_count() {
        let layout = compute(&sample(46), 1280, 720);
        assert!(layout.details_shown > 0 && layout.details_shown < 46);
        assert_eq!(layout.details_shown + layout.details_hidden, 46);
        let more = format!("... and {} more (see the log)", layout.details_hidden);
        assert!(layout.text.contains(&more), "{more}: {:?}", layout.text);
        assert!(
            layout.text.iter().any(|t| t == "DETAILS - 46"),
            "the section counts every detail, shown or not"
        );
        // The most that fit: one more would not.
        let budget = Budget::full(layout.details_shown + 1);
        let u = unit_for(1280, 720);
        let taller = compose(&sample(46), [1280.0, 720.0], u, &budget).unwrap();
        assert!(taller.height > 0.9 * 720.0, "{}", taller.height);
    }

    #[test]
    fn a_smaller_frame_shows_fewer_details() {
        let big = compute(&sample(46), 1920, 1080);
        let small = compute(&sample(46), 420, 320);
        assert!(small.details_shown < big.details_shown);
        assert!(small.details_hidden > 0);
    }

    #[test]
    fn the_dialog_never_exceeds_90_percent_of_either_side() {
        for (w, h) in [
            (200, 120),
            (320, 200),
            (420, 320),
            (640, 100),
            (100, 600),
            (333, 777),
            (1280, 720),
            (5120, 1440),
        ] {
            let mut content = sample(60);
            content.reason = "a very long reason ".repeat(30);
            content.hint = "a very long hint ".repeat(30);
            content.pipeline = Some("p/".repeat(200));
            let layout = compute(&content, w, h);
            let [_, _, dw, dh] = layout.dialog;
            assert!(dw <= 0.9 * w as f32 + 1.0, "{w}x{h}: width {dw}");
            // The header alone needs 6.8 units, and more when the title wraps
            // in a very narrow frame; below that nothing can fit.
            if w >= 300 && (h as f32) * 0.9 >= 6.8 * layout.unit + 2.0 {
                assert!(dh <= 0.9 * h as f32 + 1.0, "{w}x{h}: height {dh}");
            }
        }
    }

    #[test]
    fn details_keep_their_severity_marks_and_chips() {
        let layout = compute(&sample(8), 1280, 720);
        assert!(layout.text.iter().any(|t| t == "x"));
        assert!(layout.text.iter().any(|t| t == "!"));
        assert!(layout.text.iter().any(|t| t == "helio.pass.layer_201"));
        assert!(
            layout.rects.iter().any(|r| r.shape[1] == RECT_BORDERED),
            "the hint chip is bordered"
        );
        assert!(layout.rects.iter().any(|r| r.shape[1] == RECT_FADE));
    }

    #[test]
    fn the_idle_screen_has_no_pipeline_row_and_a_dashed_icon() {
        let layout = compute(&idle(), 1280, 720);
        assert!(!layout.text.iter().any(|t| t == "PIPELINE"));
        assert!(!layout.text.iter().any(|t| t.starts_with("Rebuilds")));
        assert_eq!(layout.icon[3], ICON_IDLE);
        let alert = compute(&sample(1), 1280, 720);
        assert_eq!(alert.icon[3], ICON_ALERT);
        assert!(alert.text.iter().any(|t| t == "PIPELINE"));
        assert!(alert.text.iter().any(|t| t.starts_with("Rebuilds")));
    }

    #[test]
    fn the_kicker_is_uppercase() {
        let layout = compute(&sample(0), 1280, 720);
        assert!(layout.text.iter().any(|t| t == "PIPELINE INVALID"));
    }
}
