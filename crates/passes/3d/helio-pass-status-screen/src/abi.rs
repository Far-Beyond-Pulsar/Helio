//! The shader's inputs: the uniform, the line and rectangle records and the
//! character buffer. `shaders/status_screen.wgsl` declares the same structs;
//! `tests/abi.rs` parses the shader and compares every offset and size, so the
//! two cannot drift.

use bytemuck::{Pod, Zeroable};

/// Most text runs one screen draws (a run is one line, or part of a line
/// between code chips).
pub const MAX_LINES: usize = 1024;
/// Most rectangles (chips, rules, dots) one screen draws.
pub const MAX_RECTS: usize = 1024;
/// Most characters one screen draws.
pub const MAX_CHARS: usize = 65_536;

/// The `Params` uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Params {
    /// The frame, in physical pixels.
    pub screen: [f32; 2],
    /// The layout unit, in pixels.
    pub unit: f32,
    /// [`FLAG_SRGB_TARGET`].
    pub flags: u32,
    /// The dialog: x, y, width, height, in pixels.
    pub dialog: [f32; 4],
    /// Corner radius, border width, shadow sigma, shadow offset down; pixels.
    pub shape: [f32; 4],
    /// First glyph of the atlas, atlas columns, texels per glyph, 0.
    pub glyphs: [u32; 4],
    /// Text runs, rectangles, 0, 0.
    pub counts: [u32; 4],
    /// Glyph width over cell height, regular and bold glyph coverage
    /// thresholds, 0.
    pub text: [f32; 4],
    /// The icon: centre x, centre y, size in pixels, [`ICON_ALERT`] or [`ICON_IDLE`].
    pub icon: [f32; 4],
    /// Colours, in display (sRGB) space; see [`crate::theme`].
    pub void_color: [f32; 4],
    pub shadow: [f32; 4],
    pub panel_a: [f32; 4],
    pub panel_b: [f32; 4],
    pub edge: [f32; 4],
    pub rim: [f32; 4],
    pub chip_border: [f32; 4],
    pub icon_ring: [f32; 4],
    pub icon_ink: [f32; 4],
}

/// `Params::flags`: the target is an sRGB format, so the shader (which works
/// in display space, as the design does) converts its result to linear.
pub const FLAG_SRGB_TARGET: u32 = 1;
/// `Params::icon.w`: the ring with an exclamation mark.
pub const ICON_ALERT: f32 = 0.0;
/// `Params::icon.w`: the dashed ring with a dash.
pub const ICON_IDLE: f32 = 1.0;

/// A run of characters on one baseline: `Line` in the shader. Characters
/// advance by `place.z`; each glyph is `place.w * Params::text.x` wide and
/// centred in its advance.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Line {
    /// Top-left of the first cell, the advance between cells, the cell height.
    pub place: [f32; 4],
    pub color: [f32; 4],
    /// Index of the first character, number of characters, [`LINE_BOLD`], 0.
    pub text: [u32; 4],
}

/// `Line::text[2]`: draw the glyphs heavier.
pub const LINE_BOLD: u32 = 1;

/// A filled rounded rectangle: `Rect` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Rect {
    /// x, y, width, height, in pixels.
    pub bounds: [f32; 4],
    pub color: [f32; 4],
    /// Corner radius in pixels, one of the `RECT_*` kinds, 0, 0.
    pub shape: [f32; 4],
}

/// A flat fill.
pub const RECT_SOLID: f32 = 0.0;
/// A fill that fades to nothing from left to right.
pub const RECT_FADE: f32 = 1.0;
/// A fill with the chip border (`Params::chip_border`) around it.
pub const RECT_BORDERED: f32 = 2.0;
