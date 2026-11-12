//! The status screen's design tokens (the approved mockup's CSS variables).
//! Colours are sRGB, as designed; the shader converts for an sRGB target.

use crate::content::Severity;

/// `#rrggbb` and an alpha, as the four floats the shader takes.
pub const fn rgba(hex: u32, alpha: f32) -> [f32; 4] {
    [
        ((hex >> 16) & 0xff) as f32 / 255.0,
        ((hex >> 8) & 0xff) as f32 / 255.0,
        (hex & 0xff) as f32 / 255.0,
        alpha,
    ]
}

/// The field around the dialog; the alpha is how dark the vignette gets.
pub const VOID: [f32; 4] = rgba(0x040303, 0.55);
/// The contact shadow under the dialog.
pub const SHADOW: [f32; 4] = rgba(0x000000, 0.55);
/// The 168 degree panel gradient.
pub const PANEL_A: [f32; 4] = rgba(0x2f0b0d, 1.0);
pub const PANEL_B: [f32; 4] = rgba(0x1c0708, 1.0);
/// The 1px edge: something failed, and there is just nothing to draw.
pub const EDGE_ALERT: [f32; 4] = rgba(0xff5a52, 0.55);
pub const EDGE_IDLE: [f32; 4] = rgba(0xff6e64, 0.34);
/// The inset rim light on the panel's top edge.
pub const RIM: [f32; 4] = rgba(0xffbeb4, 0.10);
/// The border of a code chip in the hint.
pub const CHIP_BORDER: [f32; 4] = rgba(0xff786e, 0.25);
/// The fill of a code chip in a detail, and in the hint.
pub const CHIP_FILL: [f32; 4] = rgba(0xff786e, 0.12);
pub const CHIP_FILL_HINT: [f32; 4] = rgba(0xff786e, 0.14);
/// The rule after the section label, and the divider over the hint.
pub const RULE_FADE: [f32; 4] = rgba(0xff786e, 0.28);
pub const RULE: [f32; 4] = rgba(0xff786e, 0.20);

pub const INK: [f32; 4] = rgba(0xffe3df, 1.0);
pub const INK_SOFT: [f32; 4] = rgba(0xd49a95, 1.0);
pub const INK_DIM: [f32; 4] = rgba(0x9a6b67, 1.0);
pub const TITLE: [f32; 4] = rgba(0xffffff, 1.0);
pub const ERR: [f32; 4] = rgba(0xff7a70, 1.0);
pub const WARN: [f32; 4] = rgba(0xffc15e, 1.0);
pub const NOTE: [f32; 4] = rgba(0xa9b4c2, 1.0);

/// The icon: the alert ring and mark, and the idle dashed ring and dash.
pub const ICON_RING_ALERT: [f32; 4] = rgba(0xff5a52, 0.9);
pub const ICON_INK_ALERT: [f32; 4] = rgba(0xffffff, 1.0);
pub const ICON_RING_IDLE: [f32; 4] = rgba(0xff8a80, 0.75);
pub const ICON_INK_IDLE: [f32; 4] = rgba(0xffd9d4, 1.0);

/// The colour of a detail's severity mark.
pub const fn severity(severity: Severity) -> [f32; 4] {
    match severity {
        Severity::Error => ERR,
        Severity::Warning => WARN,
        Severity::Note => NOTE,
    }
}
