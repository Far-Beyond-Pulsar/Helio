//! Shared WGSL prelude.
//!
//! naga has no `#include`, and wgpu compiles a shader only at
//! `create_shader_module` — with a live device, at runtime. The combination is
//! why every pass ended up re-deriving the camera struct and the depth/NDC math
//! by hand, and why they drifted apart without anything catching it.
//!
//! The engine already composed shaders by string concatenation (see
//! `VHS_SHADER_SNIPPET` in the examples), so this follows the same approach
//! rather than pulling in a preprocessor.
//!
//! # Use
//!
//! Mark the shader, declare your own camera binding, drop the local copies:
//!
//! ```wgsl
//! //!use helio_prelude
//! @group(0) @binding(0) var<uniform> camera: Camera;
//! ```
//!
//! and build the module through [`module`] instead of `create_shader_module`:
//!
//! ```ignore
//! let shader = helio_core::shader::module(
//!     device,
//!     "SSR Trace Shader",
//!     include_str!("../shaders/ssr_trace.wgsl"),
//! );
//! ```
//!
//! Opting in is per-shader: a shader without the marker is passed through
//! untouched, so unmigrated passes that declare their own `Camera` keep working
//! (and would otherwise collide with the prelude's).
//!
//! # Caveat
//!
//! Prepending shifts line numbers, so naga diagnostics for a prelude-using
//! shader point into the combined source, offset by [`PRELUDE_LINES`]. That is
//! the price of concatenation over a real preprocessor; keeping the prelude small
//! and stable keeps it manageable.

use std::borrow::Cow;

/// The canonical camera struct and depth/G-buffer conventions.
pub const PRELUDE: &str = include_str!("prelude.wgsl");

/// Marker opting a shader into the prelude. Must appear in the source.
pub const MARKER: &str = "//!use helio_prelude";

/// Hi-Z screen-space ray marching, shared by SSR and water.
///
/// Separate from [`PRELUDE`] because it is only wanted by the two passes that
/// march the pyramid, and prepending it everywhere would push every other
/// shader's diagnostics further out of alignment for nothing.
pub const HIZ: &str = include_str!("hiz_trace.wgsl");

/// Marker opting a shader into the Hi-Z traversal. Must appear in the source.
pub const HIZ_MARKER: &str = "//!use helio_hiz";

/// Returns `true` if `source` opts into the prelude.
pub fn uses_prelude(source: &str) -> bool {
    source.contains(MARKER)
}

/// Returns `true` if `source` opts into the Hi-Z traversal.
pub fn uses_hiz(source: &str) -> bool {
    source.contains(HIZ_MARKER)
}

/// Lines prepended ahead of `source`, for offsetting diagnostics back to the
/// original file. Depends on which markers the source opts into.
pub fn expanded_lines(source: &str) -> usize {
    let mut lines = 0;
    if uses_prelude(source) {
        lines += PRELUDE.lines().count() + 1;
    }
    if uses_hiz(source) {
        lines += HIZ.lines().count() + 1;
    }
    lines
}

/// Expands a shader source to what the GPU actually compiles.
///
/// The single point of truth for prelude expansion: [`module`] and the
/// `wgsl_validation` test both go through here, so the test validates exactly
/// what the runtime builds rather than an approximation of it.
pub fn resolve(source: &str) -> Cow<'_, str> {
    let prelude = uses_prelude(source);
    let hiz = uses_hiz(source);
    if !prelude && !hiz {
        return Cow::Borrowed(source);
    }

    let mut out = String::new();
    if prelude {
        out.push_str(PRELUDE);
        out.push('\n');
    }
    // After the prelude, so the traversal may lean on it if it ever needs to.
    if hiz {
        out.push_str(HIZ);
        out.push('\n');
    }
    out.push_str(source);
    Cow::Owned(out)
}

/// Creates a shader module, expanding the prelude if the source opts in.
pub fn module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(resolve(source)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_without_marker_is_untouched() {
        let src = "@compute @workgroup_size(1) fn main() {}";
        assert!(matches!(resolve(src), Cow::Borrowed(_)));
        assert_eq!(resolve(src), src);
    }

    #[test]
    fn source_with_marker_gets_prelude_prepended() {
        let src = "//!use helio_prelude\n@compute @workgroup_size(1) fn main() {}";
        let out = resolve(src);
        assert!(out.contains("struct Camera"));
        assert!(out.ends_with(src));
    }

    #[test]
    fn prelude_declares_the_shared_conventions() {
        // If any of these are renamed, every migrated shader breaks at runtime;
        // pin the names so that surfaces here instead.
        for symbol in [
            "struct Camera",
            "fn helio_uv_to_ndc",
            "fn helio_ndc_to_uv",
            "fn helio_world_from_depth",
            "fn helio_view_depth",
            "fn helio_gbuffer_normal",
        ] {
            assert!(PRELUDE.contains(symbol), "prelude is missing {symbol}");
        }
    }

    #[test]
    fn expanded_line_count_matches_what_resolve_prepends() {
        // Both markers, independently and together — the reported offset is
        // what maps a diagnostic back to the file the reader will open, so it
        // has to track whatever `resolve` actually prepended.
        for src in [
            "//!use helio_prelude\nfoo",
            "//!use helio_hiz\nfoo",
            "//!use helio_prelude\n//!use helio_hiz\nfoo",
        ] {
            let resolved = resolve(src);
            let offset = resolved.lines().count() - src.lines().count();
            assert_eq!(offset, expanded_lines(src), "offset wrong for {src:?}");
        }
    }

    #[test]
    fn a_shader_opting_into_neither_is_passed_through() {
        let src = "@compute @workgroup_size(1) fn main() {}";
        assert!(matches!(resolve(src), Cow::Borrowed(_)));
        assert_eq!(expanded_lines(src), 0);
    }

    #[test]
    fn hiz_declares_the_shared_traversal() {
        // Same reasoning as the prelude: renaming these breaks SSR and water at
        // runtime, so pin them.
        for symbol in ["struct HelioHizHit", "fn helio_hiz_march"] {
            assert!(HIZ.contains(symbol), "hiz include is missing {symbol}");
        }
    }
}
