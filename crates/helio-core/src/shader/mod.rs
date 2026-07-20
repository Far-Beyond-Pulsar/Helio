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

/// Lines the prelude adds ahead of a shader's own source, for offsetting
/// diagnostics back to the original file.
pub fn prelude_lines() -> usize {
    PRELUDE.lines().count() + 1
}

/// Returns `true` if `source` opts into the prelude.
pub fn uses_prelude(source: &str) -> bool {
    source.contains(MARKER)
}

/// Expands a shader source to what the GPU actually compiles.
///
/// The single point of truth for prelude expansion: [`module`] and the
/// `wgsl_validation` test both go through here, so the test validates exactly
/// what the runtime builds rather than an approximation of it.
pub fn resolve(source: &str) -> Cow<'_, str> {
    if uses_prelude(source) {
        Cow::Owned(format!("{PRELUDE}\n{source}"))
    } else {
        Cow::Borrowed(source)
    }
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
    fn prelude_line_count_matches_what_resolve_prepends() {
        let src = "//!use helio_prelude\nfoo";
        let resolved = resolve(src);
        let offset = resolved.lines().count() - src.lines().count();
        assert_eq!(offset, prelude_lines());
    }
}
