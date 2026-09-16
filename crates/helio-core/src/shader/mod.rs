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
//! @group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
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
//! # Pass-owned snippets
//!
//! [`PRELUDE`] is the one snippet `helio-core` owns directly — every pass
//! shares exactly one camera/depth convention, so it names no specific pass.
//! Anything else a pass wants pre-pended by marker (Hi-Z traversal, a
//! material PBR evaluation library, a foliage wind model, ...) is declared as
//! a [`ShaderSnippet`] by the pass crate that owns that content and passed
//! explicitly to [`resolve_with`]/[`module_with`] — `helio-core` never
//! hardcodes a specific snippet's name or source, only the generic
//! marker-plus-text shape.
//!
//! # Caveat
//!
//! Prepending shifts line numbers, so naga diagnostics for a prelude-using
//! shader point into the combined source, offset by [`expanded_lines`]. That is
//! the price of concatenation over a real preprocessor; keeping the prelude small
//! and stable keeps it manageable.

use std::borrow::Cow;

pub mod directives;
pub mod reflection;
pub use directives::{parse as parse_directives, PipelineDirectives};
pub use reflection::{
    create_bind_group_layouts, create_pipeline_layout, create_reflected_bind_groups,
    create_reflected_bind_groups_with_layouts, create_reflected_pipeline, layout_entries,
    populate_bind_group_entries, reflect, BindingKind, ReflectedBinding, ReflectedLayout,
    ReflectedPipeline, ReflectedShader, ReflectionError,
};

/// The canonical camera struct and depth/G-buffer conventions.
///
/// The one snippet `helio-core` owns directly: every pass in the graph
/// shares exactly one camera/depth-reconstruction convention, so this names
/// no specific pass (like `PassContext::camera` being a first-class field).
pub const PRELUDE: &str = include_str!("prelude.wgsl");

/// Marker opting a shader into the prelude. Must appear in the source.
pub const MARKER: &str = "//!use helio_prelude";

/// A pass-declared shader snippet: text pre-pended to a shader's source when
/// the shader contains `marker` as a WGSL comment.
///
/// Declared as a `const` by whichever pass crate owns the snippet's content
/// (e.g. `helio-pass-hiz`'s Hi-Z traversal, `helio-pass-gbuffer`'s PBR
/// evaluation library, `helio-pass-foliage-place`'s wind model) and passed
/// explicitly to [`resolve_with`]/[`module_with`] by the pass that wants it.
/// `helio-core` never declares one itself and never hardcodes a marker or
/// source string belonging to a specific domain — only this generic shape.
#[derive(Clone, Copy)]
pub struct ShaderSnippet {
    /// WGSL comment marker that must appear in a shader's source to opt in.
    pub marker: &'static str,
    /// Text pre-pended to the shader when `marker` is present.
    pub source: &'static str,
}

impl ShaderSnippet {
    pub const fn new(marker: &'static str, source: &'static str) -> Self {
        Self { marker, source }
    }

    fn used_by(&self, source: &str) -> bool {
        source.contains(self.marker)
    }
}

/// Returns `true` if `source` opts into the prelude.
pub fn uses_prelude(source: &str) -> bool {
    source.contains(MARKER)
}

/// Lines prepended ahead of `source` by [`resolve`] (prelude only).
pub fn expanded_lines(source: &str) -> usize {
    expanded_lines_with(source, &[])
}

/// Lines prepended ahead of `source` by [`resolve_with`], for offsetting
/// diagnostics back to the original file. Depends on which of `snippets`
/// (plus the prelude) the source opts into.
pub fn expanded_lines_with(source: &str, snippets: &[ShaderSnippet]) -> usize {
    let mut lines = 0;
    if uses_prelude(source) {
        lines += PRELUDE.lines().count() + 1;
    }
    for snippet in snippets {
        if snippet.used_by(source) {
            lines += snippet.source.lines().count() + 1;
        }
    }
    lines
}

/// Expands a shader source to what the GPU actually compiles, using only the
/// generic prelude. Equivalent to `resolve_with(source, &[])`.
pub fn resolve(source: &str) -> Cow<'_, str> {
    resolve_with(source, &[])
}

/// Expands a shader source, prepending the prelude (if opted into) and every
/// snippet in `snippets` whose marker the source contains, in the order
/// given.
///
/// The single point of truth for shader expansion: [`module_with`] and each
/// pass's own `wgsl_validation`-style tests should go through here, so the
/// test validates exactly what the runtime builds rather than an
/// approximation of it.
pub fn resolve_with<'a>(source: &'a str, snippets: &[ShaderSnippet]) -> Cow<'a, str> {
    let prelude = uses_prelude(source);
    let active: Vec<&ShaderSnippet> = snippets.iter().filter(|s| s.used_by(source)).collect();
    if !prelude && active.is_empty() {
        return Cow::Borrowed(source);
    }

    let mut out = String::new();

    // WGSL global directives (`enable`, `requires`) must precede every declaration in the
    // module. Prepending an include therefore *invalidates* a shader that opens with one:
    // the directive lands after the prelude's declarations and the module fails with
    // "written after first global declaration". This is inherent to concatenation-based
    // includes, and it is why `forward_lit.wgsl` — which opens `enable
    // wgpu_binding_array;` — could not compile once it also opted into `pbr_eval`.
    //
    // So directives are hoisted out of the source and re-emitted first. Each hoisted line
    // is replaced by a blank line rather than deleted, which keeps every later line of the
    // original file at its original index and so keeps `expanded_lines`/`expanded_lines_with`
    // a correct diagnostic offset.
    // Only rewritten when a directive is actually present, so every other shader is
    // concatenated byte-for-byte as before.
    let is_directive = |line: &str| line.starts_with("enable ") || line.starts_with("requires ");
    let body: Cow<'_, str> = if source.lines().any(|line| is_directive(line.trim_start())) {
        let mut hoisted = String::with_capacity(source.len());
        for line in source.lines() {
            let trimmed = line.trim_start();
            if is_directive(trimmed) {
                out.push_str(trimmed);
                out.push('\n');
                hoisted.push('\n');
            } else {
                hoisted.push_str(line);
                hoisted.push('\n');
            }
        }
        Cow::Owned(hoisted)
    } else {
        Cow::Borrowed(source)
    };

    if prelude {
        out.push_str(PRELUDE);
        out.push('\n');
    }
    // Snippets follow the prelude, in caller-supplied order — the order a
    // pass lists its own snippets in is that pass's concern, not the core's.
    for snippet in active {
        out.push_str(snippet.source);
        out.push('\n');
    }
    out.push_str(&body);
    Cow::Owned(out)
}

/// Creates a shader module, expanding only the generic prelude if the source
/// opts in. Equivalent to `module_with(device, label, source, &[])`.
pub fn module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    module_with(device, label, source, &[])
}

/// Creates a shader module, expanding the prelude and any of `snippets` the
/// source opts into.
pub fn module_with(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    snippets: &[ShaderSnippet],
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(resolve_with(source, snippets)),
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
        // Every marker, independently and in every combination — the reported
        // offset is what maps a diagnostic back to the file the reader will
        // open, so it has to track whatever `resolve`/`resolve_with` actually
        // prepended.
        // Trailing newlines match real `include_str!`-sourced snippets (every
        // `.wgsl` file on disk ends with one): `push_str(source); push('\n')`
        // then produces a blank-line separator before the body, which is
        // what `expanded_lines_with`'s `+ 1` accounts for.
        const A: ShaderSnippet =
            ShaderSnippet::new("//!use test_a", "// snippet a\n// two lines\n");
        const B: ShaderSnippet = ShaderSnippet::new("//!use test_b", "// snippet b\n");
        let snippets = [A, B];
        for src in [
            "//!use helio_prelude\nfoo",
            "//!use test_a\nfoo",
            "//!use test_b\nfoo",
            "//!use helio_prelude\n//!use test_a\nfoo",
            "//!use helio_prelude\n//!use test_b\nfoo",
            "//!use test_a\n//!use test_b\nfoo",
            "//!use helio_prelude\n//!use test_a\n//!use test_b\nfoo",
        ] {
            let resolved = resolve_with(src, &snippets);
            let offset = resolved.lines().count() - src.lines().count();
            assert_eq!(
                offset,
                expanded_lines_with(src, &snippets),
                "offset wrong for {src:?}"
            );
        }
    }

    #[test]
    fn a_shader_opting_into_neither_is_passed_through() {
        let src = "@compute @workgroup_size(1) fn main() {}";
        assert!(matches!(resolve(src), Cow::Borrowed(_)));
        assert_eq!(expanded_lines(src), 0);
    }

    #[test]
    fn unmatched_snippets_are_not_prepended() {
        const UNUSED: ShaderSnippet = ShaderSnippet::new("//!use never_used", "// never");
        let src = "@compute @workgroup_size(1) fn main() {}";
        assert!(matches!(resolve_with(src, &[UNUSED]), Cow::Borrowed(_)));
    }

    #[test]
    fn snippets_are_appended_in_the_order_given() {
        const FIRST: ShaderSnippet = ShaderSnippet::new("//!use first", "FIRST");
        const SECOND: ShaderSnippet = ShaderSnippet::new("//!use second", "SECOND");
        let src = "//!use first\n//!use second\nbody";
        let resolved = resolve_with(src, &[FIRST, SECOND]);
        let first_pos = resolved.find("FIRST").unwrap();
        let second_pos = resolved.find("SECOND").unwrap();
        assert!(first_pos < second_pos);
    }
}
