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
//!     helio_core::include_wgsl!("../shaders/ssr_trace.wgsl"),
//! );
//! ```
//!
//! [`include_wgsl!`] is `include_str!` plus the file's identity; with the
//! `shader-hot-reload` feature that identity lets [`hot`] recompile the shader
//! when the file changes. With the feature off it is exactly `include_str!`.
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
#[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
pub mod hot;
pub mod reflection;
#[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
pub use hot::ShaderFile;
pub use directives::{parse as parse_directives, PipelineDirectives};
pub use reflection::{
    create_bind_group_layouts, create_pipeline_layout, create_reflected_bind_groups,
    create_reflected_bind_groups_with_layouts, create_reflected_pipeline, layout_entries,
    reuse_or_create_reflected_bind_groups, CachedReflectedGroup,
    populate_bind_group_entries, reflect, BindingKind, ReflectedBinding, ReflectedLayout,
    ReflectedPipeline, ReflectedShader, ReflectionError,
};

/// The canonical camera struct and depth/G-buffer conventions.
///
/// The one snippet `helio-core` owns directly: every pass in the graph
/// shares exactly one camera/depth-reconstruction convention, so this names
/// no specific pass (like `PassContext::camera` being a first-class field).
pub const PRELUDE: &str = include_str!("prelude.wgsl");

/// [`PRELUDE`] with its on-disk identity, so hot reload can watch and
/// override it like any other shader file.
#[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
pub(crate) const PRELUDE_FILE: ShaderFile = ShaderFile {
    embedded: PRELUDE,
    manifest_dir: env!("CARGO_MANIFEST_DIR"),
    file: file!(),
    rel: "prelude.wgsl",
};

/// Marker opting a shader into the prelude. Must appear in the source.
pub const MARKER: &str = "//!use helio_prelude";

/// Embeds a `.wgsl` file the way `include_str!` does, but with the file's
/// identity attached so the shader can be hot reloaded.
///
/// Use it in place of `include_str!` wherever a shader is handed to
/// [`module`]/[`module_with`]. With the `shader-hot-reload` feature off it
/// *is* `include_str!` (a `&'static str`), so nothing changes.
///
/// ```ignore
/// let shader = helio_core::shader::module(
///     device,
///     "SSR Trace Shader",
///     helio_core::include_wgsl!("../shaders/ssr_trace.wgsl"),
/// );
/// ```
#[cfg(not(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))))]
#[macro_export]
macro_rules! include_wgsl {
    ($rel:expr) => {
        include_str!($rel)
    };
}

/// Embeds a `.wgsl` file the way `include_str!` does, but with the file's
/// identity attached so the shader can be hot reloaded.
///
/// Expands to a [`ShaderFile`](crate::shader::ShaderFile) carrying the
/// embedded text, the crate's manifest dir, `file!()` and the relative path,
/// from which the on-disk file is located at runtime.
#[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
#[macro_export]
macro_rules! include_wgsl {
    ($rel:expr) => {
        $crate::shader::ShaderFile {
            embedded: include_str!($rel),
            manifest_dir: env!("CARGO_MANIFEST_DIR"),
            file: file!(),
            rel: $rel,
        }
    };
}

/// Declares a [`ShaderSnippet`] backed by a `.wgsl` file, so a hot reload of
/// that file changes what the snippet expands to. With the feature off it is
/// `ShaderSnippet::new(marker, include_str!(rel))`.
#[cfg(not(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))))]
#[macro_export]
macro_rules! wgsl_snippet {
    ($marker:expr, $rel:expr) => {
        $crate::shader::ShaderSnippet::new($marker, include_str!($rel))
    };
}

/// Declares a [`ShaderSnippet`] backed by a `.wgsl` file, so a hot reload of
/// that file changes what the snippet expands to.
#[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
#[macro_export]
macro_rules! wgsl_snippet {
    ($marker:expr, $rel:expr) => {
        $crate::shader::ShaderSnippet::from_file($marker, $crate::include_wgsl!($rel))
    };
}

/// Shader text handed to [`module`]/[`module_with`].
///
/// Built from a plain `&str` (never hot reloaded) or, with the
/// `shader-hot-reload` feature, from an [`include_wgsl!`] value that carries
/// the file's identity.
#[derive(Clone, Copy)]
pub struct ShaderSource<'a> {
    /// The text compiled when no hot-reload override exists.
    pub text: &'a str,
    /// On-disk identity of `text`, when it came from `include_wgsl!`.
    #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
    pub file: Option<ShaderFile>,
}

impl<'a> From<&'a str> for ShaderSource<'a> {
    fn from(text: &'a str) -> Self {
        Self {
            text,
            #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
            file: None,
        }
    }
}

impl<'a> From<&'a String> for ShaderSource<'a> {
    fn from(text: &'a String) -> Self {
        text.as_str().into()
    }
}

#[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
impl From<ShaderFile> for ShaderSource<'static> {
    fn from(file: ShaderFile) -> Self {
        Self {
            text: file.embedded,
            file: Some(file),
        }
    }
}

/// Where expansion gets the text of the prelude and of each snippet.
///
/// [`resolve_with`] uses the live source (the embedded text, or the latest
/// hot-reload override); the hot-reload watcher expands against a candidate
/// override set to validate a change before accepting it.
pub(crate) trait TextSource {
    fn prelude(&self) -> Cow<'_, str>;
    fn snippet<'s>(&'s self, snippet: &'s ShaderSnippet) -> Cow<'s, str>;
}

/// The text compiled into the binary.
#[cfg(not(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))))]
struct Embedded;

#[cfg(not(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))))]
impl TextSource for Embedded {
    fn prelude(&self) -> Cow<'_, str> {
        Cow::Borrowed(PRELUDE)
    }

    fn snippet<'s>(&'s self, snippet: &'s ShaderSnippet) -> Cow<'s, str> {
        Cow::Borrowed(snippet.source)
    }
}

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
    /// On-disk identity of `source`, when declared through [`wgsl_snippet!`]
    /// (so hot reload can watch it and override `source`).
    #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
    pub file: Option<ShaderFile>,
}

impl ShaderSnippet {
    /// A snippet whose text is not hot reloadable. Prefer [`wgsl_snippet!`]
    /// for a snippet that lives in a `.wgsl` file.
    pub const fn new(marker: &'static str, source: &'static str) -> Self {
        Self {
            marker,
            source,
            #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
            file: None,
        }
    }

    /// A snippet backed by a `.wgsl` file; see [`wgsl_snippet!`].
    #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
    pub const fn from_file(marker: &'static str, file: ShaderFile) -> Self {
        Self {
            marker,
            source: file.embedded,
            file: Some(file),
        }
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
    expanded_lines_from(source, snippets, live_text())
}

/// [`expanded_lines_with`] against an explicit text source.
pub(crate) fn expanded_lines_from(
    source: &str,
    snippets: &[ShaderSnippet],
    text: &dyn TextSource,
) -> usize {
    let mut lines = 0;
    if uses_prelude(source) {
        lines += text.prelude().lines().count() + 1;
    }
    for snippet in snippets {
        if snippet.used_by(source) {
            lines += text.snippet(snippet).lines().count() + 1;
        }
    }
    lines
}

/// The text source [`resolve_with`] expands against.
#[cfg(not(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))))]
fn live_text() -> &'static dyn TextSource {
    &Embedded
}

/// The text source [`resolve_with`] expands against: embedded text, unless a
/// hot-reload override replaced it.
#[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
fn live_text() -> &'static dyn TextSource {
    &hot::Live
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
    resolve_from(source, snippets, live_text())
}

/// [`resolve_with`] against an explicit text source.
pub(crate) fn resolve_from<'a>(
    source: &'a str,
    snippets: &[ShaderSnippet],
    text: &dyn TextSource,
) -> Cow<'a, str> {
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
        out.push_str(&text.prelude());
        out.push('\n');
    }
    // Snippets follow the prelude, in caller-supplied order — the order a
    // pass lists its own snippets in is that pass's concern, not the core's.
    for snippet in active {
        out.push_str(&text.snippet(snippet));
        out.push('\n');
    }
    out.push_str(&body);
    Cow::Owned(out)
}

/// The text of a shader a pass rewrites in Rust before compiling it (binding
/// array substitutions, tier-specific defines, ...).
///
/// Pass the result to [`module`]/[`module_with`] as a plain `&str`. With the
/// `shader-hot-reload` feature, an [`include_wgsl!`] source is registered with
/// the watcher and this returns the latest accepted on-disk text instead of the
/// embedded text, so the rewrite is applied to the edited file. The file is
/// not validated on the CPU (it is not valid WGSL until rewritten); the
/// host's GPU error scope judges the rebuilt result. With the feature off this
/// is just the embedded text.
pub fn source_text<'a>(label: &str, source: impl Into<ShaderSource<'a>>) -> Cow<'a, str> {
    let source = source.into();
    #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
    {
        hot::current_source(&source, label, &[], false)
    }
    #[cfg(not(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))))]
    {
        let _ = label;
        Cow::Borrowed(source.text)
    }
}

/// Creates a shader module, expanding only the generic prelude if the source
/// opts in. Equivalent to `module_with(device, label, source, &[])`.
///
/// `source` is a `&str` or an [`include_wgsl!`] value; only the latter is
/// hot reloadable.
pub fn module<'a>(
    device: &wgpu::Device,
    label: &str,
    source: impl Into<ShaderSource<'a>>,
) -> wgpu::ShaderModule {
    module_with(device, label, source, &[])
}

/// Creates a shader module, expanding the prelude and any of `snippets` the
/// source opts into.
///
/// With the `shader-hot-reload` feature, an [`include_wgsl!`] source is
/// registered with the watcher and compiled from the latest on-disk text once
/// one has been accepted, instead of the text embedded in the binary.
pub fn module_with<'a>(
    device: &wgpu::Device,
    label: &str,
    source: impl Into<ShaderSource<'a>>,
    snippets: &[ShaderSnippet],
) -> wgpu::ShaderModule {
    let source = source.into();
    #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
    let text = hot::current_source(&source, label, snippets, true);
    #[cfg(not(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))))]
    let text = Cow::Borrowed(source.text);
    // The one sanctioned call: every other site goes through here so the
    // shader is hot reloadable (see clippy.toml).
    #[allow(clippy::disallowed_methods, clippy::let_and_return)]
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(resolve_with(&text, snippets)),
    });
    module
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(all(feature = "shader-hot-reload", not(target_arch = "wasm32"))))]
    #[test]
    fn include_wgsl_is_include_str_when_hot_reload_is_off() {
        // Same type and same text, so every existing `&str` call site keeps
        // compiling and behaving identically.
        let via_macro: &'static str = crate::include_wgsl!("prelude.wgsl");
        assert_eq!(via_macro, include_str!("prelude.wgsl"));
        assert_eq!(via_macro, PRELUDE);
    }

    #[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
    #[test]
    fn include_wgsl_carries_the_embedded_text_when_hot_reload_is_on() {
        let file = crate::include_wgsl!("prelude.wgsl");
        assert_eq!(file.embedded, PRELUDE);
        assert_eq!(file.rel, "prelude.wgsl");
        assert!(file.file.ends_with("mod.rs"));
        let source: ShaderSource<'_> = file.into();
        assert_eq!(source.text, PRELUDE);
        assert!(source.file.is_some());
    }

    #[test]
    fn plain_str_converts_to_a_source() {
        let source: ShaderSource<'_> = "fn x() {}".into();
        assert_eq!(source.text, "fn x() {}");
        let owned = String::from("fn y() {}");
        let source: ShaderSource<'_> = (&owned).into();
        assert_eq!(source.text, "fn y() {}");
    }

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
