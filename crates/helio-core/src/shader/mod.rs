//! Shared WGSL prelude.
//! (Module docs continue below the sub-module declarations.)
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
//! # Caveat
//!
//! Prepending shifts line numbers, so naga diagnostics for a prelude-using
//! shader point into the combined source, offset by [`PRELUDE_LINES`]. That is
//! the price of concatenation over a real preprocessor; keeping the prelude small
//! and stable keeps it manageable.

pub mod vt_binder;

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

/// The three-band foliage wind model and the `Wind` uniform layout.
///
/// Separate from [`PRELUDE`] for the same reason as [`HIZ`]: only the foliage
/// passes want it, and every shader that does not would pay for it in shifted
/// diagnostics. Shared rather than per-pass because grass geometry, tree WPO and
/// impostor cards draw the same plant inside a LOD cross-fade band and have to
/// evaluate byte-identical wind to stay in phase.
/// Shared PBR/BRDF evaluation — `fresnel_schlick`, the distribution and geometry terms.
///
/// Lives in `libhelio/shaders/` rather than here because it is the GPU half of the
/// material model `libhelio` already owns on the CPU side.
///
/// Registering a module here is the step that is easy to miss and gives no warning when
/// missed: the `//!use` marker is an ordinary WGSL comment, so an unregistered module
/// leaves the marker inert, the source passes through untouched, and the shader fails at
/// `create_shader_module` with "no definition in scope" for a function that plainly
/// exists. That is exactly how `deferred_lighting.wgsl` and `forward_lit.wgsl` came to be
/// uncompilable — which on this path takes out lighting entirely.
pub const PBR: &str = include_str!("../../../libhelio/shaders/pbr_eval.wgsl");

/// Marker opting a shader into [`PBR`]. Must appear in the source.
pub const PBR_MARKER: &str = "//!use pbr_eval";

pub const WIND: &str = include_str!("foliage_wind.wgsl");

/// Marker opting a shader into the foliage wind model. Must appear in the source.
pub const WIND_MARKER: &str = "//!use helio_foliage_wind";

/// Virtual-texturing sampling library + demand-feedback writes (Helio#238).
///
/// Unlike every module above, [`VT`] OWNS its bindings (`@group(2)` meta rows
/// and the quarter-res density target): the binding layout is part of the
/// shared sampling contract all five raster passes and the decal pass build
/// identically, so a mismatch fails at pipeline creation instead of silently
/// misbinding residency rows. See vt_sample.wgsl's header for the two-rule
/// contract this module exists to serve.
///
/// Registering a module here is the step that is easy to miss — see the
/// caution on [`PBR`] above: an unregistered marker is an inert comment and
/// the shader dies at `create_shader_module` with "no definition in scope"
/// for functions that plainly exist in the source tree.
pub const VT: &str = include_str!("vt_sample.wgsl");

/// Marker opting a shader into the VT sampling library. Must appear in the source.
pub const VT_MARKER: &str = "//!use helio_vt";

/// Returns `true` if `source` opts into the prelude.
pub fn uses_prelude(source: &str) -> bool {
    source.contains(MARKER)
}

/// Returns `true` if `source` opts into the Hi-Z traversal.
pub fn uses_hiz(source: &str) -> bool {
    source.contains(HIZ_MARKER)
}

/// Returns `true` if `source` opts into the foliage wind model.
pub fn uses_wind(source: &str) -> bool {
    source.contains(WIND_MARKER)
}

/// Returns `true` if `source` opts into the PBR evaluation module.
pub fn uses_pbr(source: &str) -> bool {
    source.contains(PBR_MARKER)
}

/// Returns `true` if `source` opts into the VT sampling library.
pub fn uses_vt(source: &str) -> bool {
    source.contains(VT_MARKER)
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
    if uses_pbr(source) {
        lines += PBR.lines().count() + 1;
    }
    if uses_wind(source) {
        lines += WIND.lines().count() + 1;
    }
    if uses_vt(source) {
        lines += VT.lines().count() + 1;
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
    let pbr = uses_pbr(source);
    let wind = uses_wind(source);
    let vt = uses_vt(source);
    if !prelude && !hiz && !pbr && !wind && !vt {
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
    // original file at its original index and so keeps `expanded_lines` a correct
    // diagnostic offset.
    // Only rewritten when a directive is actually present, so every other shader is
    // concatenated byte-for-byte as before.
    let is_directive =
        |line: &str| line.starts_with("enable ") || line.starts_with("requires ");
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
    // After the prelude, so the traversal may lean on it if it ever needs to.
    if hiz {
        out.push_str(HIZ);
        out.push('\n');
    }
    // After the prelude: the BRDF terms take view/normal vectors the prelude's
    // conventions define, and a lighting shader opting into both wants them in
    // that order.
    if pbr {
        out.push_str(PBR);
        out.push('\n');
    }
    // Last. Wind depends on neither of the above today, so the only thing the
    // order decides is whose diagnostics move: appending keeps the prelude and
    // the Hi-Z traversal at the same resolved line numbers whether or not a
    // shader also opts into wind, and leaves room for wind to lean on the
    // prelude (a foliage shader wanting `helio_view_depth` for a distance fade
    // is the obvious future case) without another reshuffle.
    if wind {
        out.push_str(WIND);
        out.push('\n');
    }
    // After wind, before the body: VT owns its own group-2 bindings and leans
    // on nothing above it, so its position only decides diagnostics offsets.
    // It must precede the body because the including shaders' entry points
    // call into it.
    if vt {
        out.push_str(VT);
        out.push('\n');
    }
    out.push_str(&body);
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
        // Every marker, independently and in every combination — the reported
        // offset is what maps a diagnostic back to the file the reader will
        // open, so it has to track whatever `resolve` actually prepended.
        for src in [
            "//!use helio_prelude\nfoo",
            "//!use helio_hiz\nfoo",
            "//!use helio_foliage_wind\nfoo",
            "//!use helio_prelude\n//!use helio_hiz\nfoo",
            "//!use helio_prelude\n//!use helio_foliage_wind\nfoo",
            "//!use helio_hiz\n//!use helio_foliage_wind\nfoo",
            "//!use helio_prelude\n//!use helio_hiz\n//!use helio_foliage_wind\nfoo",
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

    #[test]
    fn wind_declares_the_shared_model() {
        // Same reasoning as the prelude and the Hi-Z include: a rename here
        // breaks the grass, tree-WPO and impostor shaders at `create_shader_module`
        // time, i.e. only on a machine with a GPU and only once the pass is
        // actually built. Pin the names so it surfaces in `cargo test` instead.
        for symbol in [
            "struct Wind",
            "fn helio_wind_sway",
            "fn helio_wind_flutter",
            "fn helio_wind_jitter",
            "fn helio_wind_gust",
            "fn helio_wind_offset",
            "fn helio_wind_noise",
            "fn helio_wind_hash_u32",
        ] {
            assert!(WIND.contains(symbol), "wind include is missing {symbol}");
        }
    }

    #[test]
    fn wind_keeps_time_an_explicit_parameter() {
        // The composed offset must never read the clock out of the uniform: the
        // caller has to be able to ask for `t - dt` to build `prev_clip_position`,
        // and a shader that reads `wind.time_prev_time.x` internally silently
        // reports zero foliage velocity and hands TAA a full-screen smear.
        let offset = WIND
            .split_once("fn helio_wind_offset")
            .expect("wind include should define helio_wind_offset")
            .1;
        assert!(
            offset.contains("time: f32,"),
            "helio_wind_offset must take `time` explicitly"
        );
        assert!(
            !offset.contains("wind.time_prev_time"),
            "helio_wind_offset must not read the clock out of the uniform"
        );
    }

    #[test]
    fn wind_declares_no_bindings_of_its_own() {
        // The including shader owns group/binding, exactly as it does for
        // `camera`. A binding declared here would collide with whatever the
        // foliage passes already have bound and could not be relocated.
        for line in WIND.lines() {
            let line = line.trim_start();
            assert!(
                !line.starts_with("@group"),
                "wind include must not declare bindings: {line}"
            );
        }
    }

    #[test]
    fn vt_declares_the_shared_contract_surface() {
        // Same reasoning as prelude/hiz/wind: a rename breaks all five raster
        // passes and the decal pass at `create_shader_module` time — pin the
        // names so it surfaces here instead.
        for symbol in [
            "struct VtMetaRow",
            "fn vt_sample",
            "fn vt_sample_level",
            "fn vt_feedback_write",
            "fn vt_frame_begin",
            "fn vt_mip_resident",
        ] {
            assert!(VT.contains(symbol), "vt include is missing {symbol}");
        }
    }

    #[test]
    fn vt_bindings_are_exactly_group_two_zero_and_one() {
        // The binding layout IS the shared contract (unlike prelude modules):
        // every including pass builds group 2 from these exact slots. Pin them
        // so an accidental renumber fails here rather than as misbound rows.
        let mut groups = VT
            .lines()
            .filter_map(|l| l.trim().strip_prefix("@group(2) @binding("))
            .map(|rest| rest[..1].to_string())
            .collect::<Vec<_>>();
        groups.sort();
        assert_eq!(groups, ["0", "1"], "VT must own exactly group-2 slots 0 and 1");
    }

    #[test]
    fn vt_expansion_tracks_resolve_like_every_other_module() {
        let src = "//!use helio_vt\nfoo";
        let resolved = resolve(src);
        let offset = resolved.lines().count() - src.lines().count();
        assert_eq!(offset, expanded_lines(src));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn vt_module_compiles_under_naga_when_included() {
        // wgsl_validation walks the tree but skips entry-point-less fragments,
        // which vt_sample.wgsl is — so validate the exact resolved text a
        // including shader would compile here, where a regression fails in
        // `cargo test` instead of on someone's GPU at runtime.
        let src = concat!(
            "//!use helio_vt\n",
            "enable wgpu_binding_array;\n",
            "@group(1) @binding(2) var scene_textures: binding_array<texture_2d<f32>, 256>;\n",
            "@group(1) @binding(3) var scene_samplers: binding_array<sampler, 256>;\n",
            "@fragment\n",
            "fn fs_main(@builtin(position) px: vec4<f32>) -> @location(0) vec4<f32> {\n",
            "    vt_frame_begin(px.xy);\n",
            "    let row = vt_meta[0];\n",
            "    let s = vt_sample(scene_textures[0], scene_samplers[0], row, px.xy * 0.1);\n",
            "    vt_feedback_write(0u, 3u);\n",
            "    return s;\n",
            "}\n",
        );
        let resolved = resolve(src);
        let module = naga::front::wgsl::parse_str(&resolved)
            .unwrap_or_else(|e| panic!("vt-including shader failed to parse: {}", e.emit_to_string(&resolved)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("vt-including shader failed to validate");
    }
}
