//! Parses and validates every `.wgsl` shader in the workspace.
//!
//! wgpu only compiles a shader at `create_shader_module`, i.e. at runtime with a
//! live device. Nothing in `cargo check`, `cargo build`, or `cargo test` looks at
//! WGSL, so a shader that cannot possibly compile ships silently and only fails
//! on the machine that runs it — or never, if the pass is not wired up.
//!
//! That is not hypothetical: `ssr_denoise.wgsl` sat in-tree unable to compile
//! (it negated a `u32`) for as long as it existed, because `SsrPass` never built
//! a pipeline from it.
//!
//! This test walks the repo rather than taking an explicit list, so a new shader
//! is covered the moment it is added.
//!
//! Sources go through `helio_core::shader::resolve_with` first — the same call
//! (modulo which snippets a given pass registers) the runtime makes — so a
//! prelude/snippet-using shader is validated as the GPU will see it, not as
//! the bare file on disk. The snippet list here is every snippet the Helio
//! repo declares (`wgsl_snippet!`); `resolve_with` leaves a marker with no
//! snippet in place, so the test fails on any `//!use` marker the list does
//! not provide rather than reporting an unknown identifier further down.
//!
//! A shader its crate assembles from pieces (shared declarations, a prefix,
//! hook defaults) is validated as that crate compiles it: each such crate
//! offers `validation_sources()` (the file's path and its composed source,
//! built by the function the pass itself uses), and the walk takes that
//! source for the file instead of the bare one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SNIPPETS: &[helio_core::shader::ShaderSnippet] = &[
    helio_pass_hiz::HIZ_SNIPPET,
    helio_frame_api::gbuffer::OUTPUT_SNIPPET,
    helio_frame_api::lighting::INDIRECT_SPECULAR_SNIPPET,
    helio_frame_api::lighting::sky::SKY_SNIPPET,
    helio_frame_api::lighting::sky::SKY_TABLE_SNIPPET,
    helio_frame_api::lighting::sun_transmittance::SNIPPET,
    helio_frame_api::media::SNIPPET,
    helio_frame_api::rays::RAY_TRANSMISSION_SNIPPET,
];

/// Helio's own prelude marker, which `resolve_with` always provides.
const PRELUDE_MARKER: &str = "//!use helio_prelude";

/// Every composed shader, by its file's canonical path.
fn composed_sources() -> HashMap<PathBuf, String> {
    [
        helio_scene_join::validation_sources(),
        helio_scene_derivations::validation_sources(),
        helio_pass_postprocess::validation_sources(),
        helio_pass_perf_overlay::validation_sources(),
    ]
    .into_iter()
    .flatten()
    .map(|(path, source)| {
        let path = Path::new(&path)
            .canonicalize()
            .unwrap_or_else(|e| panic!("composed shader {path}: {e}"));
        (path, source)
    })
    .collect()
}

/// The `//!use` markers of `source` that neither the prelude nor a listed
/// snippet provides.
fn unknown_markers(source: &str) -> Vec<String> {
    source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("//!use "))
        .filter(|line| *line != PRELUDE_MARKER && !SNIPPETS.iter().any(|s| s.marker == *line))
        .map(str::to_string)
        .collect()
}

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn workspace_root() -> PathBuf {
    // crates/helio-core -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("helio-core should live at <root>/crates/helio-core")
        .to_path_buf()
}

fn collect_wgsl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `target` holds vendored deps' shaders, which are not ours to police.
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_wgsl(&path, out);
        } else if path.extension().is_some_and(|e| e == "wgsl") {
            out.push(path);
        }
    }
}

/// A standalone shader has at least one entry point. A file with none is a
/// fragment that gets string-concatenated into a host shader before compiling
/// (e.g. `vhs_effects.wgsl`, pulled in as `VHS_SHADER_SNIPPET`), so it refers to
/// bindings it does not declare and cannot validate on its own.
fn is_fragment(source: &str) -> bool {
    !(source.contains("@vertex") || source.contains("@fragment") || source.contains("@compute"))
}

/// Maps a line number in resolved source back to the original file, so a
/// diagnostic in a prelude-using shader points at a line that actually exists in
/// the file the reader will open.
fn describe_location(source: &str, rendered: &str) -> String {
    let prepended = helio_core::shader::expanded_lines_with(source, SNIPPETS);
    if prepended > 0 {
        format!(" (expanded: subtract {prepended} lines for the original file)")
    } else {
        String::new()
    }
    .to_string()
        + "\n"
        + rendered
}

#[test]
fn every_wgsl_shader_parses_and_validates() {
    let root = workspace_root();
    let mut shaders = Vec::new();
    collect_wgsl(&root.join("crates"), &mut shaders);
    shaders.sort();

    assert!(
        !shaders.is_empty(),
        "found no .wgsl files under {}; the walk is probably broken, and a test \
         that silently checks nothing is worse than no test",
        root.display()
    );

    let mut composed = composed_sources();
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut skipped = Vec::new();

    for path in &shaders {
        let rel = path.strip_prefix(&root).unwrap_or(path);
        let canonical = path.canonicalize().expect("shader path should resolve");
        let source = match composed.remove(&canonical) {
            Some(source) => source,
            None => std::fs::read_to_string(path).expect("shader should be readable"),
        };

        if is_fragment(&source) {
            skipped.push(rel.display().to_string());
            continue;
        }
        checked += 1;

        let unknown = unknown_markers(&source);
        if !unknown.is_empty() {
            failures.push(format!(
                "{}: no snippet in this test's list provides {} (add the owner's \
                 snippet to SNIPPETS)",
                rel.display(),
                unknown.join(", ")
            ));
            continue;
        }

        // Exactly what create_shader_module would receive.
        let resolved = helio_core::shader::resolve_with(&source, SNIPPETS);

        let module = match naga::front::wgsl::parse_str(&resolved) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!(
                    "{}:{}",
                    rel.display(),
                    describe_location(&source, &e.emit_to_string(&resolved))
                ));
                continue;
            }
        };

        if let Err(e) =
            Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)
        {
            failures.push(format!(
                "{}:{}",
                rel.display(),
                describe_location(&source, &format!("{e:?}"))
            ));
            continue;
        }

        // A table indexed through a value is copied whole per invocation
        // (`helio_core::shader::value_array_copies`).
        match helio_core::shader::value_array_copies(&resolved) {
            Ok(copies) if copies.is_empty() => {}
            Ok(copies) => failures.push(format!(
                "{}: indexes arrays through values, copying them per invocation \
                 (read them through their binding):\n  {}",
                rel.display(),
                copies.join("\n  ")
            )),
            Err(e) => failures.push(format!("{}: {e}", rel.display())),
        }
    }

    assert!(
        composed.is_empty(),
        "composed shaders whose file the walk did not find: {:?}",
        composed.keys().collect::<Vec<_>>()
    );
    assert!(
        failures.is_empty(),
        "{} of {checked} standalone shaders failed to compile:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );

    // Guard the skip heuristic: fragments are deliberate source modules and
    // composed shaders. If this trips, the source-module set has grown enough
    // to require an explicit audit. Last audited at 56 (2026-10-28): Radiant
    // templates, shared prelude/common pieces, the voxel planet's source
    // pieces (volumes.wgsl for caves and cliffs, planet_maps.wgsl for the
    // baked maps, joined them), and the virtual geometry's declaration pieces
    // (vg-format, vg-stream, vg-bvh, vg-dense, vg-raster, vg-shade,
    // helio-mats' vg_alpha). 58 (2026-10-30): Ultra-Water's snippets
    // (water-sim's water_sky.wgsl and water_surface_shared.wgsl). Every kind
    // has left the repo since (Phase B), with its pieces.
    assert!(
        skipped.len() <= 58,
        "{} shaders were skipped as fragments/composed modules, which is more than expected:\n{}",
        skipped.len(),
        skipped.join("\n")
    );
}
