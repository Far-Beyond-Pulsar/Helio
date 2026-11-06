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
//! the bare file on disk. The snippet list here (HIZ, the frame's sky and
//! view remaps) is this
//! test's own aggregation of every pass-owned snippet in the workspace, not
//! something `helio-core` itself knows about — `resolve_with` never fails
//! for a marker with no matching snippet, so listing all of them here is
//! harmless for shaders that only use a subset.

use std::path::{Path, PathBuf};

const SNIPPETS: &[helio_core::shader::ShaderSnippet] = &[
    helio_pass_hiz::HIZ_SNIPPET,
    helio_frame_api::lighting::sky::ATMOSPHERE_SNIPPET,
    helio_frame_api::lighting::sky::ATMOSPHERE_SKY_TABLE_SNIPPET,
    helio_frame_api::lighting::sun_transmittance::SNIPPET,
    helio_frame_api::lighting::VIEW_REMAP_SNIPPET,
];

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

/// Assembles the shaders their passes build from more than one piece, the
/// same way the pass does, so they are validated as the GPU sees them.
fn compose(path: &Path, source: String) -> String {
    let normalized = path.to_string_lossy().replace('\\', "/");
    // The scene derivations prefix the shared join declarations
    // (`helio_scene_join::join_source`), wherever their crate keeps them:
    // the hierarchy resolve, the row join, and each kind's own join.
    const JOINS: &[&str] = &[
        "hierarchy_resolve.wgsl",
        "row_join.wgsl",
    ];
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| JOINS.contains(&name))
    {
        return helio_scene_join::join_source(&source);
    }
    // The tone map prefixes the post settings' declarations.
    if normalized.ends_with("helio-pass-postprocess/shaders/postprocess.wgsl") {
        return helio_pass_postprocess::with_post_settings(&source);
    }
    source
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

    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut skipped = Vec::new();

    for path in &shaders {
        let rel = path.strip_prefix(&root).unwrap_or(path);
        let source = std::fs::read_to_string(path).expect("shader should be readable");
        let source = compose(path, source);

        if is_fragment(&source) {
            skipped.push(rel.display().to_string());
            continue;
        }
        checked += 1;

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
        }
    }

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

/// Shaders their crates assemble from pieces (shared declarations, the
/// terrain program, row strides), validated exactly as each crate compiles
/// them.
#[test]
fn composed_shaders_parse_and_validate() {
    let sources = helio_scene_derivations::validation_sources();
    let failures: Vec<String> = sources
        .iter()
        .filter_map(|(name, source)| {
            let source = helio_core::shader::resolve_with(source, SNIPPETS).into_owned();
            let module = match naga::front::wgsl::parse_str(&source) {
                Ok(module) => module,
                Err(e) => {
                    return Some(format!(
                        "{name}:
{}",
                        e.emit_to_string(&source)
                    ))
                }
            };
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .err()
                .map(|e| format!("{name}: {e:?}"))
        })
        .collect();
    assert!(sources.len() >= 2, "{} composed shaders", sources.len());
    assert!(
        failures.is_empty(),
        "{} composed shaders failed:

{}",
        failures.len(),
        failures.join(
            "

"
        )
    );
}
