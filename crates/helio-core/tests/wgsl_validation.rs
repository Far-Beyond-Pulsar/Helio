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
//! Sources go through `helio_core::shader::resolve` first — the same call the
//! runtime makes — so a prelude-using shader is validated as the GPU will see it,
//! not as the bare file on disk.

use std::path::{Path, PathBuf};

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
    if helio_core::shader::uses_prelude(source) {
        format!(
            " (prelude-expanded: subtract {} lines for the original file)",
            helio_core::shader::prelude_lines()
        )
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

        if is_fragment(&source) {
            skipped.push(rel.display().to_string());
            continue;
        }
        checked += 1;

        // Exactly what create_shader_module would receive.
        let resolved = helio_core::shader::resolve(&source);

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

        if let Err(e) = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)
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

    // Guard the skip heuristic: fragments are rare and deliberate. If this trips,
    // entry-point-less files are proliferating and the skip is hiding real shaders.
    assert!(
        skipped.len() < 5,
        "{} shaders were skipped as fragments, which is more than expected:\n{}",
        skipped.len(),
        skipped.join("\n")
    );
}
