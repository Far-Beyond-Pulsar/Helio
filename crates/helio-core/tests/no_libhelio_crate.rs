//! Regression guard: the `libhelio` crate is gone for good.
//!
//! `libhelio` used to centralize every pass's GPU/frame/resource contracts —
//! exactly the closed, pass-name-aware core this workspace's architecture
//! forbids. It was fully removed (see `docs/helio_3_0_migration.md`): every
//! type it held moved either into `helio-core` (if genuinely generic) or
//! into its owning `helio-pass-*` crate (or `helio-mats`/`helio-bake-types`
//! for the few types shared by several passes with no single natural
//! owner). Nothing may reintroduce it, or a same-shaped crate under a new
//! name, as a central registry of pass-specific types.
//!
//! This walks the whole Helio workspace tree (not just `helio-core`), so it
//! catches a reintroduced dependency from *any* crate, not only the core.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    // crates/helio-core -> repo root (the nested Helio workspace, not
    // Pulsar-Native's outer one).
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("helio-core should live at <helio-workspace-root>/crates/helio-core")
        .to_path_buf()
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `target` holds build output (including this test binary's own
            // compiled artifacts, which legitimately embed the string
            // "libhelio" in this very source file's bytes); vendored deps
            // are not ours to police.
            let skip = path
                .file_name()
                .is_some_and(|n| n == "target" || n == "vendor" || n == ".git");
            if !skip {
                walk(&path, out);
            }
        } else {
            out.push(path);
        }
    }
}

#[test]
fn libhelio_crate_directory_does_not_exist() {
    let root = workspace_root();
    assert!(
        !root.join("crates").join("libhelio").exists(),
        "crates/libhelio must stay deleted -- its contents were fully redistributed \
         to helio-core and the owning helio-pass-* crates (see docs/helio_3_0_migration.md)"
    );
}

#[test]
fn nothing_in_the_workspace_references_libhelio() {
    let root = workspace_root();
    let mut files = Vec::new();
    walk(&root.join("crates"), &mut files);
    walk(&root.join("scripts"), &mut files);
    walk(&root.join("crates_other"), &mut files);

    let mut hits = Vec::new();
    for path in &files {
        let is_relevant = path
            .extension()
            .is_some_and(|ext| ext == "rs" || ext == "toml");
        if !is_relevant {
            continue;
        }
        // This test's own source (the string literal "libhelio" appears
        // throughout its doc comments and assertions) is exempt from
        // scanning itself.
        if path.file_name().is_some_and(|n| n == "no_libhelio_crate.rs") {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(path) else {
            continue;
        };
        if source.contains("libhelio") {
            let rel = path.strip_prefix(&root).unwrap_or(path);
            for (line_no, line) in source.lines().enumerate() {
                if line.contains("libhelio") {
                    hits.push(format!("{}:{}: {}", rel.display(), line_no + 1, line.trim()));
                }
            }
        }
    }

    assert!(
        hits.is_empty(),
        "found {} stale reference(s) to the deleted `libhelio` crate:\n{}",
        hits.len(),
        hits.join("\n")
    );
}
