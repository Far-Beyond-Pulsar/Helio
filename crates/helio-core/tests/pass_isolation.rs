//! Mechanical backstop for Helio's pass-isolation rule.
//!
//! The registry and graph APIs are intentionally open-ended: adding a pass
//! must not require adding a resource-name literal to `helio-core`. This
//! test is deliberately small and explicit. It catches the concrete
//! regressions identified by the 3.0 audit without pretending that a
//! source-text heuristic can prove all architectural properties.
//!
//! The stronger, AST-based guard (no pass-specific symbol, no unapproved
//! data shape, no `helio-pass-*` production dependency, anywhere in
//! `helio-core`) is `scripts/core-boundary` -- run it via
//! `cargo run --manifest-path scripts/core-boundary/Cargo.toml` from the
//! Helio workspace root. The historical second "core" crate this rule
//! used to also cover has been fully removed: every type it held was
//! either genuinely generic (moved into `helio-core` itself: the resource
//! registry, camera, temporal jitter math, `Movability`, generic asset
//! types) or pass-specific (moved into its owning `helio-pass-*` crate, or
//! into `helio-mats`/`helio-bake-types` for the handful of types shared by
//! several passes without a single natural owner -- see
//! `docs/helio_3_0_migration.md` for the full per-type ownership table).

use std::fs;
use std::path::{Path, PathBuf};

const BANNED_RESOURCE_LITERALS: &[&str] = &[
    "gbuffer",
    "gbuffer_albedo",
    "gbuffer_normal",
    "gbuffer_orm",
    "gbuffer_emissive",
    "billboards",
    "vg",
    "corona_emitters",
    "depth_texture",
    "hiz_sampler",
];

fn source_files(root: &Path, files: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(root).unwrap_or_else(|error| panic!("read {}: {error}", root.display()));
    for entry in entries {
        let entry = entry.expect("read source directory entry");
        let path = entry.path();
        if path.is_dir() {
            source_files(&path, files);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

fn string_literals(source: &str) -> impl Iterator<Item = &str> {
    source
        .split('"')
        .enumerate()
        .filter_map(|(index, part)| (index % 2 == 1).then_some(part))
}

#[test]
fn core_crates_do_not_name_audited_pass_resources() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let roots = [manifest_dir.join("src")];
    let mut files = Vec::new();
    for root in roots {
        source_files(&root, &mut files);
    }

    let mut violations = Vec::new();
    for file in files {
        let source = fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("read {}: {error}", file.display()));
        for literal in string_literals(&source) {
            for banned in BANNED_RESOURCE_LITERALS {
                if literal == *banned || literal.starts_with(&format!("{banned}_")) {
                    violations.push(format!("{} contains {:?}", file.display(), literal));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "pass-isolation violations:\n{}",
        violations.join("\n")
    );
}
