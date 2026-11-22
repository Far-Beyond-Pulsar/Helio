//! Helio's process-wide state is the host's (`helio_core::runtime`), not a
//! `static` of each copy of Helio a plugin library links. This fails when a
//! Helio crate gains a `static`, `thread_local!` or `inventory::collect!`
//! that `runtime_statics_allowed.txt` does not list, or when a listed one is
//! gone (the list only shrinks).
//!
//! Each line of the list is `path count reason`: why that many statics in that
//! file are harmless in every copy (a value cache whose copies agree, a
//! constant, an environment switch) or are the one pointer to the shared
//! state. State that must be one per process goes in `runtime::Shared`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            // Not code the library links: tests, examples and benches, build output.
            if matches!(
                name.as_ref(),
                "tests" | "examples" | "benches" | "target" | "node_modules"
            ) {
                continue;
            }
            rust_files(&path, out);
        } else if name.ends_with(".rs") && !name.ends_with("_tests.rs") && name != "tests.rs" {
            out.push(path);
        }
    }
}

/// Whether `line` declares process-wide state.
fn declares_state(line: &str) -> bool {
    let line = line.trim_start();
    if line.starts_with("//") {
        return false;
    }
    if line.contains("thread_local!") || line.contains("inventory::collect!") {
        return true;
    }
    let mut rest = line;
    if let Some(after) = rest.strip_prefix("pub") {
        rest = after;
        if rest.starts_with('(') {
            rest = rest.split_once(')').map_or("", |(_, after)| after);
        }
        rest = rest.trim_start();
    }
    let Some(rest) = rest.strip_prefix("static ") else {
        return false;
    };
    let rest = rest.trim_start().trim_start_matches("mut ");
    rest.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

/// The statics in `source`, outside its `#[cfg(test)]` modules (which, in
/// this code base, close the file).
fn count(source: &str) -> usize {
    let code = source.split("#[cfg(test)]").next().unwrap_or(source);
    code.lines().filter(|line| declares_state(line)).count()
}

#[test]
fn helio_holds_no_process_state_outside_its_shared_runtime() {
    let helio = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    rust_files(&helio, &mut files);

    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    for file in files {
        let Ok(source) = fs::read_to_string(&file) else {
            continue;
        };
        let n = count(&source);
        if n > 0 {
            let relative = file
                .strip_prefix(&helio)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            found.insert(relative, n);
        }
    }

    let allowed_text = include_str!("runtime_statics_allowed.txt");
    let mut allowed: BTreeMap<String, usize> = BTreeMap::new();
    for line in allowed_text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, ' ');
        let path = parts.next().unwrap().to_string();
        let n: usize = parts.next().and_then(|n| n.parse().ok()).expect("a count");
        assert!(
            parts.next().is_some_and(|reason| !reason.trim().is_empty()),
            "{path}: say why these are harmless in every copy"
        );
        allowed.insert(path, n);
    }

    let mut problems = Vec::new();
    for (path, n) in &found {
        match allowed.get(path) {
            None => problems.push(format!(
                "{path}: {n} new static(s); state that must be one per process belongs in \
                 helio_core::runtime::Shared, and one harmless in every copy needs a reason in \
                 runtime_statics_allowed.txt"
            )),
            Some(listed) if n > listed => {
                problems.push(format!("{path}: {n} statics, {listed} listed"))
            }
            Some(listed) if n < listed => problems.push(format!(
                "{path}: {n} statics, {listed} listed (the list only shrinks: lower it)"
            )),
            Some(_) => {}
        }
    }
    for path in allowed.keys() {
        if !found.contains_key(path) {
            problems.push(format!("{path}: listed, but it has no statics now: remove the line"));
        }
    }
    assert!(problems.is_empty(), "\n{}\n", problems.join("\n"));
}
