//! Shader hot reload (feature `shader-hot-reload`, native targets only).
//!
//! Editing a `.wgsl` file while the app runs recompiles it and lets the host
//! rebuild its render graph with the new source. No pass carries hot-reload
//! logic: a pass only builds its shader through [`module`](super::module)/
//! [`module_with`](super::module_with) with an
//! [`include_wgsl!`](crate::include_wgsl) source, which is what registers the
//! file here.
//!
//! # Shape
//!
//! * A global registry records every registered shader file (and every
//!   snippet/prelude file they expand against) and holds the *override
//!   table*: path -> latest accepted source text.
//! * [`module_with`](super::module_with) consults the override table, so any
//!   shader created after an accepted change compiles the new text.
//! * [`start_watcher`] spawns a thread that watches the registered files'
//!   directories. A change is read, expanded exactly as the runtime would
//!   (`resolve_with`) and validated on the CPU with naga. If it fails, the
//!   diagnostic is stored (line mapped back to the original file), logged, and
//!   the old source is kept. If it passes, the override is stored, the
//!   [`generation`] is bumped and the dirty flag is set.
//! * The host polls [`take_dirty`] at a frame boundary and rebuilds whatever
//!   owns pipelines (Helio's `Renderer::poll_shader_reload` rebuilds the whole
//!   graph). Phase 1 has no dependency graph: any accepted change rebuilds
//!   everything.
//!
//! # Locating files
//!
//! `include_str!` embeds the text but forgets where it came from.
//! [`include_wgsl!`](crate::include_wgsl) also records the crate's manifest
//! dir, `file!()` and the relative path; [`resolve_in`] joins them against a
//! list of candidate roots (`HELIO_SHADER_ROOT` first, then the manifest dir,
//! then the ancestors of the current dir and executable) until a file exists.
//! A shader that cannot be located is skipped (not hot reloadable) and
//! logged once.
//!
//! # Limits
//!
//! * Shaders with no entry point (fragments concatenated by Rust code) cannot
//!   be validated alone, so they are accepted unvalidated; the host's GPU
//!   error scope is the safety net for those.
//! * Only text built through `include_wgsl!` is reloadable. Runtime-generated
//!   WGSL and plain `&str` sources are not.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use super::{ShaderSnippet, ShaderSource, TextSource, PRELUDE, PRELUDE_FILE};

/// Quiet period after the last file event before a change is processed.
/// Editors emit several events per save (truncate, write, rename).
const DEBOUNCE: Duration = Duration::from_millis(150);

/// A `.wgsl` file embedded into the binary, with enough identity to find it on
/// disk at runtime. Built by [`include_wgsl!`](crate::include_wgsl).
#[derive(Clone, Copy, Debug)]
pub struct ShaderFile {
    /// The text embedded at compile time (`include_str!`).
    pub embedded: &'static str,
    /// `CARGO_MANIFEST_DIR` of the crate that embedded the file.
    pub manifest_dir: &'static str,
    /// `file!()` of the embedding source file.
    pub file: &'static str,
    /// The path given to `include_str!`, relative to `file`.
    pub rel: &'static str,
}

/// A shader edit that failed validation.
#[derive(Clone, Debug)]
pub struct ShaderDiagnostic {
    /// The file that was edited and rejected.
    pub path: PathBuf,
    /// 1-based line in the original file, when it could be determined.
    pub line: Option<u32>,
    /// What naga reported.
    pub message: String,
}

impl fmt::Display for ShaderDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "{}:{}: {}", self.path.display(), line, self.message),
            None => write!(f, "{}: {}", self.path.display(), self.message),
        }
    }
}

/// Why [`validate_wgsl`] rejected a source.
#[derive(Clone, Debug)]
pub struct ShaderError {
    /// 1-based line in the original file; `None` when unknown or inside the
    /// prepended prelude/snippet text.
    pub line: Option<u32>,
    pub message: String,
}

struct ShaderEntry {
    label: String,
    embedded: &'static str,
    snippets: Vec<ShaderSnippet>,
    /// Whether the file is valid WGSL on its own once expanded. `false` for
    /// sources a pass rewrites in Rust before compiling (see
    /// [`source_text`](super::source_text)), which only the GPU can judge.
    validate: bool,
}

type PathKey = (&'static str, &'static str, &'static str);

fn path_key(file: &ShaderFile) -> PathKey {
    (file.manifest_dir, file.file, file.rel)
}

#[derive(Default)]
struct State {
    /// Located path per file identity; `None` caches a failed lookup so it is
    /// only logged once.
    paths: HashMap<PathKey, Option<PathBuf>>,
    /// Latest accepted text per file (shaders, snippets and the prelude).
    overrides: HashMap<PathBuf, String>,
    /// Registered shader files.
    shaders: HashMap<PathBuf, Vec<ShaderEntry>>,
    /// Snippet and prelude files, with their embedded text.
    snippet_files: HashMap<PathBuf, &'static str>,
    /// Last rejected edit per file, cleared when a later edit is accepted.
    diagnostics: HashMap<PathBuf, ShaderDiagnostic>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static GENERATION: AtomicU64 = AtomicU64::new(0);
static DIRTY: AtomicBool = AtomicBool::new(false);
/// Bumped whenever a new file is registered, so the watcher knows to extend
/// its watch set.
static REGISTRY_EPOCH: AtomicU64 = AtomicU64::new(0);
static WATCHER_STARTED: AtomicBool = AtomicBool::new(false);

fn state() -> MutexGuard<'static, State> {
    STATE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl State {
    /// On-disk path of `file`, located once and cached.
    fn path_of(&mut self, file: &ShaderFile) -> Option<PathBuf> {
        if let Some(found) = self.paths.get(&path_key(file)) {
            return found.clone();
        }
        let found = locate(file);
        if found.is_none() {
            log::warn!(
                "[helio-shader] cannot locate `{}` (from {}); it will not be hot reloaded. \
                 Set HELIO_SHADER_ROOT to the directory the source tree is under.",
                file.rel,
                file.file,
            );
        }
        self.paths.insert(path_key(file), found.clone());
        found
    }

    fn register_shader(
        &mut self,
        path: PathBuf,
        file: &ShaderFile,
        label: &str,
        snippets: &[ShaderSnippet],
        validate: bool,
    ) {
        let entries = self.shaders.entry(path).or_default();
        let is_new_path = entries.is_empty();
        if !entries.iter().any(|entry| entry.label == label) {
            entries.push(ShaderEntry {
                label: label.to_owned(),
                embedded: file.embedded,
                snippets: snippets.to_vec(),
                validate,
            });
        }
        let mut changed = is_new_path;

        // The prelude and the snippets this shader may expand against are
        // watched too: a change to any of them rebuilds the shader.
        if let Some(path) = self.path_of(&PRELUDE_FILE) {
            changed |= self.snippet_files.insert(path, PRELUDE).is_none();
        }
        for snippet in snippets {
            let Some(snippet_file) = snippet.file else {
                continue;
            };
            if let Some(path) = self.path_of(&snippet_file) {
                changed |= self
                    .snippet_files
                    .insert(path, snippet_file.embedded)
                    .is_none();
            }
        }
        if changed {
            REGISTRY_EPOCH.fetch_add(1, Ordering::Release);
        }
    }

    /// The text `path` currently expands to: the override if one was accepted,
    /// otherwise the embedded text.
    fn current_text(&self, path: &Path) -> Option<&str> {
        if let Some(text) = self.overrides.get(path) {
            return Some(text.as_str());
        }
        if let Some(entries) = self.shaders.get(path) {
            return entries.first().map(|entry| entry.embedded);
        }
        self.snippet_files.get(path).copied()
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            overrides: self.overrides.clone(),
            paths: self.paths.clone(),
        }
    }
}

/// A frozen copy of the override table, used to expand and validate a
/// candidate change without holding the registry lock (and without the render
/// thread seeing the candidate before it is accepted).
struct Snapshot {
    overrides: HashMap<PathBuf, String>,
    paths: HashMap<PathKey, Option<PathBuf>>,
}

impl Snapshot {
    fn path_of(&self, file: &ShaderFile) -> Option<&PathBuf> {
        self.paths.get(&path_key(file))?.as_ref()
    }

    fn text_for(&self, file: &ShaderFile) -> Option<&String> {
        self.overrides.get(self.path_of(file)?)
    }
}

impl TextSource for Snapshot {
    fn prelude(&self) -> Cow<'_, str> {
        match self.text_for(&PRELUDE_FILE) {
            Some(text) => Cow::Borrowed(text.as_str()),
            None => Cow::Borrowed(PRELUDE),
        }
    }

    fn snippet<'s>(&'s self, snippet: &'s ShaderSnippet) -> Cow<'s, str> {
        match snippet.file.as_ref().and_then(|file| self.text_for(file)) {
            Some(text) => Cow::Borrowed(text.as_str()),
            None => Cow::Borrowed(snippet.source),
        }
    }
}

/// The text source `resolve_with` uses: embedded text unless an override was
/// accepted.
pub(super) struct Live;

impl TextSource for Live {
    fn prelude(&self) -> Cow<'_, str> {
        live_text(&PRELUDE_FILE)
    }

    fn snippet<'s>(&'s self, snippet: &'s ShaderSnippet) -> Cow<'s, str> {
        match snippet.file {
            Some(file) => live_text(&file),
            None => Cow::Borrowed(snippet.source),
        }
    }
}

fn live_text(file: &ShaderFile) -> Cow<'static, str> {
    let mut state = state();
    let override_text = state
        .path_of(file)
        .and_then(|path| state.overrides.get(&path).cloned());
    match override_text {
        Some(text) => Cow::Owned(text),
        None => Cow::Borrowed(file.embedded),
    }
}

/// Registers `source` (if it came from `include_wgsl!`) and returns the text
/// to compile: the latest accepted override, otherwise the embedded text.
///
/// `validate` is `false` for text a pass rewrites before compiling, which is
/// not valid WGSL until rewritten (so the watcher cannot judge it on the CPU).
pub(super) fn current_source<'a>(
    source: &ShaderSource<'a>,
    label: &str,
    snippets: &[ShaderSnippet],
    validate: bool,
) -> Cow<'a, str> {
    let Some(file) = source.file else {
        return Cow::Borrowed(source.text);
    };
    let mut state = state();
    let Some(path) = state.path_of(&file) else {
        return Cow::Borrowed(source.text);
    };
    state.register_shader(path.clone(), &file, label, snippets, validate);
    match state.overrides.get(&path) {
        Some(text) => Cow::Owned(text.clone()),
        None => Cow::Borrowed(source.text),
    }
}

/// Finds the on-disk file for an `include_wgsl!` identity.
fn locate(file: &ShaderFile) -> Option<PathBuf> {
    let mut bases = Vec::new();
    if let Some(root) = std::env::var_os("HELIO_SHADER_ROOT") {
        bases.push(PathBuf::from(root));
    }
    bases.push(PathBuf::from(file.manifest_dir));
    if let Ok(cwd) = std::env::current_dir() {
        bases.extend(cwd.ancestors().map(Path::to_path_buf));
    }
    if let Ok(exe) = std::env::current_exe() {
        bases.extend(exe.ancestors().skip(1).map(Path::to_path_buf));
    }
    resolve_in(&bases, file.file, file.rel)
}

/// Resolves `rel` (relative to the directory of `file`, as `include_str!`
/// reads it) to a canonical on-disk path.
///
/// `file!()` is relative to wherever the compiler was invoked, which is not
/// necessarily a directory the running app can see, so each base is tried
/// with progressively fewer leading components of `file`'s directory
/// (`crates/passes/x/src`, then `passes/x/src`, ... down to `src`). With the
/// crate's manifest dir as a base the shortest suffix is enough.
pub fn resolve_in(bases: &[PathBuf], file: &str, rel: &str) -> Option<PathBuf> {
    let dir = Path::new(file).parent()?;
    if dir.is_absolute() {
        if let Ok(found) = dir.join(rel).canonicalize() {
            return Some(found);
        }
    }
    let components: Vec<_> = dir
        .components()
        .filter(|component| matches!(component, Component::Normal(_)))
        .collect();
    for base in bases {
        for skip in 0..components.len().max(1) {
            let mut candidate = base.clone();
            for component in &components[skip.min(components.len())..] {
                candidate.push(component.as_os_str());
            }
            candidate.push(rel);
            if let Ok(found) = candidate.canonicalize() {
                return Some(found);
            }
        }
    }
    None
}

/// Whether `source` has an entry point and so can be validated standalone.
/// A file with none is a fragment concatenated into a host shader by Rust
/// code, and refers to bindings it does not declare.
fn has_entry_point(source: &str) -> bool {
    source.contains("@vertex") || source.contains("@fragment") || source.contains("@compute")
}

/// Parses and validates `resolved` (what the GPU would be given) with naga.
///
/// `line_offset` is the number of lines expansion prepended ahead of the
/// original file; reported lines are mapped back by subtracting it, so they
/// point into the file the author edited.
pub fn validate_wgsl(resolved: &str, line_offset: usize) -> Result<(), ShaderError> {
    let to_original = |line: u32| -> Option<u32> {
        (line as usize)
            .checked_sub(line_offset)
            .filter(|line| *line > 0)
            .map(|line| line as u32)
    };
    // Naga's rendered report quotes expanded-source line numbers, which are
    // only meaningful to the author when nothing was prepended.
    let describe = |short: String, rendered: String, line: Option<u32>| -> ShaderError {
        let original = line.and_then(to_original);
        let message = if line_offset == 0 {
            rendered
        } else if line.is_some() && original.is_none() {
            format!("{short} (inside the prepended prelude/snippet text)")
        } else {
            short
        };
        ShaderError {
            line: original,
            message,
        }
    };

    let module = naga::front::wgsl::parse_str(resolved).map_err(|error| {
        let line = error.location(resolved).map(|location| location.line_number);
        describe(error.to_string(), error.emit_to_string(resolved), line)
    })?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| {
        let line = error
            .spans()
            .find(|context| context.0.is_defined())
            .map(|context| context.0.location(resolved).line_number);
        describe(error.as_inner().to_string(), error.emit_to_string(resolved), line)
    })?;
    Ok(())
}

/// A shader file or snippet that should be re-expanded for a candidate change.
struct Affected {
    path: PathBuf,
    label: String,
    text: String,
    snippets: Vec<ShaderSnippet>,
    validate: bool,
}

/// Processes a change to `path` (already canonical): reads it, validates every
/// shader it affects against the candidate text, and accepts or rejects it.
fn handle_change(path: &Path) {
    {
        let state = state();
        if !state.shaders.contains_key(path) && !state.snippet_files.contains_key(path) {
            return;
        }
    }
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            // Atomic-save editors briefly remove the file; the following
            // event delivers the real change.
            log::debug!("[helio-shader] cannot read {}: {error}", path.display());
            return;
        }
    };

    let (snapshot, affected) = {
        let state = state();
        if state.current_text(path) == Some(text.as_str()) {
            return; // touched but unchanged
        }
        let mut snapshot = state.snapshot();
        snapshot.overrides.insert(path.to_path_buf(), text.clone());
        let is_snippet = state.snippet_files.contains_key(path);

        let mut affected = Vec::new();
        for (shader_path, entries) in &state.shaders {
            let source = if shader_path == path {
                text.as_str()
            } else {
                state.current_text(shader_path).unwrap_or("")
            };
            for entry in entries {
                let depends = shader_path == path
                    || (is_snippet && depends_on(&snapshot, source, &entry.snippets, path));
                if depends {
                    affected.push(Affected {
                        path: shader_path.clone(),
                        label: entry.label.clone(),
                        text: source.to_owned(),
                        snippets: entry.snippets.clone(),
                        validate: entry.validate,
                    });
                }
            }
        }
        (snapshot, affected)
    };

    for item in &affected {
        if !item.validate || !has_entry_point(&item.text) {
            continue;
        }
        let resolved = super::resolve_from(&item.text, &item.snippets, &snapshot);
        let offset = resolved.lines().count().saturating_sub(item.text.lines().count());
        if let Err(error) = validate_wgsl(&resolved, offset) {
            let diagnostic = ShaderDiagnostic {
                path: item.path.clone(),
                line: error.line,
                message: error.message,
            };
            log::error!(
                "[helio-shader] rejected edit to `{}` (keeping the previous version): {diagnostic}",
                item.label
            );
            state().diagnostics.insert(item.path.clone(), diagnostic);
            return;
        }
    }

    {
        let mut state = state();
        state.overrides.insert(path.to_path_buf(), text);
        state.diagnostics.remove(path);
        for item in &affected {
            state.diagnostics.remove(&item.path);
        }
    }
    let generation = GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
    DIRTY.store(true, Ordering::Release);
    log::info!(
        "[helio-shader] accepted {} (generation {generation}, {} shader(s) affected)",
        path.display(),
        affected.len()
    );
}

/// Whether a shader with `source` expands against the prelude/snippet at
/// `path`.
fn depends_on(
    snapshot: &Snapshot,
    source: &str,
    snippets: &[ShaderSnippet],
    path: &Path,
) -> bool {
    if super::uses_prelude(source)
        && snapshot.path_of(&PRELUDE_FILE).map(PathBuf::as_path) == Some(path)
    {
        return true;
    }
    snippets.iter().any(|snippet| {
        snippet.used_by(source)
            && snippet
                .file
                .as_ref()
                .and_then(|file| snapshot.path_of(file))
                .map(PathBuf::as_path)
                == Some(path)
    })
}

/// Monotonic count of accepted shader changes.
pub fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// Returns `true` once per accepted change (or batch of changes), clearing the
/// flag. The host calls this at a frame boundary and rebuilds on `true`.
pub fn take_dirty() -> bool {
    DIRTY.swap(false, Ordering::AcqRel)
}

/// The most recent rejected edit per file, for display in an editor UI.
/// An entry disappears when a later edit to that file is accepted.
pub fn last_errors() -> Vec<ShaderDiagnostic> {
    let mut errors: Vec<_> = state().diagnostics.values().cloned().collect();
    errors.sort_by(|a, b| a.path.cmp(&b.path));
    errors
}

/// Number of shader files registered so far (hot-reloadable ones only).
pub fn registered_shader_count() -> usize {
    state().shaders.len()
}

/// Installs `text` as the accepted source of `path` without going through the
/// watcher or validation, bumping the generation and setting the dirty flag.
/// For tools and tests that drive reloads themselves.
pub fn set_override(path: impl Into<PathBuf>, text: impl Into<String>) {
    state().overrides.insert(path.into(), text.into());
    GENERATION.fetch_add(1, Ordering::AcqRel);
    DIRTY.store(true, Ordering::Release);
}

/// Starts the file watcher thread. Idempotent; returns `true` if this call
/// started it.
///
/// It watches the directory of every registered shader/snippet (extended as
/// more are registered) and, recursively, each of `extra_roots`. Only `*.wgsl`
/// events matter.
pub fn start_watcher(extra_roots: Vec<PathBuf>) -> bool {
    if WATCHER_STARTED.swap(true, Ordering::AcqRel) {
        return false;
    }
    let spawned = std::thread::Builder::new()
        .name("helio-shader-watch".into())
        .spawn(move || watch_loop(extra_roots));
    match spawned {
        Ok(_) => true,
        Err(error) => {
            log::error!("[helio-shader] could not start watcher thread: {error}");
            false
        }
    }
}

fn watch_loop(extra_roots: Vec<PathBuf>) {
    use notify::{RecursiveMode, Watcher};

    let (tx, rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher = match notify::recommended_watcher(tx) {
        Ok(watcher) => watcher,
        Err(error) => {
            log::error!("[helio-shader] file watcher unavailable: {error}");
            return;
        }
    };
    for root in &extra_roots {
        if let Err(error) = watcher.watch(root, RecursiveMode::Recursive) {
            log::warn!("[helio-shader] cannot watch {}: {error}", root.display());
        }
    }

    let mut watched: HashSet<PathBuf> = HashSet::new();
    let mut synced_epoch = u64::MAX;
    let mut pending: HashSet<PathBuf> = HashSet::new();
    let mut last_event: Option<Instant> = None;
    log::info!("[helio-shader] hot reload watcher started");

    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Ok(event)) => {
                if event.kind.is_modify() || event.kind.is_create() {
                    for path in event.paths {
                        if path.extension().is_some_and(|ext| ext == "wgsl") {
                            if let Ok(canonical) = path.canonicalize() {
                                pending.insert(canonical);
                                last_event = Some(Instant::now());
                            }
                        }
                    }
                }
            }
            Ok(Err(error)) => log::warn!("[helio-shader] watch error: {error}"),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Passes register shaders as the graph is built (and rebuilt), so the
        // set of directories to watch grows over time.
        let epoch = REGISTRY_EPOCH.load(Ordering::Acquire);
        if epoch != synced_epoch {
            synced_epoch = epoch;
            let dirs: HashSet<PathBuf> = {
                let state = state();
                state
                    .shaders
                    .keys()
                    .chain(state.snippet_files.keys())
                    .filter_map(|path| path.parent().map(Path::to_path_buf))
                    .collect()
            };
            for dir in dirs {
                if watched.contains(&dir) {
                    continue;
                }
                match watcher.watch(&dir, RecursiveMode::NonRecursive) {
                    Ok(()) => {
                        watched.insert(dir);
                    }
                    Err(error) => {
                        log::warn!("[helio-shader] cannot watch {}: {error}", dir.display());
                    }
                }
            }
        }

        if let Some(at) = last_event {
            if at.elapsed() >= DEBOUNCE {
                last_event = None;
                for path in pending.drain() {
                    handle_change(&path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::{expanded_lines_with, resolve_with};

    /// This very file, used as a stand-in shader: it exists on disk, so path
    /// resolution succeeds without shipping a fixture.
    const SELF_FILE: ShaderFile = ShaderFile {
        embedded: "EMBEDDED TEXT",
        manifest_dir: env!("CARGO_MANIFEST_DIR"),
        file: file!(),
        rel: "hot.rs",
    };

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("helio_hot_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn override_beats_embedded_text() {
        let source = ShaderSource::from(SELF_FILE);
        let before = current_source(&source, "test", &[], true);
        assert_eq!(before, "EMBEDDED TEXT");

        let path = state().path_of(&SELF_FILE).expect("hot.rs should be locatable");
        set_override(path.clone(), "OVERRIDDEN TEXT");
        let after = current_source(&source, "test", &[], true);
        assert_eq!(after, "OVERRIDDEN TEXT");

        state().overrides.remove(&path);
        assert_eq!(current_source(&source, "test", &[], true), "EMBEDDED TEXT");
    }

    #[test]
    fn plain_str_sources_are_never_overridden() {
        let source = ShaderSource::from("fn plain() {}");
        assert_eq!(current_source(&source, "plain", &[], true), "fn plain() {}");
    }

    #[test]
    fn snippet_override_changes_expansion() {
        const SNIPPET_FILE: ShaderFile = ShaderFile {
            embedded: "// embedded snippet",
            manifest_dir: env!("CARGO_MANIFEST_DIR"),
            file: file!(),
            rel: "directives.rs",
        };
        let snippet = ShaderSnippet::from_file("//!use hot_test_snippet", SNIPPET_FILE);
        let src = "//!use hot_test_snippet\nbody";
        assert!(resolve_with(src, &[snippet]).contains("// embedded snippet"));

        let path = state().path_of(&SNIPPET_FILE).expect("directives.rs should be locatable");
        set_override(path.clone(), "// overridden snippet");
        let resolved = resolve_with(src, &[snippet]);
        state().overrides.remove(&path);
        assert!(resolved.contains("// overridden snippet"));
        assert!(!resolved.contains("// embedded snippet"));
    }

    #[test]
    fn path_resolution_finds_the_file_from_a_crate_relative_suffix() {
        let root = scratch_dir("resolve");
        let crate_dir = root.join("some-pass");
        std::fs::create_dir_all(crate_dir.join("src")).unwrap();
        std::fs::create_dir_all(crate_dir.join("shaders")).unwrap();
        std::fs::write(crate_dir.join("shaders").join("a.wgsl"), "// a").unwrap();

        // `file!()` as the compiler reports it, relative to a workspace root
        // that is not the base: only the `src` suffix exists under the
        // manifest dir.
        let found = resolve_in(
            &[crate_dir.clone()],
            "crates/passes/3d/some-pass/src/lib.rs",
            "../shaders/a.wgsl",
        )
        .expect("suffix match should resolve");
        assert_eq!(found, crate_dir.join("shaders").join("a.wgsl").canonicalize().unwrap());

        assert!(resolve_in(&[crate_dir], "src/lib.rs", "../shaders/missing.wgsl").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn validation_accepts_a_good_shader() {
        let src = "@compute @workgroup_size(1) fn main() {}";
        assert!(validate_wgsl(src, 0).is_ok());
    }

    #[test]
    fn validation_reports_the_original_file_line() {
        // Prelude-using shader with the mistake on line 4 of the file the
        // author edits; naga sees it after the prelude's lines.
        let original = "//!use helio_prelude\n\n@compute @workgroup_size(1)\nfn main() { let x: u32 = ; }\n";
        let resolved = resolve_with(original, &[]);
        let offset = resolved.lines().count() - original.lines().count();
        assert_eq!(offset, expanded_lines_with(original, &[]));
        assert!(offset > 0, "the prelude must have been prepended");

        let error = validate_wgsl(&resolved, offset).expect_err("broken WGSL must be rejected");
        assert_eq!(error.line, Some(4), "message: {}", error.message);
    }

    #[test]
    fn validation_rejects_semantic_errors() {
        // Parses, but negating a u32 does not validate.
        let src = "@compute @workgroup_size(1)\nfn main() {\n    let a: u32 = 1u;\n    let b = -a;\n}\n";
        let error = validate_wgsl(src, 0).expect_err("semantic error must be rejected");
        assert_eq!(error.line, Some(4), "message: {}", error.message);
    }

    #[test]
    fn fragments_without_entry_points_are_not_validated_standalone() {
        assert!(!has_entry_point("fn helper() -> f32 { return unknown_binding; }"));
        assert!(has_entry_point("@fragment fn fs() {}"));
    }
}
