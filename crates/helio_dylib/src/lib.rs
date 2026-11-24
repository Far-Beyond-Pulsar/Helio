//! The renderer as one shared library.
//!
//! This crate contains Helio's kernel crates and every crate they link that no other shared
//! library holds. A binary or plugin library that links it (`use helio_dylib as _;`) gets those
//! crates from this library, so Helio's process-wide state (`helio_core::runtime::Shared`, the
//! shader registry, the pipeline caches, the task host) exists once per process. `wgpu` comes
//! from `wgpui_dylib`, SceneDB from `scenedb_dylib`, the profiler from `profiling_dylib`.
//!
//! Plugin-owned render features and passes (`plugins/vendor/*/feature`) stay in their plugin's
//! library: Helio knows no object kind.

// Linked, not just depended on: a crate this library shares with another shared library must
// come from that library, or it would be in both (rustc: "only shows up once").
use profiling_dylib as _;
use scenedb_dylib as _;
use wgpui_dylib as _;

pub use helio;
pub use helio_components;
pub use helio_core;
pub use helio_feature_api;
pub use helio_frame_api;
pub use helio_graph_asset;
pub use helio_presets;
pub use helio_scene_derivations;
pub use helio_scene_join;
pub use helio_settings;
pub use helio_spine;
pub use helio_tasks;
