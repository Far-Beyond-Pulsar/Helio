# Helio 3.0 migration boundary

Helio 3.0 treats the shared `SceneDB` world and its `SceneBufferProjection` as
the authored CPU/GPU world-state authority. Renderer passes receive pass-local
resource views and graph-owned transient bindings; the renderer no longer owns
the retired `generic transient resource registry`/`renderer-owned asset/projection state` contracts or the legacy static
object mutation API.

## Nested-workspace limits

This checkout is the nested Helio workspace. Its Cargo checks cover the members
listed by `crates/helio/Cargo.toml` and the local Helio crates, but they do not
prove that every target in the parent Pulsar-Native workspace builds.

`crates/helio-component` is intentionally excluded from this workspace because
its inventory registration must resolve against the parent Pulsar-Native
workspace. Validate that crate from the parent workspace when changing its
registration or cross-crate integration.

The scene-authority guard is source-based and includes unbuilt examples, while
the focused Cargo checks below validate the migrated graph and facade packages:

```text
pwsh scripts/check_scene_authority.ps1
cargo fmt --all -- --check
cargo check -p helio-core -p helio
cargo test -p helio-core
git diff --check
```

These checks do not replace a full parent-workspace build, platform-specific
WASM/Android builds, or GPU runtime validation.

The repository-wide `cargo fmt --all -- --check` also traverses vendored and
parent-workspace members that are outside this migration. Package-scoped checks
for `helio-core`, `libhelio`, and `helio` are the meaningful formatting signal
for this nested checkout; the full command may still report unrelated existing
formatting differences.
