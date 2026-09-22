# Validation of the stored-terrain path

The CPU-only checks in this directory cover capture parsing, linear HDR
reference arithmetic, motion-reference comparisons, and TSR diagnostic
metadata. Run all three test scripts from the repository root.

GPU timing and visual acceptance remain renderer-level checks. Report warmed
percentiles for matched full-frame workloads, and inspect captures for exact
nearby voxel faces, stable distant filtering, and zero unresolved rays. A
passing script or crate build alone does not establish whole-application
60 FPS or visual quality.

The old standalone viewer and its legacy `libhelio` scene integration were
removed during the Helio 3.0 rebase. Keeping that dead path would make the
branch fail the workspace's `no_libhelio_crate` guard and create a second
runtime architecture.
