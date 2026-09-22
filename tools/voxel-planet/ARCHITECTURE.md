# Validation-tool architecture

The active terrain backend is the `helio-pass-tiny-voxel` crate. This directory
only holds reproducible, CPU-only analysis code so it cannot become a second
renderer or drift from Helio's SceneDB architecture.

The scripts are deliberately split by evidence type:

- capture comparison checks byte and image differences;
- engine timing analysis reports warmed p50/p95/p99 values;
- HDR references accumulate linear samples and compare resolved output;
- TSR diagnostics check history, motion, and unresolved-pixel metadata.

Saved terrain recipes and edits remain authoritative in the runtime crate.
These tools inspect outputs; they do not generate or persist world state.
