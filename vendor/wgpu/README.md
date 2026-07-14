# Helio WebGPU wgpu slice

This directory contains the browser WebGPU portion of wgpu 30.0.0 used by Helio.
It intentionally keeps only `wgpu`, `wgpu-types`, and `naga-types`. Native backends,
WebGL, `wgpu-core`, `wgpu-hal`, shader translators, examples, tests, and upstream
workspace tooling are outside Helio's browser runtime target.

The Rust API and generated browser bindings remain derived from the vendored wgpu
revision. Helio enables only the `webgpu` and `wgsl` features and builds this slice
for `wasm32-unknown-unknown`.
