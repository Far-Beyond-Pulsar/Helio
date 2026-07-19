# Planetary voxel extraction bake-off

Status: complete on 2026-07-19. Decision: promote GPU Transvoxel.

This benchmark decides between the two GPU extraction candidates over the same
32 cubed page plus one-sample halo. It does not redefine the production gate in
`Pulsar-Native/docs/planetary-voxel-terrain.md`.

## Fixed protocol

- Hardware and backend are printed by the executable.
- Build and run in release mode.
- Measure the complete GPU extraction command sequence with timestamp queries.
  Page upload is identical for both candidates and is intentionally outside the
  timestamp interval.
- Interleave candidate order on every iteration to reduce thermal/order bias.
- Run 10 warmups and 50 recorded samples per candidate and case by default.
- Cases are plane, sphere, cave, sharp corner, thin slab, material seam, and a
  fixed-seed dense adversarial density/material page.
- Report median and p95 dirty-page GPU latency per case and pooled across cases.
- Report emitted vertices, triangles, indexed mesh bytes, persistent extractor
  bytes, and sampled RMS/maximum surface error on the six analytic fixtures.
- Sample geometric error at every triangle vertex, edge midpoint, and centroid.
- Run existing same-LOD page tests, 2:1 transition tests, manifold topology
  audits, generation, capacity, resize, and legacy-demo regressions.

## Decision rule

Manifold dual contouring is eligible only if all of these hold:

1. Its pooled dirty-page p95 is no more than 1.25 times Transvoxel's.
2. It reduces pooled indexed mesh bytes or analytic-fixture RMS error by at
   least 10 percent.
3. Its GPU output passes the two-manifold audit.
4. It passes adversarial same-LOD and 2:1 LOD crack tests.

Otherwise Transvoxel is promoted. A missing production 2:1 transition path is
a crack-gate failure, not a benchmark omission. The selected path still needs
render-graph/meshlet integration before end-to-end total frame time can be
claimed; this harness reports the isolated extraction contribution.

Run:

```text
cargo run --release -p helio-pass-planetary-voxel --example extraction_benchmark
```

Optional controls: `HELIO_EXTRACTION_BENCH_WARMUP` and
`HELIO_EXTRACTION_BENCH_SAMPLES`.

## Recorded result

Hardware: NVIDIA GeForce RTX 3060, Vulkan, 10 warmups and 50 recorded samples
per candidate and case, release build.

| Case | Extractor | Median ms | P95 ms | Vertices | Triangles | Indexed bytes | RMS error | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plane | Transvoxel | 0.052224 | 0.057344 | 4,096 | 2,048 | 155,648 | 0 | 0 |
| Plane | Manifold DC | 0.673792 | 1.262592 | 4,225 | 8,192 | 233,504 | 0.000015259 | 0.000015259 |
| Sphere | Transvoxel | 0.048128 | 0.050176 | 1,323 | 661 | 50,268 | 0.016442299 | 0.031267194 |
| Sphere | Manifold DC | 1.158144 | 1.495040 | 1,555 | 2,952 | 85,184 | 0.034192685 | 0.052314570 |
| Cave | Transvoxel | 0.049152 | 0.052224 | 1,311 | 655 | 49,812 | 0.016507972 | 0.031267194 |
| Cave | Manifold DC | 1.164288 | 1.656832 | 1,525 | 2,904 | 83,648 | 0.034594650 | 0.052314570 |
| Sharp corner | Transvoxel | 0.046080 | 0.047104 | 3 | 1 | 108 | 0 | 0 |
| Sharp corner | Manifold DC | 0.855040 | 1.075200 | 19 | 24 | 896 | 0.000015259 | 0.000015259 |
| Thin slab | Transvoxel | 0.052224 | 0.054272 | 4,096 | 2,048 | 155,648 | 0 | 0 |
| Thin slab | Manifold DC | 0.669696 | 0.881664 | 4,225 | 8,192 | 233,504 | 0.000015259 | 0.000015259 |
| Material seam | Transvoxel | 0.051200 | 0.054272 | 4,096 | 2,048 | 155,648 | 0 | 0 |
| Material seam | Manifold DC | 0.668672 | 0.869376 | 4,225 | 8,192 | 233,504 | 0.000015259 | 0.000015259 |
| Dense adversarial | Transvoxel | 0.168960 | 0.172032 | 197,067 | 105,167 | 7,568,148 | n/a | n/a |
| Dense adversarial | Manifold DC | 11.646976 | 12.311552 | 296,641 | 393,928 | 14,219,648 | n/a | n/a |

Pooled results and gates:

- Transvoxel median 0.051200 ms, p95 0.169984 ms.
- Manifold DC median 0.869376 ms, p95 11.719680 ms.
- Manifold/Transvoxel p95 ratio: 68.945783; required <= 1.25: **fail**.
- Indexed mesh-byte ratio: 1.854870; required <= 0.90 for a mesh-size
  improvement: **fail**.
- Analytic-fixture RMS ratio: 2.087642; required <= 0.90 for an error
  improvement: **fail**.
- Persistent extractor allocation: Transvoxel 15,771,248 bytes; manifold DC
  45,928,928 bytes; ratio 2.9122.
- Manifold GPU output passes the same-LOD two-manifold audit.
- Transvoxel passes the existing GPU 2:1 transition extraction and seam tests.
- Manifold DC has no 2:1 transition extractor: **crack gate fail**.

Manifold DC therefore fails every promotion discriminator except its local
topology audit. GPU Transvoxel is the production selection. This decision does
not yet claim end-to-end terrain frame time: meshlet building, publication,
draw, LOD scheduling, and full render-graph integration remain later gates.
