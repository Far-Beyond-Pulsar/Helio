# HLFS validation

Validated on 2026-09-05 with an NVIDIA RTX 3060, Vulkan, driver 616.64,
against main revision `c3106597329f8b52386dfe3cd35eda9f0b9fbf7c`.

## Correctness and rendering

The shader test parses and validates all five WGSL programs, including the
optional ray-query variant. Nine explicit GPU regressions cover pipeline
creation, light-buffer growth, tile-list overflow, removal of lights, extinction
without changing the light count, mixed colored/directional lights, offscreen
culling, minimal targets, shadow atlas lookup beyond light index 42, moving
screen-space occluders, isolated one-pixel geometry at half resolution, packed
HDR rounding and strong hidden-light discovery. The exposure regression also
checks that reducing pre-exposure does not remove a perceptible dim light.
The allocation regression creates both full- and half-resolution 1080p targets.

At 65×49, equal-light scenes with 0, 1, 2, 65, 257 and 1,024 lights retain mean
lighting within 3.2% of the full-light reference. The one- and two-light cases
match the reference exactly. The 129-light mixed scene has 0.25% mean error and
3.75% normalized RMSE at full resolution after convergence. Its unfiltered
64-frame mean differs by 0.031%, separating estimator accuracy from denoising
bias. Half-resolution shading has 6.04% mean error and 11.07% normalized RMSE; it is an
optional memory/performance tradeoff rather than the default.

| Full-light reference | Stochastic, full resolution |
| --- | --- |
| ![Mixed reference](mixed-reference.png) | ![Mixed stochastic](mixed.png) |

A light 1,280 times stronger than each local light is first fully shadowed,
then revealed without resetting guiding. Converged mean errors are 5.41% while
occluded and 0.062% after reveal. These are finite-sample denoised results; the
test does not claim zero transient error immediately after the change.

The packing test evaluates 12 values spanning zero, subnormals and HDR through
all 256 rounding ranks for both radiance mantissa sizes and E6M5 squared-luminance
moments. Every result is finite and
within one quantization step. The mean error is bounded by one step / 256.
The thin-surface test allows two R11 rounding steps instead of an RGBA16F
absolute tolerance; it still requires illumination and zero leakage into sky.

The offscreen cathedral path renders 100 moving-camera frames with 17 lights,
640×360 output and the default 0.75 render scale (480×270 internal rendering).
Both the standard HLFS graph and its FXAA variant complete this path without
GPU validation errors.
Frames 31, 63 and 99 have sRGB mean absolute errors of 0.048%, 0.057% and 0.062%
of the output range, respectively, compared with matching full-light captures.
These are image comparisons, not proof that every possible scene is artifact-free.

| Cathedral reference, frame 99 | Stochastic, frame 99 |
| --- | --- |
| ![Cathedral reference](cathedral-reference.png) | ![Cathedral stochastic](cathedral.png) |

The repository's CI commands, `cargo build` and `cargo test`, pass at the root.
The cathedral example builds, and the explicit GPU/shader checks above test the
lighting implementation beyond that root package's checks.

## GPU timing

640×360, default two shadow samples and eight candidates per sample, 16 warmup
frames and 40 measured frames per case. GPU timestamps surround HLFS on the
render encoder. Median / p95 times are milliseconds. This synthetic plane uses
spatially distributed overlapping point lights; shadowed cases use a shared
16×16 fully lit six-layer atlas and enable bounded screen-space contact tracing.
They exercise shader cost, not production shadow-map construction or bandwidth.

| Lights | Shadows | Full-light reference | Stochastic HLFS |
| ---: | :---: | ---: | ---: |
| 64 | No | 1.838 / 2.232 | 2.604 / 3.071 |
| 256 | No | 4.538 / 4.870 | 2.666 / 3.040 |
| 1,024 | No | 15.245 / 15.544 | 2.636 / 2.937 |
| 64 | Yes | 6.481 / 6.793 | 2.749 / 3.305 |
| 256 | Yes | 23.516 / 23.864 | 2.834 / 3.474 |
| 1,024 | Yes | 92.444 / 93.349 | 2.871 / 3.485 |

Low-light-count scenes can be slower because sampling and denoising have fixed
costs. These results compare the new stochastic path with its full-light oracle;
they are not measurements of the old clip-stack implementation or total game
frame time. Desktop scheduling contributes to the observed tail variability.

## Previous-renderer comparison

The previous renderer is built from `c3106597329f8b52386dfe3cd35eda9f0b9fbf7c`
with only the [capture harness patch](baseline-capture.patch). The same cathedral,
100-frame camera path, 17 movable lights, dimensions and shadow quality are used.
Each frame waits for GPU completion; image readback is excluded from the timer.
After discarding 16 warmup frames, two alternating runs give these CPU + GPU
serialized frame-latency medians / p95 values:

| Run | Previous renderer | Stochastic HLFS |
| --- | ---: | ---: |
| 1 | 14.053 / 15.835 ms | 17.281 / 21.323 ms |
| 2 | 16.452 / 18.270 ms | 16.845 / 20.441 ms |

This low-light-count scene does **not** demonstrate an end-to-end speedup.
The high-light-count scaling measurements above are separate synthetic GPU
measurements. Scheduling variability makes small completed-frame differences
inconclusive; this is not a console benchmark.

The previous image also has different indirect illumination from the removed
clip stack. It is retained for comparison, not treated as the lighting oracle:

| Previous renderer, frame 99 | Stochastic, frame 99 |
| --- | --- |
| ![Previous cathedral](cathedral-previous.png) | ![Stochastic cathedral](cathedral.png) |

To reproduce the baseline, create a detached worktree at the revision above,
apply `baseline-capture.patch`, then run the same cathedral capture command from
the pass README. The patch adds the offscreen path and its timer; it does not
modify the previous lighting shaders or graph. Run the baseline and current
executables serially so their GPU work does not overlap.

## Allocation and sampling asset

Requested pass-owned allocations are 12,593,916 bytes (12.01 MiB) at 640×360.
At 1920×1080 the allocation test reports 113,050,116 bytes (107.81 MiB) with
full-resolution shading or 47,545,476 bytes (45.34 MiB) at half resolution.
Packed lighting and metadata reduce the working set from 64 to 40 bytes per
shading pixel. These counts exclude driver
alignment, graph-owned textures, shared scene resources and pipeline memory.
Both 1080p configurations are below the removed 128 MiB clip-stack allocation.
The approximate 34 MB target in the proposal is not reached; output, grids,
geometry rejection and both history parities are included here. Allocation still
scales with resolution, so this does not establish a bound at higher resolutions.

The independently generated 32×32×32 R8 noise asset has exactly 128 entries in
each of its 256 bins. Against a seeded white-noise volume, low-frequency power
ratios are 0.00189 spatially and 0.04878 temporally (spatial FFT radii 1–4 and
temporal bins 1–4). This checks the asset, not independence of every downstream
reservoir decision.

Reproduction commands and platform choices are documented in the
[pass README](../../../crates/passes/3d/helio-pass-hlfs/README.md).

## Proposal coverage and remaining adaptations

| Proposed phase | Implemented behavior and review boundary |
| --- | --- |
| Clip-stack removal and visible lists | Clip stacks are removed. Up to 16 explicit visible IDs are sorted after parallel workgroup selection. Reprojected guiding uses stochastic bilinear tile lookup. Portable workgroup operations provide the fallback allowed by the proposal; no subgroup requirement is introduced. |
| Dual reservoirs | Visible/hidden reservoirs, 20%/50% discovery budgets, directional budget, STBN warping and stratification are implemented. Discovery doubles on disocclusion, capped at 16 candidates. An exact unshadowed bound controls exposure-relative dim-light rejection and its aggregate error. |
| Denoiser | Separate albedo/EnvBRDF-demodulated signals, temporal YCoCg rectification, distance rejection, linear luminance moments and relative-variance spatial filtering are implemented. R11G11B10 bit packing uses portable integer storage images and stochastic rounding. Filter bypass uses exact-set confidence and low temporal variance; it does not estimate the proposal's explicit 80% total-energy coverage threshold. |
| Light grid | Coarse/fine culling uses workgroup staging and current depth bounds. Fixed per-tile capacity has an explicit global fallback. Barn-door culling and area-light permutations depend on unsupported rectangular light types. |
| Screen visibility | Eight bounded steps use current 8×8 depth minima followed by full-resolution depth confirmation and shadow-map fallback. This is a two-level hierarchy, not a general variable-length HZB traversal. |
| Area guiding | Conditional phase: current engine light types are point, spot and directional. No area quadrants or barn doors are invented in the shared light layout. |
| Reduced-resolution sampling | Optional quarter-pixel-count sampling visits a jittered 2×2 pattern. Sparse geometry-weighted reconstruction and direct bounded shading handle unmatched thin surfaces. This intentionally retains the guarded fallback instead of unrectified history reuse; reconstruction combines neighbors rather than selecting a single stochastic bilinear sample. |

The memory reduction is verified at 1080p, but the full-resolution 34 MB estimate
is not met. Console timing, optional hardware-ray execution and the background
reference's volumetric/translucency/ray-reuse extensions are not validated here.
