# Native glossy sampling cost and rejected fast paths, 2026-09-23

This is a diagnostic record for PR #248, not an accepted rendering change. All
experiments ran on the local RTX 3060 through Vulkan at native 2560 × 1440
HLFS shading and output. The technology gallery had 1,024 static lights and a
moving camera. Timings are medians of GPU timestamps after 16 warmup frames in
100-frame release captures. Full graph includes the other renderer passes;
sampling and HLFS do not. These are single pilot captures, so small differences
are not repeatable speed claims. The source for every rejected variant was
restored after measurement.

## Why candidate-count sweeps understated the cost

The gallery's main floor has roughness 0.10. In presampled RT mode,
`sample.wgsl` raises such pixels to four light samples with at least 16
candidates each, even when the example is configured for one sample and eight
candidates. A temporary floor-roughness change to 0.30, with the same geometry,
lights and camera path, changed sampling from 9.484 to 5.043 ms, HLFS from
14.813 to 10.427 ms, and full graph from 21.119 to 16.772 ms. The altered
material is a cost diagnostic; it was restored and is not a quality solution.

## GPU experiments

The control below uses the production shader after the native-pixel fast path.
All captures used the same native 1440p, TSR, RT, presampling, one configured
sample, and eight configured candidates. `capture-config.json` and the raw
timing CSVs for every row are in this directory.

| Single run | Sampling ms | HLFS ms | Full graph ms | Decision |
| --- | ---: | ---: | ---: | --- |
| Control | 9.550 | 14.733 | 21.026 | Reference |
| Four glossy reservoirs sharing 32 proposals | 6.513 | 11.806 | 18.153 | Rejected: motion and grain worsened |
| Cheaper scalar glossy proposal score | 11.986 | 17.278 | 23.475 | Rejected: slower |
| Hidden-only reservoir specialization with unchanged proposals | 9.750 | 15.092 | 21.302 | Rejected: no measured gain |
| Linear-congruential proposal RNG | 9.457 | 14.733 | 20.970 | Rejected: negligible single-run difference |
| Presampled scratch-initialization guard | 9.532 | 14.934 | 21.178 | Rejected: no measured gain |

The shared-proposal experiment scored each light once for four stratified
reservoirs, using 32 proposals total in place of four independent sets of 16.
That cut the sampling cost, but it changed the noise distribution. The fixed
seed-11 glossy fixture uses 1,024 colored lights, a moving camera and a
blocker/key-light transition. Its static-frame display-change RMS rose from
0.008574 to 0.011287, crossing the existing 0.01 motion gate. Spatial residual
RMS at steady frame 63 rose from 0.003386 to 0.003887; at the frame-80
transition it rose from 0.009933 to 0.011267. Both transition images already
failed the 0.005 grain gate in the control, and the variant made them worse.
The full motion and grain CSVs, quality CSVs, test logs, and paired frames
[63 control](glossy-control-f063.png), [63 candidate](glossy-candidate-f063.png),
[80 control](glossy-control-f080.png), and [80 candidate](glossy-candidate-f080.png)
are retained. The test intentionally returned failure after writing evidence.

The scalar score retained the exact selected-light BRDF and ray query, but its
extra shader work cost about 2.4 ms in this pilot. Its glossy motion screen
passed at 0.009129 while the existing frame-80/81 grain failures remained.
The hidden-only variant preserved the control's RNG stream, candidates and
reservoir weights; its seed-11 motion RMS matched the control, with no timing
benefit. The RNG and scratch variants had no material timing gain, so no broader
visual claim is made for them. A separate direct-BRDF algebra experiment also
slowed the gallery in A/B/B/A captures and was removed before this report.

## Research direction

The [RTXDI integration guide](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md)
describes cheap initial sampling followed by surface-dependent scoring and
later visibility; its GPU-built ReGIR cells can feed screen-space ReSTIR to
reduce boiling in distributed-light scenes. Its [application bridge](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RtxdiApplicationBridge.md)
allows an approximate target but warns that a poor target becomes noisy.
That is consistent with rejecting the shared-proposal speedup on motion
evidence. The accessible [Unreal Lumen stochastic direct-light shader](https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Shaders/Private/Lumen/LumenSceneDirectLightingStochastic.usf)
generates samples with one thread per sample before later shadow work. It is
a useful scheduling idea, not a directly comparable benchmark: Lumen shades
surface-cache cards while this fixture shades the native screen.

A next architectural experiment should split the four glossy sample streams
across GPU lanes and fuse their output per pixel, keeping each stream's 16
proposals, BRDF target and light visibility unchanged. Its first gate is exact
or bounded-equivalent captured lighting and the same moving fixture; only then
should full-graph native 1440p and 4K performance be compared. Better GPU
proposal distributions and local temporal change detection remain separate
quality investigations. None of the experiments here meet the native 3–4 ms
HLFS target or clear the moving visual gate.
