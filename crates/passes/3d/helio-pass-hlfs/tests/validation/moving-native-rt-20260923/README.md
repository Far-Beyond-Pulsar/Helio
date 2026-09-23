# Moving native RT visual gate: open

RTX 3060, Vulkan, NVIDIA driver 616.64. These are GPU captures from the
unchanged production shaders at one sample and eight candidates per pixel,
with tile presampling and reactive history enabled. The captured images are
tone mapped only for inspection.

The 257x145 quality fixture shades at sample scale 1 with 1,024 colored point
lights, no ambient term, and a moving camera. A blocker enters at frame 64;
it leaves as the dominant emitter changes at frame 80. The `reference` images
evaluate the complete light set. `quality.csv` passes its mean-error and NRMSE
thresholds, but the sampled transition images have visible color speckle
against the smooth reference. **Visual acceptance fails.** The numeric test is
only a screening gate.

The `motion/` directory contains 96 consecutive rendered frames for seed 11
as animations: [reactive history](motion/reactive-seed11-96frames.gif) and
[nonreactive history](motion/nonreactive-seed11-96frames.gif). Frames 0–63 keep
the camera, lights and geometry fixed. From frame 64, the camera and lights
move and the blocker enters; at frame 80 the blocker leaves and the dominant
emitter changes. Each animation displays one engine frame for 80 ms so the
flicker and transition can be inspected. The source captures were 257x145
tone-mapped PNGs; the GIFs are visual aids, not timing measurements.

With `HLFS_RT_QUALITY_CAPTURE_MOTION=1`, the fixture now saves every Final
frame and writes `motion-metrics.csv`. It computes the RMS display-RGB change
between consecutive frames 32–63, normalized to 0–1, and screens at 0.01.
The unchanged reactive shader scored **0.03668 (FAIL)** on seed 11 despite
passing the checkpoint mean-error/NRMSE screen. Disabling reactive history
reduced static flicker to **0.00493 (PASS)** but failed ten checkpoint rows
after the blocker entered; see the paired quality and motion CSVs in `motion/`.
Neither setting clears both visual requirements. The flicker number is a
screen for this static interval, not a substitute for viewing the moving frames.
The additional exact/sampled pairs at frames 71 and 79 show that grain remains
throughout the blocked interval, beyond its first frame.

The optional capture also writes `grain-metrics.csv`: it converts the sampled
and exact images to display RGB, subtracts them, removes each residual pixel's
3x3 neighborhood mean, and reports the remaining RMS variation. The 0.005
screen is specific to this smooth-receiver fixture. On seed 11 the current
reactive path measured **0.00869** at steady frame 63, **0.00781** at blocked
frame 71, and **0.00807** at blocked frame 79 (all FAIL); the paired CSVs are
in `motion/`. Nonreactive history measured **0.00391** at frame 63 (PASS) but
still ghosts during the blocker change. This additional screen catches spatial
speckle that the mean-error and static flicker screens can miss. The frames
remain the final visual authority.
Across the four already-used development seeds, the current reactive path
failed the spatial screen on all 32 saved frames and the static flicker screen
on all four sequences; see `reactive-development-*.csv` in `motion/`. These are
regression seeds, not untouched holdouts.

A temporary diagnostic made reactive history active only on the known blocker
event frames 64 and 80. It passed the static flicker and checkpoint screens on
the four fixed development seeds (11, 29, 47, 71); the measurements are in
`motion/diagnostic-event-oracle-*.csv`. This was a hardcoded event oracle,
not a scene-change detector, and the inspected frame-71/79 images still had
visible residual grain against the exact reference. The diagnostic shader was
reverted. The result points toward local change detection and better sampling;
it does not clear the visual or performance gate.

Reproduce the reactive sequence from the repository root with these PowerShell
settings and the ignored GPU test. Change the output directory and remove
`HLFS_RT_QUALITY_REACTIVE` for the nonreactive control. Both runs intentionally
return a failed visual/quality gate after writing the captures and CSVs.

```powershell
$env:HLFS_RT_QUALITY_OUTPUT = 'target/validation/moving-native-sequence'
$env:HLFS_RT_QUALITY_SEED = '11'
$env:HLFS_RT_QUALITY_SETTING = '1:8'
$env:HLFS_RT_QUALITY_SAMPLE_SCALE = '1'
$env:HLFS_RT_QUALITY_PRESAMPLE = '1'
$env:HLFS_RT_QUALITY_REACTIVE = '1'
$env:HLFS_RT_QUALITY_CAMERA_MOTION = '1'
$env:HLFS_RT_QUALITY_DIRECT_ONLY = '1'
$env:HLFS_RT_QUALITY_SWITCH_KEY = '1'
$env:HLFS_RT_QUALITY_CAPTURE_MOTION = '1'
cargo test --release -p helio-pass-hlfs --test gpu_hlfs_rt benchmark_rt_quality_frontier -- --ignored --nocapture
```

`2560x1440-scale1-spp1-c8-frame064.png` is a separate native 1440p stress
capture after 120 warmup frames: 1,024 moving lights, 10,000 moving instances,
and one million instanced triangles from a shared 100-triangle mesh. The flat
receiver shows persistent fine grain. This capture has no full-resolution
reference and does not measure the complete game renderer. The accompanying
600-frame stress run measured TLAS plus HLFS GPU p50/p95 of 11.43/12.55 ms.
An idle-GPU repeat with the matching release test binary measured 11.51/12.62
ms; its 600 timing rows are in `release-idle-repeat-600-frames.csv`. A debug
build of the same workload measured substantially slower and is not comparable
to this release benchmark. These timings exclude the rest of the renderer.
With the native-resolution composite fast path, another matching release
600-frame run measured 11.15/12.32 ms (p50/p95), saved in
`composite-fastpath-release-600-frames.csv`. The composite stage p50 fell from
0.827 to 0.569 ms. The change does not alter sampling or scene content.
The seed-11 moving visual gate still failed: static flicker RMS was 0.03656
against a 0.01 limit, with visible grain at motion checkpoints. Its metrics
are in `motion/composite-fastpath-*-metrics.csv`. Frames 63, 64, 80 and 95
from this run are saved as `motion/composite-fastpath-f*.png` to inspect the
start of camera/blocker motion and the later light change. This speedup does
not make the moving image acceptable.

The moving cathedral is a separate workload with 12 animated local lights.
Its motion captures do not clear this 1,024-light visual gate.
