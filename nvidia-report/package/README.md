# NVIDIA bug 6939141: Vulkan render-target writes lost in 8x8 / 4x8 blocks

Intermittently, on Vulkan, a full-screen render pass that writes every pixel
of eight colour attachments and depth leaves scattered 8x8 and 4x8 pixel
blocks of **all** of them holding values written **earlier in the same
frame**. In about 1 in 50 frames, around 15% of the frame is affected. The
fragment shader provably ran for every pixel, Khronos synchronization
validation reports no hazard, and the same build on D3D12 never shows it.
It only happens while the GPU is still executing the previous frame.

`REPORT.md` (next to this folder) has the full analysis and measurements.

## What is in this package

| File | What |
|---|---|
| `voxel_flight.exe` | The repro: Helio's voxel planet example at commit `27142692` (the one the report names), release build |
| `run_repro.bat` | One run on Vulkan, about 1.5 minutes plus settled references; captures to `out_vulkan\` |
| `run_dx12.bat` | The same run on D3D12 for comparison; captures to `out_dx12\` |
| `detect_lost_blocks.py` | Optional: finds the affected captures of a run (see below) |
| `hit_616_64_capture.png`, `hit_616_64_reference.png`, `hit_616_64_zoom_8x.png` | A hit from this exact exe on driver 616.64 (2026-10-20): capture 31 of a run, 11.6% of its 8x8 blocks lost; its settled reference; an 8x crop |
| `corrupted_frame.png`, `reference_frame.png`, `corrupted_zoom_8x.png` | The images of the original report |

## Running the exe

Requirements: Windows 10/11, an NVIDIA GPU with Vulkan (reported on an RTX
3060, driver 616.64), about 2 GB free disk per run. No install is needed.
The shaders are compiled into the executable.

1. Double-click `run_repro.bat` (or run it from `cmd`). It flies a scripted,
   fixed-step descent over the planet at 2560x1440 and saves a frame every
   0.25 s of flight time as `out_vulkan\stress_descent_NN.png`. Then it
   renders a settled reference for each one: `_ref.png` and `_ref2.png`,
   the second 30 frames later, which gives the noise floor.
2. Find the affected captures. They show flat grey-purple 8x8 or 4x8
   blocks scattered over the ground, where `_ref.png` shows terrain (see
   `hit_616_64_zoom_8x.png`). High in the descent the ground is hazy, so
   they are hard to see by eye. `python detect_lost_blocks.py out_vulkan`
   (Python with numpy and Pillow) finds them: a lost block is flat, sits on
   the 8x8 grid, and differs from the settled reference by more than the
   reference's own frame-to-frame noise (`_ref2`) and more than the blocks
   around it. With `--noise` it scores the clean `_ref2` frames instead,
   which is the false-positive floor (0 in our runs).
3. Runs on Vulkan hit intermittently: about half of them in the original
   measurements. On 2026-10-20, with this exe and driver 616.64, one of
   five runs hit, with 11.6% of a capture's blocks lost. Run it a few
   times if the first is clean. Each run overwrites `out_vulkan\`, so move
   a hit aside first.
4. `run_dx12.bat` runs the same flight on D3D12 (`WGPU_BACKEND=dx12`). It
   never shows the pattern (0 of 6 runs here).

The run is deterministic in its camera path and terrain. The frames differ
only in timing: the bug needs the CPU to record a frame while the GPU is
still executing the previous one.

## Reproducing from the source repository

Helio is open source: <https://github.com/Far-Beyond-Pulsar/Helio>.
Building needs Rust (stable, via rustup); the first build takes a few
minutes.

```
git clone https://github.com/Far-Beyond-Pulsar/Helio
cd Helio
git checkout 27142692
cargo build --release --example voxel_flight -p helio-default-graphs
```

Then, in `cmd`, from the repository root:

```
set HELIO_VOXEL_FLIGHT_SEED=75
set HELIO_VOXEL_STRESS_AT=0,1,0
set HELIO_VOXEL_FLIGHT_SUN_ELEVATION=32
set HELIO_VOXEL_FLIGHT_STRESS=descent
set HELIO_VOXEL_STRESS_STEP=1
set HELIO_VOXEL_FLIGHT_HOLD_SECONDS=16
set HELIO_VOXEL_STRESS_SAMPLE=0.25
target\release\examples\voxel_flight.exe out 2560 1440
```

With `set WGPU_BACKEND=dx12` the same command runs on D3D12.

The relevant code at that commit:

- The frame graph is in `crates/helio-default-graphs/src/lib.rs`.
- The voxel pass is in `crates/passes/3d/helio-pass-voxel-planet/`. Its full-screen "publish" render pass, step 3 in `REPORT.md`, is `fn publish` in `shaders/chunk_draw.wgsl`.
- The G-buffer pass is in `crates/passes/3d/helio-pass-gbuffer/`.

## What the frame does (short)

1. **G-buffer render pass.** 8 colour attachments (RGBA8, RGBA16F, RGBA8,
   RGBA16F, RG16F, RGBA16F, RGBA16F, RGBA16F) and a D32 depth, all
   `LOAD_OP_CLEAR` / `STORE_OP_STORE`.
2. **Several milliseconds of voxel terrain work.** Compute into storage
   buffers, render passes into a separate R32Uint target with its own D32
   depth, then more compute.
3. **Voxel publish.** The same 8 attachments and depth, `LOAD_OP_LOAD` /
   `STORE_OP_STORE`, one full-screen triangle. The fragment shader writes
   all 8 outputs and `frag_depth`.
4. Deferred lighting and post-processing.

**Expected:** every pixel holds what step 3 wrote.

**Actual:** some 8x8 / 4x8 blocks of every target and of depth still hold
step 1's clear values.

The key measurements, all detailed in `REPORT.md`:

- An atomic counter in step 3's shader equals width x height on affected
  frames, and no pixel is discarded.
- The lost blocks always hold an earlier write from the same frame, never
  data from the previous frame.
- Sync validation is clean.
- D3D12 runs 0 of 6; Vulkan runs about 55%.
- It disappears when each frame waits for the previous one on the GPU
  before recording, or when the step-2 work is removed.

## Contact

We can provide a build with the in-shader write counters and a G-buffer
probe, or work on a smaller standalone repro if that helps.
