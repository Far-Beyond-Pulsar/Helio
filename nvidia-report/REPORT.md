# Vulkan: render-target writes intermittently lost in 8x8 / 4x8 pixel blocks (RTX 3060, driver 616.64)

## Summary

In a deferred renderer on Vulkan, a full-screen render pass that writes every
pixel of eight colour attachments and depth sometimes leaves scattered 8x8
and 4x8 pixel blocks of **all** of those targets holding values written
**earlier in the same frame**. In about 1 in 50 frames, roughly 15% of the
frame is affected. The fragment shader provably ran for every pixel.
Khronos synchronization validation reports no hazard. The same application
on D3D12, on the same machine, never shows it. It only happens when the
frame is recorded while the GPU is still executing the previous frame.

## Environment

- GPU: NVIDIA GeForce RTX 3060 (GA106), 12 GB
- Driver: 616.64 (Vulkan, Windows), also shown as 32.0.16.1664 under D3D12
- OS: Windows 11 Pro 10.0.26200
- API: Vulkan through wgpu 30.0.1 (Rust). Ordinary descriptor sets, no
  descriptor buffers, one graphics queue. Submissions are chained with
  binary semaphores (wgpu's relay semaphores).
- Application: Helio, open source: https://github.com/Far-Beyond-Pulsar/Helio
  (commit 27142692; later `main` also reproduces).

## What the frame does

1. **G-buffer pass.** One render pass with 8 colour attachments (RGBA8,
   RGBA16F, RGBA8, RGBA16F, RG16F, RGBA16F, RGBA16F, RGBA16F) and a D32
   depth attachment. All are `LOAD_OP_CLEAR` / `STORE_OP_STORE` (colour to 0,
   depth to 0, reversed-Z).
2. **Voxel terrain work** (several milliseconds). Compute dispatches generate
   terrain into storage buffers. Its own render passes rasterise boxes into a
   separate R32Uint target with a separate D32 depth. More compute traces rays.
3. **Voxel publish.** One render pass with the same 8 colour attachments
   and the same depth, all `LOAD_OP_LOAD` / `STORE_OP_STORE`. It draws one
   full-screen triangle. The fragment shader writes all 8 outputs and
   `frag_depth` (compare `GREATER`, also tried `ALWAYS`).
4. Deferred lighting and post-processing sample those targets.

Between steps 1 and 3 there are proper image barriers
(`COLOR_ATTACHMENT_OUTPUT` / write, then `COLOR_ATTACHMENT_OUTPUT` /
read+write; early|late fragment tests for depth; same layout).

## Expected vs. actual

**Expected:** after the frame, every pixel of the G-buffer and depth holds
what step 3 wrote.

**Actual:** intermittently, blocks of 8x8 pixels (sometimes 4x8, rarely 2x2)
hold step 1's clear values in **every** colour target **and** in depth.
They are scattered at random over about 15% of the screen. Lighting then
treats those blocks as empty sky. See `corrupted_frame.png` against
`reference_frame.png`, and `corrupted_zoom_8x.png`.

## What we measured

- **The writes were issued.**
  - An atomic counter in step 3's fragment shader equals width x height on
    affected frames.
  - A second counter shows no pixel was discarded.
  - After the frame, a compute shader reads the targets: the affected blocks
    hold step 1's clear values, so the data in memory is wrong, not only what
    later passes saw.
- **Earlier writes are what reappear.**
  - If step 1 clears with a distinctive value, or clears by drawing instead
    of a load-op clear, the lost blocks hold exactly that value.
  - If step 3's own render pass does the clear (step 1 loads instead), the
    lost blocks hold step 3's clear value. In other words, its draw writes
    are lost within the same render pass.
  - Data from the previous frame never reappears. Tested by tagging each
    frame's writes by frame parity; zero stale blocks in 8 runs, 520
    captures.
- **No API hazard.**
  - Khronos validation layer, SDK 1.4.363, with synchronization validation
    confirmed enabled (`CURRENT-VALIDATION-ENABLED: Synchronization`): no
    hazards over a full run.
  - Core validation reports an unrelated layout mismatch on a different image
    (a HiZ mip chain); nothing on the G-buffer or depth.
- **D3D12 never shows it:** 0 of 6 runs, against about 55% of runs on Vulkan.
- **It needs frames to overlap.**
  - If each frame waits for the previous one to finish on the GPU before it
    is recorded, it never happens (0 of 6 runs).
  - Recording the frame while the GPU is busy and then waiting for the GPU
    before the submit still shows it.
  - Waiting for the GPU just before the voxel work is recorded also removes
    it (0 of 8).
- **Things that do not change it:**
  - Splitting the frame into many command buffers, or recording it as one.
  - Reusable versus one-time command buffers.
  - Adding barriers at frame start that wait on all of the previous frame's
    colour output, depth tests, compute writes and transfers.
  - Keeping all per-frame buffers persistent instead of allocating them.
  - Moving the clear into step 3's pass, or clearing by drawing.
- **It needs step 2's work.** With step 2 skipped (only steps 1 and 3 every
  frame) it never happens (0 of 6 runs).

This looks like the same issue as
https://forums.developer.nvidia.com/t/382912: same driver, Ampere, writes
lost intermittently, fixed only by waiting for the GPU, barriers no help,
validation clean.

## How to reproduce

```
git clone https://github.com/Far-Beyond-Pulsar/Helio
cd Helio
git checkout 27142692
cargo build --release --example voxel_flight -p helio-default-graphs
```

Run a fixed-step descent at 2560x1440 with captures every 0.25 s (Windows
`cmd`):

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

Each `out\stress_descent_NN.png` is one frame. About half of all runs
produce at least one capture with the blocky haze pattern. The run then
renders a settled reference for each capture (`_ref.png`, `_ref2.png`), and
the `CHUNKGATE descent: sample NN differs ...` lines flag affected captures
at above 0.05. With `HELIO_VOXEL_STRESS_SAMPLE=2` the run is faster but hits
only about 1 run in 7.

We can provide a build with the in-shader write counters and the G-buffer
probe described above, or work on a smaller standalone repro if that helps.

## Attachments

- `corrupted_frame.png`: an affected capture (2560x1440).
- `reference_frame.png`: the same view, rendered settled.
- `corrupted_zoom_8x.png`: a 128x64 region of the affected capture at 8x.
