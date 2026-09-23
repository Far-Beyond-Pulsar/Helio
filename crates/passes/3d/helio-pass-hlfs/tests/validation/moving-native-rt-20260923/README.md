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

`2560x1440-scale1-spp1-c8-frame064.png` is a separate native 1440p stress
capture after 120 warmup frames: 1,024 moving lights, 10,000 moving instances,
and one million instanced triangles from a shared 100-triangle mesh. The flat
receiver shows persistent fine grain. This capture has no full-resolution
reference and does not measure the complete game renderer. The accompanying
600-frame stress run measured TLAS plus HLFS GPU p50/p95 of 11.43/12.55 ms.

The moving cathedral is a separate workload with 12 animated local lights.
Its motion captures do not clear this 1,024-light visual gate.
