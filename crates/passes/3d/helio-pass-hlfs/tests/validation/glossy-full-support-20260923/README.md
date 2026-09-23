# Full-support glossy proposal and small-light GPU shortcut, 2026-09-23

This is a retained incremental renderer change in PR #248. All proposal
construction and light selection stay on the GPU. The native 1440p target,
geometry, material, ray visibility, and output resolution were held fixed.

## Change and reason

The [grid diagnostic](../grid-overflow-20260923/README.md) found that every
coarse tile overflows in the 1,024-static-light gallery. Glossy pixels then
score four sets of 16 global candidates. An earlier shortcut through the
coarse alias table was faster but failed moving-light flicker: the table kept
one randomly selected representative from each four-light stratum. The
[rejected result](../glossy-alias-20260923/README.md) is retained separately.

The GPU coarse builder now stores the four exact proposal weights for each
stratum when the allocated population is at most 1,024. The alias table first
chooses a weighted stratum; a second draw chooses one of its actual lights.
Thus every positive-weight non-key light in a tile remains represented, with the
conditional probability used by the sampler's inverse-PDF correction. The
1/16 uniform proposal component remains for zero-weight or badly represented
receivers. Glossy pixels use four samples with eight candidates each; the
selected light still receives the original BRDF and hardware visibility.
Above 1,024 allocated slots the previous proposal path remains in use.

The proposal row grows from 24 to 40 bytes, adding about 3.6 MiB for the
920 coarse tiles at native 1440p. The GPU key-selection pass also records the
number of active emitters. When at most 32 are active, exact shading never
reads an alias table, so its GPU construction is skipped. Sparse allocated
SceneDB slots are excluded from this decision. No light, triangle, material,
or screen pixel is removed.

## GPU timing

Release binaries on the RTX 3060 used Vulkan timestamps. All medians exclude
frames 0–15. The 1,024-light gallery has **static** lights and a moving camera;
it is not the dynamic-light probe. Each row is a separate 100-frame capture
at native 2560 × 1440 output and HLFS shading, native TSR, hardware RT, one
configured sample and eight candidates. Candidate rows 1 and 2 precede the
small-light shortcut, which is inactive for 1,024 active lights. A final
capture of the complete code gives 8.982 ms sampling, 14.386 ms HLFS and
20.776 ms graph. Raw stage/graph CSVs and configurations are in [gallery](gallery/).

| Capture order | Sampling p50 | HLFS p50 | Full graph p50 |
| --- | ---: | ---: | ---: |
| Control 1 | 9.357 ms | 14.726 ms | 20.994 ms |
| Candidate 1 | 8.892 ms | 14.335 ms | 20.594 ms |
| Candidate 2 | 8.895 ms | 14.426 ms | 20.618 ms |
| Control 2 | 9.452 ms | 14.910 ms | 21.107 ms |

The separate synthetic probe uses 1,024 **moving** lights and 256 moving
single-triangle instances, with 120 warmup and 600 measured frames. Native
1440p glossy presampled RT used one configured sample, eight candidates and
reactive history. In control/candidate/candidate order, HLFS GPU medians were
23.755/22.091/22.050 ms, and serialized frame wall medians were
25.413/23.728/23.588 ms. The wall figures imply roughly 39/42/42 frames per
second for this serialized probe; they are not presented application FPS.
Its geometry and near-uniform view are too simple to validate scene-level
visuals. Full probe logs are in [moving](moving/).

The detailed cathedral has about 415,300 triangles, textured stone, a daylight
sun and 12 interior lights with colored glass transmission. It tests complex
geometry and a small active-light path, separately from the dense gallery.
Each row is one matched capture with two configured samples and candidates.
The whole graph includes other passes, while HLFS does not include the full
application's presentation loop. Raw CSVs/configurations are in
[cathedral](cathedral/).

| Large cathedral | Control HLFS | Candidate HLFS | Control graph | Candidate graph |
| --- | ---: | ---: | ---: | ---: |
| Native 1440p | 5.984 ms | 6.049 ms | 14.462 ms | 14.149 ms |
| Native 4K | 13.930 ms | 13.582 ms | 30.425 ms | 30.054 ms |

The 1440p HLFS difference is small and the cathedral pairs are single runs;
they do not establish a cathedral HLFS speedup. The candidate's GPU coarse
stage was 0.060/0.070 ms at 1440p/4K versus 0.215/0.404 ms in control.

## Visual and regression checks

The final code passed all three fixed four-seed groups (12 seeds total,
including seed 11). Each ran 96 frames with 1,024
moving colored lights, camera motion, blocker and key-light transitions,
native shading, presampling and reactive history. Every final-output,
static-frame motion and spatial grain gate passed. Maximum motion RMS across
the three groups was 0.00832 against the fixed 0.010 limit. CSVs and inspected
seed-11 frames [63](moving/seed11-f63.png), [64](moving/seed11-f64.png),
[80](moving/seed11-f80.png), and [81](moving/seed11-f81.png) are in
[moving](moving/); the complete consecutive sequences remain under ignored
`target/validation/glossy-exactgroup-final-*` on the validation host.

The large cathedral candidate matched the control **byte for byte** at all
seven saved 1440p and six saved 4K moving-camera checkpoints. I inspected
[1440p frame 99](cathedral/candidate-1440p-f099.png) and
[4K frame 49](cathedral/candidate-4k-f049.png) at full scene scale. The final
gallery's consecutive moving-camera frames 96–99 were inspected beside the
control; [candidate 96](gallery/final-candidate-technology-096.png),
[candidate 99](gallery/final-candidate-technology-099.png), and
[control 99](gallery/final-control-technology-099.png) are retained. The
gallery still shows substantial colored mottling, also present in control;
this work does not claim artifact-free moving visuals.

The release library suite passed 3/3, the RT GPU regression suite 17/17,
and the screen-space GPU suite 14/14. Their logs are in [tests](tests/).
The active-emitter count is asserted in the sparse key-selection GPU test.

The requested native 1440p 3–4 ms HLFS target is still unmet, as are final
artifact-free motion validation and a detailed dynamic-light scene. PR #248
remains a draft; this evidence supports only the incremental speed change.
