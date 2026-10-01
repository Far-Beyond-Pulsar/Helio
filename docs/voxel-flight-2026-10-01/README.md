# Voxel flight corrections — draft evidence, 2026-10-01

Windows, RTX 3060, 1920x1080 quality mode (1440x810 internal), 0.1 m recipe.
Companion: [Pulsar-Native #994](https://github.com/Far-Beyond-Pulsar/Pulsar-Native/pull/994).

CPU: 32 passed / 2 ignored. Existing GPU: 14 passed. New surface-entry and
alpine-material regressions: 2 passed. Deferred graph/sky/resize: 4 passed.
GPU suites were serial.

The additional graph regression enables temporal reconstruction after initial
construction, switches quality, and disables it again. Runtime quality was
previously lost when reconstructing the graph config, so native selection
could disagree with an initially configured harness. It also verifies that a
material edit and reset of colour history show in one frame, keep resident
columns, and restore the default palette. `PlanetPass::set_appearance` now
reports whether appearance changed so hosts can invalidate history only when
necessary. Pulsar resets temporal history on appearance edits and camera cuts.

The full 5,314-frame offscreen flight at 6b1126aa reported:

| Observation | Result |
|---|---|
| Warm full-graph sync p95 | 9.36 ms |
| Movement sync p99 | 17.85 ms |
| Terrain GPU p95 | 7.61 ms; 5 ms target unmet |
| Descent arrival | 2 frames / 8.16 ms |
| Changed pixels after 250 ms vs settled | 0% |
| Local edit visibility | 25.45 ms |
| Settled loading/exhausted rays | 0 |
| Near-field cell disagreement | 24 / 168,561; not exact agreement |
| Resize resident columns | 708,723 before and after |
| Logical terrain GPU memory | ~541.04 MiB including climate cache |

CPU compilation ran concurrently. This is not isolated native-editor timing.
The memory counter originally omitted the ~1.11 MiB climate cache; that
accounting is corrected in 08a56a92. The final fill adjustment changes
appearance only, and follows the timing run above.

Ground capture at f47bbb5f, final 1.25 hemisphere fill:

![Ground appearance](ground.png)

Mountain capture at 6b1126aa, before the final fill adjustment (0.75 fill):

![Mountain appearance](mountain.png)

These captures demonstrate the current result, including its shortcomings.
They are not production art acceptance. Coarse geometric level transitions,
small distant edit loss, representative shadow errors and worst-case native
flight timing remain open. Fine voxels and all-distance destructibility are
requirements; the current coarse field does not meet them. Palette,
roughness, grass/detail controls and planetary atmosphere are a foundation
for further art direction, not completed AAA visuals. Keep both PRs drafts.
