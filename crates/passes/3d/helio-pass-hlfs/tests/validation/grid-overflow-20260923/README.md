# Native gallery light-grid coverage, 2026-09-23

The native 2560 × 1440 technology gallery has 1,024 **static** point lights,
a moving camera, and full-resolution presampled RT. The capture helper now
supports `HLFS_GRID_DIAGNOSTIC=1`, which copies the GPU-built coarse and fine
grid buffers after selected frames and writes `grid-histogram.csv`. Each row
reports how many tiles had a given accepted-light count. A count of
`4294967295` in the fine grid means its coarse tile overflowed. This readback
is opt-in and occurs after the rendered frame; it is not a timing run.

At the production capacities (256 coarse entries, 64 fine entries), **all 920
coarse tiles exceeded capacity at frames 0, 31, 63, and 99**. All 57,600 fine
tiles carried the overflow sentinel. Thus the sampler uses the complete global
1,024-light population at every shaded pixel in this gallery. The coarse counts are
only overflow witnesses: the current GPU builder stops incrementing once
overflow is established, so its reported values do not equal the true light
counts. See [current-capacity-grid-histogram.csv](current-capacity-grid-histogram.csv).

I temporarily raised coarse capacity to 512 and GPU workgroup storage to
match, with no change to light positions, ranges or the camera. At frames
0/31/63/99, 424/543/660/644 of 920 coarse tiles still overflowed. Fine
tiles in those regions continued to use the global light set. Among tiles
whose coarse list fit, the normal 64-entry fine grid also overflowed almost
every nonempty tile. A second diagnostic counted every fine intersection
without storing past 64 entries. At frames 31/63/99 it found 22,270/15,301/
16,225 positive-count fine tiles in the nonoverflow coarse regions; none had
64 or fewer lights, and only 2,812/376/1,053 had 256 or fewer. The maximum
observed counts were 389/372/367. Tiles behind a coarse overflow are still
unknown; the figures do not characterize their fine populations. See the
[512-entry capped](coarse512-capped-grid-histogram.csv) and
[full-count](coarse512-full-fine-counts.csv) histograms.

In one release capture with the temporary 512-entry coarse grid, fine-stage
GPU median rose from the production control's 0.121 ms to 0.428 ms; HLFS
median was 15.241 ms versus the control's 14.733 ms. Sampling remained around
9.4 ms. The runs were not interleaved and selected diagnostic frames had
additional readback after timing, so these are screening measurements, not a
precise performance comparison. Merely raising coarse capacity did not make
the gallery's fine grid useful. The temporary capacity and full-count shader
edits were removed; only the opt-in readback helper is retained.

The release library suite also exposed a stale GPU test setup: its key-light
selection test supplied 128-byte SceneDB light rows directly to the shader's
64-byte compact-light binding. The test now runs the same GPU compaction pass
as the renderer before selection. All three release library tests pass.

This explains a large part of the ineffective candidate-count experiments:
the dense gallery never reaches the fine local-light list, and its glossy
floor evaluates four 16-candidate streams against the global set. A useful
next GPU design must build a more selective proposal distribution or a
compact dense-light representation while maintaining support for every moving
light. Capacity changes alone would require much larger per-tile lists and
more grid work. This static gallery diagnosis does not validate the separate
1,024-**moving**-light RT probe or the cathedral's directional-light path.

Reproduce with the release `technology_gallery_hlfs` capture binary and the
native RT settings from [capture-config.json](capture-config.json), adding
`HLFS_GRID_DIAGNOSTIC=1`. The production source generates the first histogram;
the two 512-entry histograms document rejected local diagnostic variants.
